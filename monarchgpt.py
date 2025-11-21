import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
import sys
import os
import tiktoken 

# --- Hyperparameters ---
batch_size = 8
local_block_size = 64 
manager_ctx_len = 32   
max_iters = 300000       
eval_interval = 5000
learning_rate = 3e-4    
min_lr = 1e-5           
warmup_iters = 1000     
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout = 0.2

# Dimensions
dim_worker = 256       
dim_manager = 384      

# Bucket Settings
num_buckets = 64       
num_heads = 4          
commitment_cost = 0.5 
coherence_alpha = 0.1  

# --- SKIM MEMORY SETTINGS ---
skim_capacity = 16384  # 16k slots * 64 tokens/block = ~1 Million tokens context
skim_top_k = 32        # Only attend to the top 32 most relevant past moments
skim_dim = dim_manager # The memory stores manager states

torch.set_float32_matmul_precision('high')

# --- 0. RoPE ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        if seq_len > self.cos_cached.shape[2]:
            return self.cos_cached, self.sin_cached # Fallback or dynamic resize could go here
        return self.cos_cached[..., :seq_len, :], self.sin_cached[..., :seq_len, :]

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # Basic broadcast fix for different sequence lengths if necessary
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

# --- 1. Efficient HBSL Layer ---
class HierarchicalBlockShuffleLayer(nn.Module):
    def __init__(self, dim, block_size=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        self.num_groups = (dim + block_size - 1) // block_size
        self.padded_dim_groups = self.num_groups * self.block_size
        self.block_weights = nn.Parameter(torch.randn(self.num_groups, self.block_size, self.block_size))
        nn.init.kaiming_uniform_(self.block_weights, a=math.sqrt(5))
        
        sqrt_dim_ceil = math.ceil(math.sqrt(dim))
        self.padded_dim_monarch = sqrt_dim_ceil * sqrt_dim_ceil
        self.sqrt_dim = sqrt_dim_ceil
        self.W_row = nn.Parameter(torch.randn(self.sqrt_dim, self.sqrt_dim))
        self.W_col = nn.Parameter(torch.randn(self.sqrt_dim, self.sqrt_dim))
        nn.init.kaiming_uniform_(self.W_row, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_col, a=math.sqrt(5))
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        orig_shape = x.shape
        x = x.reshape(-1, self.dim)
        
        pad_amt = self.padded_dim_groups - self.dim
        x_padded = F.pad(x, (0, pad_amt)) if pad_amt > 0 else x
        x_grouped = x_padded.reshape(-1, self.num_groups, self.block_size)
        y_perm = torch.bmm(x_grouped.permute(1, 0, 2), self.block_weights)
        y_mixed = y_perm.permute(1, 0, 2).reshape(-1, self.padded_dim_groups)
        if pad_amt > 0: y_mixed = y_mixed[..., :self.dim]
        block_out = torch.flip(y_mixed, dims=[-1])

        pad_mon = self.padded_dim_monarch - self.dim
        x_mon = F.pad(x, (0, pad_mon)) if pad_mon > 0 else x
        x_grid = x_mon.reshape(-1, self.sqrt_dim, self.sqrt_dim)
        z_grid = torch.matmul(torch.matmul(x_grid, self.W_col), self.W_row.t())
        monarch_out = z_grid.reshape(-1, self.padded_dim_monarch)
        if pad_mon > 0: monarch_out = monarch_out[..., :self.dim]
        
        out = block_out + self.alpha * monarch_out + self.bias
        return out.reshape(orig_shape)

def hbs_layer_factory(dim_in, dim_out):
    if dim_in == dim_out: return HierarchicalBlockShuffleLayer(dim=dim_in)
    return nn.Linear(dim_in, dim_out)

# --- 2. Components ---
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.c_attn = hbs_layer_factory(n_embd, 3 * n_embd)
        self.c_proj = hbs_layer_factory(n_embd, n_embd)
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

    def forward(self, x, rotary_emb=None):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        if rotary_emb is not None:
            cos, sin = rotary_emb(q, seq_len=T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.ffn_fc = hbs_layer_factory(n_embd, 4 * n_embd)
        self.ffn_proj = hbs_layer_factory(4 * n_embd, n_embd)

    def forward(self, x, rotary_emb=None):
        x = x + self.attn(self.ln_1(x), rotary_emb=rotary_emb)
        r = self.ln_2(x)
        r = self.ffn_fc(r)
        r = F.gelu(r)
        r = self.ffn_proj(r)
        x = x + r
        return x

# --- 3. Multi-Head VQ ---
class MultiHeadVectorQuantizer(nn.Module):
    def __init__(self, num_buckets, embedding_dim, num_heads, commitment_cost):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Parameter(torch.randn(num_heads, num_buckets, self.head_dim))
        nn.init.uniform_(self.embeddings, -1/num_buckets, 1/num_buckets)

    def forward(self, inputs):
        B, T, D = inputs.shape
        flat_input = inputs.view(B, T, self.num_heads, self.head_dim)
        flat_input = flat_input.view(-1, self.num_heads, self.head_dim)
        
        input_sq = torch.sum(flat_input**2, dim=2, keepdim=True)
        codebook_sq = torch.sum(self.embeddings**2, dim=2)
        
        product = torch.einsum('ihd,hbd->ihb', flat_input, self.embeddings)
        distances = input_sq + codebook_sq.unsqueeze(0) - 2 * product
        encoding_indices = torch.argmin(distances, dim=2)
        
        quantized_out = torch.zeros_like(flat_input)
        for h in range(self.num_heads):
            idx = encoding_indices[:, h] 
            quantized_out[:, h, :] = self.embeddings[h, idx, :]
            
        quantized = quantized_out.view(B, T, D)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss, encoding_indices

# --- 4. THE SKIM MEMORY (NEW) ---

class SkimMemory(nn.Module):
    def __init__(self, dim, capacity, top_k):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        self.top_k = top_k
        
        # Buffer: [1, Capacity, Dim] (We treat it as a continuous bank)
        self.register_buffer('memory', torch.zeros(1, capacity, dim))
        self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('is_full', torch.zeros(1, dtype=torch.bool))

        # Linear layers for the retrieval attention
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def write(self, states):
        """
        Writes new manager states into the ring buffer.
        states: [B, T, Dim] - We flatten B to treat them as a stream of info
        """
        # Detach to stop gradients (crucial for scalability)
        inputs = states.detach().view(-1, self.dim) 
        n = inputs.size(0)
        
        # Simple pointer arithmetic for ring buffer
        ptr = self.ptr.item()
        end = ptr + n
        
        if end <= self.capacity:
            self.memory[0, ptr:end] = inputs
            self.ptr[0] = (self.ptr[0] + n) % self.capacity
        else:
            # Wrap around
            overflow = end - self.capacity
            self.memory[0, ptr:] = inputs[:self.capacity - ptr]
            self.memory[0, :overflow] = inputs[self.capacity - ptr:]
            self.ptr[0] = overflow
            self.is_full[0] = True

    def read(self, query):
        """
        Retrieves relevant context for the query.
        query: [B, T, Dim] (Current Manager States)
        Returns: [B, T, Dim] (Retrieved Context)
        """
        B, T, C = query.shape
        
        # If memory is empty, return zeros
        if self.ptr == 0 and not self.is_full:
            return torch.zeros_like(query)

        # 1. Project Query
        q = self.q_proj(query) # [B, T, Dim]

        # 2. Prepare Keys/Values (The Memory Bank)
        # We only look at valid memory
        valid_len = self.capacity if self.is_full else self.ptr.item()
        mem_bank = self.memory[:, :valid_len, :] # [1, Mem_Len, Dim]
        
        k = self.k_proj(mem_bank) 
        v = self.v_proj(mem_bank)

        # 3. Top-K Retrieval Attention
        # Score: (B, T, Dim) @ (1, Mem, Dim)^T -> (B, T, Mem)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.dim))
        
        # We only want the Top-K strongest connections
        # For extremely large memory, we don't soft max everything.
        k_val = min(self.top_k, valid_len)
        top_scores, top_indices = torch.topk(scores, k=k_val, dim=-1)
        
        attn_weights = F.softmax(top_scores, dim=-1) # [B, T, k]
        
        # Gather values
        # v is [1, Mem, Dim]. expanded -> [B, Mem, Dim]
        # We need to gather from dimension 1
        v_expanded = v.expand(B, -1, -1)
        
        # Gather logic: we need indices [B, T, k] to pull from [B, Mem, Dim]
        # Torch gather is tricky with multi-dims, so we flatten or use specialized func
        # Easiest way: Construct [B, T, k, Dim]
        
        # (B, T, k) indices -> (B, T, k, Dim) values
        top_indices_flat = top_indices.view(B, T * k_val)
        # This gather is expensive if Mem is huge. 
        # Optimization: Batched Index Select
        
        # Let's use a simplified loop for readability/safety on consumer cards 
        # (The complexity is constrained by K, not Mem size, mostly)
        out_context = torch.zeros(B, T, self.dim, device=query.device)
        
        # Use einsum on the reduced matrices
        # We need to extract the specific V vectors corresponding to top indices
        # Since V is shared across batch (it's global memory), we can index easier.
        
        # valid_v: [Mem, Dim]
        valid_v = v[0] 
        
        # For every batch and time step, we have k indices.
        # Flatten indices to [B*T*k]
        flat_indices = top_indices.view(-1)
        selected_v = valid_v[flat_indices] # [B*T*k, Dim]
        selected_v = selected_v.view(B, T, k_val, self.dim)
        
        # Weighted sum
        # attn_weights: [B, T, k] -> [B, T, k, 1]
        context = (selected_v * attn_weights.unsqueeze(-1)).sum(dim=2)
        
        return self.out_proj(context)

# --- 5. The Manager-Worker Model (Modified) ---

class HierarchicalGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        
        # --- WORKER ---
        self.w_token_emb = nn.Embedding(vocab_size, dim_worker)
        self.w_rope = RotaryEmbedding(dim=dim_worker // 4) 
        self.w_blocks = nn.ModuleList([Block(dim_worker, n_head=4) for _ in range(4)])
        self.w_ln_f = nn.LayerNorm(dim_worker)
        self.lm_head = nn.Linear(dim_worker, vocab_size, bias=False)

        # --- MANAGER ---
        self.m_rope = RotaryEmbedding(dim=dim_manager // 8) 
        self.compressor = nn.Conv1d(dim_worker, dim_manager, kernel_size=local_block_size, stride=local_block_size)
        self.m_blocks = nn.ModuleList([Block(dim_manager, n_head=8) for _ in range(4)])
        self.m_ln_f = nn.LayerNorm(dim_manager)
        
        # --- SKIM MEMORY ---
        self.skim_memory = SkimMemory(dim_manager, skim_capacity, skim_top_k)
        self.skim_gate = nn.Parameter(torch.tensor(0.0)) # Learn how much to rely on memory

        # --- VQ ---
        self.vq_layer = MultiHeadVectorQuantizer(num_buckets, dim_manager, num_heads, commitment_cost)
        
        self.ctx_proj = nn.Linear(dim_manager, dim_worker)

    def forward(self, idx, targets=None, use_cache=True):
        B, Total_T = idx.shape
        w_emb = self.w_token_emb(idx) 
        
        # --- MANAGER FLOW ---
        x_for_conv = w_emb.transpose(1, 2)
        m_input = self.compressor(x_for_conv).transpose(1, 2) 
        
        m_out = m_input
        for block in self.m_blocks:
            m_out = block(m_out, rotary_emb=self.m_rope)
        m_out = self.m_ln_f(m_out) 
        
        # --- SKIM MEMORY RETRIEVAL ---
        # The Manager looks at its own output and asks: "Have I seen this before?"
        skim_context = self.skim_memory.read(m_out)
        
        # Inject Retrieved Context (Gated Residual)
        m_out = m_out + (torch.sigmoid(self.skim_gate) * skim_context)
        
        # --- MEMORY UPDATE (Training) ---
        # Only update memory if we are training. 
        # During inference, we update manually based on user "refresh" rate.
        if self.training and use_cache:
            self.skim_memory.write(m_out)

        # --- VQ ---
        quantized_m_out, vq_loss, bucket_indices = self.vq_layer(m_out)
        
        # Shift & Project to Worker
        global_context = torch.roll(quantized_m_out, shifts=1, dims=1)
        global_context[:, 0, :] = 0 
        global_context_proj = self.ctx_proj(global_context)
        
        # --- WORKER FLOW ---
        num_blocks = m_input.size(1)
        w_reshaped = w_emb.view(B, num_blocks, local_block_size, dim_worker)
        
        w_conditioned = w_reshaped + global_context_proj.unsqueeze(2)
        w_flat = w_conditioned.view(-1, local_block_size, dim_worker)
        
        # Coherence Loss
        worker_block_avg = w_reshaped.mean(dim=2) 
        w_norm = F.normalize(worker_block_avg, p=2, dim=-1)
        g_norm = F.normalize(global_context_proj, p=2, dim=-1)
        coherence_loss = F.mse_loss(w_norm, g_norm)
        
        x = w_flat
        for block in self.w_blocks:
            x = block(x, rotary_emb=self.w_rope)
        x = self.w_ln_f(x)
        
        logits = self.lm_head(x).view(B, Total_T, -1)
        
        loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = lm_loss + vq_loss + (coherence_alpha * coherence_loss)
            
        return logits, loss, bucket_indices, m_out

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, refresh_memory_tokens=128, temperature=1.0, top_k=None):
        """
        refresh_memory_tokens: Update skim memory after generating this many tokens.
        Note: Because of the architecture, we can only meaningfully update when a full block (64 tokens) is done.
        So we will align refresh_memory_tokens to the nearest multiple of local_block_size.
        """
        # Align refresh rate to block size
        refresh_block_count = max(1, refresh_memory_tokens // local_block_size)
        tokens_generated = 0
        pending_blocks = []

        print(f"[System] Skim Memory will refresh every {refresh_block_count * local_block_size} tokens.")

        for _ in range(max_new_tokens):
            # Crop context
            idx_cond = idx[:, - (manager_ctx_len * local_block_size):]
            
            T = idx_cond.size(1)
            remainder = T % local_block_size
            if remainder != 0:
                pad_len = local_block_size - remainder
                idx_padded = F.pad(idx_cond, (0, pad_len), value=0)
            else:
                idx_padded = idx_cond
            
            # Disable auto-write during inference step, we control it manually
            logits, _, _, m_out_full = self(idx_padded, use_cache=False)
            
            logits = logits[:, T-1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            tokens_generated += 1

            # --- MEMORY REFRESH LOGIC ---
            # We check if we just finished a block
            if idx.size(1) % local_block_size == 0:
                # We just finished a block. The last vector in m_out_full corresponds to this block.
                # m_out_full shape: [B, T_blocks, Dim]
                latest_manager_state = m_out_full[:, -1:, :] 
                pending_blocks.append(latest_manager_state)

                if len(pending_blocks) >= refresh_block_count:
                    # Commit to Skim Memory
                    print(".", end="", flush=True) # indicator
                    chunk = torch.cat(pending_blocks, dim=1) # [B, Count, Dim]
                    self.skim_memory.write(chunk)
                    pending_blocks = []

        return idx

# --- 6. Execution ---

def get_batch(data):
    num_blocks_train = 8
    train_seq_len = num_blocks_train * local_block_size
    ix = torch.randint(len(data) - train_seq_len, (batch_size,), device=device)
    x = torch.stack([data[i:i+train_seq_len] for i in ix])
    y = torch.stack([data[i+1:i+train_seq_len+1] for i in ix])
    return x, y

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (learning_rate - min_lr)

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning) 

    try:
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        print("Error: tiktoken not found. Please run 'pip install tiktoken'")
        sys.exit(1)

    if not os.path.exists('lovecraft.txt'):
        with open('lovecraft.txt', 'w', encoding='utf-8') as f:
            f.write("The oldest and strongest emotion of mankind is fear. " * 1000)
            
    with open('lovecraft.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    print("Encoding text...")
    tokens = enc.encode(text)
    vocab_size = enc.n_vocab 
    
    data = torch.tensor(tokens, dtype=torch.long, device=device)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"Running Hierarchical GPT with Skim Memory on {device}")
    
    model = HierarchicalGPT(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler() 
    
    print(f"Model Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    start_iter = 0
    ckpt_path = 'ckpt_skim.pt'
    
    if os.path.exists(ckpt_path):
        print(f"\nCheckpoint found at '{ckpt_path}'!")
        mode = input("Resume training (r), Enter chat mode (c), or Start fresh (f)? [r/c/f]: ").lower().strip()
        
        if mode == 'c':
            print("Loading checkpoint for Chat Mode...")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            model.eval()
            
            print("\n--- CHAT MODE ---")
            # Specify refresh rate here
            refresh_rate = 128 
            print(f"Skim Memory Refresh Rate: Every {refresh_rate} tokens.")
            
            while True:
                try:
                    user_in = input("\nYou: ")
                    if user_in.lower() in ['exit', 'quit']: break
                    ids = enc.encode(user_in)
                    if not ids: continue
                    ctx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
                    out = model.generate(ctx, max_new_tokens=200, refresh_memory_tokens=refresh_rate, temperature=0.8, top_k=50)
                    response_text = enc.decode(out[0].tolist())
                    print(f"Bot: {response_text[len(user_in):]}")
                except KeyboardInterrupt:
                    break
            sys.exit()
            
        elif mode == 'r':
            print("Resuming training...")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_iter = checkpoint['iter']
    
    torch.cuda.synchronize()
    iter_start = time.time()
    
    for iter in range(start_iter, max_iters + 1):
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        xb, yb = get_batch(train_data)
        
        with torch.amp.autocast(device_type=device, dtype=torch.float16):
            # use_cache=True enables automatic memory writing during training
            logits, loss, bucket_indices, _ = model(xb, yb, use_cache=True)
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if iter % eval_interval == 0 and iter > 0:
            torch.cuda.synchronize()
            dt = time.time() - iter_start
            loss_val = loss.item()
            print(f"step {iter}: loss {loss_val:.4f}, lr {lr:.5f}, time {dt:.2f}s")
            
            # Save
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter': iter
            }
            torch.save(checkpoint, ckpt_path)
            iter_start = time.time()