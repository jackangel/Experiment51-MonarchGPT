import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
import sys
import os
import random
import tiktoken 
import warnings

# --- Hyperparameters ---
batch_size = 6
local_block_size = 64 
manager_ctx_len = 32   
eval_interval = 500     
predict_interval = 2000 
learning_rate = 1.5e-4 
warmup_iters = 1000     
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout = 0.1

dim_worker = 256       
dim_manager = 384      

num_buckets = 64
num_heads = 4          
commitment_cost = 0.25
coherence_alpha = 0.1  

num_experts = 8
top_k_experts = 2

skim_capacity = 16384  
skim_top_k = 32        

train_on_parquet = True           
parquet_folder = r"D:\Datasets"  
parquet_text_col = "text"          

torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore", category=FutureWarning) 

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
            return self.cos_cached, self.sin_cached 
        return self.cos_cached[..., :seq_len, :], self.sin_cached[..., :seq_len, :]

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos[..., :q.shape[2], :]
    sin = sin[..., :q.shape[2], :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

# --- 1. Robust HBSL Layer (Pre-Norm + Float32 Core) ---
class HierarchicalBlockShuffleLayer(nn.Module):
    def __init__(self, dim, block_size=16):
        super().__init__()
        self.dim = dim
        self.block_size = block_size
        self.num_groups = (dim + block_size - 1) // block_size
        self.padded_dim_groups = self.num_groups * self.block_size
        
        # Internal Norms to prevent signal explosion/decay
        self.pre_norm = nn.LayerNorm(dim)
        
        self.block_weights = nn.Parameter(torch.randn(self.num_groups, self.block_size, self.block_size))
        nn.init.orthogonal_(self.block_weights)
        self.block_weights.data *= 0.5 # Gentle init
        
        sqrt_dim_ceil = math.ceil(math.sqrt(dim))
        self.padded_dim_monarch = sqrt_dim_ceil * sqrt_dim_ceil
        self.sqrt_dim = sqrt_dim_ceil
        self.W_row = nn.Parameter(torch.randn(self.sqrt_dim, self.sqrt_dim))
        self.W_col = nn.Parameter(torch.randn(self.sqrt_dim, self.sqrt_dim))
        nn.init.orthogonal_(self.W_row)
        nn.init.orthogonal_(self.W_col)
        self.W_row.data *= 0.1
        self.W_col.data *= 0.1
        
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # 1. Pre-Norm: Forces input to be stable range [-3, 3]
        x_norm = self.pre_norm(x)
        orig_shape = x.shape
        
        # 2. Force FP32 for Matrix Math
        with torch.amp.autocast('cuda', enabled=False):
            x_f32 = x_norm.float()
            x_flat = x_f32.reshape(-1, self.dim)
            
            # Branch 1: Block Shuffle
            pad_amt = self.padded_dim_groups - self.dim
            x_padded = F.pad(x_flat, (0, pad_amt)) if pad_amt > 0 else x_flat
            x_grouped = x_padded.reshape(-1, self.num_groups, self.block_size)
            
            w_block_f32 = self.block_weights.float()
            y_perm = torch.bmm(x_grouped.permute(1, 0, 2), w_block_f32)
            y_mixed = y_perm.permute(1, 0, 2).reshape(-1, self.padded_dim_groups)
            if pad_amt > 0: y_mixed = y_mixed[..., :self.dim]
            block_out = torch.flip(y_mixed, dims=[-1])

            # Branch 2: Monarch
            pad_mon = self.padded_dim_monarch - self.dim
            x_mon = F.pad(x_flat, (0, pad_mon)) if pad_mon > 0 else x_flat
            x_grid = x_mon.reshape(-1, self.sqrt_dim, self.sqrt_dim)
            z_grid = torch.matmul(torch.matmul(x_grid, self.W_col.float()), self.W_row.float().t())
            monarch_out = z_grid.reshape(-1, self.padded_dim_monarch)
            if pad_mon > 0: monarch_out = monarch_out[..., :self.dim]
            
            # Recombine
            out = block_out + self.alpha.float() * monarch_out + self.bias.float()
            out = out.to(x.dtype) # Cast back to FP16/BF16
            
        return out.reshape(orig_shape)

def hbs_layer_factory(dim_in, dim_out):
    if dim_in == dim_out: return HierarchicalBlockShuffleLayer(dim=dim_in)
    return nn.Linear(dim_in, dim_out)

# --- 2. Stabilized Hybrid Attention (Safe Linear) ---
class HybridFocusAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.c_attn = hbs_layer_factory(n_embd, 3 * n_embd)
        self.c_proj = hbs_layer_factory(n_embd, n_embd)
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.focus_gate = nn.Parameter(torch.tensor(0.0)) 
        
        # Norms for stability
        self.attn_norm = nn.LayerNorm(self.head_dim)
        self.out_norm = nn.LayerNorm(n_embd)

    def forward(self, x, rotary_emb=None):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        if rotary_emb is not None:
            cos, sin = rotary_emb(q, seq_len=T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # --- BRANCH A: SAFE LINEAR ATTENTION ---
        # Using ELU+1 kernel (Katharopoulos et al.) which is always positive
        # This removes the instability of "Fuzzy" Softplus Cumsum
        with torch.amp.autocast('cuda', enabled=False):
            q_f32 = q.float()
            k_f32 = k.float()
            v_f32 = v.float()
            
            # Activation: x > 0. Non-negative guarantees stability.
            scale = 1.0 / math.sqrt(self.head_dim)
            Q_prime = F.elu(q_f32 * scale) + 1.0
            K_prime = F.elu(k_f32 * scale) + 1.0
            
            # Normalizer for stability
            KV = torch.einsum('bhtd,bhte->bhtde', K_prime, v_f32)
            KV_cumsum = torch.cumsum(KV, dim=2)
            
            K_cumsum = torch.cumsum(K_prime, dim=2)
            
            numerator = torch.einsum('bhtd,bhtde->bhte', Q_prime, KV_cumsum)
            denominator = torch.einsum('bhtd,bhtd->bht', Q_prime, K_cumsum).unsqueeze(-1)
            
            # Epsilon is critical here
            y_linear = numerator / (denominator + 1e-4)
            y_linear = y_linear.to(x.dtype)

        y_linear = self.attn_norm(y_linear.transpose(1, 2)).transpose(1, 2)

        # --- BRANCH B: SHARP FOCUS ---
        y_sharp = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # Gating
        alpha = torch.sigmoid(self.focus_gate)
        y_combined = (alpha * y_sharp) + ((1.0 - alpha) * y_linear)
        
        y = y_combined.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output Norm to reset scale before projection
        return self.c_proj(self.out_norm(y))

# --- 3. VQ with Input Squash ---
class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_buckets, embedding_dim, num_heads, commitment_cost, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        self.register_buffer('embeddings', torch.randn(num_heads, num_buckets, self.head_dim))
        self.register_buffer('cluster_size', torch.zeros(num_heads, num_buckets))
        self.register_buffer('embed_avg', self.embeddings.clone())
        
        # Learnable input normalization
        self.in_norm = nn.LayerNorm(embedding_dim)
        
        nn.init.orthogonal_(self.embeddings)
        self.embeddings.data = F.normalize(self.embeddings.data, p=2, dim=-1)

    def forward(self, inputs):
        # 1. Normalize Inputs immediately (Safe Zone)
        inputs = self.in_norm(inputs)
        
        # 2. Hard Tanh Limit - prevents "fireballs"
        inputs = torch.tanh(inputs / 5.0) * 5.0 

        if not torch.isfinite(inputs).all():
            return inputs, torch.tensor(0.0, device=inputs.device, requires_grad=True), 0

        B, T, D = inputs.shape
        inputs_norm = F.normalize(inputs, p=2, dim=-1, eps=self.epsilon)
        flat_input = inputs_norm.view(-1, self.num_heads, self.head_dim)
        
        embed_t = self.embeddings.permute(0, 2, 1) 
        dist = torch.einsum('nhd,hdk->nhk', flat_input, embed_t)
        encoding_indices = torch.argmax(dist, dim=-1) 
        
        quantized = torch.zeros_like(flat_input)
        for h in range(self.num_heads):
            quantized[:, h, :] = self.embeddings[h, encoding_indices[:, h], :]
        quantized = quantized.view(B, T, D)
        
        if self.training:
             with torch.no_grad():
                flat_input_no_grad = flat_input.detach()
                encodings = F.one_hot(encoding_indices, self.num_buckets).float()
                avg_usage = encodings.sum(dim=0)
                self.cluster_size.data.mul_(self.decay).add_(avg_usage, alpha=1 - self.decay)
                embed_sum = torch.einsum('nhk,nhd->hkd', encodings, flat_input_no_grad)
                self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
                n = self.cluster_size.sum(dim=1, keepdim=True)
                cluster_size_smoothed = (self.cluster_size + self.epsilon) / (n + self.num_buckets * self.epsilon) * n
                embed_normalized = self.embed_avg / cluster_size_smoothed.unsqueeze(-1)
                self.embeddings.data.copy_(F.normalize(embed_normalized, p=2, dim=-1, eps=self.epsilon))
                
                # Random Restart for Dead Codebooks
                for h in range(self.num_heads):
                    dead_indices = torch.nonzero(self.cluster_size[h] < 0.1).squeeze(-1)
                    if dead_indices.numel() > 0:
                        rand_idx = torch.randint(0, flat_input.size(0), (dead_indices.numel(),), device=inputs.device)
                        self.embeddings.data[h, dead_indices, :] = flat_input_no_grad[rand_idx, h, :]
                        self.cluster_size.data[h, dead_indices] = 1.0
                        self.embed_avg.data[h, dead_indices, :] = flat_input_no_grad[rand_idx, h, :]

        e_latent_loss = F.mse_loss(quantized.detach(), inputs_norm)
        
        if self.training:
            encodings = F.one_hot(encoding_indices, self.num_buckets).float()
            avg_probs = encodings.mean(dim=0) 
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1)).mean()
            entropy_loss = -0.1 * perplexity 
        else:
            entropy_loss = torch.tensor(0.0, device=inputs.device)

        loss = (self.commitment_cost * e_latent_loss) + entropy_loss
        quantized = inputs + (quantized - inputs).detach()
        unique_count = torch.unique(encoding_indices).numel()
        return quantized, loss, unique_count

# --- 4. MoE ---
class Expert(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            hbs_layer_factory(n_embd, 4 * n_embd),
            nn.GELU(),
            hbs_layer_factory(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

import torch.utils.checkpoint as checkpoint
class SparseMoE(nn.Module):
    def __init__(self, n_embd, num_experts, top_k, use_checkpointing=False):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(n_embd, num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(n_embd) for _ in range(num_experts)])
        self.use_checkpointing = use_checkpointing

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        
        logits = self.gate(x_flat)
        # Clamp logits to prevent Softmax saturation
        logits = torch.clamp(logits, min=-20.0, max=20.0)
        
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        results = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.experts):
            batch_idx, rank_idx = torch.where(indices == i)
            if batch_idx.numel() == 0: continue
            
            inp_subset = x_flat[batch_idx]
            if self.training and self.use_checkpointing:
                out_subset = checkpoint.checkpoint(expert, inp_subset, use_reentrant=False)
            else:
                out_subset = expert(inp_subset)
            
            w_subset = weights[batch_idx, rank_idx].unsqueeze(-1)
            results.index_add_(0, batch_idx, out_subset * w_subset)
                    
        return results.view(B, T, C)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = HybridFocusAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.moe = SparseMoE(n_embd, num_experts=num_experts, top_k=top_k_experts, use_checkpointing=True)

    def forward(self, x, rotary_emb=None):
        x = x + self.attn(self.ln_1(x), rotary_emb=rotary_emb)
        x = x + self.moe(self.ln_2(x))
        return x

# --- 5. Skim Memory ---
class SkimMemory(nn.Module):
    def __init__(self, dim, capacity, top_k):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        self.top_k = top_k
        self.register_buffer('memory', torch.zeros(1, capacity, dim))
        self.register_buffer('ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('is_full', torch.zeros(1, dtype=torch.bool))
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def write(self, states):
        inputs = states.detach().view(-1, self.dim)
        if not torch.isfinite(inputs).all(): return
        n = inputs.size(0)
        ptr = self.ptr.item()
        end = ptr + n
        if end <= self.capacity:
            self.memory[0, ptr:end] = inputs
            self.ptr[0] = (self.ptr[0] + n) % self.capacity
        else:
            overflow = end - self.capacity
            self.memory[0, ptr:] = inputs[:self.capacity - ptr]
            self.memory[0, :overflow] = inputs[self.capacity - ptr:]
            self.ptr[0] = overflow
            self.is_full[0] = True

    def read(self, query):
        B, T, C = query.shape
        if self.ptr == 0 and not self.is_full: return torch.zeros_like(query)
        q = self.q_proj(query) 
        valid_len = self.capacity if self.is_full else self.ptr.item()
        mem_bank = self.memory[:, :valid_len, :] 
        k = self.k_proj(mem_bank) 
        v = self.v_proj(mem_bank)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.dim))
        k_val = min(self.top_k, valid_len)
        top_scores, top_indices = torch.topk(scores, k=k_val, dim=-1)
        attn_weights = F.softmax(top_scores, dim=-1) 
        valid_v = v[0] 
        selected_v = valid_v[top_indices.view(-1)].view(B, T, k_val, self.dim)
        context = (selected_v * attn_weights.unsqueeze(-1)).sum(dim=2)
        return self.out_proj(context)

# --- 6. Manager-Worker ---
class HierarchicalGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.w_token_emb = nn.Embedding(vocab_size, dim_worker)
        self.w_rope = RotaryEmbedding(dim=dim_worker // 4) 
        self.w_blocks = nn.ModuleList([Block(dim_worker, n_head=4) for _ in range(4)])
        self.w_ln_f = nn.LayerNorm(dim_worker)
        self.lm_head = nn.Linear(dim_worker, vocab_size, bias=False)

        self.m_rope = RotaryEmbedding(dim=dim_manager // 8) 
        self.compressor = nn.Conv1d(dim_worker, dim_manager, kernel_size=local_block_size, stride=local_block_size)
        self.m_blocks = nn.ModuleList([Block(dim_manager, n_head=8) for _ in range(4)])
        self.m_ln_f = nn.LayerNorm(dim_manager)
        
        self.skim_memory = SkimMemory(dim_manager, skim_capacity, skim_top_k)
        self.skim_gate = nn.Parameter(torch.tensor(-4.0))

        self.vq_layer = EMAVectorQuantizer(num_buckets, dim_manager, num_heads, commitment_cost)
        self.ctx_proj = nn.Linear(dim_manager, dim_worker)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, use_cache=True):
        B, Total_T = idx.shape
        w_emb = self.w_token_emb(idx) 
        
        x_for_conv = w_emb.transpose(1, 2)
        m_input = self.compressor(x_for_conv).transpose(1, 2) 
        m_out = m_input
        for block in self.m_blocks:
            m_out = block(m_out, rotary_emb=self.m_rope)
        m_out = self.m_ln_f(m_out) 
        
        skim_context = self.skim_memory.read(m_out)
        m_out = m_out + (torch.sigmoid(self.skim_gate) * skim_context)
        if self.training and use_cache:
            self.skim_memory.write(m_out)

        quantized_m_out, vq_loss, vq_count = self.vq_layer(m_out)
        
        global_context = torch.roll(quantized_m_out, shifts=1, dims=1)
        global_context[:, 0, :] = 0 
        global_context_proj = self.ctx_proj(global_context)
        
        num_blocks = m_input.size(1)
        w_reshaped = w_emb.view(B, num_blocks, local_block_size, dim_worker)
        w_conditioned = w_reshaped + global_context_proj.unsqueeze(2)
        w_flat = w_conditioned.view(-1, local_block_size, dim_worker)
        
        worker_block_avg = w_reshaped.mean(dim=2) 
        w_norm = F.normalize(worker_block_avg, p=2, dim=-1, eps=1e-5)
        g_norm = F.normalize(global_context_proj, p=2, dim=-1, eps=1e-5)
        coherence_loss = F.mse_loss(w_norm, g_norm)
        
        x = w_flat
        for block in self.w_blocks:
            x = block(x, rotary_emb=self.w_rope)
        x = self.w_ln_f(x)
        
        logits = self.lm_head(x).view(B, Total_T, -1)
        logits = torch.clamp(logits, min=-30.0, max=30.0)

        loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = lm_loss + vq_loss + (coherence_alpha * coherence_loss)
            
        return logits, loss, vq_count, m_out

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, refresh_memory_tokens=128, temperature=1.0, top_k=None):
        refresh_block_count = max(1, refresh_memory_tokens // local_block_size)
        pending_blocks = []
        for _ in range(max_new_tokens):
            idx_cond = idx[:, - (manager_ctx_len * local_block_size):]
            T = idx_cond.size(1)
            pad_len = 0
            remainder = T % local_block_size
            if remainder != 0:
                pad_len = local_block_size - remainder
                idx_padded = F.pad(idx_cond, (0, pad_len), value=0)
            else:
                idx_padded = idx_cond
            
            logits, _, _, m_out_full = self(idx_padded, use_cache=False)
            logits = logits[:, T-1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            if idx.size(1) % local_block_size == 0:
                latest_manager_state = m_out_full[:, -1:, :] 
                pending_blocks.append(latest_manager_state)
                if len(pending_blocks) >= refresh_block_count:
                    chunk = torch.cat(pending_blocks, dim=1)
                    self.skim_memory.write(chunk)
                    pending_blocks = []
        return idx

# --- 7. Main ---
def get_batch(data):
    num_blocks_train = 8
    train_seq_len = num_blocks_train * local_block_size
    max_idx = len(data) - train_seq_len
    if max_idx <= 0:
        data = torch.cat([data, data], dim=0)
        max_idx = len(data) - train_seq_len
    ix = torch.randint(max_idx, (batch_size,))
    x = torch.stack([data[i:i+train_seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+train_seq_len+1] for i in ix]).to(device)
    x = torch.clamp(x, max=enc.n_vocab - 1)
    y = torch.clamp(y, max=enc.n_vocab - 1)
    return x, y

def get_lr(it):
    if it < warmup_iters: return learning_rate * (it + 1) / (warmup_iters + 1)
    return learning_rate 

def load_parquet_optimized(filepath, col_name, enc):
    try: import pyarrow.parquet as pq
    except ImportError: return torch.tensor([], dtype=torch.long)
    try: table = pq.read_table(filepath, columns=[col_name])
    except: return torch.tensor([], dtype=torch.long)
    all_ids = []
    for batch in table.to_batches():
        chunk_str = "".join(batch[col_name].to_pylist())
        try: all_ids.extend(enc.encode(chunk_str, allowed_special={'<|endoftext|>'}))
        except: continue
    return torch.tensor(all_ids, dtype=torch.long)

def get_balanced_file_list(root_folder):
    files = []
    for root, _, fs in os.walk(root_folder):
        for f in fs:
            if f.endswith(".parquet"): files.append(os.path.join(root, f))
    random.shuffle(files)
    return files

if __name__ == '__main__':
    concurrent_file_load = 3   
    shuffle_block_size = 2048 
    GRAD_ACCUM_STEPS = 8      
    
    try: enc = tiktoken.get_encoding("gpt2")
    except: sys.exit(1)
    parquet_files = []
    if train_on_parquet and os.path.exists(parquet_folder):
        parquet_files = get_balanced_file_list(parquet_folder)
    vocab_size = enc.n_vocab 

    print(f"Running STABILIZED Hierarchical GPT (Pre-Norm + Safe Linear) on {device}")
    
    model = HierarchicalGPT(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Initialize Scaler
    scaler = torch.amp.GradScaler() 
    
    ckpt_path = 'ckpt_robust.pt'
    current_iter = 0 
    
    if os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}...")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            current_iter = checkpoint['iter']
            
            # --- MANDATORY RESET ---
            # We assume the previous scaler state is potentially corrupted/dead.
            # Start fresh to ensure training actually moves.
            scaler = torch.amp.GradScaler() 
            print("!!! Scaler force-reset to default (65536) to prevent dead training !!!")
            
        except Exception as e: 
            print(f"Resuming failed: {e}. Starting fresh.")

    iter_start = time.time()
    file_chunks = [parquet_files[i:i + concurrent_file_load] for i in range(0, len(parquet_files), concurrent_file_load)]
    
    for chunk_idx, current_chunk_files in enumerate(file_chunks):
        print(f"\n[Chunk {chunk_idx+1}/{len(file_chunks)}]")
        raw_tensors = []
        for fpath in current_chunk_files:
            t = load_parquet_optimized(fpath, parquet_text_col, enc)
            if t.numel() > 0: raw_tensors.append(t)
        if not raw_tensors: continue

        all_blocks = []
        for t in raw_tensors:
            blocks = torch.split(t, shuffle_block_size)
            all_blocks.extend([b for b in blocks if b.size(0) == shuffle_block_size])
        random.shuffle(all_blocks) 
        train_data = torch.cat(all_blocks)
        del raw_tensors, all_blocks
        
        seq_len = 8 * local_block_size
        tokens_per_step = batch_size * seq_len
        steps_in_buffer = len(train_data) // tokens_per_step
        if steps_in_buffer == 0: steps_in_buffer = 1
        
        for step in range(steps_in_buffer):
            model.train()
            xb, yb = get_batch(train_data)
            
            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                logits, loss, vq_count, _ = model(xb, yb, use_cache=True)
                loss = loss / GRAD_ACCUM_STEPS 
            
            # Scale Loss
            scaler.scale(loss).backward()
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                
                # Global Gradient Clipping (Stabilizer)
                # This replaces the aggressive manual hooks
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                lr = get_lr(current_iter)
                for param_group in optimizer.param_groups: param_group['lr'] = lr
                
                # Scaler step will skip if NaNs are found after unscaling (safe)
                scaler.step(optimizer)
                scaler.update()
                
                optimizer.zero_grad(set_to_none=True)
                current_iter += 1

                if current_iter % eval_interval == 0 and current_iter > 0:
                    dt = time.time() - iter_start
                    real_loss = loss.item() * GRAD_ACCUM_STEPS
                    current_scale = scaler.get_scale()
                    print(f"Iter {current_iter}: loss {real_loss:.4f}, lr {lr:.5f}, time {dt:.2f}s")
                    print(f"    VQ Usage: {vq_count} buckets.")
                    print(f"    Grad Norm: {norm:.4f} | Amp Scale: {current_scale:.1f}")
                    iter_start = time.time()

                if current_iter % predict_interval == 0:
                    model.eval()
                    ctx = torch.tensor(enc.encode("The future"), dtype=torch.long, device=device).unsqueeze(0)
                    try:
                        out = model.generate(ctx, max_new_tokens=40)
                        print(f"Gen: {enc.decode(out[0].tolist())}\n")
                    except: pass
                    model.train()
            
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'iter': current_iter}
        torch.save(checkpoint, ckpt_path)
        del train_data
        torch.cuda.empty_cache()