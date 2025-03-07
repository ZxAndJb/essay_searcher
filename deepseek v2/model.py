import torch
import math

from click.core import F
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, x):
        x_dtype = x.dtype
        normal_factor = x.pow(2).sum(dim=-1, keepdim=True) + self.eps
        hidden_states = (x * torch.rsqrt(normal_factor)) * self.weight
        return hidden_states.to(x_dtype)

class DeepseekRope(nn.Module):
    def __init__(self, hidden_dim, max_position_embeddings = 2048, base = 10000):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.hidden_dim = hidden_dim
        inv_freq = 1.0/(self.base ** (torch.arange(0, self.max_position_embeddings,2)/self.max_position_embeddings))
        self.register_buffer('inv_freq', inv_freq)
        self.set_cos_sin_cache()


    def _set_cos_sin_cache(self):
        len = torch.arange(0, self.max_position_embeddings)
        freqs = torch.outer(len, self.inv_freq)
        emb = torch.cat([freqs,freqs])
        self.register_buffer('cos_cache', emb.cos())
        self.register_buffer('sin_cache', emb.sin())

    def forward(self, position_ids):
        return self.cos_cached[:len(position_ids)], self.sin_cached[:len(position_ids)]


def rotate_half(hidden_state):
    x_1 = hidden_state[..., : hidden_state.shape[-1]//2]
    x_2 = hidden_state[..., hidden_state.shape[-1]//2:]

    return torch.cat([-x_2,x_1],dim=-1)


def apply_rope_pos_emb(q,k,cos,sin,unsqueeze_dim =1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Deepseekv2Attention(nn.Module):

    def __init__(self, config, layer_idx):
        super(Deepseekv2Attention, self).__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attn_dropout = config.attn_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_base = config.rope_base
        ## 1536 默认值
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.head_dim
        self.qk_nope_head_dim = config.head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        if self.q_lora_rank is not None:
            self.q_a_project = nn.Linear(self.hidden_size, self.q_lora_rank)
            self.q_a_RNSNorm = RMSNorm(self.q_lora_rank)
            self.q_b_project = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_nope_head_dim, bias = False)
        self.kv_a_RNSNorm = RMSNorm(self.kv_lora_rank)
        self.kv_b_project = nn.Linear(self.kv_lora_rank, self.num_heads * 2 * self.qk_nope_head_dim)
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self.rope_emb = DeepseekRope(self.qk_rope_head_dim, self.max_position_embeddings)



    def forward(self, hidden_states, attn_mask, position_ids, past_key_value, use_cache):
        bs, seq_len, hidden_dim = hidden_states.shape
        queries = self.q_b_project(self.q_a_RNSNorm(self.q_a_project(hidden_states)))
        queries = queries.view(bs, seq_len, self.num_heads, -1).transpose(1, 2).contiguous()
        q_nope, q_pe = torch.split(queries,[self.qk_nope_head_dim,self.qk_rope_head_dim],dim=-1)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_nope_head_dim],dim=-1)
        k_pe = k_pe.view(bs,seq_len, 1, self.qk_nope_head_dim).transpose(1,2)
        kv_b_proj = self.kv_b_project.weight.view(self.kv_lora_rank, self.num_heads, 2, self.qk_nope_head_dim)
        q_absorb, o_absorb = kv_b_proj[...,0,:], kv_b_proj[...,1,:]

        cos, sin = self.rope_emb(k_pe)
        q_pe, k_pe = apply_rope_pos_emb(q_pe,k_pe, cos,sin)

        if past_key_value is not None:
            compressed_kv = past_key_value.update(compressed_kv, self.layer_idx)

        qk_head_dim = self.kv_lora_rank + self.qk_nope_head_dim
        query_states = torch.new_empty(bs, self.num_heads, seq_len, qk_head_dim)
        query_states[:,:,:,:self.kv_lora_rank] = torch.einsum('chd, bhsd -> bhsc', q_absorb.squeeze(), q_nope)
        query_states[:,:,:,self.kv_lora_rank:] = q_pe

        key_states = torch.new_empty(bs, self.num_heads, seq_len, qk_head_dim)
        key_states[:, :, :, : self.kv_lora_rank] = compressed_kv.unsqueeze(1)
        key_states[:, :, :, self.kv_lora_rank:] = k_pe

        attn_weights = torch.matmul(qk_head_dim, key_states.transpose(2, 3))/ math.sqrt(qk_head_dim)
        if attn_mask is not None:
            attn_weights += attn_mask
        attn_weights = F.softmax(attn_weights)
        attn_outputs = torch.einsum('bhss, bsc -> bhsc', attn_weights, compressed_kv)
        attn_outputs = torch.einsum('bhsc, chd -> bhsd ', attn_outputs, o_absorb)

        attn_outputs = attn_outputs.transpose(1,2).view(bs, seq_len, -1).contiguous()

        return self.o_proj(attn_outputs)





















