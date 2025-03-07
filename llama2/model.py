import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps= 1e-6):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        # ensuring the data stability
        hidden_states = hidden_states.to(torch.float32)
        normalization_factor = hidden_states.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon
        hidden_states = hidden_states * torch.rsqrt(normalization_factor) * self.weight
        return hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings = 2048, base = 10000):
        super(LlamaRotaryEmbedding, self).__init__()
        self.inv_freq = self.compute_default_rope_parameter(dim, base)
        self.hidden_dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer('inv_freq', self.inv_freq)

    def compute_default_rope_parameter(self,dim, base):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float()/dim))
        return inv_freq

    def forward(self, x, position_ids):
        # [dim/2] -> [b, dim/2, 1]
        inv_freq_expand = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        # [b,s] -> [b, 1, s]
        position_ids_expand = position_ids.unsqueeze(1).float()
        with torch.autocast(enabled=False):
            # [b,s,dim/2]
            freqs = torch.matmul(inv_freq_expand,position_ids_expand).transpose(1,2)
            emb = torch.concat(freqs, freqs)
            #[b,s,dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2,x1), dim=-1)

def apply_rotary_embedding(q, k, cos, sin, unsqueeze_dim = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_emb = q * cos + rotate_half(q) * sin
    k_emb = k * cos + rotate_half(k) * sin
    return q_emb, k_emb

class LlamaMLP(nn.Module):
    def __init__(self,hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.act_fn = nn.functional.silu


    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states, repeat_time):
    b, num_heads, s, head_dim = hidden_states.shape
    if repeat_time == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,None,:,:].expand(b, num_heads, repeat_time, s, head_dim)
    hidden_states = hidden_states.view(b,-1, s, head_dim)
    return hidden_states

def generate_casual_mask(seq_len):
    casual_mask = torch.ones((seq_len,seq_len)).tril()
    casual_mask = casual_mask.masked_fill(casual_mask == 0, float('-inf') )
    return casual_mask


class LlamaAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, num_key_value_head, attn_dropout,layer_ids):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_head = num_key_value_head
        self.num_key_value_group = num_heads // num_key_value_head
        self.layer_ids = layer_ids
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_size, num_key_value_head * head_dim)
        self.v_proj = nn.Linear(hidden_size, num_key_value_head * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)
        self.rope_emb = LlamaRotaryEmbedding(head_dim)
        self.dropout = nn.Dropout(attn_dropout)


    def forward(self, hidden_states, attn_mask, position_ids, past_key_values, use_cache):
        b, s, h = hidden_states.shape
        queries = self.q_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.k_proj(hidden_states).view(b, s, self.num_key_value_head, self.head_dim).transpose(1, 2)
        values = self.v_proj(hidden_states).view(b, s, self.num_key_value_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rope_emb(values, position_ids)
        queries, keys = apply_rotary_embedding(queries,keys,cos,sin)
        if past_key_values is not None:
            keys, values = past_key_values.update(keys, values, self.layer_ids)

        keys = repeat_kv(keys, self.num_key_value_group)
        values = repeat_kv(values, self.num_key_value_group)

        attn_weights = torch.matmul(queries, keys.t())/torch.sqrt(self.num_heads)
        attn_weights = self.dropout(attn_weights)
        # b, num_heads, s, s

        if attn_mask is not None:
            # [b, s]
            attn_mask = attn_mask[:, None, None, :].expand(-1, self.num_heads, 1, -1)
            attn_mask = attn_weights.masked_fill(attn_mask == 0, float('inf'))
            attn_weights += attn_mask

        attn_outputs = nn.functional.softmax(attn_weights, dim=-1)
        attn_outputs = torch.matmul(attn_outputs, values)
        attn_outputs = attn_outputs.transpose(1,2).contiguous().view(b,s,h)
        attn_outputs = self.o_proj(attn_outputs)
        return attn_outputs



if __name__ == '__main__':
    model = RMSNorm(512)
    inputs = torch.randn(8,30,512)
    hidden_states = model(inputs)





