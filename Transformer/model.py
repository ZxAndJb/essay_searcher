import math

import torch
import torch.nn as nn
import argparse
from torch.nn.functional import log_softmax, pad
from model_helper import *
import torch.nn.functional as F



DEFAULT_MAX_SOURCE_POSITIONS = 512
DEFAULT_MAX_TARGET_POSITIONS = 512

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, encoder_embed, decoder_embed, **kwargs):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_embed = encoder_embed
        self.decoder_embed = decoder_embed
        if getattr(kwargs, 'max_source_positions', None) is None:
            kwargs.max_src_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(kwargs, 'max_target_positions', None) is None:
            kwargs.max_tgt_positions = DEFAULT_MAX_TARGET_POSITIONS


    def forward(self, scr_x, target_x, scr_mask, target_mask):
        return self.decode(self.encode(scr_x, scr_mask), target_x, target_mask)

    def encode(self, scr_x, scr_mask):
        return self.encoder(self.encoder_embed(scr_x), scr_mask)

    def decode(self, encoder_hidden_state, scr_mask, target_x, target_mask):
        return self.decoder(encoder_hidden_state, scr_mask, self.decoder_embed(target_x), target_mask)



class Encoder(nn.Module):
    def __init__(self, args: argparse.Namespace, dictionary):
        super(Encoder, self).__init__()
        self.dropout = args.dropout
        self.embedding_dim = args.embedding_dim
        self.padding_idx = dictionary.pad_idx
        self.nlayers = args.nlayers
        self.max_src_positions = args.max_src_positions
        self.embedding = generate_embeddings(dictionary.vocab_size, self.embedding_dim, self.padding_idx)
        self.embed_scale = math.sqrt(self.embedding_dim) # scaling 的目的？ 1.make postional encoding relatively small 2. stable training
        self.embed_positions = PositionEncoding(self.embedding_dim, padding_idx=self.padding_idx,
                                                base=10000, init_size=self.max_src_positions + 1)


        self.layers = nn.ModuleList()

        self.layers.extend([EncoderLayer(args) for _ in range(self.nlayers)])

    def forward(self, scr_tokens, scr_mask):
        embeddings = self.embed_scale *  self.embedding(scr_tokens)
        # [bs, src_seq_len, h]
        forward_states = embeddings + self.embed_positions(scr_tokens)


        for layer in self.layers:
            forward_states = layer(forward_states, encoder_padding_mask = scr_mask)

        return {
            "scr_out": forward_states,
            "scr_padding_mask": scr_mask
        }
        #

class EncoderLayer(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(EncoderLayer, self).__init__()
        self.embed_dim = args.embed_dim
        self.attn_dropout = args.attn_dropout
        self.ffn_dropout = args.ffn_dropout
        self.attn_layernorm = nn.LayerNorm(normalized_shape=self.embed_dim, eps=1e-5, elementwise_affine=True)
        self.ffn_layernorm = nn.LayerNorm(normalized_shape=self.embed_dim, eps=1e-5, elementwise_affine=True)
        self.multihead_attention = MultiHeadAttention(self.embed_dim, args.num_heads, dropout=self.attn_dropout,self_attention=True)
        self.FFN = nn.Sequential(
            nn.Linear(self.embed_dim, args.ffn_dim),
            nn.ReLU(),
            nn.Linear(args.ffn_dim, self.embed_dim)
        )

    def forward(self, x, scr_mask):
        residual = x.clone()
        x, _ = self.multihead_attention(query=x, key=x, value=x, key_padding_mask=scr_mask)
        x = F.dropout(x,p=self.attn_dropout, training=self.training)
        x += residual
        x = self.attn_layernorm(x)

        residual = x.clone()
        x = F.dropout(self.FFN(x), p =self.ffn_dropout, training = self.training)
        x += residual
        x = self.ffn_layernorm(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, k_dim=None, v_dim=None, self_attention=True):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.k_dim = embed_dim if k_dim is None else k_dim
        self.v_dim = embed_dim if v_dim is None else v_dim
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, k_dim)
        self.W_v = nn.Linear(embed_dim, v_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.self_attention = self_attention
        self.dropout = dropout

        assert self.head_dim * self.num_heads == embed_dim, "Embed dim must be divisible by num_heads!"

    def forward(self,q, k, v, key_padding_mask = None, casual_mask=None):
        b, query_s_l, h = q.size()
        b, key_s_l, h = k.size()
        # [b,s,h] -> [b,num_heads,s,head_dim]
        queries = self.W_q(q).reshape(b, query_s_l, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        keys = self.W_k(k).reshape(b, key_s_l, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        values = self.W_v(v).reshape(b, key_s_l, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        # [b,num_heads,query_s_l, key_s_l]
        scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.embed_dim)

        if key_padding_mask is not None:
            # [b,s]
            key_padding_mask = key_padding_mask.view(b, 1, 1,key_s_l)
            key_padding_mask = key_padding_mask.expand(b, self.num_heads,1,key_s_l)

            key_padding_mask = torch.zeros_like(scores).masked_fill(key_padding_mask, float('-inf'))
            if casual_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = key_padding_mask + casual_mask

            scores = scores + attn_mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_outputs = F.dropout(torch.matmul(attn_weights, values), p=self.dropout, training=self.training)
        attn_outputs = attn_outputs.permute(0,2,1,3).contiguous().reshape(b, query_s_l, self.embed_dim)
        attn_outputs = self.W_o(attn_outputs)

        return attn_outputs



class PositionEncoding(nn.Module):
    def __init__(self,embed_dim, padding_idx, base, init_size = 1024):
        super(PositionEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.base = base
        self.init_size = init_size

    def get_weights(self, init_size, embed_dim, padding_idx, base=10000):
        # i in [0, embed_dim//2]
        half_dim = embed_dim//2
        emb = math.log(base)/ (half_dim - 1)
        # create a value for each dim
        emb = torch.exp(torch.arange(0, half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(0, init_size).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)])

        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb


    def forward(self, x):
        batch_size, sequence_length = x.size()
        mask = x.ne(self.padding_idx).int()
        positions = torch.cusum(mask, dim=1).type_as(x) * mask + self.padding_idx
        return self.weights.index_select(0, positions.view(-1)).view(batch_size, sequence_length,-1).detach()

