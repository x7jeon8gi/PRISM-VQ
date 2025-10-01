import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from module.layers.src.encoding import RotaryPE


def _get_activation_fn(activation):
    if callable(activation):
        return activation
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise ValueError(f"Unsupported activation: {activation}")


class RotaryTransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer with RoPE-aware self-attention."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        layer_norm_eps=1e-5,
        batch_first=True,
        norm_first=False,
        activation="relu",
        rotary_pe=None,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead for RoPE layer")

        self.d_model = d_model
        self.num_heads = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.rotary_pe = rotary_pe

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if not self.batch_first:
            src = src.transpose(0, 1)

        if self.norm_first:
            src = src + self._self_attn_block(
                self.norm1(src), src_mask, src_key_padding_mask, is_causal
            )
            src = src + self._ff_block(self.norm2(src))
        else:
            src = self.norm1(
                src
                + self._self_attn_block(src, src_mask, src_key_padding_mask, is_causal)
            )
            src = self.norm2(src + self._ff_block(src))

        if not self.batch_first:
            src = src.transpose(0, 1)
        return src

    def _self_attn_block(self, x, attn_mask, key_padding_mask, is_causal):
        attn_output = self._scaled_dot_product_attention(
            x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, is_causal=is_causal
        )
        return self.dropout1(attn_output)

    def _ff_block(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout_ff(x)
        x = self.linear2(x)
        return self.dropout2(x)

    def _scaled_dot_product_attention(
        self,
        x,
        attn_mask=None,
        key_padding_mask=None,
        is_causal=False,
    ):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.rotary_pe is not None:
            q = self._apply_rope(q, seq_len)
            k = self._apply_rope(k, seq_len)

        attn_scores = torch.matmul(q * self.scale, k.transpose(-2, -1))

        if attn_mask is not None:
            attn_scores = attn_scores + self._expand_attn_mask(attn_mask, attn_scores)

        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(
                dtype=torch.bool, device=attn_scores.device
            )
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=attn_scores.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )
        return self.out_proj(attn_output)

    def _apply_rope(self, tensor, seq_len):
        batch_size, num_heads, seq_len_tensor, head_dim = tensor.shape
        if seq_len_tensor != seq_len:
            raise ValueError("Sequence length mismatch when applying RoPE")

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        tensor = self.rotary_pe.apply_rotary(tensor, seq_len)
        tensor = tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        return tensor

    def _expand_attn_mask(self, attn_mask, attn_scores):
        if attn_mask.dim() == 2:
            expanded = attn_mask.unsqueeze(0).unsqueeze(0)
        elif attn_mask.dim() == 3:
            expanded = attn_mask.unsqueeze(1)
        else:
            raise ValueError("Unsupported attn_mask dimensions for RoPE layer")

        expanded = expanded.to(attn_scores.device)
        if expanded.dtype == torch.bool:
            expanded = expanded.masked_fill(expanded, float("-inf"))
        return expanded

# ----------------------------------
# Temporal Transformer Encoder
# ----------------------------------


class RotaryTransformerEncoder(nn.Module):
    """Lightweight encoder stack that works with RotaryTransformerEncoderLayer."""

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=False):
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class TemporalTransformerEncoder(nn.Module):
    """
    CLS 토큰 + RoPE positional encoding
    (다른 PE는 더 이상 지원하지 않음)
    """
    def __init__(self, config_predictor):
        super().__init__()
        self.max_len = config_predictor['pred_len']+1
        self.pe_kind = config_predictor['transformer'].get('pe_kind', 'rope')
        self.d_model = config_predictor['transformer']['d_model']
        self.num_heads = config_predictor['transformer']['num_heads']
        self.dim_feedforward = config_predictor['transformer']['dim_feedforward']
        self.dropout = config_predictor['transformer']['dropout']
        self.batch_first = config_predictor['transformer']['batch_first']
        self.num_layers = config_predictor['transformer']['num_layers']
        self.z_q_dim = config_predictor['vq_embed_dim']
        self.num_features = config_predictor['num_features']
        # 시퀀스 전체 반환 여부
        self.return_sequence = config_predictor['transformer'].get('return_sequence', True)

        self.rotary_pe = RotaryPE(self.d_model, self.max_len)

        if self.pe_kind != 'rope':
            raise ValueError(f"TemporalTransformerEncoder requires pe_kind='rope', got {self.pe_kind}")

        layer = RotaryTransformerEncoderLayer(
            self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=self.batch_first,
            rotary_pe=self.rotary_pe,
        )

        self.z_q_proj = nn.Linear(self.z_q_dim, self.d_model)
        self.normalization = nn.LayerNorm(self.d_model)
        self.transformer = RotaryTransformerEncoder(layer, num_layers=self.num_layers)
        
        # Input projection layer - 입력 차원을 d_model에 맞춤
        self.input_proj = nn.Linear(self.num_features, self.d_model)

    def forward(self, x, z_q):                     # x: (B,T,D)
        B, T, D = x.shape
        
        # 입력 차원이 d_model과 일치하지 않으면 projection으로 변환
        if D != self.d_model:
            x = self.input_proj(x)  # (B, T, D) -> (B, T, d_model)
            D = self.d_model  # 차원 업데이트

        # 1) z_q를 전체 시퀀스에 더하기 (CLS 대신 additive fusion)
        if z_q.shape[1] != D:
            z_q = self.z_q_proj(z_q)
        # z_q를 CLS 토큰처럼 시퀀스 맨 앞에 추가
        x = torch.cat((z_q.unsqueeze(1), x), dim=1)  # (B, 1+T, D)
        # x = self.normalization(x) # ! 일단 normalization을 꺼보자. 

        # 2) Positional Encoding (RoPE handled inside attention)
        out = self.transformer(x)           # (B,T+1,D)
        
        # 3) CLS 토큰만 반환 (시계열 대표값)
        return out[:, 0]                   # (B,D)
    