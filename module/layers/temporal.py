import torch, math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from module.layers.src.encoding import *

# ----------------------------------
# 2. PE Factory - 최신 고성능 방법들 추가!
# ----------------------------------
def build_pe(kind, d_model, max_len):
    if kind == "sin":
        return SinusoidalPE(d_model, max_len)
    elif kind == "learnable":
        return LearnableAbsPE(d_model, max_len)
    elif kind == "relative":
        return RelativePE(d_model, max_len)
    elif kind == "rope":
        return RotaryPE(d_model, max_len)
    elif kind == "time2vec":
        return Time2VecPE(d_model)
    elif kind == "time2vec2":
        return Time2VecPE2(d_model)
    elif kind == "tape":  # time Absolute Position Encoding
        return tAPE(d_model, max_len)
    elif kind == "tupe":  # Transformer with Untied Positional Encoding  
        return TUPE(d_model, max_len)
    elif kind == "conv_spe":  # Convolutional Sinusoidal PE
        return ConvSPE(d_model, max_len=max_len)
    elif kind == "temporal_pe":  # Temporal PE (T-PE)
        return TemporalPE(d_model, max_len)
    elif kind == "adaptive":  # 🚀 우리의 혁신적 방법!
        return AdaptiveTemporalPE(d_model, max_len)
    elif kind == "fourier":  # Fourier-based PE
        return FourierPE(d_model, max_len)
    else:
        raise ValueError(f"Unknown PE: {kind}")

# ----------------------------------
# 3. Temporal Transformer Encoder
# ----------------------------------
class TemporalTransformerEncoder(nn.Module):
    """
    CLS 토큰 + 선택형 Positional Encoding
    🚀 새로운 고성능 PE 방법들 지원!
    
    kind ∈ {sin, learnable, relative, rope, time2vec, tape, tupe, conv_spe, temporal_pe, adaptive, fourier}
    """
    def __init__(self, config_predictor):
        super().__init__()
        self.max_len = config_predictor['pred_len']+1
        self.pe_kind = config_predictor['transformer']['pe_kind']      # 유저 설정
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

        # Positional Encoding 모듈 - 새로운 방법들 포함!
        self.pe = build_pe(self.pe_kind, self.d_model, self.max_len)

        # Transformer 블록 - 항상 기본 layer 사용
        layer = nn.TransformerEncoderLayer(
            self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=self.batch_first,
            # norm_first = True, # True하는 순간 성능 떡락
            # activation = nn.GELU(),
        )
        self.z_q_proj = nn.Linear(self.z_q_dim, self.d_model)
        self.normalization = nn.LayerNorm(self.d_model)
        self.transformer = nn.TransformerEncoder(layer, num_layers=self.num_layers)
        
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

        # 2) Positional Encoding
        if self.pe_kind in {"sin", "learnable", "time2vec", "tape", "conv_spe", "temporal_pe", "adaptive", "fourier"}:
            x = self.pe(x)
        elif self.pe_kind == "rope":
            # RoPE: 쿼리·키 회전에 직접 적용
            x_pe = self.pe.apply_rotary(x, T+1)  # (B,T+1,D)
            x = x_pe
        elif self.pe_kind == "tupe":
            # TUPE: 사전 적용 방식으로 변경 (호환성 보장)
            x = self._apply_tupe_pe(x)
        # relative PE는 attention 내부서 사용 → 생략

        # 3) Transform
        out = self.transformer(x)           # (B,T+1,D)
        
        # 4) CLS 토큰만 반환 (시계열 대표값)
        return out[:, 0]                   # (B,D)
    
    def _apply_tupe_pe(self, x):
        """TUPE PE를 미리 적용하는 안전한 방법"""
        if hasattr(self.pe, 'pos_q') and hasattr(self.pe, 'pos_k'):
            seq_len = x.size(1)
            # 단순화: query position encoding만 적용
            pos_encoding = self.pe.pos_q[:seq_len].unsqueeze(0)  # (1, T, d_model)
            return x + pos_encoding
        else:
            return x  # fallback
