import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .moe import FactorGatedMoE
from typing import Tuple, Optional


class GEGLU(nn.Module):
    """
    Gated Linear Unit with GELU activation
    
    Args:
        dim_in: Input dimension
        dim_out: Output dimension  
    """
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.proj(x).chunk(2, dim=-1)
        return a * F.gelu(b)
    
class ResBlock(nn.Module):
    """
    Residual Block with Pre-Norm → GEGLU → Linear → Dropout → Residual add
    
    Args:
        dim: Input/output dimension
        hidden: Hidden dimension (should be > dim for expansion)
        drop: Dropout probability
    """
    def __init__(self, dim: int, hidden: int, drop: float = 0.2):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.ff    = GEGLU(dim, hidden)
        self.proj  = nn.Linear(hidden, dim)
        self.drop  = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(self.ff(self.norm(x)))
        return x + self.drop(y)
    


class HyperFusion(nn.Module):
    """
    (h, z) → α, β_prior, β_latent (개선된 버전)
    - h로부터 동적 기저(base) β를 생성
    - z 기반 MoE의 출력으로 기저 β를 FiLM 방식으로 조절 (γ, δ)
    - α는 MoE 출력을 기반으로 예측
    """
    def __init__(self,
                 d_h: int,
                 d_z: int,
                 k_prior: int,
                 k_latent: int,
                 drop: float = 0.2,
                 num_experts: int = 4,
                 moe_k: int = 1,
                 hidden_size: int = 64): # moe_hidden은 작은 값(32 or 64) 유지
        super().__init__()
        
        # 1. 입력 프로젝션 (이전과 동일)
        input_dim = d_h + d_z
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_h), # hidden_dim -> d_h로 통일
            ResBlock(dim=d_h, hidden=d_h * 2, drop=drop),
            nn.Linear(d_h, d_h)
        )

        self.norm_h = nn.LayerNorm(d_h)
        self.norm_z = nn.LayerNorm(d_z)
        
        # 2. MoE
        self.moe = FactorGatedMoE(
            gate_input_size=d_z,
            expert_input_size=d_h,
            hidden_size=hidden_size, # MoE 내부 전문가의 hidden size
            num_experts=num_experts,
            k=moe_k
        )

        # 3. 동적 기저(Base) 생성 헤드 (h로부터 생성)
        # h의 시간적 정보가 전체적인 트렌드/기저를 형성
        self.base_beta_prior_head = nn.Linear(d_h, k_prior)
        self.base_beta_latent_head = nn.Linear(d_h, k_latent)
        
        # 4. FiLM 파라미터(γ, δ) 생성 헤드 (MoE 출력으로부터 생성)
        # MoE의 공간 전문성이 기저를 어떻게 조절할지 결정
        self.film_prior_head = nn.Linear(hidden_size, k_prior * 2)
        self.film_latent_head = nn.Linear(hidden_size, k_latent * 2)

        # 5. Alpha 예측 헤드
        self.alpha_head = nn.Linear(hidden_size, 1)

    def forward(self, h, z):

        h_norm = self.norm_h(h)
        z_norm = self.norm_z(z)

        x_fused = torch.cat([h_norm, z_norm], dim=-1)
        x_proj = self.input_proj(x_fused)
        moe_out, moe_loss = self.moe(x=x_proj, z=z_norm)
        
        # 2. h로부터 동적 기저 베타 생성
        base_beta_p = self.base_beta_prior_head(h)
        base_beta_l = self.base_beta_latent_head(h)

        # 3. MoE 출력으로 FiLM 파라미터(gamma, delta) 생성 및 분리
        gamma_p, delta_p = self.film_prior_head(moe_out).chunk(2, dim=-1)
        gamma_l, delta_l = self.film_latent_head(moe_out).chunk(2, dim=-1)

        # 4. FiLM을 이용해 최종 베타 계산: y = γ * x + δ
        beta_p = (gamma_p * base_beta_p) + delta_p
        beta_l = (gamma_l * base_beta_l) + delta_l
        
        # 5. Alpha 생성
        alpha = self.alpha_head(moe_out).squeeze(-1)

        # 베타 규제 loss는 여기서도 계산하여 반환
        beta_reg_loss = (torch.norm(beta_p, p=2) + torch.norm(beta_l, p=2))
        
        return alpha, beta_p, beta_l, moe_loss + beta_reg_loss