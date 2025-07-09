import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, n_feature, hidden_dim):
        super().__init__()
        self.n_feature = n_feature
        self.hidden_dim = hidden_dim

        # LayerNorm은 특징 차원(C)을 인수로 받음
        self.normalize = nn.LayerNorm(n_feature)
        self.linear = nn.Linear(n_feature, n_feature)
        self.leakyrelu = nn.LeakyReLU()
        self.get_h = nn.GRU(n_feature, hidden_dim, batch_first=True)
       
    def forward(self, x):
        # x: (B, T, C)
        x = self.linear(x)
        # LayerNorm은 (B, T, C) 형태에 바로 적용 가능 (마지막 차원에 대해 정규화)
        x = self.normalize(x)
        x = self.leakyrelu(x)

        # GRU 입력
        _, h_n = self.get_h(x) # h_n shape: (1, B, H) for num_layers=1
        return h_n.squeeze(0) # (B, H) - 활성화 없이 반환하는 것이 일반적일 수 있음

class CrossAssetTransformerEncoder(nn.Module):
    """ 주식 간의 관계를 모델링하는 Transformer 인코더 """
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 num_layers, 
                 d_out,
                 dim_feedforward=None):
        
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * embed_dim # 일반적인 설정
        # RMSNorm 구현
        class RMSNorm(nn.Module):
            def __init__(self, dim, eps=1e-6):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim))
                self.bias = nn.Parameter(torch.zeros(dim))  # 더미 bias 추가
            
            def forward(self, x):
                # x: [..., dim]
                # RMS 계산
                rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
                x = x / rms * self.weight
                return x
        
        # 커스텀 TransformerEncoderLayer 생성 (RMSNorm 사용)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            activation='gelu',
            batch_first=True,
            dropout=0.1,
            norm_first=True
        )
        
        # 기본 LayerNorm을 RMSNorm으로 교체
        encoder_layer.norm1 = RMSNorm(embed_dim)
        encoder_layer.norm2 = RMSNorm(embed_dim)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 출력 차원 변환을 위한 feed forward 네트워크
        self.out_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, d_out)
        )


    def forward(self, temporal_summaries):
        # temporal_summaries shape: (Batch=1 or N_t, embed_dim)
        # Transformer는 (Batch, Seq, Feature) 입력을 기대하므로 차원 조정 필요
        # 만약 N_t가 가변적이라면 패딩/마스킹 필요할 수 있음
        # 여기서는 입력이 (N_t, embed_dim)이고 Batch=1, Seq=N_t 라고 가정
        if temporal_summaries.dim() == 2:
            temporal_summaries = temporal_summaries.unsqueeze(0) # (1, N_t, embed_dim)

        # Add optional stock embeddings/positional encodings here if used

        # Transformer 인코더 적용
        # Output shape: (1, N_t, embed_dim)
        refined_representation = self.transformer_encoder(temporal_summaries)

        # Remove batch dimension if added
        if refined_representation.shape[0] == 1:
            refined_representation = refined_representation.squeeze(0) # (N_t, embed_dim)
        out = self.out_layer(refined_representation) # vq_dim 으로 변경 해줌 (factor의 차원)
        return out


class SpatialEncoder(nn.Module):
    """ 사전 학습 VQ-VAE의 인코더: Temporal GRU + Cross-Asset Transformer """
    def __init__(self, 
                 input_features_C,
                 T_window, 
                 gru_hidden_size, 
                 num_transformer_heads,
                 num_transformer_layers,
                 final_embed_dim_d
                 ):
        super().__init__()
        self.T_window = T_window
        self.C = input_features_C # 158
        self.gru_hidden_size = gru_hidden_size # 64 or 32 ?

        # 1. Temporal Feature Extractor (GRU)
        self.temporal_extractor = FeatureExtractor(input_features_C, gru_hidden_size)

        # 2. Cross-Asset Attention (Transformer)
        # Transformer의 입력 차원은 GRU의 출력 차원과 같아야 함
        self.cross_asset_transformer = CrossAssetTransformerEncoder(embed_dim=gru_hidden_size,
                                                                    num_heads=num_transformer_heads,
                                                                    num_layers=num_transformer_layers,
                                                                    d_out=final_embed_dim_d)


    def forward(self, x_batch):
        # x_batch shape: (N_t, T_window, C) - 특정 시점 t의 N_t개 주식 데이터
        N_t = x_batch.shape[0]

        # 1단계: 각 주식별로 Temporal GRU 적용
        # GRU는 batch 처리가 가능하므로 N_t를 batch 차원으로 사용
        # Input shape for GRU: (N_t, T_window, C)
        temporal_summaries = self.temporal_extractor(x_batch) # Output shape: (N_t, gru_hidden_size)

        # 2단계: 주식 간 관계 모델링 (Transformer)
        # Input shape for Transformer: (N_t, gru_hidden_size) -> 내부적으로 (1, N_t, gru_hidden_size)로 처리
        refined_representation = self.cross_asset_transformer(temporal_summaries) # Output shape: (N_t, gru_hidden_size)


        # h_batch의 각 행이 h_i 가 됨
        return refined_representation