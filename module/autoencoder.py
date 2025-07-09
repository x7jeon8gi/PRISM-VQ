import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from module.quantise import VectorQuantiser
from module.layers import ReconstructionDecoder
from module.layers import SpatialEncoder
from module.layers import SequencePredictorGRU
from module.layers.src import RevIN
# from module.layers.decoder import orthogonality_loss

class FVQVAE(nn.Module):
    def __init__(self, config):
        super(FVQVAE, self).__init__()
        self.config = config

        vqvae_cfg = config['vqvae']

        # VQVAE 파라미터
        self.num_prior_factors = vqvae_cfg['num_prior_factors'] # P
        self.num_embed     = vqvae_cfg['num_embed']      # K (코드북 크기)
        self.num_features  = vqvae_cfg['num_features'] # C
        self.hidden_size   = vqvae_cfg['hidden_size']  # H (GRU 출력 차원)
        self.vq_embed_dim  = vqvae_cfg['vq_embed_dim'] # d (VQ 차원, Encoder 출력 차원)
        self.seq_len       = vqvae_cfg['seq_len'] # T_window for reconstruction

        # Quantizer 파라미터
        self.decay         = vqvae_cfg['quantizer']['decay']
        self.commit_weight = vqvae_cfg['quantizer']['commit_weight']  # beta
        
        # Encoder 파라미터
        self.transformer_heads = vqvae_cfg['encoder']['num_heads'] 
        self.transformer_layers = vqvae_cfg['encoder']['num_layers']

        # Decoder 파라미터
        self.initial_T = vqvae_cfg['decoder']['initial_T']
        self.hidden_channels = vqvae_cfg['decoder']['hidden_channels']
        self.pred_weight = vqvae_cfg['predictor']['pred_weight']
        # self.ortho_loss_weight = vqvae_cfg['decoder']['ortho_loss_weight']

        self.spatial_encoder = SpatialEncoder(
            input_features_C=self.num_features,
            T_window=self.seq_len,
            gru_hidden_size=self.hidden_size,
            num_transformer_heads=self.transformer_heads,
            num_transformer_layers=self.transformer_layers,
            final_embed_dim_d=self.vq_embed_dim
        )

        self.quantizer = VectorQuantiser(
            num_embed=self.num_embed,         # K
            embed_dim=self.vq_embed_dim,      # d
            beta=self.commit_weight,
            # ... (distance, anchor 등 config에서 읽어오기) ...
            distance=vqvae_cfg['quantizer']['distance'],
            anchor=vqvae_cfg['quantizer']['anchor'],
            first_batch=vqvae_cfg['quantizer']['first_batch'],
            contras_loss=vqvae_cfg['quantizer']['contras_loss']
        )

        self.decoder = ReconstructionDecoder(
            latent_dim=self.vq_embed_dim,      # d
            prior_factor_dim=self.num_prior_factors, # P
            output_T=self.seq_len,             # T_window
            output_C=self.num_features,        # C
            initial_T=self.initial_T,
            hidden_channels=self.hidden_channels,
            norm_type=vqvae_cfg['decoder'].get('norm_type', 'none'),
            num_groups=vqvae_cfg['decoder'].get('num_groups', 8)
        )
        
        # 새로운 디코더 설계에서는 차원 독립적으로 작동
        print(f"✅ Decoder design: VQ embed dim ({self.vq_embed_dim}) → Hidden channels ({self.hidden_channels})")
        
        self.revin = RevIN(self.num_features)

        self.predictor = SequencePredictorGRU(
            latent_dim =self.vq_embed_dim,
            prior_factor_dim=self.num_prior_factors,
            hidden_dim =self.vq_embed_dim,
            output_seq_len = vqvae_cfg['predictor'].get('pred_len', 10),
            output_dim = 1, # 무조건 return series는 1개의 값만 있다.
            num_gru_layers = vqvae_cfg['predictor'].get('num_gru_layers', 1),
            dropout = vqvae_cfg['predictor'].get('dropout', 0.1)
        )

        self.layer_norm = nn.LayerNorm(self.num_prior_factors)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label) # ignore nan values
        loss = F.mse_loss(pred[mask], label[mask])
        return loss

    def forward(self, feature, prior_factor, future_returns):
        # feature: (B, T, C) - 여기서 T는 GRU 입력 길이
        # prior_factor: (B, P) 또는 (1, P)
        # future_returns: (B, H) - 예측 대상 정답


        prior_factor_normed = self.layer_norm(prior_factor)
        # 1. Reverse Instance Normalization
        feature_normalized = self.revin(feature, mode="norm")

        # 2. Cross-asset interaction (shared across assets)
        h_batch = self.spatial_encoder(feature_normalized)  # (B, d) - VQ 입력 준비

        # 3. Quantize factors
        # z_q는 STE가 적용된, 값은 e_k와 같고 그래디언트는 h_batch로부터 받는 벡터 (B, d)
        # vq_loss는 Commitment Loss + Codebook Loss 
        z_q, vq_loss, (perplexity, min_encodings, encoding_indices) = self.quantizer(h_batch)

        # 4. Decode back to features and compute reconstruction loss
        reconstruction  = self.decoder(z_q, prior_factor_normed)  # (B, T, C), hat_z_q, hat_z_p 반환 (B, d)
        decoded_feature = self.revin(reconstruction, mode="denorm")
        
        # 5. Compute reconstruction loss
        #ortho_loss = orthogonality_loss(hat_z_q, hat_z_p)
        recon_loss = self.loss_fn(decoded_feature, feature) # firm_feature는 (B, T, C), T==T_window 확인 필요

        # 6. Compute predictor loss
        future_returns_pred  = self.predictor(z_q, prior_factor_normed)
        pred_loss = self.loss_fn(future_returns_pred, future_returns)

        # 5. Compute total loss (VQ Loss 추가)
        total_loss = recon_loss +  vq_loss + self.pred_weight * pred_loss
        
        return recon_loss, vq_loss, pred_loss,total_loss , z_q, (perplexity, min_encodings, encoding_indices)