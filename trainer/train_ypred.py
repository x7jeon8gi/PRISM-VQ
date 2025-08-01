import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from module.autoregressive import LoadingGenerator
from utils import get_root_dir, calc_ic
from utils.rankloss import RankLoss
from module.quantise import VectorQuantiser
from module.layers.encoder import SpatialEncoder
from module.layers.decoder import ReconstructionDecoder
from module.layers.src import RevIN
from utils import corr_cluster_order
from torch.optim.lr_scheduler import LambdaLR
import math
from module.layers.src import ListNetLoss

# def quantile_regression_loss(y_pred: torch.Tensor, y_true: torch.Tensor, tau: float):
#     """
#     배치 단위로 Quantile Regression Loss(tilted loss) 계산
#     Args:
#         y_pred: (B,) 예측값
#         y_true: (B,) 실제값
#         tau: 분위수 (0 < tau < 1), 예: 0.1 또는 0.9
#     Returns:
#         loss: 스칼라 텐서
#     """
#     diff = y_true - y_pred  # (B,)
#     # tau * max(diff,0) + (1 - tau) * max(-diff,0)
#     # → 구형 함수(tilted), 좌/우 다른 기울기를 줌
#     loss = torch.max(tau * diff, (tau - 1) * diff)
#     # loss는 (B,) 형태, 배치 평균 반환
#     return torch.mean(loss)

# def combined_quantile_loss(y_pred: torch.Tensor, y_true: torch.Tensor, quantiles: list = [0.1, 0.5, 0.9]):
#     """
#     여러 분위수에 대한 Quantile Loss의 가중 평균
#     Args:
#         y_pred: (B,) 예측값 
#         y_true: (B,) 실제값
#         quantiles: 사용할 분위수 리스트
#     Returns:
#         loss: 스칼라 텐서
#     """
#     total_loss = 0.0
#     for tau in quantiles:
#         total_loss += quantile_regression_loss(y_pred, y_true, tau)
#     return total_loss / len(quantiles)

class LatentValueHead(nn.Module):
    def __init__(self, d_latent, K):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_latent, K),
            nn.LayerNorm(K),
            nn.GELU(),
            nn.Linear(K, K)
        )
    def forward(self, z_q):
        return self.head(z_q)    # (B, K)

def check_vq_idx(idx, K):
    if torch.any(idx >= K) or torch.any(idx < 0):
        bad = idx[(idx >= K) | (idx < 0)]
        raise ValueError(f"[VQ-IDX] out-of-range values: {bad.tolist()} (K_latent={K})")
    
class GenerateReturn(pl.LightningModule):
    def __init__(self,
                 config,
                 T_max,
                 ):
        super().__init__()
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

        # 1. Encoder
        self.encoder = SpatialEncoder(
            input_features_C = self.num_features,
            T_window=self.seq_len,
            gru_hidden_size=self.hidden_size,
            num_transformer_heads=self.transformer_heads,
            num_transformer_layers=self.transformer_layers,
            final_embed_dim_d=self.vq_embed_dim
        )

        # 2. Vector Quantizer
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
        
        # 3. RevIN
        self.revin = RevIN(self.num_features)

        # # 4. Decoder
        # self.decoder = ReconstructionDecoder(
        #     latent_dim=self.vq_embed_dim,      # d
        #     prior_factor_dim=self.num_prior_factors, # P
        #     output_T=self.seq_len,             # T_window
        #     output_C=self.num_features,        # C
        #     initial_T=self.initial_T,
        #     hidden_channels=self.hidden_channels,
        #     norm_type=vqvae_cfg['decoder'].get('norm_type', 'none'),
        #     num_groups=vqvae_cfg['decoder'].get('num_groups', 8)
        # )

        self.saved_model = config['predictor']['saved_model']
        self.load_pretrained_vqvae(checkpoint_path=os.path.join('checkpoints', f"{self.saved_model}"))
        self.freeze_vqvae()

        # 4. Factor Loading
        self.z_prior_norm = nn.LayerNorm(self.num_prior_factors)
        self.loadings = LoadingGenerator(config)
        self.latent_value_head = LatentValueHead(
            d_latent=self.vq_embed_dim,
            K = self.vq_embed_dim
        )

        self.use_prior = config['predictor']['use_prior']
        # 5. Return Predictor
        self.return_predictor = ReturnPredictor(num_prior  = self.num_prior_factors, 
                                                num_latent = self.vq_embed_dim,
                                                use_prior = self.use_prior)

        self.n_features = config['vqvae']['num_features']
        self.n_prior_factors = config['vqvae']['num_prior_factors']

        self.T_max = T_max
        self.target_index = config['predictor']['target_day'] - 1 # ex. 5 -> 4 (start from 0)
        
        self.aux_weight = config['predictor']['aux_weight']

        self.ic = []
        self.ric = []
        self.best_val_loss = float('inf')
        self.best_metrics_at_min_loss = {}
        self.rank = config['predictor']['rank']
        self.rank_loss = RankLoss(alpha=self.rank)
        self.listNet_loss = ListNetLoss(temperature=1.0)

    def configure_optimizers(self):
        optimizer  = torch.optim.AdamW(self.parameters(), lr=self.config['train']['learning_rate'], weight_decay=1e-5)
        # 1) Warm-up 단계: 총 T_warmup 스텝 동안 선형으로 lr 증가
        total_steps = self.T_max
        warmup_steps = int(0.05 * total_steps)  # 예: 전체 스텝의 5%를 워밍업으로 할당

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # 선형 워밍업: 초기 lr * (current_step / warmup_steps)
                return float(current_step) / max(1, warmup_steps)
            # 워밍업 이후에는 Cosine 스케줄러
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # scheduler  = CosineAnnealingLR(optimizer, T_max=self.T_max)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        sch_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [sch_config]
    
    def _get_data(self, batch, batch_idx):
        batch   = batch.squeeze(0)
        batch   = batch.float()
        feature = batch[:, :, 0:self.n_features] # (300, 20, 158)
        prior_factor = batch[:, -1, self.n_features : self.n_features+self.n_prior_factors] # (300, 13)
        future_returns = batch[:, -1, self.n_features+self.n_prior_factors  : ] # (300, 1, 10)
        future_returns = future_returns.squeeze(-1) # (300, 10)
        
        label = future_returns[:, self.target_index] # (300, 1)

        return feature, prior_factor, label
    
    def forward(self, feature, prior_factor):
        
        ####### STAGE1: VQVAE #######
        # 1. RevIN + Encoder
        feature_normalized = self.revin(feature, mode="norm")
        h_batch = self.encoder(feature_normalized)  # (B, H)
        # 2. Quantization
        z_q, _, (_, min_encodings, vq_idx) = self.quantizer(h_batch)
        z_q = z_q.detach()
        # 3. Decoder + RevIN
        #recon_feature = self.decoder(z_q, prior_factor) # prior factor에 대한 norm은 decoder 내부에서 처리
        #denorm_recon_feature = self.revin(recon_feature, mode="denorm")

        ####### STAGE2: Loading Generator #######
        # 1. Loading Generator 모델 호출 방식 변경
        alpha, beta_p, beta_l, loss_imp = self.loadings(feature, z_q) # todo: 그냥 feature 넣는 것 고려
        prior_factor_normed = self.z_prior_norm(prior_factor)
        
        f_latent = self.latent_value_head(z_q)

        # 2. 예측
        y_pred = self.return_predictor(
            alpha    = alpha,
            beta_p   = beta_p,
            beta_l   = beta_l,              # soft weights?
            f_prior  = prior_factor_normed, # (B,P)
            f_latent = f_latent,            # (B,K)
        )
        loss_imp = torch.clamp(loss_imp, min=0, max=1)
        return y_pred, beta_p, beta_l, z_q, loss_imp


    def training_step(self, batch, batch_idx):
        feature, prior_factor, label = self._get_data(batch, batch_idx)
        y_pred, beta_p, beta_l, z_q, aux_loss = self.forward(feature, prior_factor)

        # 기본 MSE 손실 (RankLoss 사용)
        mse_loss = self.rank_loss(y_pred, label)
        rank_loss = self.listNet_loss(y_pred, label)
        
        # Quantile Loss 추가 (옵션)
        main_loss = mse_loss + rank_loss
            
        loss = main_loss + self.aux_weight * aux_loss
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_mse_loss', mse_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_rank_loss', rank_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('train_aux_loss', aux_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        feature, prior_factor, label = self._get_data(batch, batch_idx)
        y_pred, beta_p, beta_l, z_q, aux_loss = self.forward(feature, prior_factor)

        # 기본 MSE 손실 (RankLoss 사용)
        mse_loss = self.rank_loss(y_pred, label)
        rank_loss = self.listNet_loss(y_pred, label)
        
        # Quantile Loss 추가 (옵션)
        main_loss = mse_loss + rank_loss
            
        loss = main_loss + self.aux_weight * aux_loss
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_mse_loss', mse_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_rank_loss', rank_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_aux_loss', aux_loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        
        daily_ic, daily_ric = calc_ic(y_pred.cpu().numpy(), label.cpu().numpy())
        self.ic.append(daily_ic)
        self.ric.append(daily_ric)
        return {"loss": loss}
            
    def on_validation_epoch_end(self):
        current_ic = np.mean(self.ic)
        current_ric = np.mean(self.ric)
        current_icir = np.mean(self.ic) / np.std(self.ic) if np.std(self.ic) != 0 else 0
        current_ricir = np.mean(self.ric) / np.std(self.ric) if np.std(self.ric) != 0 else 0

        metric = {
            'Val_IC': current_ic,
            'Val_ICIR': current_icir,
            'Val_RIC': current_ric,
            'Val_RICIR': current_ricir,
        }
        self.log_dict(metric, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        
        # Reset the IC and RIC lists
        self.ic = []
        self.ric = []

        val_loss_epoch = self.trainer.callback_metrics.get('val_loss')
        val_rank_loss_epoch = self.trainer.callback_metrics.get('val_rank_loss')
        
        if val_loss_epoch is not None and val_loss_epoch < self.best_val_loss:
            # 현재 에폭이 이전까지의 최소 validation loss보다 낮다면 업데이트
            self.best_val_loss = val_loss_epoch
            self.best_metrics_at_min_loss = {
                'Best_Val_Loss': float(val_loss_epoch),
                'Best_Val_IC': current_ic,
                'Best_Val_ICIR': current_icir,
                'Best_Val_RIC': current_ric,
                'Best_Val_RICIR': current_ricir,
            }
            self.log_dict(self.best_metrics_at_min_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        if val_loss_epoch is not None:
            self.log('val_loss_epoch', val_loss_epoch, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        if val_rank_loss_epoch is not None:
            self.log('val_rank_loss_epoch', val_rank_loss_epoch, on_step=False, on_epoch=True, logger=True, sync_dist=True)


    def init_from_ckpt(self, path, ignore_keys=list()):
        """사전 훈련된 모델 가중치를 로드하는 함수"""
        sd = torch.load(path, map_location="cuda")
        
        # state_dict가 여러 형태로 저장되었을 수 있음
        if "state_dict" in sd:
            sd = sd["state_dict"]
        
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        
        self.load_state_dict(sd, strict=False)
        print(f"모델이 {path}에서 복원되었습니다.")
    
    def load_pretrained_vqvae(self, checkpoint_path=None):
        """
        사전 훈련된 FVQVAE 모델에서 필요한 컴포넌트(GRU, Encoder, Quantizer)를 로드하는 함수
        """
        # 체크포인트 로드
        print(f"사전 훈련된 FVQVAE 모델을 {checkpoint_path}에서 로드합니다...")
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # FVQVAE 모델에서 필요한 컴포넌트 가중치만 추출하여 할당
        revin_state_dict = {}
        encoder_state_dict = {}
        quantizer_state_dict = {}
        decoder_state_dict = {}
        
        for k, v in state_dict.items():
            # Encoder 관련 가중치
            if k.startswith('vqvae.spatial_encoder.'):
                encoder_state_dict[k.replace('vqvae.spatial_encoder.', '')] = v
            # Quantizer 관련 가중치
            elif k.startswith('vqvae.quantizer.'):
                quantizer_state_dict[k.replace('vqvae.quantizer.', '')] = v
            # RevIN 관련 가중치
            elif k.startswith('vqvae.revin.'):
                revin_state_dict[k.replace('vqvae.revin.', '')] = v
            # # Decoder 관련 가중치
            # elif k.startswith('vqvae.decoder.'):
            #     decoder_state_dict[k.replace('vqvae.decoder.', '')] = v

        # 각 컴포넌트에 가중치 로드
        missing_encoder, unexpected_encoder = self.encoder.load_state_dict(encoder_state_dict, strict=True)
        missing_quantizer, unexpected_quantizer = self.quantizer.load_state_dict(quantizer_state_dict, strict=True)
        missing_revin, unexpected_revin = self.revin.load_state_dict(revin_state_dict, strict=True)
        # missing_decoder, unexpected_decoder = self.decoder.load_state_dict(decoder_state_dict, strict=True)
        # 로드 결과 출력
        print(f"--- Encoder 로드 완료: missing={len(missing_encoder)}, unexpected={len(unexpected_encoder)}")
        print(f"--- Quantizer 로드 완료: missing={len(missing_quantizer)}, unexpected={len(unexpected_quantizer)}")
        print(f"--- RevIN 로드 완료: missing={len(missing_revin)}, unexpected={len(unexpected_revin)}")
        # print(f"--- Decoder 로드 완료: missing={len(missing_decoder)}, unexpected={len(unexpected_decoder)}")
        # # 코드북 로딩 확인
        # if hasattr(self.quantizer, 'embedding') and '_embedding' in quantizer_state_dict:
        #     print("코드북 가중치 로드 완료! 코드북 크기:", self.quantizer.embedding.weight.shape)
        
        # return True

    def freeze_vqvae(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.quantizer.parameters():
            param.requires_grad = False
        for param in self.revin.parameters():
            param.requires_grad = False
        print("== FVQ-VAE 모델 가중치 고정 완료 ==")

        self.encoder.eval()
        self.quantizer.eval()
        self.revin.eval()
        # self.decoder.eval() -> further training 필요

class ReturnPredictor(nn.Module):
    def __init__(self, num_prior, num_latent, use_prior=True):
        super().__init__()
        self.num_prior = num_prior
        self.num_latent = num_latent
        self.use_prior = use_prior
        
    def forward(self, alpha, beta_p, beta_l, f_prior, f_latent):
        prior_term = (beta_p * f_prior).sum(dim=1)
        latent_term = (beta_l * f_latent).sum(dim=1) # [B, K] 요소 곱

        combined = torch.cat([prior_term.unsqueeze(1), latent_term.unsqueeze(1)], dim=1)
        # intercept_term = self.final_layer(combined).squeeze(-1)
        
        if self.use_prior:  
            output = alpha + prior_term + latent_term # + intercept_term
        else:
            output = alpha + latent_term

        return output