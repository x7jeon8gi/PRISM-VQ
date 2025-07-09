import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange


class VectorQuantiser(nn.Module):
    """
    Improved version over vector quantiser, with the dynamic initialisation
    for these unoptimised "dead" points.
    num_embed: number of codebook entry
    embed_dim: dimensionality of codebook entry
    beta: weight for the commitment loss
    distance: distance for looking up the closest code
    anchor: anchor sampled methods
    first_batch: if true, the offline version of our model
    contras_loss: if true, use the contras_loss to further improve the performance
    """
    def __init__(self, num_embed, embed_dim, beta, distance='cos', 
                 anchor='probrandom', first_batch=False, contras_loss=False):
        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False

        self.pool = FeaturePool(self.num_embed, self.embed_dim)
        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
        self.register_buffer("embed_prob", torch.zeros(self.num_embed))

    
# forward 함수 내부의 수정 필요한 부분들:
    def forward(self, h_batch):
        # h_batch: 입력 텐서, shape: (B, D) == (N_t, d)

        # --- 1. 입력 Reshape 제거 ---
        # 기존 코드: z = rearrange(z, 'b c h w -> b h w c').contiguous()
        # 기존 코드: z_flattened = z.view(-1, self.embed_dim)
        # 수정 후: 입력 h_batch를 그대로 사용 (이미 (B, D) 형태)

        # --- 2. 거리 계산 시 입력 변수 수정 ---
        # 기존 코드: d = - torch.sum(z_flattened.detach() ** 2, ...) or d = torch.einsum('bd,dn->bn', normed_z_flattened, ...)
        # 수정 후: z_flattened.detach() -> h_batch.detach() 사용
        if self.distance == 'l2':
            # h_batch.detach() 사용
            d = - torch.sum(h_batch.detach() ** 2, dim=1, keepdim=True) - \
                torch.sum(self.embedding.weight ** 2, dim=1) + \
                2 * torch.einsum('bd, dn-> bn', h_batch.detach(), rearrange(self.embedding.weight, 'n d-> d n'))
        elif self.distance == 'cos':
            # h_batch.detach() 사용 (normed_z_flattened 대신)
            normed_h_batch = F.normalize(h_batch, dim=1).detach()
            normed_codebook = F.normalize(self.embedding.weight, dim=1)
            d = torch.einsum('bd,dn->bn', normed_h_batch, rearrange(normed_codebook, 'n d -> d n'))

        # --- 3. 양자화 벡터 생성 및 Reshape 수정 ---
        # encoding_indices 계산은 동일
        sort_distance, indices = d.sort(dim=1)
        encoding_indices = indices[:,-1]
        encodings = torch.zeros(encoding_indices.unsqueeze(1).shape[0], self.num_embed, device=h_batch.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # 기존 코드: z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        # 수정 후: view(z.shape) 제거. 결과는 이미 (B, D) 형태임. 변수명 변경 권장.
        z_q_vectors = torch.matmul(encodings, self.embedding.weight) # shape: (B, D)

        # --- 4. 손실 함수 계산 변수 수정 ---
        # 기존 코드: loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)
        # 수정 후: z -> h_batch, z_q -> z_q_vectors 사용
        loss = self.beta * torch.mean((z_q_vectors.detach() - h_batch)**2) + torch.mean((z_q_vectors - h_batch.detach()) ** 2)

        # --- 5. STE 적용 변수 및 최종 반환값 수정 ---
        # 기존 코드: z_q = z + (z_q - z).detach()
        # 기존 코드: z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        # 수정 후: 올바른 변수 사용 및 최종 reshape 제거. 반환값 형태는 (B, D).
        z_q_output = h_batch + (z_q_vectors - h_batch).detach() # STE 적용

        # perplexity, min_encodings 등 계산은 그대로 유지 가능 (encodings 사용)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        min_encodings = encodings # Note: min_encodings 변수명은 약간 오해의 소지가 있음. 실제로는 one-hot 인코딩임.

        # --- 6. Online Reinitialization / Contrastive Loss 입력 수정 ---
        if self.training:
            # calculate the average usage of code entries
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)
            if self.anchor in ['closest', 'random', 'probrandom'] and (not self.init):
                # 기존 코드: random_feat = z_flattened.detach()[...] 등
                # 수정 후: h_batch.detach() 사용
                if self.anchor == 'closest':
                    sort_distance_reinit, indices_reinit = d.sort(dim=0) # 여기서 d는 위에서 계산한 거리
                    random_feat = h_batch.detach()[indices_reinit[-1,:]]
                elif self.anchor == 'random':
                    random_feat = self.pool.query(h_batch.detach()) # FeaturePool 입력 확인
                elif self.anchor == 'probrandom':
                    norm_distance = F.softmax(d.t(), dim=1)
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = h_batch.detach()[prob]

                # decay parameter based on the average usage
                decay = torch.exp(-(self.embed_prob*self.num_embed*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.embed_dim)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
                if self.first_batch:
                    self.init = True

            if self.contras_loss:
                sort_distance_contra, indices_contra = d.sort(dim=0)
                dis_pos = sort_distance_contra[-max(1, int(sort_distance_contra.size(0)/self.num_embed)):,:].mean(dim=0, keepdim=True)
                dis_neg = sort_distance_contra[:int(sort_distance_contra.size(0)*1/2),:]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
                loss += contra_loss

        # 최종 반환: 양자화된 벡터 (STE 적용됨), 손실, 기타 정보
        return z_q_output, loss, (perplexity, min_encodings, encoding_indices)


class FeaturePool():
    """
    This class implements a feature buffer that stores previously encoded features

    This buffer enables us to initialize the codebook using a history of generated features
    rather than the ones produced by the latest encoders
    """
    def __init__(self, pool_size, dim=64):
        """
        Initialize the FeaturePool class

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1)/ pool_size

    def query(self, features):
        """
        return features from the pool
        """
        self.features = self.features.to(features.device)    
        if self.nums_features < self.pool_size:
            if features.size(0) > self.pool_size: # if the batch size is large enough, directly update the whole codebook
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features:num] = features
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[:features.size(0)]] = features

        return self.features
