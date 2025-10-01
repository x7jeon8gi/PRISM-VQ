from pathlib import Path
from typing import List, Optional

import hydra
import pickle
import pytorch_lightning as pl
import qlib
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from qlib.constant import REG_CN, REG_US
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DataHandlerLP, TSDatasetH

from dataset.dataset import init_data_loader
from trainer.train_vqvae import FactorVQVAE
from utils import get_root_dir, load_yaml_param_settings, seed_everything
from utils.wandb import make_wandb_config

torch.set_float32_matmul_precision('high')

# OmegaConf resolver 등록 - n_expert의 절반을 계산하는 함수
OmegaConf.register_new_resolver("half", lambda x: int(x) // 2)

def _build_run_name(cfg: DictConfig) -> str:
    if cfg.train.run_name != "auto":
        return cfg.train.run_name

    hidden_size = cfg.vqvae.hidden_size
    num_embed = cfg.vqvae.num_embed
    hidden_channels = cfg.vqvae.decoder.hidden_channels
    vq_embed_dim = cfg.vqvae.vq_embed_dim
    distance = cfg.vqvae.quantizer.distance
    pred_len = cfg.vqvae.predictor.pred_len
    seed = 42  # stage1.py는 항상 seed 42로 고정
    universe = cfg.data.universe

    return (
        f"infu{universe}_h{hidden_size}_VQK{num_embed}_C{hidden_channels}_"
        f"emb{vq_embed_dim}_d{distance}p{pred_len}_s{seed}"
    )

def _build_callbacks(cfg: DictConfig, run_name: str) -> List[Callback]:
    checkpoint_dir = Path(get_root_dir()) / cfg.train.save_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    early_cfg = cfg.train.early_stopping
    callbacks_cfg = [
        {
            "_target_": "pytorch_lightning.callbacks.LearningRateMonitor",
            "logging_interval": "step",
        },
        {
            "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
            "save_top_k": 1,
            "monitor": early_cfg.monitor,
            "mode": early_cfg.mode,
            "dirpath": str(checkpoint_dir),
            "filename": f"{run_name}" + "-{epoch}-{val_loss:.4f}",
        },
        {
            "_target_": "pytorch_lightning.callbacks.EarlyStopping",
            "monitor": early_cfg.monitor,
            "min_delta": early_cfg.min_delta,
            "patience": early_cfg.patience,
            "verbose": early_cfg.verbose,
            "mode": early_cfg.mode,
        },
    ]

    return [instantiate(cb_cfg) for cb_cfg in callbacks_cfg]

def _build_wandb_logger(cfg: DictConfig, run_name: str) -> WandbLogger:
    # config는 나중에 수동으로 업데이트할 거임
    logger_cfg = {
        "_target_": "pytorch_lightning.loggers.wandb.WandbLogger",
        "project": cfg.train.project_name,
        "name": run_name,
        "group": cfg.data.universe,
        "entity": cfg.train.get("wandb_entity"),
        "save_dir": str(Path(get_root_dir()) / cfg.train.save_dir),
        "log_model": cfg.train.get("wandb_log_model", False),
        "reinit": True,
    }

    logger_cfg = {k: v for k, v in logger_cfg.items() if v is not None}
    logger = instantiate(logger_cfg)
    
    # wandb가 초기화된 후에 config 업데이트
    try:
        clean_config = make_wandb_config(cfg)
        if hasattr(logger, 'experiment') and clean_config:
            logger.experiment.config.update(clean_config)
    except Exception:
        print("config 업데이트 실패")
        pass  # config 업데이트 실패해도 상관없음
    
    return logger


def train(cfg: DictConfig,
          config_dict: dict,
          train_loader,
          valid_loader,
          num_batches_per_epoch_train: int) -> None:
    run_name = _build_run_name(cfg)

    T_max = num_batches_per_epoch_train * cfg.train.num_epochs
    model = FactorVQVAE(config_dict, T_max)

    wandb_logger = _build_wandb_logger(cfg, run_name)
    wandb_logger.watch(model, log='all')
    callbacks = _build_callbacks(cfg, run_name)

    trainer = pl.Trainer(
        logger=wandb_logger,
        enable_checkpointing=True,
        callbacks=callbacks,
        max_epochs=cfg.train.num_epochs,
        accelerator=cfg.train.get("accelerator", "gpu"),
        devices=cfg.train.get("gpu_counts", 1),
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.get("gradient_clip_val", 5),
        deterministic=True,
        log_every_n_steps=cfg.train.get("log_every_n_steps", 50),
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    experiment = getattr(wandb_logger, "experiment", None)
    if experiment is not None:
        experiment.finish()


def get_region(region_code: str):
    """지역 코드에 따라 적절한 qlib 지역 상수를 반환합니다."""
    region_map = {
        'CN': REG_CN,
        'US': REG_US,
    }
    return region_map.get(region_code, REG_CN)  # 기본값은 REG_CN


def _load_data_handler_config(cfg: DictConfig) -> dict:
    default_path = Path(get_root_dir()) / "configs" / "data_handler_config.yaml"
    cfg_path = cfg.data.get("handler_config_path", str(default_path))
    abs_path = to_absolute_path(str(cfg_path))
    config = load_yaml_param_settings(abs_path)
    return config or {}


def _resolve_data_path(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    return Path(to_absolute_path(path))


def _prepare_dataset(cfg: DictConfig,
                     region_code: str,
                     universe_prefix: str,
                     data_handler_config: dict):
    data_path = _resolve_data_path(cfg.data.get('data_path'))
    pred_horizon = cfg.vqvae.predictor.pred_len
    window_size = cfg.data.window_size

    if not data_path:
        raise ValueError("data_path가 설정되지 않았습니다. config.yaml에서 data.data_path를 설정해주세요.")
    
    expected_files = {
        'train': data_path / region_code / f"{universe_prefix}_{window_size}_h{pred_horizon}_dl2_train.pkl",
        'valid': data_path / region_code / f"{universe_prefix}_{window_size}_h{pred_horizon}_dl2_valid.pkl",
    }
    
    missing = [str(p) for p in expected_files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"필요한 pickle 파일들이 존재하지 않습니다: {', '.join(missing)}")
    
    print(f"========== Loading data from pickle: {data_path} ==========")
    train_prepare = pickle.load(open(expected_files['train'], 'rb'))
    valid_prepare = pickle.load(open(expected_files['valid'], 'rb'))
    return train_prepare, valid_prepare


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # 실행 시점의 config를 메모리에 고정하여 실행 중 변경에 영향받지 않게 함
    frozen_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_readonly(frozen_cfg, True)
    
    config_dict = OmegaConf.to_container(frozen_cfg, resolve=True)
    data_handler_config = _load_data_handler_config(frozen_cfg)

    # stage1.py는 항상 seed 42로 고정
    fixed_seed = 42
    seed_everything(fixed_seed)
    pl.seed_everything(fixed_seed, workers=True)

    universe = frozen_cfg.data.universe
    if universe == 'csi300':
        region_code = 'CN'
        universe_prefix = 'csi300'
    elif universe == 'csi500':
        region_code = 'CN'
        universe_prefix = 'csi500'
    elif universe == 'sp500':
        region_code = 'US'
        universe_prefix = 'sp500'
    elif universe == 'nasdaq':
        region_code = 'US'
        universe_prefix = 'nasdaq'
    else:
        raise ValueError(f"Invalid universe: {universe}")

    print(f"Seed value: {fixed_seed}")
    print(f"Region: {region_code}")
    print(f"Universe: {universe}")

    train_prepare, valid_prepare = _prepare_dataset(frozen_cfg, region_code, universe_prefix, data_handler_config)

    num_workers = frozen_cfg.train.num_workers
    train_loader, num_batches_per_epoch_train = init_data_loader(train_prepare, shuffle=True, num_workers=num_workers)
    valid_loader, _ = init_data_loader(valid_prepare, shuffle=False, num_workers=num_workers)

    train(frozen_cfg, config_dict, train_loader, valid_loader, num_batches_per_epoch_train)


if __name__ == "__main__":
    main()
