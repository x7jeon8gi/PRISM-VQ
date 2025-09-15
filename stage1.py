import torch
import wandb
import pandas as pd
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from qlib.data.dataset import TSDatasetH, DataHandlerLP
from trainer.train_vqvae import FactorVQVAE
import os
from utils import load_yaml_param_settings, load_args, get_root_dir, save_model, seed_everything
from dataset.dataset import init_data_loader
from utils.logger import set_logger
from qlib.constant import REG_CN, REG_US
import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data import D
import pickle
torch.set_float32_matmul_precision('high')

args = load_args()
config = load_yaml_param_settings(args.config)

config['train']['seed'] = 42
config['train']['num_epochs'] = 50
config['train']['early_stopping']['patience'] = 20
config['train']['learning_rate'] = 0.0001
config['train']['gradient_clip_val'] = 5

def train(config, train_loader, valid_loader, num_batches_per_epoch_train, num_batches_per_epoch_valid):

    hidden_size = config['vqvae']['hidden_size']
    num_embed = config['vqvae']['num_embed'] # ?코드북 크기
    decay = config['vqvae']['quantizer']['decay']
    project_name = config['train']['project_name']
    hidden_channels = config['vqvae']['decoder']['hidden_channels']
    seq_len = config['vqvae']['seq_len']
    vq_embed_dim = config['vqvae']['vq_embed_dim']
    distance = config['vqvae']['quantizer']['distance']
    pred_len = config['vqvae']['predictor']['pred_len']
    seed = config['train']['seed']
    universe = config['data']['universe']

    if config['train']['run_name'] == "auto":
        run_name = f'aaai{universe}_h{hidden_size}_VQK{num_embed}_C{hidden_channels}_emb{vq_embed_dim}_d{distance}p{pred_len}_s{seed}'
    else:
        run_name = config['train']['run_name']

    #* Init model
    T_max = num_batches_per_epoch_train * config['train']['num_epochs']
    model = FactorVQVAE(config, T_max)

    #* Init logger
    wandb.init(project=project_name, name=run_name, config=config, group=universe, entity="x7jeon8gi")
    wandb_logger = WandbLogger(project=project_name, name=run_name, config=config)
    wandb_logger.watch(model, log='all')

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        dirpath=os.path.join(get_root_dir(), 'checkpoints'),
        filename=f'{run_name}'+'-{epoch}-{val_loss:.4f}'
    )

    early_stopping_config = config['train']['early_stopping']
    early_stop_callback = EarlyStopping(
        monitor=early_stopping_config['monitor'],
        min_delta=early_stopping_config['min_delta'],
        patience=early_stopping_config['patience'], 
        verbose=early_stopping_config['verbose'],
        mode=early_stopping_config['mode']
    )

    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=True,
                         callbacks=[LearningRateMonitor(logging_interval='step'), 
                                   checkpoint_callback, 
                                   early_stop_callback],
                         max_epochs=config['train']['num_epochs'],
                         accelerator='gpu',
                         devices=1, 
                         precision=config['train']['precision'],
                         gradient_clip_val=config['train'].get('gradient_clip_val', 5),
                         deterministic=True
                         )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    wandb.finish()

def get_region(region_code):
    """지역 코드에 따라 적절한 qlib 지역 상수를 반환합니다."""
    region_map = {
        'CN': REG_CN,
        'US': REG_US,
    }
    return region_map.get(region_code, REG_CN)  # 기본값은 REG_CN

if __name__ == "__main__":
    #* Load config
    data_handler_config = load_yaml_param_settings(args.data_handler_config)
    
    # * Set seed
    seed_everything(config['train']['seed'])
    pl.seed_everything(config['train']['seed'])
    # # * Set logger
    # logger = set_logger(config['train']['run_name'], os.path.join(get_root_dir(), 'logs'))

    # * Set qlib
    if config['data']['universe'] == 'csi300':  
        region_code = 'CN'
        universe_prefix = 'csi300'
    elif config['data']['universe'] == 'csi500':
        region_code = 'CN'
        universe_prefix = 'csi500'
    elif config['data']['universe'] == 'sp500':
        region_code = 'US'
        universe_prefix = 'sp500'
    elif config['data']['universe'] == 'nasdaq':
        region_code = 'US'
        universe_prefix = 'nasdaq'
    else:
        raise ValueError(f"Invalid universe: {config['data']['universe']}")
    region = get_region(region_code)
    
    # * Load dataset
    print(f"Seed value: {config['train']['seed']}")
    print(f"Region: {region_code}")
    print(f"Universe: {config['data']['universe']}")
    
    ###### Load dataset ######
    if universe_prefix != 'nasdaq':
        # 피클 파일이 존재하면 로드하고, 없으면 Alpha158 사용
        pickle_path = config['data'].get('data_path')
        pred_horizon = config['vqvae']['predictor']['pred_len']
        if pickle_path and os.path.exists(pickle_path):
            print(f"========== Loading data from pickle: {pickle_path} ==========")
            train_prepare = pickle.load(open(f"{pickle_path}/{region_code}/{universe_prefix}_{config['data']['window_size']}_h{pred_horizon}_dl_train.pkl", 'rb'))
            valid_prepare = pickle.load(open(f"{pickle_path}/{region_code}/{universe_prefix}_{config['data']['window_size']}_h{pred_horizon}_dl_valid.pkl", 'rb'))
            test_prepare = pickle.load(open(f"{pickle_path}/{region_code}/{universe_prefix}_{config['data']['window_size']}_h{pred_horizon}_dl_test.pkl", 'rb'))
        
        else:
            print(f"Using Alpha158 handler with qlib data")
            qlib.init(provider_uri=config['data'].get('provider_uri', "./qlib_data/cn_data"), region=region)
            dataset = Alpha158(**data_handler_config)

            segments = {
                'train': config['data']['train_period'],
                'valid': config['data']['valid_period'],
                'test': config['data']['test_period'],
            }

            TsDataset = TSDatasetH(
                handler=dataset, 
                segments=segments, 
                step_len=config['data']['window_size'], 
            )

            train_prepare = TsDataset.prepare(segments='train', data_key=DataHandlerLP.DK_L)
            valid_prepare = TsDataset.prepare(segments='valid', data_key=DataHandlerLP.DK_L)
            test_prepare = TsDataset.prepare(segments='test', data_key=DataHandlerLP.DK_I)
            train_prepare.config(fillna_type='ffill+bfill')
            valid_prepare.config(fillna_type='ffill+bfill')
            test_prepare.config(fillna_type='ffill+bfill')
    else:
        dataset = pickle.load(open(f"{config['data']['data_path']}/{region_code}/{universe_prefix}_data.pkl", 'rb'))
        train_prepare = dataset.prepare(segments='train', data_key=DataHandlerLP.DK_L)
        valid_prepare = dataset.prepare(segments='valid', data_key=DataHandlerLP.DK_L)
        test_prepare = dataset.prepare(segments='test', data_key=DataHandlerLP.DK_I)
        train_prepare.config(fillna_type='ffill+bfill')
        valid_prepare.config(fillna_type='ffill+bfill')
        test_prepare.config(fillna_type='ffill+bfill')
        
    num_workers = config['train']['num_workers']
    train_loader, num_batches_per_epoch_train = init_data_loader(train_prepare, shuffle=True, num_workers=num_workers)
    valid_loader, num_batches_per_epoch_valid = init_data_loader(valid_prepare, shuffle=False, num_workers=num_workers)

    train(config, train_loader, valid_loader, num_batches_per_epoch_train, num_batches_per_epoch_valid)