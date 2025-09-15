import os
# PyTorch를 import 하기 전에 환경 변수를 설정합니다.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
import torch
import wandb
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from trainer.train_ypred_wo_prior import GenerateReturn
from utils import load_yaml_param_settings, load_args, get_root_dir, seed_everything, log_metrics_as_bar_chart
from dataset.dataset import init_data_loader
from qlib.constant import REG_CN, REG_US
import pickle
from utils import run_inference

torch.set_float32_matmul_precision('high')


def get_region(region_code):
    region_map = {
        'CN': REG_CN,
        'US': REG_US,
    }
    return region_map.get(region_code, REG_CN)


def generate_run_name(config):
    n_expert   = config['predictor']['n_expert']
    k          = config['predictor']['k']
    dim        = config['predictor']['transformer']['d_model']
    n_heads    = config['predictor']['transformer']['num_heads']
    n_layer    = config['predictor']['transformer']['num_layers']
    dropout    = config['predictor']['transformer']['dropout']
    aux_weight = config['predictor']['aux_weight']
    kernel_size = config['predictor']['kernel_size']
    seed        = config['train']['seed']

    saved_model = config['predictor']['saved_model']
    model_name_part = saved_model.split('-')[0]
    params = model_name_part.split('_')

    param_dict = {}
    for part in params:
        if part.startswith('VQ'):
            param_dict['num_embed'] = part[2:]
        elif part.startswith('n'):
            param_dict['enc_heads'] = part[1:]
        elif part.startswith('e'):
            param_dict['vq_embed_dim'] = part[1:]
        elif part.startswith('d'):
            param_dict['dropout_pred'] = part[1:]
        elif part in ['l2', 'cos']:
            param_dict['distance'] = part

    num_embed = param_dict.get('num_embed', config['vqvae']['num_embed'])
    enc_heads = param_dict.get('enc_heads', config['vqvae']['encoder']['num_heads'])
    vq_embed_dim = param_dict.get('vq_embed_dim', config['vqvae']['vq_embed_dim'])
    dropout_pred = param_dict.get('dropout_pred', config['vqvae']['predictor']['dropout'])
    distance = param_dict.get('distance', config['vqvae']['quantizer']['distance'])
    moe_hidden = config['predictor']['moe_hidden']
    moe_drop = config['predictor']['dropout']
    horizon = config['predictor']['pred_len']
    aux_imp = config['predictor']['aux_imp']

    run_name = (
        f'{seed}_VQ{num_embed}_{config["data"]["universe"]}_woPRIOR_'
        f'mo{n_expert}_k{k}_mh{moe_hidden}_md{moe_drop}_'
        f'dm{dim}_nh{n_heads}_l{n_layer}_d{dropout}_'
        f'au{aux_weight}_1h{enc_heads}_1e{vq_embed_dim}_1d{dropout_pred}_1{distance}_p{horizon}_ai{aux_imp}'
    )
    return run_name


def train(config, train_loader, valid_loader, num_batches_per_epoch_train, test_loader, wandb_run=None):
    run_name = generate_run_name(config)
    T_max = num_batches_per_epoch_train * config['train']['num_epochs']
    model = GenerateReturn(config, T_max=T_max)

    if wandb_run:
        wandb_logger = WandbLogger(experiment=wandb_run)
    else:
        project_name = config['train']['project_name']+'_'+config['data']['universe']+'_stage2_wo_prior'
        group_name = config['data']['universe']
        wandb_logger = WandbLogger(project=project_name, name=run_name, config=config, group=group_name, entity="x7jeon8gi")
    wandb_logger.watch(model, log='all')

    early_stopping_config = config['train']['early_stopping']

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor= early_stopping_config['monitor'],
        mode= early_stopping_config['mode'],
        dirpath=os.path.join(get_root_dir(), 'checkpoints'),
        filename=f'{run_name}'+'-{epoch}-{val_loss:.4f}',
    )

    early_stop_callback = EarlyStopping(
        monitor=early_stopping_config['monitor'],
        min_delta=early_stopping_config['min_delta'],
        patience=early_stopping_config['patience'],
        verbose=early_stopping_config['verbose'],
        mode=early_stopping_config['mode']
    )

    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=True,
                         callbacks=[LearningRateMonitor(logging_interval='step'), checkpoint_callback, early_stop_callback],
                         max_epochs=config['train']['num_epochs'],
                         accelerator='gpu', devices=1,
                         precision=config['train']['precision'],
                         gradient_clip_val=config['train'].get('gradient_clip_val', 2))

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    best_checkpoint_path = checkpoint_callback.best_model_path
    print(f"========== Best checkpoint path: {best_checkpoint_path} ==========")
    print(f"========== Best validation loss: {checkpoint_callback.best_model_score} ==========")

    best_model = GenerateReturn.load_from_checkpoint(best_checkpoint_path, config=config, T_max=T_max)
    best_model.freeze_vqvae()
    best_model.eval()

    with torch.no_grad():
        print(f"========== Validation 데이터 평가 시작 ==========")
        _, _, val_metric = run_inference(best_model, valid_loader, config)
        print(f"Validation RIC: {val_metric['RankIC']:.4f}")

        print(f"========== Test 데이터 평가 시작 ==========")
        pred_df, _, metric = run_inference(best_model, test_loader, config)
        print(f"Test RIC: {metric['RankIC']:.4f}")

    # wo_prior 폴더 생성 및 저장
    wo_prior_dir = os.path.join(get_root_dir(), config['train']['save_res'], 'wo_prior')
    # run_name에서 seed (0_, 1_, 2_, 3_, 4_) 제거 후 폴더로 만든 뒤, seed별 파일 저장
    parts = run_name.split('_')
    seed_part = parts[0]  # 첫 번째 부분이 seed (0, 1, 2, 3, 4 등)
    model_name = '_'.join(parts[1:])  # seed 제거한 나머지 부분
    wo_prior_dir = os.path.join(wo_prior_dir, model_name)
    os.makedirs(wo_prior_dir, exist_ok=True)
    
    output_path = os.path.join(wo_prior_dir, f"{seed_part}_best.pkl")
    pred_df.to_pickle(output_path)
    output_csv_path = os.path.join(wo_prior_dir, f"{seed_part}_metric.csv")
    pd.DataFrame([metric], index=['values']).transpose().to_csv(output_csv_path)
    print(f"Results saved to {output_path} and {output_csv_path}")

    log_metrics_as_bar_chart(metric, model_name=run_name)
    wandb.log({'metrics_best': metric})
    wandb.finish()
    return metric['RankIC']


def main(config, wandb_run=None):
    seed_everything(config['train']['seed'])

    if config['data']['universe'] == 'csi300':
        region_code = 'CN'; universe_prefix = 'csi300'
    elif config['data']['universe'] == 'sp500':
        region_code = 'US'; universe_prefix = 'sp500'
    elif config['data']['universe'] == 'csi500':
        region_code = 'CN'; universe_prefix = 'csi500'
    else:
        raise ValueError(f"Invalid universe: {config['data']['universe']}")

    print(f"Seed value: {config['train']['seed']}")
    print(f"********** Region: {region_code} **********")
    print(f"********** Universe: {config['data']['universe']} **********")

    pickle_path = config['data'].get('data_path')
    if pickle_path and os.path.exists(pickle_path):
        print(f"========== Loading data from pickle: {pickle_path} ==========")
        train_pickle_path = f"{pickle_path}/{region_code}/{universe_prefix}_{config['data']['window_size']}_h{config['vqvae']['predictor']['pred_len']}_dl_train.pkl"
        valid_pickle_path = f"{pickle_path}/{region_code}/{universe_prefix}_{config['data']['window_size']}_h{config['vqvae']['predictor']['pred_len']}_dl_valid.pkl"
        test_pickle_path = f"{pickle_path}/{region_code}/{universe_prefix}_{config['data']['window_size']}_h{config['vqvae']['predictor']['pred_len']}_dl_test.pkl"
        required_files = [train_pickle_path, valid_pickle_path, test_pickle_path]
        if not all(os.path.exists(p) for p in required_files):
            missing_files = [p for p in required_files if not os.path.exists(p)]
            raise FileNotFoundError(f"Missing required pickle files: {', '.join(missing_files)}. Please ensure data is pre-processed.")
        train_prepare = pickle.load(open(train_pickle_path, 'rb'))
        valid_prepare = pickle.load(open(valid_pickle_path, 'rb'))
        test_prepare = pickle.load(open(test_pickle_path, 'rb'))
    else:
        raise FileNotFoundError(f"Data path '{pickle_path}' not found or not specified in config. This script requires pre-processed pickle data.")

    num_workers = config['train']['num_workers']
    train_loader, num_batches_per_epoch_train = init_data_loader(train_prepare, shuffle=True, num_workers=num_workers)
    valid_loader, _ = init_data_loader(valid_prepare, shuffle=False, num_workers=num_workers)
    test_loader, _ = init_data_loader(test_prepare, shuffle=False, num_workers=num_workers)

    ric_score = train(config, train_loader, valid_loader, num_batches_per_epoch_train, test_loader, wandb_run=wandb_run)
    return ric_score


if __name__ == "__main__":
    args = load_args()
    config = load_yaml_param_settings(args.config, args.seed)
    metric = main(config)
    print(metric)

