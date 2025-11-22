import os
import yaml
import argparse
from pytorch_lightning import Trainer, seed_everything # Updated import for seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models import * # pylint: disable=wildcard-import, unused-wildcard-import
from dataset import VAEDataset

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config_path):
    config = load_config(config_path)

    # Seed everything for reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    # Instantiate the model
    model_name = config['model_params']['name']
    model_kwargs = {k: v for k, v in config['model_params'].items() if k != 'name'}
    
    try:
        # Pass experimental parameters to the model constructor
        model = vae_models[model_name](exp_params=config['exp_params'], **model_kwargs)
    except KeyError:
        raise ValueError(f"Model '{model_name}' not found. Available models are: {list(vae_models.keys())}")

    # Instantiate the data module
    data_module = VAEDataset(**config['data_params'])
    data_module.setup() # Call setup to prepare datasets

    # Loggers and Callbacks
    log_dir = os.path.join(config['logging_params']['save_dir'], config['logging_params']['name'])
    os.makedirs(log_dir, exist_ok=True)
    tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                  name=config['logging_params']['name'],
                                  default_hp_metric=False)

    checkpoint_callback = ModelCheckpoint(
        monitor=config['logging_params']['monitor_metric'],
        dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=config['logging_params']['save_top_k'],
        mode=config['logging_params']['mode']
    )

    # Trainer
    # Check for accelerator and devices based on PyTorch Lightning version
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = 1 if torch.cuda.is_available() else 'auto'

    runner = Trainer(logger=tb_logger,
                     callbacks=[checkpoint_callback],
                     accelerator=accelerator,
                     devices=devices,
                     max_epochs=config['trainer_params']['max_epochs'],
                     min_epochs=config['trainer_params']['min_epochs'],
                     num_sanity_val_steps=config['trainer_params']['num_sanity_val_steps'],
                     gradient_clip_val=config['trainer_params']['gradient_clip_val'],
                    )

    # Training
    print(f"Missing logger folder: {tb_logger.log_dir}")
    runner.fit(model, datamodule=data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c',
                        dest='filename',
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/cvae.yaml')

    args = parser.parse_args()
    main(args.filename)
