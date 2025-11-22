from .types_ import *
from torch import nn
from abc import abstractmethod
from typing import Any, List

import pytorch_lightning as pl
import torch.optim as optim


class BaseVAE(pl.LightningModule):

    def __init__(self, exp_params: dict = None) -> None:
        super(BaseVAE, self).__init__()
        self.hparams.update(exp_params)

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

    def training_step(self, batch, batch_idx):
        results = self.forward(batch[0], labels=batch[1])
        train_loss = self.loss_function(*results, M_N = self.hparams['kld_weight'],
                                        optimizer_idx=0, batch_idx=batch_idx)
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        results = self.forward(batch[0], labels=batch[1])
        val_loss = self.loss_function(*results, M_N = self.hparams['kld_weight'],
                                      optimizer_idx=0, batch_idx=batch_idx)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams['LR'],
                               weight_decay=self.hparams['weight_decay'])
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=self.hparams['scheduler_gamma'])
        return [
            {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_loss', # Required for learning rate schedule and checkpointing
            }
        ]
