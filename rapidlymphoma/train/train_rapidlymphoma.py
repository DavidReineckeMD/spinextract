"""Contrastive learning experiment training script.

Copyright (c) 2022 University of Michigan. All rights reserved.
Licensed under the MIT License. See LICENSE for license information.
"""

import yaml
import logging
from functools import partial
from typing import Dict, Any

import torch

import pytorch_lightning as pl
import torchmetrics

from rapidlymphoma.models import MLP, resnet_backbone, ContrastiveLearningNetwork, vit_backbone, Network
from rapidlymphoma.train.common import (setup_output_dirs, parse_args, get_exp_name,
                                  get_contrastive_dataloaders, config_loggers,
                                  get_optimizer_func, get_scheduler_func)
from rapidlymphoma.losses.loss import SimilarityLoss


class ContrastiveSystem(pl.LightningModule):
    """Lightning system for contrastive learning experiments."""

    def __init__(self, cf: Dict[str, Any], num_it_per_ep: int):
        super().__init__()
        self.cf_ = cf

        if cf["model"]["backbone"] == "resnet50":
            bb = partial(resnet_backbone, arch=cf["model"]["backbone"])
        elif cf["model"]["backbone"] == "vit":
            bb = partial(vit_backbone, cf["model"]["backbone_params"])
        else:
            raise NotImplementedError()

        mlp = partial(MLP,
                      n_in=bb().num_out,
                      hidden_layers=cf["model"]["mlp_hidden"],
                      n_out=cf["model"]["num_embedding_out"])
        self.model = ContrastiveLearningNetwork(bb, mlp)
        self.criterion = SupConLoss()
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()

        self.num_it_per_ep_ = num_it_per_ep

    def predict_step(self, batch, batch_idx):
        out = self.model.bb(batch["image"])
        return {
            "path": batch["path"],
            "label": batch["label"],
            "embeddings": out
        }

    def on_train_epoch_end(self):
        train_loss = self.train_loss.compute()
        self.log("train/contrastive_manualepoch",
                 train_loss,
                 on_epoch=True,
                 sync_dist=False,
                 rank_zero_only=True)
        logging.info(f"train/contrastive_manualepoch {train_loss}")
        self.train_loss.reset()

    def on_validation_epoch_end(self):
        val_loss = self.val_loss.compute()
        self.log("val/contrastive_manualepoch",
                 val_loss,
                 on_epoch=True,
                 sync_dist=False,
                 rank_zero_only=True)
        logging.info(f"val/contrastive_manualepoch {val_loss}")
        self.val_loss.reset()

    def configure_optimizers(self):
        # if not training, no optimizer
        if "training" not in self.cf_:
            return None

        # get optimizer
        opt = get_optimizer_func(self.cf_)(self.model.parameters())

        # check if use a learn rate scheduler
        sched_func = get_scheduler_func(self.cf_, self.num_it_per_ep_)
        if not sched_func:
            return opt

        # get learn rate scheduler
        lr_scheduler_config = {
            "scheduler": sched_func(opt),
            "interval": "step",
            "frequency": 1,
            "name": "lr"
        }

        return [opt], lr_scheduler_config

    def configure_ddp(self, *args, **kwargs):
        logging.basicConfig(level=logging.INFO)
        return super().configure_ddp(*args, **kwargs)

class BYOLSystem(ContrastiveSystem):
    """BYOL in Pytorch lightning implementation"""

    def __init__(self, cf, num_it_per_ep):
        super().__init__(cf, num_it_per_ep)
        self.beta = cf["training"]["objective"]["params"]["ema_beta"]
        self.model = Network(get_backbone(cf))
        self.target_model = self._get_target_model()

    def forward(self, batch):
        online_out1 = self.model(batch["image"][:, 0, ...])
        online_out2 = self.model(batch["image"][:, 1, ...])

        with torch.no_grad():
            target_out1 = self.target_model(batch["image"][:, 0, ...])
            target_out2 = self.target_model(batch["image"][:, 1, ...])
            target_out1["proj"].detach_()
            target_out2["proj"].detach_()

        loss1 = self.criterion(online_out1["pred"],
                               target_out2["proj"].detach())
        loss2 = self.criterion(online_out2["pred"],
                               target_out1["proj"].detach())
        loss = loss1 + loss2
        return loss.mean()

    def training_step(self, batch, _):

        loss = self.forward(batch)
        bs = batch["image"].shape[0]
        self.log("train/contrastive",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 batch_size=bs,
                 rank_zero_only=True)
        self.train_loss.update(loss, weight=bs)
        return loss

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        bs = batch["image"].shape[0]
        self.val_loss.update(loss, weight=bs)

    def on_before_zero_grad(self, _):
        self.update_target_encoder()


    def _get_target_model(self):
        target_model = copy.deepcopy(self.model)
        for p in target_model.parameters():
            p.requires_grad = False
        return target_model

    def update_target_encoder(self):
        for p, pt in zip(self.model.parameters(),
                         self.target_model.parameters()):
            pt.data = self.beta * pt.data + (1 - self.beta) * p.data



def main():
    cf_fd = parse_args()
    cf = yaml.load(cf_fd, Loader=yaml.FullLoader)
    exp_root, model_dir, cp_config = setup_output_dirs(cf, get_exp_name, "")
    pl.seed_everything(cf["infra"]["seed"])

    # logging and copying config files
    cp_config(cf_fd.name)
    config_loggers(exp_root)

    # get dataloaders
    train_loader, valid_loader = get_contrastive_dataloaders(cf)
    logging.info(f"num devices: {torch.cuda.device_count()}")
    logging.info(f"num workers in dataloader: {train_loader.num_workers}")

    num_it_per_ep = len(train_loader)
    if torch.cuda.device_count() > 1:
        num_it_per_ep //= torch.cuda.device_count()

    if cf["training"]["objective"] == "byol":
        system_func = BYOLSystem
    else:
        raise NotImplementedError()

    ce_exp = system_func(cf, num_it_per_ep)

    # config loggers
    logger = [
        pl.loggers.TensorBoardLogger(save_dir=exp_root, name="tb"),
        pl.loggers.CSVLogger(save_dir=exp_root, name="csv")
    ]

    # config callbacks
    epoch_ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        save_top_k=-1,
        save_on_train_epoch_end=True,
        filename="ckpt-epoch{epoch}-loss{val/contrastive_manualepoch:.2f}",
        auto_insert_metric_name=False)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step",
                                                  log_momentum=False)

    # create trainer
    trainer = pl.Trainer(accelerator="gpu",
                         devices=-1,
                         default_root_dir=exp_root,
                         strategy=pl.strategies.DDPStrategy(
                             find_unused_parameters=False, static_graph=True),
                         logger=logger,
                         log_every_n_steps=10,
                         callbacks=[epoch_ckpt, lr_monitor],
                         max_epochs=cf["training"]["num_epochs"])
    trainer.fit(ce_exp,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader)


if __name__ == '__main__':
    main()
