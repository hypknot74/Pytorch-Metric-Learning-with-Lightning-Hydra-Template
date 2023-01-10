from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MinMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.reducers import ThresholdReducer

class FlowerLitModule(LightningModule):
    """LightningModule for Flower classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        distance = CosineSimilarity()
        reducer = ThresholdReducer(low = 0)
        # WARNING: reducerをThresholdReducer(low = 0)に設定すると、バッチ内のlossで0より大きな値の
        # 平均を取るので、戻り値はスカラーであり、バッチサイズに限らずオーダーは同じになると考える。
        # しかし、実際はバッチサイズに応じてオーダーが異なる。原因は分かっていない。
        self.criterion = TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)

        # miner for triplet loss
        self.miner = TripletMarginMiner(margin=0.2, distance=distance)

        # metric objects for calculating and averaging accuracy across batches
        # self.train_acc = Accuracy(task="multiclass", num_classes=10)
        # self.val_acc = Accuracy(task="multiclass", num_classes=10)
        # self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acloss_best doesn't store accuracy from these checks
        self.val_loss_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        embeddings = self.forward(x)
        indices_tuple = self.miner(embeddings, y)
        loss = self.criterion(embeddings, y, indices_tuple)
        return loss, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        # self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        # self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        loss = self.val_loss.compute()  # get current valloss
        self.val_loss_best(loss)  # update best so far valloss
        # log `val_acloss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

    # def test_step(self, batch: Any, batch_idx: int):
    #     loss, targets = self.model_step(batch)

    #     # update and log metrics
    #     self.test_loss(loss)
    #     # self.test_acc(preds, targets)
    #     self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
    #     self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    #     return {"loss": loss, "targets": targets}

    # def test_epoch_end(self, outputs: List[Any]):
    #     pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)
