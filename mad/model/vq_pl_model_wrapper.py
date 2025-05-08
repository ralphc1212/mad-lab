import torch
import typing as tp
import pytorch_lightning as pl
import torchmetrics as met
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from mad.metrics import Accuracy



    
class VQPLModelWrap(pl.LightningModule):
    """
    PyTorch Lightning wrapper for VQ-based language models.

    Args:
        model (nn.Module): VQTransformerMAD or similar model.
        mad_config (MADConfig): configuration object.
        metrics (list, optional): list of metrics to compute (e.g., ["acc", "ppl"]).
    """
    def __init__(self, model, mad_config, metrics: list = ['acc', 'ppl']):
        super().__init__()
        self.model = model
        self.mad_config = mad_config
        self.loss_fn = nn.NLLLoss(ignore_index=self.mad_config.target_ignore_index)
        self.vq_loss_weight = getattr(self.mad_config, 'vq_loss_weight', 0.01)
        self.instantiate_metrics(metrics)
        self.save_hyperparameters('mad_config')

    def instantiate_metrics(self, metrics: list) -> None:
        mets = []
        for m in metrics:
            if m == 'acc':
                mets.append(Accuracy(
                    num_classes=self.model.vocab_size,
                    ignore_index=self.mad_config.target_ignore_index
                ))
            elif m == 'ppl':
                mets.append(met.text.Perplexity(ignore_index=self.mad_config.target_ignore_index))
            elif isinstance(m, met.Metric):
                mets.append(m)
            else:
                raise ValueError(f"Invalid metric: {m}. Use 'acc', 'ppl', or torchmetrics.Metric.")

        mets = met.MetricCollection(mets)
        self.train_metrics = mets.clone(prefix='train/')
        self.test_metrics = mets.clone(prefix='test/')

    def forward(self, input_ids, loss_mask=None, targets=None):
        return self.model(input_ids, loss_mask=loss_mask, targets=targets)

    def step(self, batch: tuple, batch_idx: int):
        input_ids, targets = batch
        
        model_out = self(input_ids, loss_mask=(input_ids != self.mad_config.target_ignore_index), targets=targets)
        log_probs = model_out["logprobs"]

        total_loss = model_out["loss"]
        log_probs = model_out["logprobs"]

        return total_loss, log_probs, targets, {
            'l_ce': model_out['l_lm_unscaled'].detach(),
            'l_vq': (model_out['l_commit'] + model_out['l_codebook']).detach(),
            **{f"metrics/{k}": v for k, v in model_out['metrics'].items()}
        }

    def phase_step(self, batch, batch_idx, phase='train'):
        loss, log_probs, targets, extras = self.step(batch, batch_idx)
        # if phase == 'train':
        #     phase_str = 'tr'
        # else:
        #     phase_str = 'te'

        self.log(f'{phase}/Loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f'{phase}/LCe', extras['l_ce'], on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f'{phase}/LVq', extras['l_vq'], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        metrics = getattr(self, f'{phase}_metrics')(log_probs, targets)

        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict({f"{phase}/{k}": v for k, v in extras.items() if k.startswith('metrics/')},
                      on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)


        return {'loss': loss, 'outputs': log_probs, 'targets': targets}

# train/Loss_step=-109., train/Accuracy_step=0.0395, train/Perplexity_step=inf.0, test/Loss_step=-3.32, test/Accuracy_step=0.0458, test/Perplexity_step=inf.

    def training_step(self, batch, batch_idx):
        return self.phase_step(batch, batch_idx, phase='train')

    def validation_step(self, batch, batch_idx):
        return self.phase_step(batch, batch_idx, phase='test')

    def test_step(self, batch, batch_idx):
        return self.phase_step(batch, batch_idx, phase='test')

    def configure_optimizers(self):
        if self.mad_config.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.mad_config.lr,
                weight_decay=self.mad_config.weight_decay
            )
        elif self.mad_config.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.mad_config.lr,
                weight_decay=self.mad_config.weight_decay
            )
        else:
            raise ValueError(f"invalid optimizer: {self.mad_config.optimizer}")

        if self.mad_config.scheduler == 'none':
            return optimizer
        elif self.mad_config.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.mad_config.epochs,
                eta_min=self.mad_config.min_lr,
                last_epoch=-1
            )
            return {'optimizer': optimizer, 'scheduler': scheduler}
        elif self.mad_config.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=self.mad_config.plateau_patience,
                factor=self.mad_config.plateau_factor,
                min_lr=self.mad_config.min_lr,
                verbose=True
            )
            return {'optimizer': optimizer, 'scheduler': scheduler, 'monitor': "test/Loss_epoch"}
        else:
            raise ValueError(f"invalid scheduler: {self.mad_config.scheduler}")
