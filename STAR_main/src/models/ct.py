from pytorch_lightning import LightningModule
from omegaconf import DictConfig
import torch
from torch import nn
from omegaconf.errors import MissingMandatoryValue
import torch.nn.functional as F
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Dataset, Subset
import logging
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from functools import partial
import seaborn as sns
from sklearn.manifold import TSNE

from src.models.edct import EDCT
from src.models.utils_transformer import TransformerMultiInputBlock, LayerNorm
from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.utils import BRTreatmentOutcomeHead

logger = logging.getLogger(__name__)


class CT(EDCT):

    model_type = 'multi'
    possible_model_types = {'multi'}

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 projection_horizon: int = None,
                 bce_weights: np.array = None, **kwargs):

        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        if self.dataset_collection is not None:
            self.projection_horizon = self.dataset_collection.projection_horizon
        else:
            self.projection_horizon = projection_horizon

        # Used in hparam tuning
        self.input_size = max(self.dim_treatments, self.dim_static_features, self.dim_vitals, self.dim_outcome)
        logger.info(f'Max input size of {self.model_type}: {self.input_size}')
        assert self.autoregressive  # prev_outcomes are obligatory

        self.basic_block_cls = TransformerMultiInputBlock
        self._init_specific(args.model.multi)
        self.save_hyperparameters(args)

        self.lr = args.model.multi.optimizer.learning_rate

    # =========================================================================
    # PL >= 1.8 patch #1: force the single-optimizer (grad_reverse, no-EMA) path
    # =========================================================================
    def configure_optimizers(self):
        if self.balancing != 'grad_reverse':
            raise RuntimeError(
                f"CT (PL>=1.8 patched) requires exp.balancing='grad_reverse', "
                f"got {self.balancing!r}. Either switch to grad_reverse or downgrade "
                f"pytorch-lightning to 1.7.x to use domain_confusion."
            )
        if self.hparams.exp.weights_ema:
            raise RuntimeError(
                "CT (PL>=1.8 patched) requires exp.weights_ema=False. "
                "EMA averaging needs two optimizers; downgrade pytorch-lightning "
                "to 1.7.x if you need EMA."
            )

        optimizer = self._get_optimizer(list(self.named_parameters()))
        if self.hparams.model[self.model_type]['optimizer']['lr_scheduler']:
            return self._get_lr_schedulers(optimizer)
        return optimizer

    # =========================================================================
    # PL >= 1.8 patch #2: training_step without optimizer_idx,
    # functionally identical to BRCausalModel.training_step(optimizer_idx=0)
    # under the grad_reverse + no-EMA configuration.
    # =========================================================================
    def training_step(self, batch, batch_ind):
        for par in self.parameters():
            par.requires_grad = True

        treatment_pred, outcome_pred, _ = self(batch)

        mse_loss = F.mse_loss(outcome_pred, batch['outputs'].float(), reduce=False)
        t_loss = self.treatment_loss(
            treatment_pred, batch['current_treatments'].float(), kind='predict'
        )

        # Masking for shorter sequences — matches BRCausalModel.training_step
        t_loss = (batch['active_entries'].squeeze(-1) * t_loss).sum() / batch['active_entries'].sum()
        mse_loss = (batch['active_entries'] * mse_loss).sum() / batch['active_entries'].sum()
        loss = t_loss + mse_loss

        self.log(f'{self.model_type}_train_loss', loss,
                 on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{self.model_type}_train_treatment_loss', t_loss,
                 on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{self.model_type}_train_mse_loss', mse_loss,
                 on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{self.model_type}_alpha', self.br_treatment_outcome_head.alpha,
                 on_epoch=True, on_step=False, sync_dist=True)

        return loss

    # =========================================================================
    # Model architecture (unchanged from original CT)
    # =========================================================================
    def _init_specific(self, sub_args: DictConfig):
        try:
            super(CT, self)._init_specific(sub_args)

            if self.seq_hidden_units is None or self.br_size is None or self.fc_hidden_units is None \
                    or self.dropout_rate is None:
                raise MissingMandatoryValue()

            self.treatments_input_transformation = nn.Linear(self.dim_treatments, self.seq_hidden_units)
            self.vitals_input_transformation = \
                nn.Linear(self.dim_vitals, self.seq_hidden_units) if self.has_vitals else None
            self.outputs_input_transformation = nn.Linear(self.dim_outcome, self.seq_hidden_units)
            self.static_input_transformation = nn.Linear(self.dim_static_features, self.seq_hidden_units)

            self.n_inputs = 3 if self.has_vitals else 2  # prev_outcomes and prev_treatments

            self.transformer_blocks = nn.ModuleList(
                [self.basic_block_cls(self.seq_hidden_units, self.num_heads, self.head_size,
                                      self.seq_hidden_units * 4,
                                      self.dropout_rate,
                                      self.dropout_rate if sub_args.attn_dropout else 0.0,
                                      self_positional_encoding_k=self.self_positional_encoding_k,
                                      self_positional_encoding_v=self.self_positional_encoding_v,
                                      n_inputs=self.n_inputs,
                                      disable_cross_attention=sub_args.disable_cross_attention,
                                      isolate_subnetwork=sub_args.isolate_subnetwork) for _ in range(self.num_layer)])

            self.br_treatment_outcome_head = BRTreatmentOutcomeHead(
                self.seq_hidden_units, self.br_size,
                self.fc_hidden_units, self.dim_treatments, self.dim_outcome,
                self.alpha, self.update_alpha, self.balancing,
                # NOTE: treatment_mode 由 EDCT.__init__ 设置（仅支持 'binary' / 'continuous'）
                treatment_mode=self.treatment_mode,
            )
        except MissingMandatoryValue:
            logger.warning(f"{self.model_type} not fully initialised - some mandatory args are missing! "
                           f"(It's ok, if one will perform hyperparameters search afterward).")

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            self.dataset_collection.process_data_multi()

        if self.treatment_mode == 'binary' \
                and self.bce_weights is None \
                and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    def forward(self, batch, detach_treatment=False):
        dtype = next(self.parameters()).dtype
        batch = {
            k: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in batch.items()
        }
        fixed_split = batch['future_past_split'] if 'future_past_split' in batch else None

        if self.training and self.hparams.model.multi.augment_with_masked_vitals and self.has_vitals:
            # Augmenting original batch with vitals-masked copy
            assert fixed_split is None  # Only for training data
            fixed_split = torch.empty((2 * len(batch['active_entries']),)).type_as(batch['active_entries'])
            for i, seq_len in enumerate(batch['active_entries'].sum(1).int()):
                fixed_split[i] = seq_len  # Original batch
                fixed_split[len(batch['active_entries']) + i] = torch.randint(0, int(seq_len) + 1, (1,)).item()

            for (k, v) in batch.items():
                batch[k] = torch.cat((v, v), dim=0)

        prev_treatments = batch['prev_treatments']
        vitals = batch['vitals'] if self.has_vitals else None
        prev_outputs = batch['prev_outputs']
        static_features = batch['static_features']
        curr_treatments = batch['current_treatments']
        active_entries = batch['active_entries']

        br = self.build_br(prev_treatments, vitals, prev_outputs, static_features, active_entries, fixed_split)
        treatment_pred = self.br_treatment_outcome_head.build_treatment(br, detach_treatment)
        outcome_pred = self.br_treatment_outcome_head.build_outcome(br, curr_treatments)

        return treatment_pred, outcome_pred, br

    def build_br(self, prev_treatments, vitals, prev_outputs, static_features, active_entries, fixed_split):

        dtype = next(self.parameters()).dtype
        prev_treatments = prev_treatments.to(dtype)
        prev_outputs = prev_outputs.to(dtype)
        static_features = static_features.to(dtype)

        if vitals is not None:
            vitals = vitals.to(dtype)

        active_entries_treat_outcomes = torch.clone(active_entries)
        active_entries_vitals = torch.clone(active_entries)

        if fixed_split is not None and self.has_vitals:  # Test sequence data / Train augmented data
            for i in range(len(active_entries)):
                # Masking vitals in range [fixed_split: ]
                active_entries_vitals[i, int(fixed_split[i]):, :] = 0.0
                vitals[i, int(fixed_split[i]):] = 0.0

        x_t = self.treatments_input_transformation(prev_treatments)
        x_o = self.outputs_input_transformation(prev_outputs)
        x_v = self.vitals_input_transformation(vitals) if self.has_vitals else None
        x_s = self.static_input_transformation(static_features.unsqueeze(1))

        for block in self.transformer_blocks:

            if self.self_positional_encoding is not None:
                x_t = x_t + self.self_positional_encoding(x_t)
                x_o = x_o + self.self_positional_encoding(x_o)
                x_v = x_v + self.self_positional_encoding(x_v) if self.has_vitals else None

            if self.has_vitals:
                x_t, x_o, x_v = block((x_t, x_o, x_v), x_s, active_entries_treat_outcomes, active_entries_vitals)
            else:
                x_t, x_o = block((x_t, x_o), x_s, active_entries_treat_outcomes)

        if not self.has_vitals:
            x = (x_o + x_t) / 2
        else:
            if fixed_split is not None:  # Test seq data
                x = torch.empty_like(x_o)
                for i in range(len(active_entries)):
                    # Masking vitals in range [fixed_split: ]
                    x[i, :int(fixed_split[i])] = \
                        (x_o[i, :int(fixed_split[i])] + x_t[i, :int(fixed_split[i])] + x_v[i, :int(fixed_split[i])]) / 3
                    x[i, int(fixed_split[i]):] = (x_o[i, int(fixed_split[i]):] + x_t[i, int(fixed_split[i]):]) / 2
            else:  # Train data always has vitals
                x = (x_o + x_t + x_v) / 3

        output = self.output_dropout(x)
        br = self.br_treatment_outcome_head.build_br(output)
        return br

    def get_autoregressive_predictions(self, dataset: Dataset) -> np.array:

        logger.info(f'Autoregressive Prediction for {dataset.subset_name}.')

        predicted_outputs = np.zeros((len(dataset), self.hparams.dataset.projection_horizon, self.dim_outcome))

        for t in range(self.hparams.dataset.projection_horizon + 1):
            logger.info(f't = {t + 1}')
            outputs_scaled = self.get_predictions(dataset)

            for i in range(len(dataset)):
                split = int(dataset.data['future_past_split'][i])
                if t < self.hparams.dataset.projection_horizon:
                    dataset.data['prev_outputs'][i, split + t, :] = outputs_scaled[i, split - 1 + t, :]
                if t > 0:
                    predicted_outputs[i, t - 1, :] = outputs_scaled[i, split - 1 + t, :]

        return predicted_outputs

    def visualize(self, dataset: Dataset, index=0, artifacts_path=None):

        fig_keys = ['self_attention_o', 'self_attention_t', 'cross_attention_ot', 'cross_attention_to']
        if self.has_vitals:
            fig_keys += ['cross_attention_vo', 'cross_attention_ov', 'cross_attention_vt', 'cross_attention_tv',
                         'self_attention_v']
        self._visualize(fig_keys, dataset, index, artifacts_path)