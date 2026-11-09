from pytorch_lightning import LightningModule
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
import torch
import math
from typing import Union
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging
import numpy as np
import os
import csv
from copy import deepcopy
from pytorch_lightning import Trainer
import ray
from ray import tune
from ray import ray_constants

from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.time_varying_model import BRCausalModel
from src.models.utils import grad_reverse, BRTreatmentOutcomeHead, AlphaRise
from src.models.utils_lstm import VariationalLSTM


logger = logging.getLogger(__name__)


class CRN(BRCausalModel):
    """
    Pytorch-Lightning implementation of Counterfactual Recurrent Network (CRN)
    (https://arxiv.org/abs/2002.04083, https://github.com/ioanabica/Counterfactual-Recurrent-Network)
    """

    model_type = None  # Will be defined in subclasses
    possible_model_types = {'encoder', 'decoder'}

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 bce_weights: np.array = None,
                 **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)


    def _patient_error_export_enabled(self):
        """Whether to export patient-level counterfactual sequence errors."""
        try:
            return bool(self.hparams.exp.get('export_patient_level_predictions', True))
        except Exception:
            return True

    def _patient_error_export_dir(self):
        exp_cfg = getattr(self.hparams, 'exp', None)
        configured = None
        phase_root = None
        seed = None
        try:
            configured = exp_cfg.get('patient_level_output_dir', None)
            phase_root = exp_cfg.get('phase_b_output_dir', None)
            seed = exp_cfg.get('seed', None)
        except Exception:
            pass

        if configured:
            out_dir = str(configured)
        elif phase_root:
            seed_tag = f'run_seed{seed}' if seed is not None else 'run'
            out_dir = os.path.join(str(phase_root), seed_tag, 'patient_level_errors')
        else:
            trainer_root = getattr(getattr(self, 'trainer', None), 'default_root_dir', '.')
            out_dir = os.path.join(str(trainer_root), 'patient_level_errors')

        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    @staticmethod
    def _patient_error_id_array(dataset, key, n_rows, fallback=None):
        for container_name in ('data_processed_seq', 'data'):
            container = getattr(dataset, container_name, None)
            if isinstance(container, dict) and key in container and container[key] is not None:
                arr = np.asarray(container[key])
                if arr.ndim > 0 and arr.shape[0] == n_rows:
                    return arr
        if fallback is None:
            return np.full(n_rows, '', dtype=object)
        return np.asarray(fallback)

    @staticmethod
    def _patient_error_scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _export_patient_level_sequence_errors(
            self, dataset, predictions_unscaled, targets_unscaled,
            active_entries, valid_rows):
        """Export CRN decoder errors for every patient-policy and every horizon.

        ``squared_error_sum`` is the exact row/horizon contribution to the numerator
        of the standard n-step RMSE calculation (before averaging over patients).
        """
        if not self._patient_error_export_enabled():
            return None

        pred = np.asarray(predictions_unscaled)
        true = np.asarray(targets_unscaled)
        active = np.asarray(active_entries)
        if pred.ndim == 2:
            pred = pred[:, :, None]
        if true.ndim == 2:
            true = true[:, :, None]
        if pred.shape != true.shape:
            raise RuntimeError(
                f'CRN patient-error export shape mismatch: pred={pred.shape}, true={true.shape}.'
            )

        n_rows, n_horizons, d_out = pred.shape
        if active.shape[0] != n_rows or active.shape[1] != n_horizons:
            raise RuntimeError(
                f'CRN active-entry shape {active.shape} incompatible with predictions {pred.shape}.'
            )
        if active.ndim == 2:
            active = active[:, :, None]

        valid = np.asarray(valid_rows, dtype=bool).reshape(-1)
        if valid.shape[0] != n_rows:
            raise RuntimeError(
                f'CRN valid mask length {valid.shape[0]} != {n_rows}.'
            )

        patient_ids = self._patient_error_id_array(
            dataset, 'patient_ids', n_rows, fallback=np.arange(n_rows)
        )
        trajectory_ids = self._patient_error_id_array(dataset, 'trajectory_ids', n_rows)
        policy_ids = self._patient_error_id_array(dataset, 'policy_ids', n_rows)
        trajectory_names = getattr(dataset, 'trajectory_names', None)

        subset = str(getattr(dataset, 'subset_name', 'evaluation'))
        safe_subset = ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in subset)
        try:
            seed = self.hparams.exp.get('seed', None)
        except Exception:
            seed = None
        seed_tag = f'_seed{seed}' if seed is not None else ''
        out_path = os.path.join(
            self._patient_error_export_dir(),
            f'CRN_{safe_subset}_H1-H{n_horizons}{seed_tag}_patient_errors.csv',
        )

        fieldnames = [
            'model', 'subset', 'horizon', 'row_index', 'patient_id',
            'trajectory_id', 'trajectory_name', 'policy_id', 'active_weight',
            'norm_const', 'squared_error_sum', 'absolute_error_mean', 'rmse_row',
        ]
        for k in range(d_out):
            fieldnames += [f'y_true_{k}', f'y_pred_{k}', f'error_{k}']

        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in np.flatnonzero(valid):
                traj_id = self._patient_error_scalar(trajectory_ids[i])
                traj_name = ''
                try:
                    if trajectory_names is not None and str(traj_id) != '':
                        traj_name = str(trajectory_names[int(traj_id)])
                except Exception:
                    traj_name = ''

                for h in range(n_horizons):
                    # active_entries is usually (N,K,1); broadcast across outputs.
                    a = active[i, h]
                    if a.size == 1:
                        mask = np.full(d_out, float(a.reshape(-1)[0]), dtype=float)
                    else:
                        mask = np.asarray(a, dtype=float).reshape(-1)
                        if mask.shape[0] != d_out:
                            mask = np.full(d_out, float(np.max(mask)), dtype=float)
                    if np.sum(mask) <= 0:
                        continue

                    err = pred[i, h] - true[i, h]
                    masked_se = (err ** 2) * mask
                    active_dims = mask > 0
                    row = {
                        'model': 'CRN',
                        'subset': subset,
                        'horizon': h + 1,
                        'row_index': int(i),
                        'patient_id': self._patient_error_scalar(patient_ids[i]),
                        'trajectory_id': traj_id,
                        'trajectory_name': traj_name,
                        'policy_id': self._patient_error_scalar(policy_ids[i]),
                        'active_weight': float(np.sum(mask)),
                        'norm_const': float(getattr(dataset, 'norm_const', 1.0)),
                        'squared_error_sum': float(np.sum(masked_se)),
                        'absolute_error_mean': float(np.mean(np.abs(err[active_dims]))),
                        'rmse_row': float(np.sqrt(np.mean((err[active_dims]) ** 2))),
                    }
                    for k in range(d_out):
                        row[f'y_true_{k}'] = float(true[i, h, k])
                        row[f'y_pred_{k}'] = float(pred[i, h, k])
                        row[f'error_{k}'] = float(err[k])
                    writer.writerow(row)

        logger.info('Saved CRN patient-level error export: %s', out_path)
        print(f'[patient-error-export] CRN -> {out_path}', flush=True)
        return out_path

    def _init_specific(self, sub_args: DictConfig):
        try:
            self.br_size = sub_args.br_size
            self.seq_hidden_units = sub_args.seq_hidden_units
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.dropout_rate = sub_args.dropout_rate
            self.num_layer = sub_args.num_layer

            required = (
                self.seq_hidden_units, self.br_size, self.fc_hidden_units,
                self.dropout_rate, self.num_layer,
            )
            if any(value is None for value in required):
                raise MissingMandatoryValue()

            self.lstm = VariationalLSTM(self.input_size, self.seq_hidden_units, self.num_layer, self.dropout_rate)

            self.br_treatment_outcome_head = BRTreatmentOutcomeHead(self.seq_hidden_units, self.br_size, self.fc_hidden_units,
                                                                    self.dim_treatments, self.dim_outcome, self.alpha,
                                                                    self.update_alpha, self.balancing)

        except MissingMandatoryValue:
            logger.warning(f"{self.model_type} not fully initialised - some mandatory args are missing! "
                           f"(It's ok, if one will perform hyperparameters search afterward).")

    def configure_optimizers(self):
        # Use the shared optimizer implementation so optimizer_cls, weight decay,
        # and scheduler behaviour are consistent across all baselines.
        return super().configure_optimizers()

    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        """
        Used for hyperparameter tuning and model reinitialisation
        :param model_args: Sub DictConfig, with encoder/decoder parameters
        :param new_args: New hyperparameters
        :param input_size: Input size of the model
        :param model_type: Submodel specification
        """
        sub_args = model_args[model_type]
        sub_args.optimizer.learning_rate = new_args['learning_rate']
        sub_args.batch_size = new_args['batch_size']
        if 'seq_hidden_units' in new_args:  # Only relevant for encoder
            sub_args.seq_hidden_units = int(input_size * new_args['seq_hidden_units'])
        sub_args.br_size = int(input_size * new_args['br_size'])
        sub_args.fc_hidden_units = int(sub_args.br_size * new_args['fc_hidden_units'])
        sub_args.dropout_rate = new_args['dropout_rate']
        sub_args.num_layer = new_args['num_layer']

    def build_br(self, prev_treatments, vitals_or_prev_outputs, static_features, init_states=None):

        dtype = next(self.parameters()).dtype
        prev_treatments = prev_treatments.to(dtype)
        vitals_or_prev_outputs = vitals_or_prev_outputs.to(dtype)
        static_features = static_features.to(dtype)

        if init_states is not None:
            init_states = init_states.to(dtype)

        x = torch.cat((prev_treatments, vitals_or_prev_outputs), dim=-1)
        x = torch.cat(
            (x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)),
            dim=-1
        )
        x = self.lstm(x, init_states=init_states)
        br = self.br_treatment_outcome_head.build_br(x)
        return br


class CRNEncoder(CRN):

    model_type = 'encoder'

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None, has_vitals: bool = None, bce_weights: np.array = None,
                 **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)
        self.input_size = self.dim_treatments + self.dim_static_features
        self.input_size += self.dim_vitals if self.has_vitals else 0
        self.input_size += self.dim_outcome if self.autoregressive else 0
        logger.info(f'Input size of {self.model_type}: {self.input_size}')

        self._init_specific(args.model.encoder)

    def prepare_data(self) -> None:
        # Datasets normalisation etc.
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_encoder:
            self.dataset_collection.process_data_encoder()
        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    def forward(self, batch, detach_treatment=False):
        dtype = next(self.parameters()).dtype
        batch = {
            k: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in batch.items()
        }
        prev_treatments = batch['prev_treatments']
        sequential_features = []
        if self.has_vitals:
            sequential_features.append(batch['vitals'])
        if self.autoregressive:
            sequential_features.append(batch['prev_outputs'])
        if not sequential_features:
            raise ValueError(
                'CRNEncoder requires at least one of vitals or previous outcomes.'
            )

        vitals_or_prev_outputs = torch.cat(sequential_features, dim=-1)
        static_features = batch['static_features']
        curr_treatments = batch['current_treatments']

        br = self.build_br(
            prev_treatments,
            vitals_or_prev_outputs,
            static_features,
            init_states=None,
        )
        treatment_pred = self.br_treatment_outcome_head.build_treatment(
            br, detach_treatment
        )
        outcome_pred = self.br_treatment_outcome_head.build_outcome(
            br, curr_treatments
        )
        return treatment_pred, outcome_pred, br


class CRNDecoder(CRN):

    model_type = 'decoder'

    def __init__(self, args: DictConfig,
                 encoder: CRNEncoder = None,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 encoder_r_size: int = None, autoregressive: bool = None, has_vitals: bool = None, bce_weights: np.array = None,
                 **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        self.input_size = self.dim_treatments + self.dim_static_features + self.dim_outcome
        logger.info(f'Input size of {self.model_type}: {self.input_size}')

        self.encoder = encoder
        resolved_encoder_size = (
            self.encoder.br_size if self.encoder is not None else encoder_r_size
        )
        if resolved_encoder_size is None:
            raise ValueError(
                'CRNDecoder requires either an encoder or encoder_r_size.'
            )

        # Do not mutate the shared Hydra/OmegaConf object in place.  In-place
        # mutation can leak hyperparameters across seeds and Ray Tune trials.
        decoder_args = deepcopy(args.model.decoder)
        decoder_args.seq_hidden_units = int(resolved_encoder_size)
        self._init_specific(decoder_args)

    def prepare_data(self) -> None:
        # Datasets normalisation etc.
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_decoder:
            self.dataset_collection.process_data_decoder(self.encoder)
        if self.bce_weights is None and self.hparams.exp.bce_weight:
            self._calculate_bce_weights()

    def forward(self, batch, detach_treatment=False):
        dtype = next(self.parameters()).dtype
        batch = {
            k: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in batch.items()
        }
        prev_treatments = batch['prev_treatments']
        prev_outputs = batch['prev_outputs']
        static_features = batch['static_features']
        curr_treatments = batch['current_treatments']
        init_states = batch['init_state']

        br = self.build_br(prev_treatments, prev_outputs, static_features, init_states)
        treatment_pred = self.br_treatment_outcome_head.build_treatment(br, detach_treatment)
        outcome_pred = self.br_treatment_outcome_head.build_outcome(br, curr_treatments)

        return treatment_pred, outcome_pred, br


    def get_normalised_n_step_rmses(self, dataset: Dataset, datasets_mc=None):
        """Standard CRN n-step RMSE plus automatic patient-level error export.

        This mirrors TimeVaryingCausalModel.get_normalised_n_step_rmses, but keeps
        the unreduced predictions/errors long enough to write the diagnostic CSV.
        """
        logger.info(f'RMSE calculation for {dataset.subset_name}.')
        assert hasattr(dataset, 'data_processed_seq')

        unscale = self.hparams.exp.unscale_rmse
        percentage = self.hparams.exp.percentage_rmse
        outputs_scaled = self.get_autoregressive_predictions(
            dataset if datasets_mc is None else datasets_mc
        )

        output_stds = dataset.scaling_params['output_stds']
        output_means = dataset.scaling_params['output_means']
        outputs_unscaled = outputs_scaled * output_stds + output_means
        targets_unscaled = np.asarray(dataset.data_processed_seq['unscaled_outputs'])
        active_entries = np.asarray(dataset.data_processed_seq['active_entries'])

        if unscale:
            mse = ((outputs_unscaled - targets_unscaled) ** 2) * active_entries
        else:
            mse = ((outputs_scaled - dataset.data_processed_seq['outputs']) ** 2) * active_entries

        output_ref = np.asarray(dataset.data_processed_seq['outputs'])
        nan_idx = np.unique(np.where(np.isnan(output_ref))[0])
        valid_rows = np.ones(outputs_scaled.shape[0], dtype=bool)
        valid_rows[nan_idx] = False
        not_nan = np.flatnonzero(valid_rows)

        # Export original-scale errors from the same autoregressive predictions used
        # for the official H1..HK RMSE calculation.  No extra model training is done.
        self._export_patient_level_sequence_errors(
            dataset, outputs_unscaled, targets_unscaled, active_entries, valid_rows
        )

        mse_orig = (
            mse[not_nan].sum(0).sum(-1)
            / active_entries[not_nan].sum(0).sum(-1)
        )
        rmses_normalised_orig = np.sqrt(mse_orig) / dataset.norm_const
        if percentage:
            rmses_normalised_orig *= 100.0
        return rmses_normalised_orig
