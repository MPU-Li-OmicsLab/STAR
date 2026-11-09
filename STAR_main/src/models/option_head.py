from ray import ray_constants

from src.models.utils_transformer import AbsolutePositionalEncoding, RelativePositionalEncoding
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging
import numpy as np
from typing import Union

from src.models.utils_transformer import TransformerMultiInputBlock
from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.utils import OutcomeHead
from src.models.time_varying_model import TimeVaryingCausalModel
from src.models.mechanism.target_drift_monitor import TargetDriftMonitor

logger = logging.getLogger(__name__)
ray_constants.FUNCTION_SIZE_ERROR_THRESHOLD = 10 ** 8


class IGCNET(TimeVaryingCausalModel):

    model_type = 'igcnet'
    possible_model_types = {'igcnet'}

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 projection_horizon: int = None,
                 bce_weights: np.array = None,
                 path: str = None,
                 **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        if projection_horizon is not None:
            self.projection_horizon = projection_horizon
        elif args.model.igcnet.projection_horizon is not None:
            self.projection_horizon = args.model.igcnet.projection_horizon
        elif self.dataset_collection is not None:
            self.projection_horizon = args.dataset.projection_horizon
        else:
            raise MissingMandatoryValue()

        # Fairness-first: one fixed strategy per trained model, matching IGC-Net.
        treatment_sequence = args.dataset.get('treatment_sequence', None)
        if self.projection_horizon > 0 and treatment_sequence is None:
            raise MissingMandatoryValue(
                'Policy-specific training requires dataset.treatment_sequence.'
            )

        self.treatment_sequence = (
            torch.as_tensor(treatment_sequence, dtype=torch.float32)[:self.projection_horizon + 1, :]
            if treatment_sequence is not None else None
        )

        if self.treatment_sequence is not None:
            expected = int(self.projection_horizon) + 1
            if self.treatment_sequence.ndim != 2:
                raise ValueError(
                    'dataset.treatment_sequence must be 2D (K+1,D); '
                    f'got {tuple(self.treatment_sequence.shape)}.'
                )
            if self.treatment_sequence.shape[0] != expected:
                raise ValueError(
                    f'dataset.treatment_sequence must contain K+1={expected} '
                    f'interventions; got {self.treatment_sequence.shape[0]}.'
                )

        self.max_projection = (
            args.dataset.projection_horizon
            if self.dataset_collection is not None else self.projection_horizon
        )
        assert self.projection_horizon <= self.max_projection

        self.input_size = max(
            self.dim_treatments, self.dim_static_features, self.dim_vitals, self.dim_outcome
        )
        logger.info(f'Max input size of {self.model_type}: {self.input_size}')

        sub_args = args.model.igcnet

        self.stagewise_training = bool(sub_args.get('stagewise_training', True))
        self.stagewise_terminal_epochs = int(sub_args.get('stagewise_terminal_epochs', 25))
        self.stagewise_epochs_per_head = int(sub_args.get('stagewise_epochs_per_head', 40))
        self.stagewise_terminal_lr = float(
            sub_args.get('stagewise_terminal_lr', sub_args.optimizer.learning_rate)
        )
        self.stagewise_head_lr = float(
            sub_args.get('stagewise_head_lr', sub_args.optimizer.learning_rate)
        )
        self.stagewise_history_lr_scale = float(sub_args.get('stagewise_history_lr_scale', 0.25))
        self.stagewise_history_anchor_weight = float(
            sub_args.get('stagewise_history_anchor_weight', 0.0)
        )
        self.stagewise_log_history_drift = bool(sub_args.get('stagewise_log_history_drift', True))
        self.stagewise_fast_mode = bool(sub_args.get('stagewise_fast_mode', True))
        self.stagewise_require_full_schedule = bool(
            sub_args.get('stagewise_require_full_schedule', True)
        )
        self.stagewise_stop_after_schedule = bool(
            sub_args.get('stagewise_stop_after_schedule', True)
        )
        self.stagewise_fail_on_early_stopping = bool(
            sub_args.get('stagewise_fail_on_early_stopping', False)
        )
        self.predict_return_history = bool(sub_args.get('predict_return_history', False))
        self.monitor_max_samples = int(sub_args.get('monitor_max_samples', 200_000))

        if self.stagewise_history_lr_scale <= 0:
            raise ValueError('stagewise_history_lr_scale must be > 0.')
        if self.stagewise_history_anchor_weight != 0:
            raise ValueError(
                'SFTR+Naive Adaptation has no frozen representation anchor; '
                'set stagewise_history_anchor_weight=0.'
            )
        if self.stagewise_training:
            if self.stagewise_terminal_epochs <= 0:
                raise ValueError('stagewise_terminal_epochs must be >= 1.')
            if self.stagewise_epochs_per_head <= 0:
                raise ValueError('stagewise_epochs_per_head must be >= 1.')

        self._stagewise_active_stage = None
        self._stagewise_last_stage_logged = None
        self._stagewise_restored_stage = None

        self._eval_metric_cache = {'val': [], 'test': []}
        self._eval_metric_count = {'val': 0, 'test': 0}

        self.basic_block_cls = TransformerMultiInputBlock
        self._init_specific(args)
        self.tuning_criterion = 'rmse'
        self.save_hyperparameters(args)
        self.lr = args.model.igcnet.optimizer.learning_rate

        # Training-time mechanism instrumentation (disabled unless configured).
        # SFTR + Naive Adaptation:
        #   - downstream operator q_{j+1} is stage-frozen
        #   - target representation uses the current live backbone
        # Hence operator drift should vanish within a stage, while
        # representation-induced target drift can remain.
        self._mechanism_kind = "frozen_head"
        self.mechanism_monitor = TargetDriftMonitor.from_args(
            args, model_label="Frozen Head", mechanism_kind="frozen_head"
        )

    def _init_specific(self, args: DictConfig):
        try:
            sub_args = args.model.igcnet
            self.max_seq_length = sub_args.max_seq_length
            self.hr_size = sub_args.hr_size
            self.seq_hidden_units = sub_args.seq_hidden_units
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.dropout_rate = sub_args.dropout_rate
            self.num_layer = sub_args.num_layer
            self.num_heads = sub_args.num_heads

            if (self.seq_hidden_units is None or self.hr_size is None
                    or self.fc_hidden_units is None or self.dropout_rate is None):
                raise MissingMandatoryValue()

            self.head_size = sub_args.seq_hidden_units // sub_args.num_heads

            self.self_positional_encoding = None
            self.self_positional_encoding_k = None
            self.self_positional_encoding_v = None
            if sub_args.self_positional_encoding.absolute:
                self.self_positional_encoding = AbsolutePositionalEncoding(
                    self.max_seq_length, self.seq_hidden_units,
                    sub_args.self_positional_encoding.trainable,
                )
            else:
                self.self_positional_encoding_k = RelativePositionalEncoding(
                    sub_args.self_positional_encoding.max_relative_position,
                    self.head_size, sub_args.self_positional_encoding.trainable,
                )
                self.self_positional_encoding_v = RelativePositionalEncoding(
                    sub_args.self_positional_encoding.max_relative_position,
                    self.head_size, sub_args.self_positional_encoding.trainable,
                )

            self.cross_positional_encoding = None
            self.cross_positional_encoding_k = None
            self.cross_positional_encoding_v = None
            if ('cross_positional_encoding' in sub_args
                    and sub_args.cross_positional_encoding.absolute):
                self.cross_positional_encoding = AbsolutePositionalEncoding(
                    self.max_seq_length, self.seq_hidden_units,
                    sub_args.cross_positional_encoding.trainable,
                )
            elif 'cross_positional_encoding' in sub_args:
                self.cross_positional_encoding_k = RelativePositionalEncoding(
                    sub_args.cross_positional_encoding.max_relative_position,
                    self.head_size, sub_args.cross_positional_encoding.trainable,
                    cross_attn=True,
                )
                self.cross_positional_encoding_v = RelativePositionalEncoding(
                    sub_args.cross_positional_encoding.max_relative_position,
                    self.head_size, sub_args.cross_positional_encoding.trainable,
                    cross_attn=True,
                )

            self.treatments_input_transformation = nn.Linear(self.dim_treatments, self.seq_hidden_units)
            self.vitals_input_transformation = (
                nn.Linear(self.dim_vitals, self.seq_hidden_units) if self.has_vitals else None
            )
            self.outputs_input_transformation = nn.Linear(self.dim_outcome, self.seq_hidden_units)
            self.static_input_transformation = nn.Linear(self.dim_static_features, self.seq_hidden_units)
            self.n_inputs = 3 if self.has_vitals else 2

            self.transformer_blocks = nn.ModuleList([
                self.basic_block_cls(
                    self.seq_hidden_units, self.num_heads, self.head_size,
                    self.seq_hidden_units * 4, self.dropout_rate,
                    self.dropout_rate if sub_args.attn_dropout else 0.0,
                    self_positional_encoding_k=self.self_positional_encoding_k,
                    self_positional_encoding_v=self.self_positional_encoding_v,
                    n_inputs=self.n_inputs,
                    disable_cross_attention=sub_args.disable_cross_attention,
                    isolate_subnetwork=sub_args.isolate_subnetwork,
                )
                for _ in range(self.num_layer)
            ])
            self.hr_output_transformation = nn.Linear(self.seq_hidden_units, self.hr_size)
            self.output_dropout = nn.Dropout(self.dropout_rate)

            # IMPORTANT: standard IGC-Net head, not plan-conditioned head.
            self.G_comp_heads = nn.ModuleList([
                OutcomeHead(
                    self.seq_hidden_units, self.hr_size, self.fc_hidden_units,
                    self.dim_treatments, self.dim_outcome,
                )
                for _ in range(self.projection_horizon + 1)
            ])

        except MissingMandatoryValue:
            logger.warning(
                f'{self.model_type} not fully initialised - some mandatory args are missing! '
                '(OK during hyperparameter search).'
            )

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            self.dataset_collection.process_data_multi()

    def _elog(self, name, value, prog_bar=False):
        if torch.is_tensor(value):
            value = value.to(self.device)
        self.log(name, value, on_epoch=True, on_step=False, sync_dist=True, prog_bar=prog_bar)

    def _elog_many(self, metrics, prog_bar_suffixes=()):
        for name, value in metrics.items():
            self._elog(name, value, prog_bar=name.endswith(prog_bar_suffixes))

    def build_hr(self, prev_treatments, vitals, prev_outputs, static_features, active_entries):
        active_entries_treat_outcomes = active_entries
        active_entries_vitals = active_entries

        x_t = self.treatments_input_transformation(prev_treatments)
        x_o = self.outputs_input_transformation(prev_outputs)
        x_v = self.vitals_input_transformation(vitals) if self.has_vitals else None
        x_s = self.static_input_transformation(static_features.unsqueeze(1))

        for block in self.transformer_blocks:
            if self.self_positional_encoding is not None:
                x_t = x_t + self.self_positional_encoding(x_t)
                x_o = x_o + self.self_positional_encoding(x_o)
                if self.has_vitals:
                    x_v = x_v + self.self_positional_encoding(x_v)

            if self.has_vitals:
                x_t, x_o, x_v = block(
                    (x_t, x_o, x_v), x_s, active_entries_treat_outcomes, active_entries_vitals
                )
            else:
                x_t, x_o = block((x_t, x_o), x_s, active_entries_treat_outcomes)

        x = (x_o + x_t + x_v) / 3 if self.has_vitals else (x_o + x_t) / 2
        output = self.output_dropout(x)
        return F.elu(self.hr_output_transformation(output))

    def _stagewise_required_epochs(self):
        K = int(self.projection_horizon)
        if K <= 0:
            return 0
        return int(self.stagewise_terminal_epochs) + K * int(self.stagewise_epochs_per_head)

    def _stagewise_stage_for_epoch(self, epoch=None):
        """Map epoch -> q_K, q_{K-1}, ..., q_0."""
        K = int(self.projection_horizon)
        if K <= 0:
            return 0

        e = int(self.current_epoch if epoch is None else epoch)
        terminal_epochs = int(self.stagewise_terminal_epochs)
        per_head = int(self.stagewise_epochs_per_head)

        if e < terminal_epochs:
            return K
        block = (e - terminal_epochs) // per_head
        return max(K - 1 - int(block), 0)

    @staticmethod
    def _set_module_requires_grad(module, flag):
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = bool(flag)

    def _backbone_modules(self):
        """All registered modules that can alter H."""
        names = (
            'treatments_input_transformation', 'vitals_input_transformation',
            'outputs_input_transformation', 'static_input_transformation',
            'transformer_blocks', 'hr_output_transformation', 'output_dropout',
            'self_positional_encoding',
            'self_positional_encoding_k', 'self_positional_encoding_v',
            'cross_positional_encoding',
            'cross_positional_encoding_k', 'cross_positional_encoding_v',
        )
        out, seen = [], set()
        for name in names:
            mod = getattr(self, name, None)
            if isinstance(mod, nn.Module) and id(mod) not in seen:
                out.append(mod)
                seen.add(id(mod))
        return out

    def _stagewise_set_lr(self, stage):
        is_terminal = int(stage) == int(self.projection_horizon)
        base_lr = float(self.stagewise_terminal_lr) if is_terminal else float(self.stagewise_head_lr)
        history_lr = base_lr if is_terminal else base_lr * float(self.stagewise_history_lr_scale)

        try:
            optim = self.optimizers()
            optims = optim if isinstance(optim, (list, tuple)) else [optim]
            for opt in optims:
                for group in opt.param_groups:
                    if group.get('group_name', 'other') == 'history_backbone':
                        group['lr'] = history_lr
                    else:
                        group['lr'] = base_lr
        except Exception:
            pass
        return base_lr, history_lr

    def _stagewise_build_target_hr(self, prev_treatments, vitals, prev_outputs,
                                   static_features, active_entries):
        """Build the pseudo-target representation from the CURRENT live backbone.

        This is the defining Naive-Adaptation ablation: the downstream head is
        frozen stagewise (SFTR), but the teacher input representation is shared
        with the adaptive student backbone. ``torch.no_grad`` stops gradients
        through the target branch, but does NOT make its representation
        stationary across optimizer steps.
        """
        with torch.no_grad():
            return self.build_hr(
                prev_treatments, vitals, prev_outputs, static_features, active_entries
            )

    def _stagewise_apply_trainability(self, stage):
        if not self.stagewise_training or self.projection_horizon <= 0:
            return

        K = int(self.projection_horizon)
        j = int(stage)
        if not (0 <= j <= K):
            raise RuntimeError(f'Invalid stage q_{j} for K={K}.')

        # Freeze everything first.
        for p in self.parameters():
            p.requires_grad = False

        # Only the current recursive head is trainable.
        self._set_module_requires_grad(self.G_comp_heads[j], True)

        # Naive adaptation: the shared live history backbone remains adaptive.
        for mod in self._backbone_modules():
            self._set_module_requires_grad(mod, True)

        self._stagewise_active_stage = j
        base_lr, history_lr = self._stagewise_set_lr(j)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        logger.info(
            'Active=q_%d/%d trainable=%d/%d head_lr=%.6g history_lr=%.6g',
            j, K, n_trainable, n_total, base_lr, history_lr,
        )

    def _stagewise_enforce_modes(self, stage):
        """Current q_j/backbone adapt; downstream q_{j+1} stays frozen."""
        if not self.stagewise_training or self.projection_horizon <= 0:
            return

        K = int(self.projection_horizon)
        j = int(stage)

        for h, head in enumerate(self.G_comp_heads):
            if h == j:
                head.train()
            else:
                head.eval()

        for mod in self._backbone_modules():
            if j == K:
                mod.train()
            else:
                mod.eval()


    def on_save_checkpoint(self, checkpoint):
        parent = getattr(super(), 'on_save_checkpoint', None)
        if callable(parent):
            parent(checkpoint)
        checkpoint['_igc_stagewise_active_stage'] = self._stagewise_active_stage

    def on_load_checkpoint(self, checkpoint):
        parent = getattr(super(), 'on_load_checkpoint', None)
        if callable(parent):
            parent(checkpoint)
        restored_stage = checkpoint.get('_igc_stagewise_active_stage', None)
        self._stagewise_restored_stage = int(restored_stage) if restored_stage is not None else None
        self._stagewise_active_stage = None

    def on_fit_start(self):
        parent = getattr(super(), 'on_fit_start', None)
        if callable(parent):
            parent()

        if not self.stagewise_training or self.projection_horizon <= 0:
            return

        required = self._stagewise_required_epochs()
        max_epochs = int(getattr(self.trainer, 'max_epochs', -1))

        if self.stagewise_require_full_schedule and 0 < max_epochs < required:
            raise RuntimeError(
                f'Trainer.max_epochs is insufficient: need >= {required}, got {max_epochs}.'
            )

        early_callbacks = [
            cb for cb in getattr(self.trainer, 'callbacks', [])
            if 'earlystopping' in cb.__class__.__name__.lower()
        ]
        if early_callbacks:
            warning = (
                'Global EarlyStopping detected. Before q_0 is reached, final CF validation '
                'loss is not the active-stage objective. Disable it for this run.'
            )
            if self.stagewise_fail_on_early_stopping:
                raise RuntimeError(warning)
            print(warning, flush=True)
            logger.warning(warning)

    def on_train_epoch_start(self):
        parent = getattr(super(), 'on_train_epoch_start', None)
        if callable(parent):
            parent()

        if not self.stagewise_training or self.projection_horizon <= 0:
            return

        j = self._stagewise_stage_for_epoch()
        if self._stagewise_restored_stage == j:
            self._stagewise_apply_trainability(j)
            self._stagewise_restored_stage = None
        elif self._stagewise_active_stage != j:
            # SFTR only: freeze downstream heads stagewise.
            # There is deliberately NO frozen history snapshot here.
            self._stagewise_apply_trainability(j)

        self._stagewise_enforce_modes(j)

        if self._stagewise_last_stage_logged != j:
            print(
                f'Epoch={int(self.current_epoch)} --> fitting q_{j} with frozen downstream head + live target representation',
                flush=True,
            )
            self._stagewise_last_stage_logged = j

    def _policy_batch(self, batch_size, device, dtype):
        """Return the one configured K+1 intervention strategy for every sample."""
        if self.treatment_sequence is None:
            raise RuntimeError('Fixed treatment_sequence is unavailable.')
        seq = self.treatment_sequence.to(device=device, dtype=dtype)
        return seq.unsqueeze(0).expand(int(batch_size), -1, -1)

    def _build_counterfactual_treatment_history(self, ct, pt, t, K, policy_batch):
        """Original IGC-style fixed-policy replacement for one origin t.

        Replaces current_treatments[t-1 : t+K] with the configured K+1 policy,
        then shifts it to obtain the corresponding prev_treatments_cf.
        """
        T = ct.shape[1]
        start = int(t) - 1
        stop = int(t) + int(K)
        if start < 0 or stop > T:
            raise RuntimeError(f'Invalid CF replacement window [{start},{stop}) for T={T}.')

        ct_cf = ct.clone()
        ct_cf[:, start:stop, :] = policy_batch
        pt_cf = torch.cat((pt[:, :1, :], ct_cf[:, :-1, :]), dim=1)
        return ct_cf, pt_cf

    def _stagewise_training_step(self, batch, batch_ind):
        """Stagewise SFTR with a shared live representation in the target branch.

        q_K is first fitted to observed outcomes. For j<K, only q_j and the
        shared history backbone are trainable. q_{j+1} is frozen, but its input
        representation is recomputed from the CURRENT live backbone. Therefore:

            operator drift       -> removed within stage
            representation drift -> intentionally retained

        This is the exact SFTR + Naive Adaptation ablation.
        """
        j = int(self._stagewise_stage_for_epoch())
        K = int(self.projection_horizon)
        self._stagewise_enforce_modes(j)

        ct = torch.where(
            batch['current_treatments'].abs() < 1e-6,
            torch.zeros_like(batch['current_treatments']),
            batch['current_treatments'],
        )
        pt = torch.where(
            batch['prev_treatments'].abs() < 1e-6,
            torch.zeros_like(batch['prev_treatments']),
            batch['prev_treatments'],
        )
        vitals = batch['vitals'] if self.has_vitals else None
        prev_outputs = batch['prev_outputs']
        static_features = batch['static_features']
        active_entries = batch['active_entries']

        B, T, _ = pt.shape
        if T - K - 1 <= 0:
            raise RuntimeError(f'T={T} too short for K={K}.')

        policy = self._policy_batch(B, ct.device, ct.dtype) if K > 0 else None

        total_se = torch.zeros((), device=self.device)
        total_den = torch.zeros((), device=self.device)
        target_values = []
        pred_values = []

        for t in range(1, T - K):
            current_active = active_entries.clone()
            current_active[:, int(t + K):] = 0.0
            terminal_mask = current_active[:, t + K - 1, :]

            # Student path: factual history, live/adaptive representation.
            hr_live = self.build_hr(
                pt, vitals, prev_outputs, static_features, current_active
            )

            if j == K:
                stage_time = t + K - 1
                pred = self.G_comp_heads[K].build_outcome(
                    hr_live, ct
                )[:, stage_time, :]
                target = batch['outputs'][:, stage_time, :]
            else:
                current_time = t + j - 1
                target_time = t + j

                pred = self.G_comp_heads[j].build_outcome(
                    hr_live, ct
                )[:, current_time, :]

                ct_cf, pt_cf = self._build_counterfactual_treatment_history(
                    ct, pt, t, K, policy
                )

                # Teacher path:
                #   live representation parameters + frozen downstream q_{j+1}.
                # no_grad != stationary representation; after each optimizer
                # update the target representation may move.
                with torch.no_grad():
                    hr_teacher_cf = self._stagewise_build_target_hr(
                        pt_cf, vitals, prev_outputs, static_features, current_active
                    )
                    target = self.G_comp_heads[j + 1].build_outcome(
                        hr_teacher_cf, ct_cf
                    )[:, target_time, :]

            finite = (
                torch.isfinite(pred).all(dim=-1, keepdim=True)
                & torch.isfinite(target).all(dim=-1, keepdim=True)
            )
            mask = terminal_mask * finite.to(terminal_mask.dtype)

            pred_safe = torch.nan_to_num(
                pred, nan=0.0, posinf=0.0, neginf=0.0
            )
            target_safe = torch.nan_to_num(
                target, nan=0.0, posinf=0.0, neginf=0.0
            )

            total_se = total_se + ((pred_safe - target_safe).pow(2) * mask).sum()
            total_den = total_den + mask.sum() * self.dim_outcome

            if not self.stagewise_fast_mode:
                target_values.append(target_safe.detach())
                pred_values.append(pred_safe.detach())

        loss = total_se / (total_den + 1e-8)

        self._elog(f'{self.model_type}_stagewise_active_head', float(j))
        self._elog(f'{self.model_type}_stagewise_loss_q{j}', loss, prog_bar=True)
        self._elog(f'{self.model_type}_stagewise_level_loss_q{j}', loss)

        if not self.stagewise_fast_mode and target_values:
            target_all = torch.cat(target_values, dim=0)
            pred_all = torch.cat(pred_values, dim=0)
            self._elog(
                f'{self.model_type}_stagewise_target_std_q{j}',
                target_all.float().std(unbiased=False),
            )
            self._elog(
                f'{self.model_type}_stagewise_pred_std_q{j}',
                pred_all.float().std(unbiased=False),
            )

        self._elog(f'{self.model_type}_train_loss', loss, prog_bar=True)
        return loss

    def forward(self, batch):
        """Prediction forward; K>0 training is handled stagewise."""
        dtype = torch.float32
        batch = {
            k: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in batch.items()
        }
        ct = torch.where(
            batch['current_treatments'].abs() < 1e-6,
            torch.zeros_like(batch['current_treatments']),
            batch['current_treatments'],
        )
        pt = torch.where(
            batch['prev_treatments'].abs() < 1e-6,
            torch.zeros_like(batch['prev_treatments']),
            batch['prev_treatments'],
        )
        vitals = batch['vitals'] if self.has_vitals else None
        prev_outputs = batch['prev_outputs']
        static_features = batch['static_features']
        active_entries = batch['active_entries'].clone()
        B, T, _ = pt.shape

        if self.training:
            if self.projection_horizon > 0:
                raise RuntimeError('K>0 training is stagewise; call training_step().')
            hr = self.build_hr(pt, vitals, prev_outputs, static_features, active_entries)
            pred = self.G_comp_heads[0].build_outcome(hr, ct)
            return pred, None, None, active_entries

        fixed_split = (
            batch['sequence_lengths'] - self.max_projection
            if self.projection_horizon > 0 else batch['sequence_lengths']
        )

        cutoff = fixed_split.long() + int(self.projection_horizon)
        positions = torch.arange(T, device=active_entries.device).view(1, T, 1)
        active_entries = active_entries * (positions < cutoff.view(B, 1, 1)).to(active_entries.dtype)

        hr = self.build_hr(pt, vitals, prev_outputs, static_features, active_entries)

        if self.projection_horizon > 0:
            pred_all = self.G_comp_heads[0].build_outcome(hr, ct)
            pred_idx = fixed_split.long() - 1
            b = torch.arange(B, device=self.device)
            pred = pred_all[b, pred_idx, :]
        else:
            pred = self.G_comp_heads[0].build_outcome(hr, ct)

        return pred, hr

    def _cache_eval_predictions(self, split, predictions, targets):
        if split not in self._eval_metric_cache:
            self._eval_metric_cache[split] = []
        predictions = predictions.detach().reshape(-1, self.dim_outcome)
        targets = targets.detach().reshape(-1, self.dim_outcome)
        valid = (
            torch.isfinite(predictions).all(dim=-1) & torch.isfinite(targets).all(dim=-1)
        )
        if not valid.any():
            return

        predictions = predictions[valid].float().cpu()
        targets = targets[valid].float().cpu()
        current_n = int(self._eval_metric_count.get(split, 0))
        remaining = max(self.monitor_max_samples - current_n, 0)
        if remaining <= 0:
            return
        kept = min(int(predictions.shape[0]), int(remaining))
        if kept > 0:
            self._eval_metric_cache[split].append((predictions[:kept], targets[:kept]))
            self._eval_metric_count[split] = current_n + kept

    def _flush_eval_predictions(self, split):
        cache = self._eval_metric_cache.get(split, [])
        if len(cache) == 0:
            return
        predictions = torch.cat([item[0] for item in cache], dim=0).float()
        targets = torch.cat([item[1] for item in cache], dim=0).float()
        errors = predictions - targets
        pred_flat = predictions.reshape(-1)
        err_flat = errors.reshape(-1)

        self._elog_many({
            f'{self.model_type}_{split}_rmse': err_flat.pow(2).mean().sqrt(),
            f'{self.model_type}_{split}_mae': err_flat.abs().mean(),
            f'{self.model_type}_{split}_bias': err_flat.mean(),
            f'{self.model_type}_{split}_error_var': err_flat.var(unbiased=False),
            f'{self.model_type}_{split}_pred_mean': pred_flat.mean(),
            f'{self.model_type}_{split}_pred_std': pred_flat.std(unbiased=False),
            f'{self.model_type}_{split}_pred_p05': torch.quantile(pred_flat, 0.05),
            f'{self.model_type}_{split}_pred_p95': torch.quantile(pred_flat, 0.95),
        }, prog_bar_suffixes=('_rmse', '_mae'))

        self._eval_metric_cache[split] = []
        self._eval_metric_count[split] = 0

    def training_step(self, batch, batch_ind):
        if self.projection_horizon > 0:
            if not self.stagewise_training:
                raise RuntimeError('Requires stagewise_training=True for K>0.')
            return self._stagewise_training_step(batch, batch_ind)

        pred, _, _, active = self(batch)
        mse = F.mse_loss(pred, batch['outputs'], reduction='none')
        loss = (mse * active).sum() / (active.sum() * self.dim_outcome + 1e-8)
        self._elog(f'{self.model_type}_train_loss', loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if (self.stagewise_training and self.projection_horizon > 0
                and self.stagewise_stop_after_schedule
                and int(self.current_epoch) + 1 >= self._stagewise_required_epochs()):
            logger.info('Completed full stagewise schedule; stopping trainer.')
            self.trainer.should_stop = True

    def _eval_step(self, split, batch):
        pred_outcomes, _ = self(batch)
        if self.projection_horizon > 0:
            fixed_split = (batch['sequence_lengths'] - self.max_projection).long()
            target_idx = fixed_split + self.projection_horizon - 1
            b = torch.arange(pred_outcomes.size(0), device=self.device)
            targets = batch['outputs'][b, target_idx]
            valid = (
                torch.isfinite(pred_outcomes).all(dim=-1)
                & torch.isfinite(targets).all(dim=-1)
            )
        else:
            targets = batch['outputs']
            valid = (
                (batch['active_entries'].squeeze(-1) > 0)
                & torch.isfinite(pred_outcomes).all(dim=-1)
                & torch.isfinite(targets).all(dim=-1)
            )

        if valid.any():
            loss = F.mse_loss(pred_outcomes[valid], targets[valid])
            self._cache_eval_predictions(split, pred_outcomes[valid], targets[valid])
        else:
            loss = torch.zeros((), device=self.device)
        self._elog(f'{self.model_type}_{split}_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, dataset_idx=None):
        return self._eval_step('test', batch)

    def validation_step(self, batch, batch_idx):
        return self._eval_step('val', batch)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._eval_metric_cache['val'] = []
        self._eval_metric_count['val'] = 0

    def on_validation_epoch_end(self):
        self._flush_eval_predictions('val')
        super().on_validation_epoch_end()

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self._eval_metric_cache['test'] = []
        self._eval_metric_count['test'] = 0

    def on_test_epoch_end(self):
        self._flush_eval_predictions('test')
        super().on_test_epoch_end()

    def predict_step(self, batch, batch_idx, dataset_idx=None):
        outcome_pred, hr = self(batch)
        outcome_pred = outcome_pred.detach().cpu()
        if self.predict_return_history:
            return outcome_pred, hr.detach().cpu()
        return outcome_pred

    def get_predictions(self, dataset: Dataset, _oracle_internal: bool = False) -> np.array:
        logger.info(f'Predictions for {dataset.subset_name}.')
        loader = DataLoader(
            dataset,
            batch_size=self.hparams.dataset.val_batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        batches = self.trainer.predict(self, loader)
        if len(batches) == 0:
            return np.empty((0, self.dim_outcome), dtype=np.float32)
        if isinstance(batches[0], (tuple, list)):
            outcome_pred = torch.cat([x[0] for x in batches], dim=0)
        else:
            outcome_pred = torch.cat(batches, dim=0)
        return outcome_pred.numpy()

    def on_train_batch_start(self, batch, batch_idx):
        parent = getattr(super(), 'on_train_batch_start', None)
        if callable(parent):
            parent(batch, batch_idx)
        if getattr(self, 'mechanism_monitor', None) is not None:
            self.mechanism_monitor.collect_probe(self, batch)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        parent = getattr(super(), 'on_train_batch_end', None)
        if callable(parent):
            parent(outputs, batch, batch_idx)
        if getattr(self, 'mechanism_monitor', None) is not None:
            self.mechanism_monitor.observe(self)

    def on_fit_end(self):
        if getattr(self, 'mechanism_monitor', None) is not None:
            self.mechanism_monitor.finalize()
        parent = getattr(super(), 'on_fit_end', None)
        if callable(parent):
            parent()

    def get_normalised_n_step_rmses(self, dataset: Dataset):
        logger.info(f'RMSE calculation for {dataset.subset_name}.')
        outputs_scaled = self.get_predictions(dataset)
        unscale = self.hparams.exp.unscale_rmse
        percentage = self.hparams.exp.percentage_rmse

        if unscale:
            output_stds = dataset.scaling_params['output_stds']
            output_means = dataset.scaling_params['output_means']
            outputs_unscaled = outputs_scaled * output_stds + output_means
            mse = (outputs_unscaled
                   - dataset.data_processed_seq['unscaled_outputs'][:, self.projection_horizon - 1]) ** 2
        else:
            mse = (outputs_scaled
                   - dataset.data_processed_seq['outputs'][:, self.projection_horizon - 1]) ** 2

        output_ref = np.asarray(dataset.data_processed_seq['outputs'])
        valid_rows = ~np.isnan(output_ref).reshape(output_ref.shape[0], -1).any(axis=1)
        mse = mse[valid_rows].mean()
        rmse_normalised = np.sqrt(mse) / dataset.norm_const
        if percentage:
            rmse_normalised *= 100.0
        if getattr(self, 'mechanism_monitor', None) is not None:
            split_name = str(getattr(dataset, 'subset_name', 'evaluation'))
            self.mechanism_monitor.record_performance(
                split_name, rmse_normalised, horizon=int(self.projection_horizon)
            )
        return rmse_normalised

    def configure_optimizers(self):
        backbone_ids = set()
        backbone_params = []
        for mod in self._backbone_modules():
            for p in mod.parameters():
                if id(p) not in backbone_ids:
                    backbone_ids.add(id(p))
                    backbone_params.append(p)

        other_params = [p for p in self.parameters() if id(p) not in backbone_ids]

        groups = []
        if backbone_params:
            groups.append({'params': backbone_params, 'lr': self.lr, 'group_name': 'history_backbone'})
        if other_params:
            groups.append({'params': other_params, 'lr': self.lr, 'group_name': 'other'})
        return torch.optim.AdamW(groups, lr=self.lr)

    def visualize(self, dataset: Dataset, index=0, artifacts_path=None):
        fig_keys = ['self_attention_o', 'self_attention_t', 'cross_attention_ot', 'cross_attention_to']
        if self.has_vitals:
            fig_keys += [
                'cross_attention_vo', 'cross_attention_ov',
                'cross_attention_vt', 'cross_attention_tv',
                'self_attention_v',
            ]
        self._visualize(fig_keys, dataset, index, artifacts_path)

    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        sub_args = model_args[model_type]
        sub_args.optimizer.learning_rate = new_args['learning_rate']
        sub_args.batch_size = new_args['batch_size']
        sub_args.num_heads = new_args['num_heads']
        sub_args.projection_horizon = 0

        if 'seq_hidden_units' in new_args:
            sub_args.seq_hidden_units = int(input_size * new_args['seq_hidden_units'])
            common_multiplier = np.lcm.reduce([sub_args.num_heads, 2]).item()
            if sub_args.seq_hidden_units % common_multiplier != 0:
                sub_args.seq_hidden_units = (
                    sub_args.seq_hidden_units
                    + (common_multiplier - sub_args.seq_hidden_units % common_multiplier)
                )
            print(f'Factual seq_hidden_units of {model_type}: {sub_args.seq_hidden_units}.')

        sub_args.hr_size = int(input_size * new_args['hr_size'])
        sub_args.fc_hidden_units = int(sub_args.hr_size * new_args['fc_hidden_units'])
        sub_args.dropout_rate = new_args['dropout_rate']
        sub_args.num_layer = new_args['num_layer']