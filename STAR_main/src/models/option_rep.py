from ray import ray_constants

from src.models.utils_transformer import AbsolutePositionalEncoding, RelativePositionalEncoding
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging
import copy
import numpy as np
from typing import Union

from src.models.utils_transformer import TransformerMultiInputBlock
from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.utils import OutcomeHead
from src.models.time_varying_model import TimeVaryingCausalModel
from src.models.mechanism.target_drift_monitor import TargetDriftMonitor

logger = logging.getLogger(__name__)
ray_constants.FUNCTION_SIZE_ERROR_THRESHOLD = 10 ** 8


class FrozenHistoryBackboneSnapshot(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.has_vitals = bool(model.has_vitals)

        self.treatments_input_transformation = copy.deepcopy(model.treatments_input_transformation)
        self.vitals_input_transformation = (
            copy.deepcopy(model.vitals_input_transformation)
            if model.vitals_input_transformation is not None else None
        )
        self.outputs_input_transformation = copy.deepcopy(model.outputs_input_transformation)
        self.static_input_transformation = copy.deepcopy(model.static_input_transformation)
        self.transformer_blocks = copy.deepcopy(model.transformer_blocks)

        self.self_positional_encoding = (
            copy.deepcopy(model.self_positional_encoding)
            if model.self_positional_encoding is not None else None
        )
        self.hr_output_transformation = copy.deepcopy(model.hr_output_transformation)
        self.output_dropout = copy.deepcopy(model.output_dropout)

        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, prev_treatments, vitals, prev_outputs, static_features, active_entries):
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
        self.stagewise_history_lr_scale = float(sub_args.get('stagewise_history_lr_scale', 1.0))
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
        if abs(self.stagewise_history_lr_scale - 1.0) > 1e-12:
            logger.warning(
                'FrozenRep main ablation is cleanest with stagewise_history_lr_scale=1.0; '
                'current value %.6g changes the live-backbone LR relative to the head.',
                self.stagewise_history_lr_scale,
            )
        if self.stagewise_history_anchor_weight != 0:
            raise ValueError(
                'FrozenRep-only ablation freezes ONLY the teacher representation; '
                'set stagewise_history_anchor_weight=0 so the live student backbone '
                'is not additionally anchored.'
            )
        if self.stagewise_training:
            if self.stagewise_terminal_epochs <= 0:
                raise ValueError('stagewise_terminal_epochs must be >= 1.')
            if self.stagewise_epochs_per_head <= 0:
                raise ValueError('stagewise_epochs_per_head must be >= 1.')

        self._stagewise_active_stage = None
        self._stagewise_last_stage_logged = None
        self._stagewise_target_history_backbone_box = [None]
        # Explicit monitor-facing handle to the actual target-side representation.
        self._mechanism_teacher_representation_module = None
        self._stagewise_restored_stage = None

        self._eval_metric_cache = {'val': [], 'test': []}
        self._eval_metric_count = {'val': 0, 'test': 0}

        self.basic_block_cls = TransformerMultiInputBlock
        self._init_specific(args)
        self.tuning_criterion = 'rmse'
        self.save_hyperparameters(args)
        self.lr = args.model.igcnet.optimizer.learning_rate

        # Training-time mechanism instrumentation (disabled unless configured).
        self._mechanism_kind = "frozen_rep"
        self.mechanism_monitor = TargetDriftMonitor.from_args(
            args, model_label="FrozenRep+LiveHead", mechanism_kind="frozen_rep"
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

    def _stagewise_refresh_target_history_backbone(self, stage):
        if int(stage) == int(self.projection_horizon):
            self._stagewise_target_history_backbone_box[0] = None
            self._mechanism_teacher_representation_module = None
            return

        snapshot = FrozenHistoryBackboneSnapshot(self).to(self.device)
        snapshot.eval()
        for p in snapshot.parameters():
            p.requires_grad = False
        self._stagewise_target_history_backbone_box[0] = snapshot
        self._mechanism_teacher_representation_module = snapshot

    def _stagewise_build_target_hr(self, prev_treatments, vitals, prev_outputs,
                                   static_features, active_entries):
        snapshot = self._stagewise_target_history_backbone_box[0]
        if snapshot is None:
            raise RuntimeError('Target history backbone snapshot is missing.')
        snapshot.eval()
        with torch.inference_mode():
            return snapshot(prev_treatments, vitals, prev_outputs, static_features, active_entries)

    def _stagewise_apply_trainability(self, stage):
        """Frozen teacher representation + live downstream heads.

        This is the *representation-only* ablation in the 2x2 target-drift design.
        At stage q_j:

          * the live/student history backbone remains trainable;
          * q_j, q_{j+1}, ..., q_K remain trainable;
          * q_0, ..., q_{j-1} stay frozen until their stage is introduced;
          * pseudo-target representations are produced by the frozen stage snapshot.

        Thus the teacher representation cannot drift within a stage, while the
        downstream target operator q_{j+1} can still move across optimizer steps.
        """
        if not self.stagewise_training or self.projection_horizon <= 0:
            return

        K = int(self.projection_horizon)
        j = int(stage)
        if not (0 <= j <= K):
            raise RuntimeError(f'Invalid stage q_{j} for K={K}.')

        # Start from a known state.
        for p in self.parameters():
            p.requires_grad = False

        # LIVE HEADS: current q_j and all already-introduced downstream heads
        # continue to update. This is what preserves operator/head drift.
        for h in range(j, K + 1):
            self._set_module_requires_grad(self.G_comp_heads[h], True)

        # LIVE STUDENT REPRESENTATION: FrozenRep refers only to the TEACHER
        # representation snapshot, not to the student backbone.
        for mod in self._backbone_modules():
            self._set_module_requires_grad(mod, True)

        self._stagewise_active_stage = j
        base_lr, history_lr = self._stagewise_set_lr(j)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        logger.info(
            'Active=q_%d/%d live_heads=q_%d..q_%d trainable=%d/%d '
            'head_lr=%.6g history_lr=%.6g',
            j, K, j, K, n_trainable, n_total, base_lr, history_lr,
        )

    def _stagewise_enforce_modes(self, stage):
        """Enforce the FrozenRep ablation without conflating it with eval-mode training.

        The *student* representation is genuinely live: it is trainable and stays
        in train mode at every stage.  Only the target-side representation snapshot
        is frozen/eval.  Downstream live heads stay in eval mode while receiving
        gradients from their auxiliary head-only losses; this keeps pseudo-target
        evaluation deterministic while still allowing operator parameters to move.
        """
        if not self.stagewise_training or self.projection_horizon <= 0:
            return

        K = int(self.projection_horizon)
        j = int(stage)

        # Active q_j is trained normally.  Downstream heads q_{j+1:K} are live
        # parameters but evaluated deterministically; gradients still flow in
        # eval mode because requires_grad=True is independent of module.training.
        for h, head in enumerate(self.G_comp_heads):
            if h == j:
                head.train()
            else:
                head.eval()

        # IMPORTANT: the live/student backbone must remain in TRAIN mode.
        # Otherwise FrozenRep would be confounded with dropout/normalization-mode
        # changes relative to joint IGC-Net.
        for mod in self._backbone_modules():
            mod.train()

        # Only the teacher-side representation snapshot is frozen/deterministic.
        target_history = self._stagewise_target_history_backbone_box[0]
        if target_history is not None:
            target_history.eval()

        self._assert_frozenrep_invariants(j)

    def _assert_frozenrep_invariants(self, stage):
        """Fail fast if the implemented ablation no longer matches its definition."""
        K = int(self.projection_horizon)
        j = int(stage)

        # Student representation: trainable + train mode.
        for mod in self._backbone_modules():
            if not mod.training:
                raise RuntimeError('FrozenRep invariant violated: live student backbone is not in train mode.')
            for p in mod.parameters():
                if not p.requires_grad:
                    raise RuntimeError('FrozenRep invariant violated: live student backbone parameter is frozen.')

        # Current and downstream heads must remain live so operator drift is possible.
        for h in range(j, K + 1):
            for p in self.G_comp_heads[h].parameters():
                if not p.requires_grad:
                    raise RuntimeError(
                        f'FrozenRep invariant violated: live head q_{h} has a frozen parameter.'
                    )

        # Target-side representation must be a separate, frozen, deterministic snapshot.
        if j < K:
            target_history = self._stagewise_target_history_backbone_box[0]
            if target_history is None:
                raise RuntimeError('FrozenRep invariant violated: target representation snapshot is missing.')
            if target_history.training:
                raise RuntimeError('FrozenRep invariant violated: target representation snapshot is in train mode.')
            if any(p.requires_grad for p in target_history.parameters()):
                raise RuntimeError('FrozenRep invariant violated: target representation snapshot requires gradients.')

            # Snapshot lives in a plain holder and must never be optimized with the model.
            try:
                optim = self.optimizers()
                optims = optim if isinstance(optim, (list, tuple)) else [optim]
                opt_ids = {
                    id(p)
                    for opt in optims
                    for group in opt.param_groups
                    for p in group.get('params', [])
                }
                overlap = [id(p) for p in target_history.parameters() if id(p) in opt_ids]
                if overlap:
                    raise RuntimeError(
                        'FrozenRep invariant violated: target representation snapshot appears in optimizer.'
                    )
            except RuntimeError:
                raise
            except Exception:
                # Optimizer may not yet be attached during very early lifecycle hooks.
                pass

    def mechanism_teacher_representation(self):
        """Explicit accessor for mechanism monitors: this, not build_hr(), is the teacher rep."""
        return self._stagewise_target_history_backbone_box[0]

    def mechanism_student_representation_modules(self):
        """Explicit accessor for mechanism monitors."""
        return tuple(self._backbone_modules())

    # Backward-compatible aliases for external monitors.
    def get_teacher_representation_module(self):
        return self.mechanism_teacher_representation()

    def get_student_representation_modules(self):
        return self.mechanism_student_representation_modules()

    def on_save_checkpoint(self, checkpoint):
        parent = getattr(super(), 'on_save_checkpoint', None)
        if callable(parent):
            parent(checkpoint)

        checkpoint['_igc_stagewise_active_stage'] = self._stagewise_active_stage
        target_history = self._stagewise_target_history_backbone_box[0]
        if target_history is not None:
            checkpoint['_igc_target_history_state'] = {
                k: v.detach().cpu() for k, v in target_history.state_dict().items()
            }

    def on_load_checkpoint(self, checkpoint):
        parent = getattr(super(), 'on_load_checkpoint', None)
        if callable(parent):
            parent(checkpoint)

        restored_stage = checkpoint.get('_igc_stagewise_active_stage', None)
        self._stagewise_restored_stage = int(restored_stage) if restored_stage is not None else None
        self._stagewise_active_stage = None

        self._stagewise_target_history_backbone_box[0] = None
        self._mechanism_teacher_representation_module = None
        target_history_state = checkpoint.get('_igc_target_history_state', None)
        if target_history_state is not None:
            target_history = FrozenHistoryBackboneSnapshot(self)
            target_history.load_state_dict(target_history_state, strict=True)
            target_history.eval()
            for p in target_history.parameters():
                p.requires_grad = False
            self._stagewise_target_history_backbone_box[0] = target_history
            self._mechanism_teacher_representation_module = target_history

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
            target_history = self._stagewise_target_history_backbone_box[0]
            if target_history is not None:
                target_history.to(self.device)
            self._stagewise_apply_trainability(j)
            self._stagewise_restored_stage = None
        elif self._stagewise_active_stage != j:
            # Snapshot TEACHER history before the new stage; student history stays live.
            self._stagewise_refresh_target_history_backbone(j)
            self._stagewise_apply_trainability(j)

        self._stagewise_enforce_modes(j)

        if self._stagewise_last_stage_logged != j:
            print(
                f'Epoch={int(self.current_epoch)} --> fitting q_{j} with frozen TEACHER representation + live heads q_{j}..q_{int(self.projection_horizon)}',
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
        """Stagewise FrozenRep-only ablation.

        Student path (live):
            H_live -> q_j -> prediction

        Teacher/target path (representation frozen, head live):
            H_snapshot -> q_{j+1}^{live} -> stop-gradient pseudo-target

        q_{j+1:K} remain genuinely live by receiving their own auxiliary
        regression losses.  Those auxiliary head losses use DETACHED student
        representations, so they update the heads without adding extra gradient
        terms to the live student backbone.  Consequently, relative to SFTR the
        intended target-side change is isolated as closely as possible:

            teacher representation: frozen
            teacher/downstream head: live
            student representation: live
            current student head q_j: live
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

        # q_j is the active regression objective.  q_{j+1:K} get auxiliary
        # head-only objectives so the teacher operator remains live.
        active_se = torch.zeros((), device=self.device)
        active_den = torch.zeros((), device=self.device)
        aux_se = [torch.zeros((), device=self.device) for _ in range(K + 1)]
        aux_den = [torch.zeros((), device=self.device) for _ in range(K + 1)]

        # Diagnostics: live student rep is allowed to move away from the frozen
        # teacher snapshot; this quantity therefore need NOT be zero.
        history_drift_abs = torch.zeros((), device=self.device)
        history_drift_den = torch.zeros((), device=self.device)
        target_values = []
        pred_values = []

        for t in range(1, T - K):
            current_active = active_entries.clone()
            current_active[:, int(t + K):] = 0.0
            terminal_mask = current_active[:, t + K - 1, :]

            # LIVE student representation. Gradients from the active q_j loss
            # are allowed to update this backbone exactly as in SFTR.
            hr_live = self.build_hr(
                pt, vitals, prev_outputs, static_features, current_active,
            )

            # Counterfactual history is shared by all recursive pseudo-targets.
            if j < K:
                ct_cf, pt_cf = self._build_counterfactual_treatment_history(
                    ct, pt, t, K, policy,
                )
                with torch.no_grad():
                    # FROZEN TEACHER REPRESENTATION. The snapshot is fixed for
                    # this complete q_j stage, even while hr_live keeps adapting.
                    hr_teacher_cf = self._stagewise_build_target_hr(
                        pt_cf, vitals, prev_outputs, static_features, current_active,
                    )
                    hr_teacher_factual = self._stagewise_build_target_hr(
                        pt, vitals, prev_outputs, static_features, current_active,
                    )

                current_time_j = t + j - 1
                h_live_j = hr_live[:, current_time_j:current_time_j + 1, :]
                h_ref_j = hr_teacher_factual[:, current_time_j:current_time_j + 1, :].detach()
                h_diff = h_live_j - h_ref_j
                h_mask = terminal_mask.unsqueeze(-1).to(h_diff.dtype)
                history_drift_abs = history_drift_abs + (
                    h_diff.detach().abs() * h_mask
                ).sum()
                history_drift_den = history_drift_den + h_mask.sum() * h_diff.shape[-1]
            else:
                ct_cf = None
                hr_teacher_cf = None

            # ---------------------------------------------------------------
            # 1) ACTIVE q_j LOSS: updates q_j + LIVE STUDENT BACKBONE.
            # ---------------------------------------------------------------
            if j == K:
                active_pred_time = t + K - 1
                pred_j = self.G_comp_heads[K].build_outcome(
                    hr_live, ct,
                )[:, active_pred_time, :]
                target_j = batch['outputs'][:, active_pred_time, :]
            else:
                active_pred_time = t + j - 1
                active_target_time = t + j
                pred_j = self.G_comp_heads[j].build_outcome(
                    hr_live, ct,
                )[:, active_pred_time, :]

                # q_{j+1} is LIVE across optimizer steps, but target construction
                # for q_j is stop-gradient within this particular backward pass.
                with torch.no_grad():
                    target_j = self.G_comp_heads[j + 1].build_outcome(
                        hr_teacher_cf, ct_cf,
                    )[:, active_target_time, :]

            finite_j = (
                torch.isfinite(pred_j).all(dim=-1, keepdim=True)
                & torch.isfinite(target_j).all(dim=-1, keepdim=True)
            )
            mask_j = terminal_mask * finite_j.to(terminal_mask.dtype)
            pred_j_safe = torch.nan_to_num(pred_j, nan=0.0, posinf=0.0, neginf=0.0)
            target_j_safe = torch.nan_to_num(target_j, nan=0.0, posinf=0.0, neginf=0.0)

            active_se = active_se + (
                (pred_j_safe - target_j_safe).pow(2) * mask_j
            ).sum()
            active_den = active_den + mask_j.sum() * self.dim_outcome

            if not self.stagewise_fast_mode:
                target_values.append(target_j_safe.detach())
                pred_values.append(pred_j_safe.detach())

            # ---------------------------------------------------------------
            # 2) AUXILIARY LIVE-HEAD LOSSES: update q_{j+1:K} ONLY.
            #    hr_live.detach() prevents these losses from changing the
            #    student-backbone objective relative to the active stage.
            # ---------------------------------------------------------------
            if j < K:
                hr_head_only = hr_live.detach()

                for h in range(j + 1, K + 1):
                    if h == K:
                        pred_time_h = t + K - 1
                        pred_h = self.G_comp_heads[K].build_outcome(
                            hr_head_only, ct,
                        )[:, pred_time_h, :]
                        target_h = batch['outputs'][:, pred_time_h, :]
                    else:
                        pred_time_h = t + h - 1
                        target_time_h = t + h
                        pred_h = self.G_comp_heads[h].build_outcome(
                            hr_head_only, ct,
                        )[:, pred_time_h, :]
                        with torch.no_grad():
                            target_h = self.G_comp_heads[h + 1].build_outcome(
                                hr_teacher_cf, ct_cf,
                            )[:, target_time_h, :]

                    finite_h = (
                        torch.isfinite(pred_h).all(dim=-1, keepdim=True)
                        & torch.isfinite(target_h).all(dim=-1, keepdim=True)
                    )
                    mask_h = terminal_mask * finite_h.to(terminal_mask.dtype)
                    pred_h_safe = torch.nan_to_num(
                        pred_h, nan=0.0, posinf=0.0, neginf=0.0,
                    )
                    target_h_safe = torch.nan_to_num(
                        target_h, nan=0.0, posinf=0.0, neginf=0.0,
                    )

                    aux_se[h] = aux_se[h] + (
                        (pred_h_safe - target_h_safe).pow(2) * mask_h
                    ).sum()
                    aux_den[h] = aux_den[h] + mask_h.sum() * self.dim_outcome

        active_loss = active_se / (active_den + 1e-8)

        # Keep the active q_j/backbone gradient at exactly its own loss scale.
        # Auxiliary terms are ADDED (not averaged with q_j), and because their
        # representations are detached they affect only downstream head params.
        aux_losses = []
        for h in range(j + 1, K + 1):
            loss_h = aux_se[h] / (aux_den[h] + 1e-8)
            aux_losses.append(loss_h)
            self._elog(f'{self.model_type}_stagewise_aux_live_head_loss_q{h}', loss_h)

        aux_loss = (
            torch.stack(aux_losses).sum()
            if aux_losses else torch.zeros((), device=self.device)
        )
        loss = active_loss + aux_loss

        self._elog(f'{self.model_type}_stagewise_active_head', float(j))
        self._elog(f'{self.model_type}_stagewise_num_live_heads', float(K - j + 1))
        self._elog(f'{self.model_type}_stagewise_level_loss_q{j}', active_loss)
        self._elog(f'{self.model_type}_stagewise_loss_q{j}', loss, prog_bar=True)

        if j < K:
            history_drift_mean = history_drift_abs / (history_drift_den + 1e-8)
            if self.stagewise_log_history_drift:
                self._elog(
                    f'{self.model_type}_frozen_teacher_rep_student_drift_abs_q{j}',
                    history_drift_mean,
                )

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
        dtype = next(self.parameters()).dtype
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