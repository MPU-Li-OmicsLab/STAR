import logging
import math
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.time_varying_model import TimeVaryingCausalModel

logger = logging.getLogger(__name__)

COMPONENT_TYPES = ('continuous', 'binary', 'categorical')


@dataclass
class ComponentSpec:

    index: int
    offset: int
    size: int
    comp_type: str
    num_classes: int = None

    @property
    def end(self) -> int:
        return self.offset + self.size

    @property
    def is_continuous(self) -> bool:
        return self.comp_type == 'continuous'


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_seq_length: int, hidden_size: int):
        super().__init__()
        pe = torch.zeros(max_seq_length, hidden_size)
        position = torch.arange(0, max_seq_length).unsqueeze(1).to(torch.get_default_dtype())
        div_term = torch.exp(torch.arange(0, hidden_size, 2).to(torch.get_default_dtype())
                             * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        n_cos = pe[:, 1::2].size(1)
        pe[:, 1::2] = torch.cos(position * div_term)[:, :n_cos]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(1) <= self.pe.size(0), 'max_seq_length 小于输入序列长度'
        return x + self.pe[:x.size(1)].unsqueeze(0)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_seq_length: int, hidden_size: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_seq_length, hidden_size))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(1) <= self.pe.size(0), 'max_seq_length 小于输入序列长度'
        return x + self.pe[:x.size(1)].unsqueeze(0)


class RelativePositionalEncoding(nn.Module):

    def __init__(self, max_relative_position: int, head_size: int):
        super().__init__()
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.empty(2 * max_relative_position + 1, head_size))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q: int, length_k: int) -> torch.Tensor:
        device = self.embeddings_table.device
        range_q = torch.arange(length_q, device=device)
        range_k = torch.arange(length_k, device=device)
        distance = range_k.unsqueeze(0) - range_q.unsqueeze(1)
        distance = torch.clamp(distance, -self.max_relative_position, self.max_relative_position)
        return self.embeddings_table[distance + self.max_relative_position]


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_size: int, attn_dropout_rate: float = 0.0,
                 positional_encoding_k: nn.Module = None, positional_encoding_v: nn.Module = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.inner_size = num_heads * head_size

        self.W_q = nn.Linear(hidden_size, self.inner_size)
        self.W_k = nn.Linear(hidden_size, self.inner_size)
        self.W_v = nn.Linear(hidden_size, self.inner_size)
        self.W_o = nn.Linear(self.inner_size, hidden_size)

        self.attn_dropout = nn.Dropout(attn_dropout_rate)
        self.positional_encoding_k = positional_encoding_k
        self.positional_encoding_v = positional_encoding_v

    def _split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        return x.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

    def forward(self, query, key, value, mask: torch.Tensor = None) -> torch.Tensor:
        b, l_q, _ = query.size()
        l_k = key.size(1)

        q = self._split_heads(self.W_q(query), b)
        k = self._split_heads(self.W_k(key), b)
        v = self._split_heads(self.W_v(value), b)

        scores = torch.matmul(q, k.transpose(-2, -1))

        if self.positional_encoding_k is not None:
            r_k = self.positional_encoding_k(l_q, l_k)
            q_r = q.permute(2, 0, 1, 3).reshape(l_q, b * self.num_heads, self.head_size)
            scores_rel = torch.matmul(q_r, r_k.transpose(-2, -1))
            scores = scores + scores_rel.view(l_q, b, self.num_heads, l_k).permute(1, 2, 0, 3)

        scores = scores / math.sqrt(self.head_size)

        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = self.attn_dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)

        if self.positional_encoding_v is not None:
            r_v = self.positional_encoding_v(l_q, l_k)
            attn_r = attn.permute(2, 0, 1, 3).reshape(l_q, b * self.num_heads, l_k)
            out_rel = torch.matmul(attn_r, r_v)
            out = out + out_rel.view(l_q, b, self.num_heads, self.head_size).permute(1, 2, 0, 3)

        out = out.transpose(1, 2).contiguous().view(b, l_q, self.inner_size)
        return self.W_o(out)


class TransformerEncoderBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, head_size, ff_hidden_size,
                 dropout_rate, attn_dropout_rate,
                 positional_encoding_k=None, positional_encoding_v=None):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_size, num_heads, head_size, attn_dropout_rate,
                                            positional_encoding_k, positional_encoding_v)
        self.layer_norm_attn = nn.LayerNorm(hidden_size)
        self.layer_norm_ff = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, ff_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_hidden_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        h = self.layer_norm_attn(x)
        x = x + self.dropout(self.self_attn(h, h, h, mask))
        h = self.layer_norm_ff(x)
        x = x + self.dropout(self.feed_forward(h))
        return x


class CausalTransformerEncoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_layer: int, num_heads: int, head_size: int,
                 ff_hidden_size: int, dropout_rate: float, attn_dropout_rate: float = 0.0,
                 positional_encoding: str = 'sinusoidal', max_seq_length: int = None,
                 max_relative_position: int = 15):
        super().__init__()
        self.input_transformation = nn.Linear(input_size, hidden_size)
        self.input_dropout = nn.Dropout(dropout_rate)

        self.absolute_pe = None
        pe_k = pe_v = None
        if positional_encoding == 'relative':
            pe_k = RelativePositionalEncoding(max_relative_position, head_size)
            pe_v = RelativePositionalEncoding(max_relative_position, head_size)
        elif positional_encoding in ('sinusoidal', 'absolute'):
            if max_seq_length is None:
                raise MissingMandatoryValue()
            self.absolute_pe = SinusoidalPositionalEncoding(max_seq_length, hidden_size) \
                if positional_encoding == 'sinusoidal' else LearnedPositionalEncoding(max_seq_length, hidden_size)
        else:
            raise ValueError(f'未知的 positional_encoding: {positional_encoding}')

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(hidden_size, num_heads, head_size, ff_hidden_size,
                                    dropout_rate, attn_dropout_rate, pe_k, pe_v)
            for _ in range(num_layer)
        ])
        self.output_layer_norm = nn.LayerNorm(hidden_size)

    @staticmethod
    def build_mask(seq_length: int, active_entries: torch.Tensor, device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device, dtype=torch.bool))
        mask = mask.unsqueeze(0).unsqueeze(0)
        if active_entries is not None:
            key_padding = active_entries.reshape(active_entries.size(0), -1).bool()
            mask = mask & key_padding.unsqueeze(1).unsqueeze(1)
        return mask

    def forward(self, x: torch.Tensor, active_entries: torch.Tensor = None) -> torch.Tensor:
        x = self.input_transformation(x)
        if self.absolute_pe is not None:
            x = self.absolute_pe(x)
        x = self.input_dropout(x)

        mask = self.build_mask(x.size(1), active_entries, x.device)
        for block in self.blocks:
            x = block(x, mask)
        return self.output_layer_norm(x)


class ComponentEncoder(nn.Module):

    def __init__(self, spec: ComponentSpec, base_input_size: int, hidden_size: int, num_layer: int,
                 num_heads: int, head_size: int, ff_hidden_size: int, fc_hidden_units: int,
                 dropout_rate: float, attn_dropout_rate: float, positional_encoding: str,
                 max_seq_length: int, max_relative_position: int, categorical_relu: bool = False):
        super().__init__()
        self.spec = spec
        self.categorical_relu = categorical_relu
        self.encoder = CausalTransformerEncoder(base_input_size + spec.offset, hidden_size, num_layer,
                                                num_heads, head_size, ff_hidden_size, dropout_rate,
                                                attn_dropout_rate, positional_encoding, max_seq_length,
                                                max_relative_position)
        if fc_hidden_units:
            self.head = nn.Sequential(nn.Linear(hidden_size, fc_hidden_units), nn.ReLU(),
                                      nn.Dropout(dropout_rate), nn.Linear(fc_hidden_units, spec.size))
        else:
            self.head = nn.Linear(hidden_size, spec.size)

    def forward(self, base_x: torch.Tensor, prev_comp: torch.Tensor = None,
                active_entries: torch.Tensor = None) -> torch.Tensor:
        x = base_x if self.spec.offset == 0 else torch.cat((base_x, prev_comp), dim=-1)
        h = self.encoder(x, active_entries)
        out = self.head(h)
        if self.categorical_relu and not self.spec.is_continuous:
            out = F.relu(out)
        return out


class TreatmentStrategy:

    def __call__(self, data: dict, index: np.ndarray, dim_treatments: int) -> np.ndarray:
        raise NotImplementedError


class StaticTreatmentStrategy(TreatmentStrategy):

    def __call__(self, data, index, dim_treatments):
        return None


class ThresholdTreatmentStrategy(TreatmentStrategy):

    def __init__(self, vital_index: int, threshold: float, dose_below: float, dose_above: float = 0.0,
                 treatment_index: int = 0):
        self.vital_index = vital_index
        self.threshold = threshold
        self.dose_below = dose_below
        self.dose_above = dose_above
        self.treatment_index = treatment_index

    def __call__(self, data, index, dim_treatments):
        rows = np.arange(len(index))
        vital = data['vitals'][rows, index, self.vital_index]
        dose = np.where(vital < self.threshold, self.dose_below, self.dose_above)
        treatments = np.zeros((len(index), dim_treatments), dtype=data['current_treatments'].dtype)
        treatments[:, self.treatment_index] = dose
        return treatments


class GTransformer(TimeVaryingCausalModel):

    model_type = 'g_transformer'
    possible_model_types = {'g_transformer'}
    tuning_criterion = 'rmse'

    def __init__(self, args: DictConfig,
                 dataset_collection: Union[RealDatasetCollection, SyntheticDatasetCollection] = None,
                 autoregressive: bool = None,
                 has_vitals: bool = None,
                 projection_horizon: int = None,
                 bce_weights: np.array = None,
                 treatment_strategy: TreatmentStrategy = None, **kwargs):
        super().__init__(args, dataset_collection, autoregressive, has_vitals, bce_weights)

        if self.dataset_collection is not None:
            self.projection_horizon = self.dataset_collection.projection_horizon
        else:
            self.projection_horizon = projection_horizon

        assert self.autoregressive

        self.input_size = self.dim_treatments + self.dim_static_features + self.dim_outcome
        self.input_size += self.dim_vitals if self.has_vitals else 0
        logger.info(f'Input size of {self.model_type}: {self.input_size}')

        self.output_size = self.dim_vitals + self.dim_outcome
        self.treatment_mode = args.dataset.get('treatment_mode', 'continuous')
        self.treatment_strategy = treatment_strategy

        self._init_specific(args.model[self.model_type])
        self.save_hyperparameters(args)

    def _build_comp_specs(self, sub_args: DictConfig) -> List[ComponentSpec]:
        comp_sizes = [int(s) for s in sub_args.comp_sizes]
        num_comp = int(sub_args.num_comp)
        comp_types = sub_args.get('comp_types', None)
        comp_types = ['continuous'] * num_comp if comp_types is None else [str(t) for t in comp_types]
        comp_num_classes = sub_args.get('comp_num_classes', None)
        comp_num_classes = [None] * num_comp if comp_num_classes is None else list(comp_num_classes)

        assert len(comp_sizes) == num_comp, 'comp_sizes 长度必须等于 num_comp'
        assert len(comp_types) == num_comp, 'comp_types 长度必须等于 num_comp'
        assert sum(comp_sizes) == self.output_size, \
            f'comp_sizes 之和 {sum(comp_sizes)} 必须等于 dim_outcome + dim_vitals = {self.output_size}'

        specs, offset = [], 0
        for i, (size, comp_type) in enumerate(zip(comp_sizes, comp_types)):
            assert comp_type in COMPONENT_TYPES, f'未知的分量类型 {comp_type}'
            num_classes = comp_num_classes[i]
            if comp_type == 'categorical':
                num_classes = size if num_classes is None else int(num_classes)
                assert num_classes == size, 'categorical 分量的 size 必须等于类别数(一个 one-hot 变量)'
            specs.append(ComponentSpec(index=i, offset=offset, size=size,
                                       comp_type=comp_type, num_classes=num_classes))
            offset += size
        return specs

    def _init_specific(self, sub_args: DictConfig):
        try:
            self.dropout_rate = sub_args.dropout_rate
            self.attn_dropout_rate = sub_args.get('attn_dropout_rate', sub_args.dropout_rate)
            self.seq_hidden_units = sub_args.seq_hidden_units
            self.num_layer = sub_args.num_layer
            self.num_heads = sub_args.get('num_heads', 4)
            self.fc_hidden_units = sub_args.get('fc_hidden_units', 0)
            self.mc_samples = sub_args.mc_samples

            self.positional_encoding = sub_args.get('positional_encoding', 'sinusoidal')
            self.max_seq_length = sub_args.get('max_seq_length', None)
            self.max_relative_position = sub_args.get('max_relative_position', 15)
            self.categorical_relu = sub_args.get('categorical_relu', False)
            self.iterative_training = sub_args.get('iterative_training', False)
            self.fit_treatment = sub_args.get('fit_treatment', False)
            self.treatment_loss_weight = sub_args.get('treatment_loss_weight', 1.0)

            if self.seq_hidden_units is None or self.dropout_rate is None or self.num_layer is None:
                raise MissingMandatoryValue()

            self.comp_specs = self._build_comp_specs(sub_args)
            comp_loss_weights = sub_args.get('comp_loss_weights', None)
            self.comp_loss_weights = [1.0] * len(self.comp_specs) if comp_loss_weights is None \
                else [float(w) for w in comp_loss_weights]

            self.head_size = max(1, int(self.seq_hidden_units) // int(self.num_heads))
            self.seq_hidden_units = int(self.num_heads) * self.head_size
            self.ff_hidden_units = sub_args.get('ff_hidden_units', 4 * self.seq_hidden_units)

            enc_kwargs = dict(hidden_size=self.seq_hidden_units, num_layer=self.num_layer,
                              num_heads=self.num_heads, head_size=self.head_size,
                              ff_hidden_size=self.ff_hidden_units, fc_hidden_units=self.fc_hidden_units,
                              dropout_rate=self.dropout_rate, attn_dropout_rate=self.attn_dropout_rate,
                              positional_encoding=self.positional_encoding,
                              max_seq_length=self.max_seq_length,
                              max_relative_position=self.max_relative_position,
                              categorical_relu=self.categorical_relu)

            self.comp_encoders = nn.ModuleList([
                ComponentEncoder(spec, self.input_size, **enc_kwargs) for spec in self.comp_specs
            ])

            if self.fit_treatment:
                treat_spec = ComponentSpec(index=-1, offset=self.output_size, size=self.dim_treatments,
                                           comp_type='continuous' if self.treatment_continuous else 'binary')
                self.treatment_encoder = ComponentEncoder(treat_spec, self.input_size, **enc_kwargs)
            else:
                self.treatment_encoder = None

            logger.info(f'{self.model_type} components: '
                        + ', '.join(f'[{s.index}] {s.comp_type} x{s.size}' for s in self.comp_specs))

        except MissingMandatoryValue:
            logger.warning(f"{self.model_type} not fully initialised - some mandatory args are missing! "
                           f"(It's ok, if one will perform hyperparameters search afterward).")

    @property
    def treatment_continuous(self) -> bool:
        return self.treatment_mode == 'continuous'

    def prepare_data(self) -> None:
        if self.dataset_collection is not None and not self.dataset_collection.processed_data_multi:
            self.dataset_collection.process_data_multi()
        if self.dataset_collection is not None:
            self.dataset_collection.split_train_f_holdout(self.hparams.dataset.holdout_ratio)
            self.dataset_collection.explode_cf_treatment_seq(self.mc_samples)

    @staticmethod
    def set_hparams(model_args: DictConfig, new_args: dict, input_size: int, model_type: str):
        sub_args = model_args[model_type]
        sub_args.optimizer.learning_rate = new_args['learning_rate']
        sub_args.batch_size = new_args['batch_size']
        sub_args.num_heads = new_args.get('num_heads', sub_args.num_heads)

        seq_hidden_units = int(input_size * new_args['seq_hidden_units'])
        head_size = max(1, seq_hidden_units // int(sub_args.num_heads))
        sub_args.seq_hidden_units = int(sub_args.num_heads) * head_size

        sub_args.ff_hidden_units = int(sub_args.seq_hidden_units * new_args.get('ff_hidden_units', 4.0))
        sub_args.fc_hidden_units = int(sub_args.seq_hidden_units * new_args.get('fc_hidden_units', 0.0))
        sub_args.dropout_rate = new_args['dropout_rate']
        sub_args.attn_dropout_rate = new_args.get('attn_dropout_rate', new_args['dropout_rate'])
        sub_args.num_layer = new_args['num_layer']

    def _concat_base(self, curr_treatments, vitals, prev_outputs, static_features):
        vitals_prev_outputs = []
        vitals_prev_outputs.append(vitals) if self.has_vitals else None
        vitals_prev_outputs.append(prev_outputs) if self.autoregressive else None
        vitals_prev_outputs = torch.cat(vitals_prev_outputs, dim=-1)

        x = torch.cat((curr_treatments, vitals_prev_outputs), dim=-1)
        x = torch.cat((x, static_features.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)
        return x

    def build_base_input(self, batch):
        return self._concat_base(batch['current_treatments'],
                                 batch['vitals'] if self.has_vitals else None,
                                 batch['prev_outputs'],
                                 batch['static_features'])

    def build_comp_targets(self, batch):
        if self.has_vitals:
            return torch.cat((batch['outputs'], batch['next_vitals']), dim=-1)
        return batch['outputs']

    def _forward_raw(self, base_x, comp_targets, active_entries):
        raw = []
        for spec, encoder in zip(self.comp_specs, self.comp_encoders):
            prev_comp = comp_targets[:, :, :spec.offset] if spec.offset > 0 else None
            raw.append(encoder(base_x, prev_comp, active_entries))
        return torch.cat(raw, dim=-1)

    def _forward_raw_iterative(self, base_x, comp_targets, active_entries):
        outs = []
        for t in range(1, base_x.size(1) + 1):
            act = active_entries[:, :t] if active_entries is not None else None
            out_t = self._forward_raw(base_x[:, :t], comp_targets[:, :t], act)
            outs.append(out_t[:, -1])
        return torch.stack(outs, dim=1)

    def raw_to_point(self, raw: torch.Tensor) -> torch.Tensor:
        pieces = []
        for spec in self.comp_specs:
            r = raw[:, :, spec.offset:spec.end]
            if spec.comp_type == 'binary':
                r = torch.sigmoid(r)
            elif spec.comp_type == 'categorical':
                r = torch.softmax(r, dim=-1)
            pieces.append(r)
        return torch.cat(pieces, dim=-1)

    def forward(self, batch, sample=False):
        dtype = next(self.parameters()).dtype
        batch = {
            k: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in batch.items()
        }
        base_x = self.build_base_input(batch)
        comp_targets = self.build_comp_targets(batch)
        active_entries = batch['active_entries'] if 'active_entries' in batch else None
        raw = self._forward_raw(base_x, comp_targets, active_entries)
        return self.raw_to_point(raw)

    def _component_losses(self, raw, comp_targets, active_entries):
        losses = {}
        total = 0.0
        denom = active_entries.sum().clamp(min=1.0)
        for spec, weight in zip(self.comp_specs, self.comp_loss_weights):
            r = raw[:, :, spec.offset:spec.end]
            y = comp_targets[:, :, spec.offset:spec.end]
            if spec.comp_type == 'continuous':
                per_elem = F.mse_loss(r, y, reduction='none')
                loss = (active_entries * per_elem).sum() / (denom * spec.size)
            elif spec.comp_type == 'binary':
                per_elem = F.binary_cross_entropy_with_logits(r, y, reduction='none')
                loss = (active_entries * per_elem).sum() / (denom * spec.size)
            else:
                log_p = torch.log_softmax(r, dim=-1)
                per_step = -(y * log_p).sum(-1, keepdim=True)
                loss = (active_entries * per_step).sum() / denom
            losses[f'comp{spec.index}_{spec.comp_type}'] = loss
            total = total + weight * loss
        return total, losses

    def _treatment_loss(self, base_x, comp_targets, curr_treatments, active_entries):
        raw = self.treatment_encoder(base_x, comp_targets, active_entries)
        pred = raw[:, :-1]
        target = curr_treatments[:, 1:]
        act = active_entries[:, 1:]
        if self.treatment_continuous:
            per_elem = F.mse_loss(pred, target, reduction='none')
        else:
            per_elem = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        return (act * per_elem).sum() / (act.sum().clamp(min=1.0) * self.dim_treatments)

    def training_step(self, batch, batch_ind):
        base_x = self.build_base_input(batch)
        comp_targets = self.build_comp_targets(batch)
        active_entries = batch['active_entries']

        if self.iterative_training:
            raw = self._forward_raw_iterative(base_x, comp_targets, active_entries)
        else:
            raw = self._forward_raw(base_x, comp_targets, active_entries)

        loss, comp_losses = self._component_losses(raw, comp_targets, active_entries)

        if self.treatment_encoder is not None:
            treat_loss = self._treatment_loss(base_x, comp_targets, batch['current_treatments'], active_entries)
            loss = loss + self.treatment_loss_weight * treat_loss
            self.log(f'{self.model_type}_train_treatment_loss', treat_loss,
                     on_epoch=True, on_step=False, sync_dist=True)

        for name, value in comp_losses.items():
            self.log(f'{self.model_type}_train_loss_{name}', value, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f'{self.model_type}_train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_ind, dataset_idx=None):
        return self(batch).cpu()

    def on_fit_end(self) -> None:
        if self.dataset_collection is not None and hasattr(self.dataset_collection, 'train_f_holdout') \
                and len(self.dataset_collection.train_f_holdout) > 0:
            logger.info('Fitting residuals based on train_f_holdout.')
            self.eval()
            outcome_next_vitals_pred = self.get_predictions(self.dataset_collection.train_f_holdout, vitals=True)

            outcomes_next_vitals = self.dataset_collection.train_f_holdout.data['outputs']
            if self.has_vitals:
                vitals = self.dataset_collection.train_f_holdout.data['next_vitals']
                outcomes_next_vitals = np.concatenate((outcomes_next_vitals, vitals), axis=-1)

            self.holdout_resid = outcomes_next_vitals - outcome_next_vitals_pred
            self.holdout_resid_len = self.dataset_collection.train_f_holdout.data['sequence_lengths']
        else:
            self.holdout_resid = self.holdout_resid_len = None

    def get_predictions(self, dataset: Union[Dataset, List[Dataset]], vitals=False) -> np.array:
        if not isinstance(dataset, list):
            logger.info(f'Predictions for {dataset.subset_name}.')
        if isinstance(dataset, list):
            data_loader = \
                [DataLoader(d, batch_size=self.hparams.dataset.val_batch_size, shuffle=False, num_workers=2)
                 for d in dataset]
        else:
            data_loader = DataLoader(dataset, batch_size=self.hparams.dataset.val_batch_size, shuffle=False)

        outcome_vitals_pred = self.trainer.predict(self, data_loader)

        if isinstance(dataset, list):
            outcome_vitals_pred = np.stack([torch.cat(pred).numpy() for pred in outcome_vitals_pred], axis=0)
            return outcome_vitals_pred if vitals else outcome_vitals_pred[:, :, :, :self.dim_outcome]
        else:
            outcome_vitals_pred = torch.cat(outcome_vitals_pred).numpy()
            return outcome_vitals_pred if vitals else outcome_vitals_pred[:, :, :self.dim_outcome]

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.get_default_dtype(), device=self.device)

    @torch.no_grad()
    def _encoder_raw_on_arrays(self, encoder: ComponentEncoder, data: dict, cond: np.ndarray,
                               t_end: int, batch_size: int) -> np.ndarray:
        n = data['current_treatments'].shape[0]
        offset = encoder.spec.offset
        outs = []
        for lo in range(0, n, batch_size):
            hi = min(lo + batch_size, n)
            base_x = self._concat_base(
                self._to_tensor(data['current_treatments'][lo:hi, :t_end]),
                self._to_tensor(data['vitals'][lo:hi, :t_end]) if self.has_vitals else None,
                self._to_tensor(data['prev_outputs'][lo:hi, :t_end]),
                self._to_tensor(data['static_features'][lo:hi]))
            prev_comp = self._to_tensor(cond[lo:hi, :t_end, :offset]) if offset > 0 else None
            active = self._to_tensor(data['active_entries'][lo:hi, :t_end]) \
                if 'active_entries' in data else None
            outs.append(encoder(base_x, prev_comp, active).cpu().numpy())
        return np.concatenate(outs, axis=0)

    @staticmethod
    def _softmax_np(x: np.ndarray) -> np.ndarray:
        shifted = x - x.max(axis=-1, keepdims=True)
        e = np.exp(shifted)
        return e / e.sum(axis=-1, keepdims=True)

    def _sample_component(self, spec: ComponentSpec, raw_at_pos: np.ndarray,
                          resid_at_pos: np.ndarray) -> np.ndarray:
        if spec.comp_type == 'continuous':
            return raw_at_pos if resid_at_pos is None else raw_at_pos + resid_at_pos
        if spec.comp_type == 'binary':
            probs = 1.0 / (1.0 + np.exp(-raw_at_pos))
            return (np.random.random_sample(probs.shape) < probs).astype(raw_at_pos.dtype)
        probs = self._softmax_np(raw_at_pos)
        cum = probs.cumsum(axis=-1)
        draws = np.random.random_sample((probs.shape[0], 1))
        choice = np.minimum((draws > cum).sum(axis=-1), probs.shape[-1] - 1)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(probs.shape[0]), choice] = 1.0
        return one_hot

    def _draw_residuals(self, n: int, pos: np.ndarray) -> np.ndarray:
        if self.holdout_resid is None:
            return None
        rand_ind = np.random.randint(len(self.holdout_resid), size=n)
        resid_len = self.holdout_resid_len[rand_ind].astype(int)
        return self.holdout_resid[rand_ind, np.minimum(pos, resid_len - 1), :]

    @torch.no_grad()
    def _simulate_position(self, data: dict, cond: np.ndarray, pos: np.ndarray, batch_size: int):
        n = cond.shape[0]
        rows = np.arange(n)
        t_end = int(pos.max()) + 1
        mean_pred = np.zeros((n, self.output_size), dtype=cond.dtype)
        sampled = np.zeros((n, self.output_size), dtype=cond.dtype)
        resid_row = self._draw_residuals(n, pos)

        for spec, encoder in zip(self.comp_specs, self.comp_encoders):
            raw = self._encoder_raw_on_arrays(encoder, data, cond, t_end, batch_size)
            raw_at_pos = raw[rows, pos, :]

            if spec.comp_type == 'binary':
                mean_at_pos = 1.0 / (1.0 + np.exp(-raw_at_pos))
            elif spec.comp_type == 'categorical':
                mean_at_pos = self._softmax_np(raw_at_pos)
            else:
                mean_at_pos = raw_at_pos

            resid_slice = resid_row[:, spec.offset:spec.end] \
                if (resid_row is not None and spec.is_continuous) else None
            s = self._sample_component(spec, raw_at_pos, resid_slice)

            mean_pred[:, spec.offset:spec.end] = mean_at_pos
            sampled[:, spec.offset:spec.end] = s
            cond[rows, pos, spec.offset:spec.end] = s

        return mean_pred, sampled

    @torch.no_grad()
    def _simulate_treatment(self, data: dict, cond: np.ndarray, index: np.ndarray, batch_size: int) -> np.ndarray:
        rows = np.arange(len(index))
        t_end = int(index.max()) + 1
        raw = self._encoder_raw_on_arrays(self.treatment_encoder, data, cond, t_end, batch_size)
        raw_at_pos = raw[rows, np.maximum(index - 1, 0), :]
        if self.treatment_continuous:
            return raw_at_pos
        probs = 1.0 / (1.0 + np.exp(-raw_at_pos))
        return (np.random.random_sample(probs.shape) < probs).astype(raw_at_pos.dtype)

    def get_autoregressive_predictions(self, datasets: List[Dataset], treatment_sequence=None) -> np.array:
        """
        treatment_sequence: 长度 >= projection_horizon + 1 的二维数组。
            [0]              -> 写入锚点位 split - 1;
            [1..horizon]     -> 写入未来 horizon 步 split..split+horizon-1;
            优先级高于 treatment_strategy 和 treatment_encoder。
        """
        assert hasattr(self, 'holdout_resid') and hasattr(self, 'holdout_resid_len')
        assert len(datasets) == self.mc_samples
        logger.info(f'Autoregressive Prediction for {datasets[0].subset_name} with MC-sampling of trajectories.')

        self.eval()
        horizon = self.hparams.dataset.projection_horizon
        batch_size = self.hparams.dataset.val_batch_size
        n = len(datasets[0])
        predicted_outputs = np.zeros((self.mc_samples, n, horizon, self.dim_outcome))

        if treatment_sequence is not None:
            treatment_sequence = np.asarray(treatment_sequence, dtype=np.float32)
            if (
                treatment_sequence.ndim != 2
                or treatment_sequence.shape[0] < horizon + 1
                or treatment_sequence.shape[1] != self.dim_treatments
            ):
                raise ValueError(
                    "treatment_sequence must have shape "
                    f"(>= {horizon + 1}, {self.dim_treatments}), got "
                    f"{treatment_sequence.shape}."
                )
            treatment_sequence = treatment_sequence[: horizon + 1]

        for m in range(self.mc_samples):
            logger.info(f'MC sample {m + 1} / {self.mc_samples}')
            data = datasets[m].data
            rows = np.arange(n)
            split = data['future_past_split'].astype(int)

            cond = np.concatenate((data['outputs'], data['next_vitals']), axis=-1) if self.has_vitals \
                else data['outputs']
            cond = np.array(cond, copy=True)

            # 锚点:反事实开始前一步。split=0 时兜底到 0 位置。
            if treatment_sequence is not None:
                anchor_pos = np.maximum(split - 1, 0)
                data['current_treatments'][rows, anchor_pos, :] = treatment_sequence[0]

            for t in range(horizon + 1):
                pos = split - 1 + t
                mean_pred, sampled = self._simulate_position(data, cond, pos, batch_size)

                if t > 0:
                    predicted_outputs[m, :, t - 1, :] = mean_pred[:, :self.dim_outcome]

                if t < horizon:
                    data['prev_outputs'][rows, pos + 1, :] = sampled[:, :self.dim_outcome]
                    if self.has_vitals:
                        data['vitals'][rows, pos + 1, :] = sampled[:, self.dim_outcome:]

                    # 优先级:显式序列 > strategy > encoder
                    treatments = None
                    if treatment_sequence is not None:
                        treatments = np.broadcast_to(
                            treatment_sequence[t + 1], (len(rows), self.dim_treatments)
                        ).astype(data['current_treatments'].dtype)
                    elif self.treatment_strategy is not None:
                        treatments = self.treatment_strategy(data, pos + 1, self.dim_treatments)
                    elif self.treatment_encoder is not None:
                        treatments = self._simulate_treatment(data, cond, pos + 1, batch_size)
                    if treatments is not None:
                        data['current_treatments'][rows, pos + 1, :] = treatments

        return predicted_outputs.mean(0)