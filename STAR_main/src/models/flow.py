import logging
from copy import deepcopy
from typing import Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from omegaconf.errors import MissingMandatoryValue
from torch.utils.data import DataLoader, Dataset

from src.data import RealDatasetCollection, SyntheticDatasetCollection
from src.models.time_varying_model import TimeVaryingCausalModel
from src.models.utils_lstm import VariationalLSTM

logger = logging.getLogger(__name__)


class ConditionalPiecewiseLinearSpline(nn.Module):
    """Context-conditioned monotone piecewise-linear spline on the real line."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        hidden: int,
        n_bins: int = 16,
        tail_bound: float = 6.0,
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
    ):
        super().__init__()

        if n_bins < 2:
            raise ValueError("n_bins must be at least 2.")
        if min_bin_width * n_bins >= 2.0 * tail_bound:
            raise ValueError("min_bin_width is too large.")
        if min_bin_height * n_bins >= 2.0 * tail_bound:
            raise ValueError("min_bin_height is too large.")

        self.dim = dim
        self.n_bins = n_bins
        self.tail_bound = float(tail_bound)
        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)

        self.parameter_net = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, dim * 2 * n_bins),
        )

    def _knots(self, context: torch.Tensor):
        params = self.parameter_net(context)
        params = params.view(
            *context.shape[:-1], self.dim, 2 * self.n_bins
        )
        width_logits, height_logits = params.split(
            self.n_bins, dim=-1
        )

        total_range = 2.0 * self.tail_bound

        widths = F.softmax(width_logits, dim=-1)
        widths = (
            self.min_bin_width
            + (total_range - self.min_bin_width * self.n_bins)
            * widths
        )

        heights = F.softmax(height_logits, dim=-1)
        heights = (
            self.min_bin_height
            + (total_range - self.min_bin_height * self.n_bins)
            * heights
        )

        cum_widths = F.pad(
            torch.cumsum(widths, dim=-1), (1, 0), value=0.0
        ) - self.tail_bound
        cum_heights = F.pad(
            torch.cumsum(heights, dim=-1), (1, 0), value=0.0
        ) - self.tail_bound

        cum_widths[..., 0] = -self.tail_bound
        cum_widths[..., -1] = self.tail_bound
        cum_heights[..., 0] = -self.tail_bound
        cum_heights[..., -1] = self.tail_bound

        return cum_widths, cum_heights

    @staticmethod
    def _bin_index(knots: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        flat_knots = knots.reshape(-1, knots.size(-1))
        flat_values = values.reshape(-1, 1)
        indices = torch.searchsorted(
            flat_knots, flat_values, right=True
        ).squeeze(-1) - 1
        return indices.clamp(
            min=0, max=knots.size(-1) - 2
        ).view_as(values)

    @staticmethod
    def _gather(values: torch.Tensor, indices: torch.Tensor):
        return torch.gather(
            values, dim=-1, index=indices.unsqueeze(-1)
        ).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ):
        cum_widths, cum_heights = self._knots(context)
        inside = x.abs().le(self.tail_bound)
        indices = self._bin_index(cum_widths, x)

        x0 = self._gather(cum_widths, indices)
        x1 = self._gather(cum_widths, indices + 1)
        y0 = self._gather(cum_heights, indices)
        y1 = self._gather(cum_heights, indices + 1)

        slope = (y1 - y0) / (x1 - x0).clamp_min(1e-12)
        transformed = y0 + slope * (x - x0)
        log_abs_det = torch.log(slope.clamp_min(1e-12))

        y = torch.where(inside, transformed, x)
        log_abs_det = torch.where(
            inside, log_abs_det, torch.zeros_like(log_abs_det)
        )
        return y, log_abs_det

    def inverse(
        self,
        y: torch.Tensor,
        context: torch.Tensor,
    ):
        cum_widths, cum_heights = self._knots(context)
        inside = y.abs().le(self.tail_bound)
        indices = self._bin_index(cum_heights, y)

        x0 = self._gather(cum_widths, indices)
        x1 = self._gather(cum_widths, indices + 1)
        y0 = self._gather(cum_heights, indices)
        y1 = self._gather(cum_heights, indices + 1)

        slope = (y1 - y0) / (x1 - x0).clamp_min(1e-12)
        transformed = x0 + (y - y0) / slope.clamp_min(1e-12)
        log_abs_det_inverse = -torch.log(
            slope.clamp_min(1e-12)
        )

        x = torch.where(inside, transformed, y)
        log_abs_det_inverse = torch.where(
            inside,
            log_abs_det_inverse,
            torch.zeros_like(log_abs_det_inverse),
        )
        return x, log_abs_det_inverse


class ZeroInflatedConditionalFlow(nn.Module):
    """Zero-inflated conditional spline flow for normalized continuous dose."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        n_layers: int,
        hidden: int,
        n_bins: int = 16,
        tail_bound: float = 6.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.gate_net = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, dim),
        )

        self.layers = nn.ModuleList(
            [
                ConditionalPiecewiseLinearSpline(
                    dim=dim,
                    context_dim=context_dim,
                    hidden=hidden,
                    n_bins=n_bins,
                    tail_bound=tail_bound,
                )
                for _ in range(n_layers)
            ]
        )

    def log_prob(
        self,
        dose: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        p_treat = torch.sigmoid(self.gate_net(context))
        p_treat = torch.clamp(
            p_treat, self.eps, 1.0 - self.eps
        )

        is_zero = dose.le(self.eps)
        log_p_zero = torch.log1p(-p_treat)

        dose_safe = torch.clamp(
            dose, self.eps, 1.0 - self.eps
        )
        y = torch.log(dose_safe) - torch.log1p(-dose_safe)

        z = y
        log_det_inverse = torch.zeros_like(z)
        for layer in reversed(self.layers):
            z, layer_log_det_inverse = layer.inverse(z, context)
            log_det_inverse = (
                log_det_inverse + layer_log_det_inverse
            )

        log_base = (
            -0.5 * z.square()
            - 0.5 * np.log(2.0 * np.pi)
        )

        logit_log_jacobian = (
            -torch.log(dose_safe)
            - torch.log1p(-dose_safe)
        )

        log_p_nonzero = (
            torch.log(p_treat)
            + log_base
            + log_det_inverse
            + logit_log_jacobian
        )

        return torch.where(
            is_zero, log_p_zero, log_p_nonzero
        )

    def sample(
        self,
        context: torch.Tensor,
        n: int = 1,
    ) -> torch.Tensor:
        p_treat = torch.sigmoid(self.gate_net(context))
        p_treat = torch.clamp(
            p_treat, self.eps, 1.0 - self.eps
        )

        expanded_p = p_treat.unsqueeze(0).expand(
            n, *p_treat.shape
        )
        gate = torch.bernoulli(expanded_p)

        z_shape = (n,) + context.shape[:-1] + (self.dim,)
        z = torch.randn(
            z_shape, device=context.device, dtype=context.dtype
        )
        expanded_context = context.unsqueeze(0).expand(
            n, *context.shape
        )

        y = z
        for layer in self.layers:
            y, _ = layer(y, expanded_context)

        dose = torch.sigmoid(y)
        return gate * dose


class FLOW(TimeVaryingCausalModel):
    """
    Longitudinal continuous-dose baseline using conditional spline flows.

    treatment_sequence 约定(与 IGCNET 对齐):
        长度为 projection_horizon + 1;
        [0]              -> 写入锚点位 split - 1(反事实开始前一步);
        [1..horizon]     -> 写入未来 horizon 步 split..split+horizon-1。
    """

    model_type = "flow"
    possible_model_types = {"flow"}
    tuning_criterion = "rmse"

    def __init__(
        self,
        args: DictConfig,
        dataset_collection: Union[
            RealDatasetCollection, SyntheticDatasetCollection
        ] = None,
        autoregressive: bool = None,
        has_vitals: bool = None,
        projection_horizon: int = None,
        bce_weights: np.ndarray = None,
        **kwargs,
    ):
        super().__init__(
            args, dataset_collection, autoregressive, has_vitals, bce_weights
        )

        self.projection_horizon = (
            self.dataset_collection.projection_horizon
            if self.dataset_collection is not None
            else projection_horizon
        )
        assert self.autoregressive, "FLOW requires autoregressive=True"

        self.input_size = (
            self.dim_treatments
            + self.dim_outcome
            + (self.dim_vitals if self.has_vitals else 0)
            + self.dim_static_features
        )
        logger.info("Input size of %s: %s", self.model_type, self.input_size)

        self._init_specific(args.model.flow)
        self.save_hyperparameters(args)

    def _init_specific(self, sub_args: DictConfig):
        try:
            self.seq_hidden_units = sub_args.seq_hidden_units
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.num_layer = sub_args.num_layer
            self.dropout_rate = sub_args.dropout_rate

            self.flow_n_layers = int(
                sub_args.get("flow_n_layers", 3)
            )
            self.flow_n_bins = int(
                sub_args.get("flow_n_bins", 16)
            )
            self.flow_tail_bound = float(
                sub_args.get("flow_tail_bound", 6.0)
            )
            self.mc_samples = int(
                sub_args.get("mc_samples", 1)
            )
            self.alpha_flow = float(
                sub_args.get("alpha_flow", 1.0)
            )
            self.alpha_outcome = float(
                sub_args.get("alpha_outcome", 1.0)
            )
            self.use_ipw = bool(
                sub_args.get("use_ipw", True)
            )
            self.max_ipw_weight = float(
                sub_args.get("max_ipw_weight", 10.0)
            )

            if None in (
                self.seq_hidden_units,
                self.fc_hidden_units,
                self.dropout_rate,
            ):
                raise MissingMandatoryValue()

            self.history_lstm = VariationalLSTM(
                self.input_size,
                self.seq_hidden_units,
                self.num_layer,
                self.dropout_rate,
            )

            self.conditional_flow = ZeroInflatedConditionalFlow(
                dim=self.dim_treatments,
                context_dim=self.seq_hidden_units,
                n_layers=self.flow_n_layers,
                hidden=self.fc_hidden_units,
                n_bins=self.flow_n_bins,
                tail_bound=self.flow_tail_bound,
            )

            self.marginal_flow = ZeroInflatedConditionalFlow(
                dim=self.dim_treatments,
                context_dim=self.dim_static_features + 1,
                n_layers=self.flow_n_layers,
                hidden=self.fc_hidden_units,
                n_bins=self.flow_n_bins,
                tail_bound=self.flow_tail_bound,
            )

            self.outcome_head = nn.Sequential(
                nn.Linear(
                    self.seq_hidden_units + self.dim_treatments,
                    self.fc_hidden_units,
                ),
                nn.ELU(),
                nn.Linear(
                    self.fc_hidden_units, self.dim_outcome
                ),
            )

        except MissingMandatoryValue:
            logger.warning(
                "%s not fully initialised; mandatory arguments are missing.",
                self.model_type,
            )

    @staticmethod
    def set_hparams(
        model_args: DictConfig,
        new_args: dict,
        input_size: int,
        model_type: str,
    ):
        sub = model_args[model_type]
        sub.optimizer.learning_rate = new_args["learning_rate"]
        sub.batch_size = new_args["batch_size"]
        sub.seq_hidden_units = int(
            input_size * new_args["seq_hidden_units"]
        )
        sub.fc_hidden_units = int(
            sub.seq_hidden_units * new_args["fc_hidden_units"]
        )
        sub.dropout_rate = new_args["dropout_rate"]
        sub.num_layer = new_args["num_layer"]
        sub.flow_n_layers = new_args.get("flow_n_layers", 3)
        sub.flow_n_bins = new_args.get("flow_n_bins", 16)

    def _build_history(self, batch: dict) -> torch.Tensor:
        prev_a = batch["prev_treatments"]
        prev_y = batch["prev_outputs"]
        static = batch["static_features"]

        parts = [prev_a, prev_y]
        if self.has_vitals:
            parts.append(batch["vitals"])

        x = torch.cat(parts, dim=-1)
        x = torch.cat(
            [
                x,
                static.unsqueeze(1).expand(-1, x.size(1), -1),
            ],
            dim=-1,
        )
        return self.history_lstm(x)

    def _marginal_input(self, batch: dict, time_steps: int):
        batch_size = batch["static_features"].size(0)
        device = batch["static_features"].device
        dtype = batch["static_features"].dtype

        time_index = torch.arange(
            time_steps, device=device, dtype=dtype
        ) / max(time_steps - 1, 1)
        time_index = (
            time_index.unsqueeze(0)
            .unsqueeze(-1)
            .expand(batch_size, -1, 1)
        )
        static = (
            batch["static_features"]
            .unsqueeze(1)
            .expand(-1, time_steps, -1)
        )
        return torch.cat([static, time_index], dim=-1)

    def _compute_flow_ipw(
        self, batch: dict, history: torch.Tensor
    ) -> torch.Tensor:
        treatment = batch["current_treatments"]

        if not self.use_ipw:
            return torch.ones(
                *treatment.shape[:-1],
                1,
                device=treatment.device,
                dtype=treatment.dtype,
            )

        conditional_log_prob = self.conditional_flow.log_prob(
            treatment, history
        ).sum(dim=-1, keepdim=True)

        marginal_context = self._marginal_input(
            batch, treatment.size(1)
        )
        marginal_log_prob = self.marginal_flow.log_prob(
            treatment, marginal_context
        ).sum(dim=-1, keepdim=True)

        log_increment = (
            marginal_log_prob - conditional_log_prob
        )
        cumulative_log_weight = torch.cumsum(
            log_increment, dim=1
        )

        max_log_weight = float(np.log(max(self.max_ipw_weight, 1.0)))
        cumulative_log_weight = cumulative_log_weight.clamp(
            min=-20.0, max=max_log_weight
        )
        return torch.exp(cumulative_log_weight)

    def forward(self, batch: dict) -> torch.Tensor:
        history = self._build_history(batch)
        history_and_dose = torch.cat(
            [history, batch["current_treatments"]], dim=-1
        )
        return self.outcome_head(history_and_dose)

    def training_step(self, batch, batch_idx):
        history = self._build_history(batch)
        treatment = batch["current_treatments"]
        active = batch["active_entries"]

        conditional_log_prob = self.conditional_flow.log_prob(
            treatment, history
        ).sum(dim=-1, keepdim=True)

        marginal_context = self._marginal_input(
            batch, treatment.size(1)
        )
        marginal_log_prob = self.marginal_flow.log_prob(
            treatment, marginal_context
        ).sum(dim=-1, keepdim=True)

        active_denominator = active.sum().clamp_min(1.0)
        conditional_nll = -(
            conditional_log_prob * active
        ).sum() / active_denominator
        marginal_nll = -(
            marginal_log_prob * active
        ).sum() / active_denominator

        flow_nll = 0.5 * (conditional_nll + marginal_nll)

        history_and_dose = torch.cat(
            [history, treatment], dim=-1
        )
        prediction = self.outcome_head(history_and_dose)
        squared_error = F.mse_loss(
            prediction, batch["outputs"], reduction="none"
        ).mean(dim=-1, keepdim=True)

        with torch.no_grad():
            weights = self._compute_flow_ipw(
                batch, history
            )

        denominator = (weights * active).sum().clamp_min(1e-8)
        outcome_mse = (
            weights * active * squared_error
        ).sum() / denominator

        loss = (
            self.alpha_flow * flow_nll
            + self.alpha_outcome * outcome_mse
        )

        self.log(
            "flow_train_conditional_nll",
            conditional_nll,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "flow_train_marginal_nll",
            marginal_nll,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "flow_train_nll",
            flow_nll,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "flow_train_outcome_mse",
            outcome_mse,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "flow_train_loss",
            loss,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return loss

    def predict_step(self, batch, batch_idx, dataset_idx=None):
        return self(batch).detach().cpu()

    def prepare_data(self) -> None:
        if (
            self.dataset_collection is not None
            and not self.dataset_collection.processed_data_multi
        ):
            self.dataset_collection.process_data_multi()

        if self.dataset_collection is not None:
            self.dataset_collection.split_train_f_holdout(
                self.hparams.dataset.holdout_ratio
            )
            self.dataset_collection.explode_cf_treatment_seq(
                self.mc_samples
            )

    def _predict_one_dataset(self, dataset: Dataset) -> np.ndarray:
        loader = DataLoader(
            dataset,
            batch_size=self.hparams.dataset.val_batch_size,
            shuffle=False,
            num_workers=0,
        )
        batches = self.trainer.predict(self, loader)
        return torch.cat(batches, dim=0).numpy()

    def get_predictions(
        self,
        dataset: Union[Dataset, List[Dataset]],
        vitals: bool = False,
    ) -> np.ndarray:
        del vitals

        if isinstance(dataset, list):
            return np.stack(
                [self._predict_one_dataset(d) for d in dataset],
                axis=0,
            )

        logger.info(
            "FLOW predictions for %s.",
            getattr(dataset, "subset_name", "dataset"),
        )
        return self._predict_one_dataset(dataset)

    def get_autoregressive_predictions(
        self,
        datasets,
        treatment_sequence=None,
    ) -> np.ndarray:
        """
        Autoregressive counterfactual rollout performed on deep copies.

        treatment_sequence 长度必须 >= projection_horizon + 1:
            [0]              -> 写入锚点位 split - 1(反事实开始前一步);
            [1..horizon]     -> 写入未来 horizon 步 split..split+horizon-1。
        """
        if isinstance(datasets, list):
            work_datasets = [deepcopy(d) for d in datasets]
        else:
            work_datasets = [
                deepcopy(datasets) for _ in range(self.mc_samples)
            ]

        if len(work_datasets) != self.mc_samples:
            raise ValueError(
                "Expected "
                f"{self.mc_samples} datasets, got {len(work_datasets)}."
            )

        horizon = self.hparams.dataset.projection_horizon

        if treatment_sequence is not None:
            treatment_sequence = np.asarray(
                treatment_sequence, dtype=np.float32
            )
            if (
                treatment_sequence.ndim != 2
                or treatment_sequence.shape[0] < horizon + 1
                or treatment_sequence.shape[1]
                != self.dim_treatments
            ):
                raise ValueError(
                    "treatment_sequence must have shape "
                    f"(>= {horizon + 1}, {self.dim_treatments}), got "
                    f"{treatment_sequence.shape}."
                )

            treatment_sequence = treatment_sequence[: horizon + 1]

            for dataset in work_datasets:
                split = dataset.data["future_past_split"].astype(int)
                row_index = np.arange(len(dataset))

                # 锚点:反事实开始前一步。split=0 时兜底到 0 位置。
                anchor_pos = np.maximum(split - 1, 0)
                dataset.data["current_treatments"][
                    row_index, anchor_pos, :
                ] = treatment_sequence[0]

                # 未来 horizon 步
                for step in range(horizon):
                    dataset.data["current_treatments"][
                        row_index, split + step, :
                    ] = treatment_sequence[step + 1]

        predictions = np.zeros(
            (
                self.mc_samples,
                len(work_datasets[0]),
                horizon,
                self.dim_outcome,
            ),
            dtype=np.float32,
        )

        for step in range(horizon + 1):
            for mc_index, dataset in enumerate(work_datasets):
                all_predictions = self.get_predictions(dataset)
                split = dataset.data["future_past_split"].astype(int)
                row_index = np.arange(len(dataset))

                if step > 0:
                    predictions[
                        mc_index, :, step - 1, :
                    ] = all_predictions[
                        row_index, split - 1 + step, :
                    ]

                if step < horizon:
                    dataset.data["prev_outputs"][
                        row_index, split + step, :
                    ] = all_predictions[
                        row_index, split - 1 + step, :
                    ]

        return predictions.mean(axis=0)