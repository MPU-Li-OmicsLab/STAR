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


def zi_beta_logprob(
    dose: torch.Tensor,
    p_treat: torch.Tensor,
    mu: torch.Tensor,
    log_conc: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Per-dimension log probability under a zero-inflated Beta distribution.

    P(A=0 | h) = 1 - p_treat
    p(A=a | h) = p_treat * Beta(a; alpha, beta),  a in (0, 1)
    """
    conc = F.softplus(log_conc) + eps
    mu_c = torch.clamp(mu, eps, 1.0 - eps)
    alpha = conc * mu_c
    beta = conc * (1.0 - mu_c)
    p_c = torch.clamp(p_treat, eps, 1.0 - eps)

    log_p0 = torch.log1p(-p_c)
    log_p1 = torch.log(p_c)

    dose_safe = torch.clamp(dose, eps, 1.0 - eps)
    log_beta_density = torch.distributions.Beta(alpha, beta).log_prob(dose_safe)

    is_zero = dose.le(eps)
    return torch.where(is_zero, log_p0, log_p1 + log_beta_density)


class ZIBetaHead(nn.Module):
    """Conditional zero-inflated Beta parameters for D treatment dimensions."""

    def __init__(self, in_features: int, dim: int, hidden: int):
        super().__init__()
        self.fc = nn.Linear(in_features, hidden)
        self.fc_p = nn.Linear(hidden, dim)
        self.fc_mu = nn.Linear(hidden, dim)
        self.fc_conc = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor):
        h = F.elu(self.fc(x))
        return (
            torch.sigmoid(self.fc_p(h)),
            torch.sigmoid(self.fc_mu(h)),
            self.fc_conc(h),
        )


class OutcomeHead(nn.Module):
    """MLP mapping (pre-treatment history, current dose) to next outcome."""

    def __init__(self, in_features: int, dim_outcome: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ELU(),
            nn.Linear(hidden, dim_outcome),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GPS(TimeVaryingCausalModel):
    """
    Longitudinal generalized propensity score baseline for continuous dose.

    treatment_sequence 约定(与 IGCNET 对齐):
        长度为 projection_horizon + 1;
        treatment_sequence[0] 写入反事实开始前一步(锚点位, split - 1);
        treatment_sequence[1..horizon] 写入未来 horizon 步(split..split+horizon-1)。
    """

    model_type = "gps"
    possible_model_types = {"gps"}
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
        assert self.autoregressive, "GPS requires autoregressive=True"

        self.input_size = (
            self.dim_treatments
            + self.dim_outcome
            + (self.dim_vitals if self.has_vitals else 0)
            + self.dim_static_features
        )
        logger.info("Input size of %s: %s", self.model_type, self.input_size)

        self._init_specific(args.model.gps)
        self.save_hyperparameters(args)

    def _init_specific(self, sub_args: DictConfig):
        try:
            self.seq_hidden_units = sub_args.seq_hidden_units
            self.fc_hidden_units = sub_args.fc_hidden_units
            self.num_layer = sub_args.num_layer
            self.dropout_rate = sub_args.dropout_rate
            self.mc_samples = int(sub_args.get("mc_samples", 1))
            self.alpha_prop = float(sub_args.get("alpha_prop", 1.0))
            self.alpha_outcome = float(sub_args.get("alpha_outcome", 1.0))
            self.max_ipw_weight = float(
                sub_args.get("max_ipw_weight", 10.0)
            )
            self.stabilised = bool(
                sub_args.get("stabilised_weights", True)
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

            self.cond_gps = ZIBetaHead(
                self.seq_hidden_units,
                self.dim_treatments,
                self.fc_hidden_units,
            )

            self.marg_gps = ZIBetaHead(
                self.dim_static_features + 1,
                self.dim_treatments,
                self.fc_hidden_units,
            )

            self.outcome_head = OutcomeHead(
                self.seq_hidden_units + self.dim_treatments,
                self.dim_outcome,
                self.fc_hidden_units,
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

    def _marginal_input(self, batch: dict, time_steps: int) -> torch.Tensor:
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

    def _compute_stabilised_ipw(
        self, batch: dict, history: torch.Tensor
    ) -> torch.Tensor:
        treatment = batch["current_treatments"]
        time_steps = treatment.size(1)

        p_c, mu_c, lc_c = self.cond_gps(history)
        log_cond = zi_beta_logprob(
            treatment, p_c, mu_c, lc_c
        ).sum(dim=-1, keepdim=True)

        if self.stabilised:
            p_m, mu_m, lc_m = self.marg_gps(
                self._marginal_input(batch, time_steps)
            )
            log_marg = zi_beta_logprob(
                treatment, p_m, mu_m, lc_m
            ).sum(dim=-1, keepdim=True)
            log_increment = log_marg - log_cond
        else:
            log_increment = -log_cond

        cumulative_log_weight = torch.cumsum(
            log_increment, dim=1
        )

        max_log_weight = float(np.log(max(self.max_ipw_weight, 1.0)))
        cumulative_log_weight = cumulative_log_weight.clamp(
            min=-20.0, max=max_log_weight
        )
        return torch.exp(cumulative_log_weight)

    def forward(self, batch: dict) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        batch = {
            k: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in batch.items()
        }
        history = self._build_history(batch)
        history_and_dose = torch.cat(
            [history, batch["current_treatments"]], dim=-1
        )
        return self.outcome_head(history_and_dose)

    def training_step(self, batch, batch_idx):
        history = self._build_history(batch)
        treatment = batch["current_treatments"]
        active = batch["active_entries"]

        p_c, mu_c, lc_c = self.cond_gps(history)
        cond_log_prob = zi_beta_logprob(
            treatment, p_c, mu_c, lc_c
        ).sum(dim=-1, keepdim=True)

        marginal_context = self._marginal_input(
            batch, treatment.size(1)
        )
        p_m, mu_m, lc_m = self.marg_gps(marginal_context)
        marginal_log_prob = zi_beta_logprob(
            treatment, p_m, mu_m, lc_m
        ).sum(dim=-1, keepdim=True)

        active_denominator = active.sum().clamp_min(1.0)
        conditional_nll = -(
            cond_log_prob * active
        ).sum() / active_denominator
        marginal_nll = -(
            marginal_log_prob * active
        ).sum() / active_denominator

        prop_nll = 0.5 * (conditional_nll + marginal_nll)

        history_and_dose = torch.cat(
            [history, treatment], dim=-1
        )
        prediction = self.outcome_head(history_and_dose)
        squared_error = F.mse_loss(
            prediction, batch["outputs"], reduction="none"
        ).mean(dim=-1, keepdim=True)

        with torch.no_grad():
            weights = self._compute_stabilised_ipw(
                batch, history
            )

        denominator = (weights * active).sum().clamp_min(1e-8)
        outcome_mse = (
            weights * active * squared_error
        ).sum() / denominator

        loss = (
            self.alpha_prop * prop_nll
            + self.alpha_outcome * outcome_mse
        )

        self.log(
            "gps_train_conditional_nll",
            conditional_nll,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "gps_train_marginal_nll",
            marginal_nll,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "gps_train_prop_nll",
            prop_nll,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "gps_train_outcome_mse",
            outcome_mse,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "gps_train_loss",
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
            "GPS predictions for %s.",
            getattr(dataset, "subset_name", "dataset"),
        )
        return self._predict_one_dataset(dataset)

    def get_autoregressive_predictions(
        self,
        datasets,
        treatment_sequence=None,
    ) -> np.ndarray:
        """
        Autoregressive counterfactual rollout.

        treatment_sequence 长度必须 >= projection_horizon + 1:
            [0]              -> 写入锚点位 split - 1(反事实开始前一步)
            [1..horizon]     -> 写入未来 horizon 步 split..split+horizon-1
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