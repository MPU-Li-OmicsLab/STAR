from __future__ import annotations

import copy
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _cfg_get(obj: Any, path: Sequence[str], default: Any = None) -> Any:
    """Safely read nested DictConfig/dict/object attributes."""
    cur = obj
    for key in path:
        if cur is None:
            return default
        try:
            if isinstance(cur, dict):
                if key not in cur:
                    return default
                cur = cur[key]
            elif hasattr(cur, "get"):
                sentinel = object()
                value = cur.get(key, sentinel)
                if value is sentinel:
                    return default
                cur = value
            else:
                cur = getattr(cur, key)
        except Exception:
            try:
                cur = getattr(cur, key)
            except Exception:
                return default
    return cur


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None and value != "" else int(default)


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value is not None and value != "" else str(default)


def _to_cpu_detached(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _to_cpu_detached(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu_detached(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_detached(v) for v in value)
    return copy.deepcopy(value)


def _infer_batch_size(batch: Dict[str, Any]) -> int:
    for key in ("current_treatments", "prev_treatments", "outputs", "active_entries"):
        value = batch.get(key)
        if torch.is_tensor(value) and value.ndim >= 1:
            return int(value.shape[0])
    for value in batch.values():
        if torch.is_tensor(value) and value.ndim >= 1:
            return int(value.shape[0])
    raise ValueError("Unable to infer batch size from probe batch.")


def _slice_batch(batch: Dict[str, Any], n: int) -> Dict[str, Any]:
    batch_size = _infer_batch_size(batch)
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[0]) == batch_size:
            out[key] = value[:n].detach().cpu().clone()
        else:
            out[key] = _to_cpu_detached(value)
    return out


def _module_to_cpu_eval(module: nn.Module) -> nn.Module:
    module = copy.deepcopy(module).cpu()
    module.eval()
    for p in module.parameters():
        p.requires_grad = False
    return module


def _state_distance(a: Optional[nn.Module], b: Optional[nn.Module]) -> Tuple[float, float]:
    """Return raw L2 and per-element RMS parameter/buffer displacement."""
    if a is None or b is None:
        return float("nan"), float("nan")

    sa = a.state_dict()
    sb = b.state_dict()
    common = [k for k in sa.keys() if k in sb and torch.is_tensor(sa[k]) and torch.is_tensor(sb[k])]
    if not common:
        return 0.0, 0.0

    ss = 0.0
    n = 0
    for key in common:
        xa = sa[key].detach().float().cpu()
        xb = sb[key].detach().float().cpu()
        if xa.shape != xb.shape:
            continue
        d = xa - xb
        ss += float(d.pow(2).sum().item())
        n += int(d.numel())
    if n == 0:
        return 0.0, 0.0
    return math.sqrt(ss), math.sqrt(ss / n)


def _safe_float(x: Any) -> float:
    if x is None:
        return float("nan")
    if torch.is_tensor(x):
        if x.numel() == 0:
            return float("nan")
        return float(x.detach().float().mean().cpu().item())
    try:
        return float(x)
    except Exception:
        return float("nan")


# -----------------------------------------------------------------------------
# Deterministic history-backbone snapshot
# -----------------------------------------------------------------------------


class HistoryBackboneSnapshot(nn.Module):
    """
    Minimal deterministic copy of the IGC/STAR history representation path.

    It intentionally mirrors model.build_hr() without keeping any G-computation
    head. All modules are deep-copied and placed in eval mode so the mechanism
    measurement itself does not contain dropout noise.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.has_vitals = bool(getattr(model, "has_vitals", False))

        self.treatments_input_transformation = copy.deepcopy(model.treatments_input_transformation)
        self.vitals_input_transformation = (
            copy.deepcopy(model.vitals_input_transformation)
            if getattr(model, "vitals_input_transformation", None) is not None
            else None
        )
        self.outputs_input_transformation = copy.deepcopy(model.outputs_input_transformation)
        self.static_input_transformation = copy.deepcopy(model.static_input_transformation)
        self.transformer_blocks = copy.deepcopy(model.transformer_blocks)
        self.self_positional_encoding = (
            copy.deepcopy(model.self_positional_encoding)
            if getattr(model, "self_positional_encoding", None) is not None
            else None
        )
        self.hr_output_transformation = copy.deepcopy(model.hr_output_transformation)
        self.output_dropout = copy.deepcopy(model.output_dropout)

        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(
        self,
        prev_treatments: torch.Tensor,
        vitals: Optional[torch.Tensor],
        prev_outputs: torch.Tensor,
        static_features: torch.Tensor,
        active_entries: torch.Tensor,
    ) -> torch.Tensor:
        active_entries_treat_outcomes = active_entries
        active_entries_vitals = active_entries

        prev_treatments = prev_treatments.to(dtype=torch.float32)
        prev_outputs = prev_outputs.to(dtype=torch.float32)
        static_features = static_features.to(dtype=torch.float32)
        vitals = vitals.to(dtype=torch.float32)

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
                    (x_t, x_o, x_v),
                    x_s,
                    active_entries_treat_outcomes,
                    active_entries_vitals,
                )
            else:
                x_t, x_o = block((x_t, x_o), x_s, active_entries_treat_outcomes)

        x = (x_o + x_t + x_v) / 3 if self.has_vitals else (x_o + x_t) / 2
        output = self.output_dropout(x)
        return torch.nn.functional.elu(self.hr_output_transformation(output))


@dataclass
class GeneratorSnapshot:
    backbone: nn.Module
    head: nn.Module


@dataclass
class PreviousMonitorState:
    generator: GeneratorSnapshot
    live_backbone: nn.Module


# -----------------------------------------------------------------------------
# Main monitor
# -----------------------------------------------------------------------------


class TargetDriftMonitor:
    """
    Training-time mechanism instrumentation for IGC-Net, SFTR+Naive, and STAR.

    No checkpoint is required. At each monitoring point it keeps only the
    previous target-generator snapshot in CPU memory and evaluates the same
    fixed probe bank under:

        Y00 = q^s       ( f^s       (H_cf) )
        Y10 = q^{s+1}   ( f^s       (H_cf) )
        Y01 = q^s       ( f^{s+1}   (H_cf) )
        Y11 = q^{s+1}   ( f^{s+1}   (H_cf) )

    Ordered path (operator then representation):
        delta_op    = Y10 - Y00
        delta_rep   = Y11 - Y10
        delta_total = Y11 - Y00

    Symmetric two-factor / Shapley attribution:
        delta_op_sym  = 0.5 * [(Y10-Y00) + (Y11-Y01)]
        delta_rep_sym = 0.5 * [(Y01-Y00) + (Y11-Y10)]

    Both signed decompositions close exactly to delta_total. Norm-based
    magnitudes are reported separately and are NOT assumed additive.
    """

    TRACE_COLUMNS = [
        "run_id",
        "model",
        "mechanism_kind",
        "strategy",
        "seed",
        "horizon",
        "stage",
        "q_j",
        "epoch",
        "global_step",
        "monitor_idx",
        "n_probe",
        "n_target_values",
        "stage_start",
        "D_op_mae",
        "D_rep_mae",
        "D_total_mae",
        "D_op_rmse",
        "D_rep_rmse",
        "D_total_rmse",
        "signed_op_mean",
        "signed_rep_mean",
        "signed_total_mean",
        "decomp_residual_mae",
        "decomp_residual_max",
        "T00_mean",
        "T10_mean",
        "T01_mean",
        "T11_mean",
        "D_op_sym_mae",
        "D_rep_sym_mae",
        "D_op_sym_rmse",
        "D_rep_sym_rmse",
        "signed_op_sym_mean",
        "signed_rep_sym_mean",
        "sym_decomp_residual_mae",
        "sym_decomp_residual_max",
        "target_mean",
        "target_std",
        "student_target_rmse",
        "student_target_mae",
        "teacher_head_param_delta_l2",
        "teacher_head_param_delta_rms",
        "teacher_repr_param_delta_l2",
        "teacher_repr_param_delta_rms",
        "student_repr_param_delta_l2",
        "student_repr_param_delta_rms",
        "cf_integrity_pass",
        "student_backbone_train_mode_fraction",
        "trainable_params",
        "total_params",
        "head_lr",
        "backbone_lr",
    ]

    def __init__(
        self,
        enabled: bool,
        model_label: str,
        mechanism_kind: str,
        output_dir: str,
        run_id: str,
        strategy: str,
        seed: int,
        every_n_steps: int,
        probe_size: int,
        probe_cache_path: str,
        eval_device: str = "auto",
    ):
        self.enabled = bool(enabled)
        self.model_label = str(model_label)
        self.mechanism_kind = str(mechanism_kind).lower()
        if self.mechanism_kind not in {"star", "igc", "frozen_head", "frozen_rep"}:
            raise ValueError(
                "mechanism_kind must be one of {'star', 'igc', 'frozen_head', 'frozen_rep'}, "
                f"got {mechanism_kind!r}"
            )

        self.output_dir = Path(output_dir)
        self.run_id = str(run_id)
        self.strategy = str(strategy)
        self.seed = int(seed)
        self.every_n_steps = max(int(every_n_steps), 1)
        self.probe_size = max(int(probe_size), 1)
        self.probe_cache_path = Path(probe_cache_path) if probe_cache_path else None
        self.eval_device = str(eval_device).lower().strip()
        if self.eval_device not in {"auto", "cpu", "cuda", "model"}:
            raise ValueError(
                "mechanism.eval_device must be one of {'auto','cpu','cuda','model'}, "
                f"got {eval_device!r}"
            )

        self.run_dir = self.output_dir / self.run_id
        self.trace_path = self.run_dir / "mechanism_trace.csv"
        self.performance_path = self.run_dir / "performance.csv"
        self.summary_path = self.run_dir / "mechanism_summary.csv"
        self.metadata_path = self.run_dir / "mechanism_metadata.json"

        self.probe_batches: List[Dict[str, Any]] = []
        self.probe_count = 0
        self.probe_ready = False
        self._last_monitor_step = -1
        self._monitor_idx = 0
        self._last_stage_key: Optional[int] = None
        self._previous: Dict[int, PreviousMonitorState] = {}

        if self.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._load_probe_if_available()
            self._write_metadata(initial=True)

    @classmethod
    def from_args(cls, args: Any, model_label: str, mechanism_kind: str) -> "TargetDriftMonitor":
        cfg = _cfg_get(args, ["model", "igcnet", "mechanism"], {})

        cfg_enabled = bool(_cfg_get(cfg, ["enabled"], False))
        enabled = _env_bool("MECH_ENABLED", cfg_enabled)

        default_output = str(_cfg_get(cfg, ["output_dir"], "results/mechanism"))
        output_dir = _env_str("MECH_OUTPUT_DIR", default_output)

        default_run_id = str(_cfg_get(cfg, ["run_id"], "run"))
        run_id = _env_str("MECH_RUN_ID", default_run_id)

        default_strategy = str(
            _cfg_get(cfg, ["strategy"], _cfg_get(args, ["dataset", "strategy_name"], "unknown"))
        )
        strategy = _env_str("MECH_STRATEGY", default_strategy)

        default_seed = int(
            _cfg_get(cfg, ["seed"], _cfg_get(args, ["exp", "seed"], _cfg_get(args, ["seed"], 0)))
            or 0
        )
        seed = _env_int("MECH_SEED", default_seed)

        every_n_steps = _env_int(
            "MECH_EVERY_N_STEPS",
            int(_cfg_get(cfg, ["every_n_steps"], 100)),
        )
        probe_size = _env_int(
            "MECH_PROBE_SIZE",
            int(_cfg_get(cfg, ["probe_size"], 128)),
        )

        default_probe = str(_cfg_get(cfg, ["probe_cache_path"], ""))
        probe_cache_path = _env_str("MECH_PROBE_CACHE", default_probe)

        default_eval_device = str(_cfg_get(cfg, ["eval_device"], "auto"))
        eval_device = _env_str("MECH_EVAL_DEVICE", default_eval_device)

        model_label = _env_str("MECH_MODEL_LABEL", model_label)

        return cls(
            enabled=enabled,
            model_label=model_label,
            mechanism_kind=mechanism_kind,
            output_dir=output_dir,
            run_id=run_id,
            strategy=strategy,
            seed=seed,
            every_n_steps=every_n_steps,
            probe_size=probe_size,
            probe_cache_path=probe_cache_path,
            eval_device=eval_device,
        )

    # ------------------------------------------------------------------
    # Probe-bank lifecycle
    # ------------------------------------------------------------------

    def _load_probe_if_available(self) -> None:
        if not self.enabled or self.probe_cache_path is None:
            return
        if not self.probe_cache_path.exists():
            return
        try:
            loaded = torch.load(self.probe_cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            loaded = torch.load(self.probe_cache_path, map_location="cpu")
        if isinstance(loaded, dict) and "probe_batches" in loaded:
            batches = loaded["probe_batches"]
        else:
            batches = loaded
        if not isinstance(batches, list):
            raise ValueError(f"Invalid probe cache: {self.probe_cache_path}")
        self.probe_batches = batches
        self.probe_count = sum(_infer_batch_size(b) for b in self.probe_batches)
        self.probe_ready = self.probe_count > 0

    def collect_probe(self, model: nn.Module, batch: Dict[str, Any]) -> None:
        """Call from on_train_batch_start, before model.forward can mutate batch."""
        if not self.enabled or self.probe_ready:
            return
        if not self._is_global_zero(model):
            return
        if not isinstance(batch, dict):
            raise TypeError("Mechanism monitor currently expects dict batches.")

        remaining = self.probe_size - self.probe_count
        if remaining <= 0:
            self._finalize_probe_bank()
            return

        batch_size = _infer_batch_size(batch)
        take = min(batch_size, remaining)
        self.probe_batches.append(_slice_batch(batch, take))
        self.probe_count += take

        if self.probe_count >= self.probe_size:
            self._finalize_probe_bank()

    def _finalize_probe_bank(self) -> None:
        if self.probe_count <= 0:
            return
        self.probe_ready = True
        if self.probe_cache_path is not None:
            self.probe_cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.probe_cache_path.with_suffix(self.probe_cache_path.suffix + ".tmp")
            payload = {
                "probe_batches": self.probe_batches,
                "probe_count": self.probe_count,
                "probe_size_requested": self.probe_size,
                "seed": self.seed,
            }
            torch.save(payload, tmp_path)
            # Atomic rename on a single filesystem.
            try:
                os.replace(tmp_path, self.probe_cache_path)
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                if not self.probe_cache_path.exists():
                    raise

    # ------------------------------------------------------------------
    # Hook entry point
    # ------------------------------------------------------------------

    def observe(self, model: nn.Module) -> None:
        if not self.enabled or not self.probe_ready:
            return
        if not self._is_global_zero(model):
            return
        trainer = getattr(model, "trainer", None)
        if trainer is None:
            return

        step = int(getattr(trainer, "global_step", 0))
        if step <= 0 or step == self._last_monitor_step:
            return
        if step % self.every_n_steps != 0:
            return

        K = int(getattr(model, "projection_horizon", 0))
        if K <= 0:
            return

        epoch = int(getattr(model, "current_epoch", 0))
        if self.mechanism_kind in {"star", "frozen_head", "frozen_rep"}:
            stage = getattr(model, "_stagewise_active_stage", None)
            if stage is None and hasattr(model, "_stagewise_stage_for_epoch"):
                stage = int(model._stagewise_stage_for_epoch())
            if stage is None:
                return
            stage = int(stage)
            stage_key = stage
            j_list = [stage] if stage < K else []
        else:
            stage = -1
            stage_key = -1
            j_list = list(range(K))

        # STAR changes the target-generating path deliberately at stage boundaries;
        # do not count cross-stage changes as within-stage non-stationarity.
        if self._last_stage_key is None or stage_key != self._last_stage_key:
            self._previous.clear()
            self._last_stage_key = stage_key

        if not j_list:
            self._last_monitor_step = step
            self._monitor_idx += 1
            return

        current_live_backbone = self._capture_live_backbone(model)
        rows: List[Dict[str, Any]] = []

        for j in j_list:
            current_generator = self._capture_actual_target_generator(
                model=model,
                j=j,
                current_live_backbone=current_live_backbone,
            )
            if current_generator is None:
                continue

            current_student_head = _module_to_cpu_eval(model.G_comp_heads[j])
            y11, integrity = self._evaluate_target_generator(model, current_generator, j)
            student_pred = self._evaluate_student(
                model=model,
                live_backbone=current_live_backbone,
                student_head=current_student_head,
                j=j,
            )

            n = min(int(y11.shape[0]), int(student_pred.shape[0]))
            y11 = y11[:n]
            student_pred = student_pred[:n]
            joint_finite = torch.isfinite(y11).all(dim=-1) & torch.isfinite(student_pred).all(dim=-1)
            if joint_finite.any():
                y11_valid = y11[joint_finite]
                student_valid = student_pred[joint_finite]
                student_err = student_valid - y11_valid
                student_rmse = float(student_err.pow(2).mean().sqrt().item())
                student_mae = float(student_err.abs().mean().item())
            else:
                y11_valid = y11[torch.isfinite(y11).all(dim=-1)]
                student_rmse = float("nan")
                student_mae = float("nan")

            row = self._base_row(model, j, stage, epoch, step)
            row.update(
                {
                    "n_probe": int(self.probe_count),
                    "n_target_values": int(y11.shape[0]),
                    "target_mean": float(y11_valid.mean().item()) if y11_valid.numel() else float("nan"),
                    "target_std": float(y11_valid.float().std(unbiased=False).item()) if y11_valid.numel() else float("nan"),
                    "student_target_rmse": student_rmse,
                    "student_target_mae": student_mae,
                    "cf_integrity_pass": int(bool(integrity)),
                }
            )

            previous = self._previous.get(j)
            if previous is None:
                row.update(self._nan_drift_fields(stage_start=1))
            else:
                # Four-corner evaluation of the same target generator:
                #
                #   Y00 = old head(old representation)
                #   Y10 = new head(old representation)
                #   Y01 = old head(new representation)
                #   Y11 = new head(new representation)
                #
                # The original ordered decomposition follows Y00 -> Y10 -> Y11.
                # Y01 supplies the reverse-order corner and makes the symmetric
                # two-factor / Shapley attribution identifiable without changing
                # model training in any way.
                y00, integrity00 = self._evaluate_target_generator(model, previous.generator, j)

                hybrid10 = GeneratorSnapshot(
                    backbone=previous.generator.backbone,
                    head=current_generator.head,
                )
                y10, integrity10 = self._evaluate_target_generator(model, hybrid10, j)

                hybrid01 = GeneratorSnapshot(
                    backbone=current_generator.backbone,
                    head=previous.generator.head,
                )
                y01, integrity01 = self._evaluate_target_generator(model, hybrid01, j)

                n_common = min(
                    int(y00.shape[0]),
                    int(y10.shape[0]),
                    int(y01.shape[0]),
                    int(y11.shape[0]),
                )
                y00 = y00[:n_common]
                y10 = y10[:n_common]
                y01 = y01[:n_common]
                y11_c = y11[:n_common]
                finite = (
                    torch.isfinite(y00).all(dim=-1)
                    & torch.isfinite(y10).all(dim=-1)
                    & torch.isfinite(y01).all(dim=-1)
                    & torch.isfinite(y11_c).all(dim=-1)
                )
                y00 = y00[finite]
                y10 = y10[finite]
                y01 = y01[finite]
                y11_c = y11_c[finite]

                if y00.numel() > 0:
                    # Ordered attribution: operator first, representation second.
                    d_op = y10 - y00
                    d_rep = y11_c - y10
                    d_total = y11_c - y00
                    residual = d_total - d_op - d_rep

                    # Reverse-order paths are averaged to remove arbitrary path
                    # ordering from source attribution.
                    d_op_sym = 0.5 * ((y10 - y00) + (y11_c - y01))
                    d_rep_sym = 0.5 * ((y01 - y00) + (y11_c - y10))
                    sym_residual = d_total - d_op_sym - d_rep_sym

                    row.update(
                        {
                            "stage_start": 0,
                            "D_op_mae": float(d_op.abs().mean().item()),
                            "D_rep_mae": float(d_rep.abs().mean().item()),
                            "D_total_mae": float(d_total.abs().mean().item()),
                            "D_op_rmse": float(d_op.pow(2).mean().sqrt().item()),
                            "D_rep_rmse": float(d_rep.pow(2).mean().sqrt().item()),
                            "D_total_rmse": float(d_total.pow(2).mean().sqrt().item()),
                            "signed_op_mean": float(d_op.mean().item()),
                            "signed_rep_mean": float(d_rep.mean().item()),
                            "signed_total_mean": float(d_total.mean().item()),
                            "decomp_residual_mae": float(residual.abs().mean().item()),
                            "decomp_residual_max": float(residual.abs().max().item()),
                            "T00_mean": float(y00.mean().item()),
                            "T10_mean": float(y10.mean().item()),
                            "T01_mean": float(y01.mean().item()),
                            "T11_mean": float(y11_c.mean().item()),
                            "D_op_sym_mae": float(d_op_sym.abs().mean().item()),
                            "D_rep_sym_mae": float(d_rep_sym.abs().mean().item()),
                            "D_op_sym_rmse": float(d_op_sym.pow(2).mean().sqrt().item()),
                            "D_rep_sym_rmse": float(d_rep_sym.pow(2).mean().sqrt().item()),
                            "signed_op_sym_mean": float(d_op_sym.mean().item()),
                            "signed_rep_sym_mean": float(d_rep_sym.mean().item()),
                            "sym_decomp_residual_mae": float(sym_residual.abs().mean().item()),
                            "sym_decomp_residual_max": float(sym_residual.abs().max().item()),
                        }
                    )
                else:
                    row.update(self._nan_drift_fields(stage_start=0))

                head_l2, head_rms = _state_distance(previous.generator.head, current_generator.head)
                teacher_repr_l2, teacher_repr_rms = _state_distance(
                    previous.generator.backbone, current_generator.backbone
                )
                student_repr_l2, student_repr_rms = _state_distance(
                    previous.live_backbone, current_live_backbone
                )
                row.update(
                    {
                        "teacher_head_param_delta_l2": head_l2,
                        "teacher_head_param_delta_rms": head_rms,
                        "teacher_repr_param_delta_l2": teacher_repr_l2,
                        "teacher_repr_param_delta_rms": teacher_repr_rms,
                        "student_repr_param_delta_l2": student_repr_l2,
                        "student_repr_param_delta_rms": student_repr_rms,
                        "cf_integrity_pass": int(
                            bool(integrity and integrity00 and integrity10 and integrity01)
                        ),
                    }
                )

            # If no previous state, parameter-delta columns are NaN.
            for key in (
                "teacher_head_param_delta_l2",
                "teacher_head_param_delta_rms",
                "teacher_repr_param_delta_l2",
                "teacher_repr_param_delta_rms",
                "student_repr_param_delta_l2",
                "student_repr_param_delta_rms",
            ):
                row.setdefault(key, float("nan"))

            rows.append(row)
            self._previous[j] = PreviousMonitorState(
                generator=current_generator,
                live_backbone=current_live_backbone,
            )

        if rows:
            self._append_rows(self.trace_path, rows, self.TRACE_COLUMNS)
            self._write_summary()

        self._last_monitor_step = step
        self._monitor_idx += 1

    # ------------------------------------------------------------------
    # Capture and evaluation
    # ------------------------------------------------------------------

    def _capture_live_backbone(self, model: nn.Module) -> nn.Module:
        return HistoryBackboneSnapshot(model).cpu().eval()

    def _capture_actual_target_generator(
        self,
        model: nn.Module,
        j: int,
        current_live_backbone: nn.Module,
    ) -> Optional[GeneratorSnapshot]:
        K = int(model.projection_horizon)
        if not (0 <= j < K):
            return None

        if self.mechanism_kind in {"star", "frozen_rep"}:
            # STAR and FrozenRep both generate pseudo-targets from a stage-start
            # target-side representation snapshot. Prefer an explicit model
            # accessor when available; otherwise fall back to the historical box.
            target_backbone = None
            getter = getattr(model, "get_teacher_representation_module", None)
            if callable(getter):
                try:
                    target_backbone = getter()
                except Exception:
                    target_backbone = None

            if target_backbone is None:
                getter = getattr(model, "mechanism_teacher_representation", None)
                if callable(getter):
                    try:
                        target_backbone = getter()
                    except Exception:
                        target_backbone = None

            if target_backbone is None:
                box = getattr(model, "_stagewise_target_history_backbone_box", None)
                target_backbone = (
                    box[0] if isinstance(box, (list, tuple)) and len(box) else None
                )

            if target_backbone is None:
                return None
            backbone = _module_to_cpu_eval(target_backbone)
        else:
            # Original IGC and SFTR+Naive generate pseudo-targets from the
            # CURRENT live history representation. SFTR differs from IGC in
            # whether the downstream teacher head is frozen, not in this path.
            backbone = current_live_backbone

        # For every condition, the pseudo-target supervising q_j is produced by
        # the downstream teacher operator q_{j+1}. In FrozenRep this head remains
        # live through its auxiliary head-only losses, so its movement is exactly
        # the operator drift we intend to measure.
        head = _module_to_cpu_eval(model.G_comp_heads[j + 1])

        return GeneratorSnapshot(backbone=backbone, head=head)

    def _resolve_eval_device(self, model: nn.Module) -> torch.device:
        """Resolve the device used only for mechanism forward evaluation.

        Snapshots remain stored on CPU between monitor events.  During an
        observation they are temporarily moved to this device, evaluated under
        torch.inference_mode(), then moved back to CPU.
        """
        if self.eval_device == "cpu":
            return torch.device("cpu")

        model_device = getattr(model, "device", None)
        try:
            model_device = torch.device(model_device)
        except Exception:
            model_device = None

        if self.eval_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "mechanism.eval_device='cuda' requested but CUDA is unavailable."
                )
            if model_device is not None and model_device.type == "cuda":
                return model_device
            return torch.device("cuda", torch.cuda.current_device())

        if self.eval_device == "model":
            return model_device if model_device is not None else torch.device("cpu")

        # auto: prefer the model's CUDA device; otherwise CPU.
        if model_device is not None and model_device.type == "cuda":
            return model_device
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    @staticmethod
    def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                out[key] = value.to(device=device, non_blocking=(device.type == "cuda"))
            else:
                out[key] = value
        return out

    def _evaluate_target_generator(
        self,
        model: nn.Module,
        generator: GeneratorSnapshot,
        j: int,
    ) -> Tuple[torch.Tensor, bool]:
        values: List[torch.Tensor] = []
        integrity_pass = True
        K = int(model.projection_horizon)
        device = self._resolve_eval_device(model)

        backbone = generator.backbone.to(device).eval()
        head = generator.head.to(device).eval()

        try:
            with torch.inference_mode():
                for batch_cpu in self.probe_batches:
                    batch = self._batch_to_device(batch_cpu, device)
                    ct = self._clean_tensor(batch["current_treatments"])
                    pt = self._clean_tensor(batch["prev_treatments"])
                    vitals = (
                        batch.get("vitals")
                        if bool(getattr(model, "has_vitals", False))
                        else None
                    )
                    prev_outputs = batch["prev_outputs"]
                    static_features = batch["static_features"]
                    active_entries = batch["active_entries"]

                    B, T, _ = pt.shape
                    if T - K - 1 <= 0:
                        continue
                    policy = self._policy_batch(model, B, ct.dtype, device=device)

                    for t in range(1, T - K):
                        current_active = active_entries.clone()
                        current_active[:, int(t + K):] = 0.0
                        terminal_mask = current_active[:, t + K - 1, :].squeeze(-1) > 0
                        if not terminal_mask.any():
                            continue

                        ct_cf, pt_cf = self._build_cf_history(ct, pt, t, K, policy)
                        actual_window = ct_cf[:, t - 1:t + K, :]
                        integrity_pass = integrity_pass and bool(
                            torch.allclose(actual_window, policy, atol=1e-6, rtol=0.0)
                        )

                        hr_cf = backbone(
                            pt_cf, vitals, prev_outputs, static_features, current_active
                        )
                        target_time = t + j
                        y = head.build_outcome(hr_cf, ct_cf)[:, target_time, :]
                        values.append(y[terminal_mask].detach().float().cpu())
        finally:
            # Keep snapshots in CPU memory between observation events.
            backbone.cpu()
            head.cpu()

        if not values:
            return (
                torch.empty(
                    (0, int(getattr(model, "dim_outcome", 1))),
                    dtype=torch.float32,
                ),
                integrity_pass,
            )
        return torch.cat(values, dim=0), integrity_pass

    def _evaluate_student(
        self,
        model: nn.Module,
        live_backbone: nn.Module,
        student_head: nn.Module,
        j: int,
    ) -> torch.Tensor:
        values: List[torch.Tensor] = []
        K = int(model.projection_horizon)
        device = self._resolve_eval_device(model)

        backbone = live_backbone.to(device).eval()
        head = student_head.to(device).eval()

        try:
            with torch.inference_mode():
                for batch_cpu in self.probe_batches:
                    batch = self._batch_to_device(batch_cpu, device)
                    ct = self._clean_tensor(batch["current_treatments"])
                    pt = self._clean_tensor(batch["prev_treatments"])
                    vitals = (
                        batch.get("vitals")
                        if bool(getattr(model, "has_vitals", False))
                        else None
                    )
                    prev_outputs = batch["prev_outputs"]
                    static_features = batch["static_features"]
                    active_entries = batch["active_entries"]

                    B, T, _ = pt.shape
                    if T - K - 1 <= 0:
                        continue

                    for t in range(1, T - K):
                        current_active = active_entries.clone()
                        current_active[:, int(t + K):] = 0.0
                        terminal_mask = current_active[:, t + K - 1, :].squeeze(-1) > 0
                        if not terminal_mask.any():
                            continue

                        hr = backbone(
                            pt, vitals, prev_outputs, static_features, current_active
                        )
                        current_time = t + j - 1
                        pred = head.build_outcome(hr, ct)[:, current_time, :]
                        values.append(pred[terminal_mask].detach().float().cpu())
        finally:
            backbone.cpu()
            head.cpu()

        if not values:
            return torch.empty(
                (0, int(getattr(model, "dim_outcome", 1))), dtype=torch.float32
            )
        return torch.cat(values, dim=0)

    @staticmethod
    def _clean_tensor(x: torch.Tensor) -> torch.Tensor:
        x = x.detach().clone()
        return torch.where(x.abs() < 1e-6, torch.zeros_like(x), x)

    @staticmethod
    def _policy_batch(
        model: nn.Module,
        batch_size: int,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        seq = getattr(model, "treatment_sequence", None)
        if seq is None:
            raise RuntimeError("Mechanism monitoring requires model.treatment_sequence.")
        seq = torch.as_tensor(seq, dtype=dtype, device=device)
        expected = int(model.projection_horizon) + 1
        if seq.ndim != 2 or int(seq.shape[0]) != expected:
            raise ValueError(
                f"treatment_sequence must have shape (K+1,D)=({expected},D), got {tuple(seq.shape)}"
            )
        return seq.unsqueeze(0).expand(int(batch_size), -1, -1).clone()

    @staticmethod
    def _build_cf_history(
        ct: torch.Tensor,
        pt: torch.Tensor,
        t: int,
        K: int,
        policy: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        start = int(t) - 1
        stop = int(t) + int(K)
        if start < 0 or stop > int(ct.shape[1]):
            raise RuntimeError(f"Invalid CF replacement window [{start},{stop}) for T={ct.shape[1]}.")
        ct_cf = ct.clone()
        ct_cf[:, start:stop, :] = policy
        pt_cf = torch.cat((pt[:, :1, :], ct_cf[:, :-1, :]), dim=1)
        return ct_cf, pt_cf

    # ------------------------------------------------------------------
    # Logging and summaries
    # ------------------------------------------------------------------

    def _base_row(self, model: nn.Module, j: int, stage: int, epoch: int, step: int) -> Dict[str, Any]:
        head_lr, backbone_lr = self._get_lrs(model)
        modules = []
        if hasattr(model, "_backbone_modules"):
            try:
                modules = list(model._backbone_modules())
            except Exception:
                modules = []
        if not modules:
            modules = [
                getattr(model, name, None)
                for name in (
                    "treatments_input_transformation",
                    "vitals_input_transformation",
                    "outputs_input_transformation",
                    "static_input_transformation",
                    "transformer_blocks",
                    "hr_output_transformation",
                    "output_dropout",
                )
            ]
            modules = [m for m in modules if isinstance(m, nn.Module)]
        train_fraction = (
            float(np.mean([float(bool(m.training)) for m in modules])) if modules else float("nan")
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        return {
            "run_id": self.run_id,
            "model": self.model_label,
            "mechanism_kind": self.mechanism_kind,
            "strategy": self.strategy,
            "seed": self.seed,
            "horizon": int(getattr(model, "projection_horizon", -1)),
            "stage": int(stage),
            "q_j": int(j),
            "epoch": int(epoch),
            "global_step": int(step),
            "monitor_idx": int(self._monitor_idx),
            "student_backbone_train_mode_fraction": train_fraction,
            "trainable_params": int(trainable),
            "total_params": int(total),
            "head_lr": head_lr,
            "backbone_lr": backbone_lr,
        }

    @staticmethod
    def _nan_drift_fields(stage_start: int) -> Dict[str, Any]:
        return {
            "stage_start": int(stage_start),
            "D_op_mae": float("nan"),
            "D_rep_mae": float("nan"),
            "D_total_mae": float("nan"),
            "D_op_rmse": float("nan"),
            "D_rep_rmse": float("nan"),
            "D_total_rmse": float("nan"),
            "signed_op_mean": float("nan"),
            "signed_rep_mean": float("nan"),
            "signed_total_mean": float("nan"),
            "decomp_residual_mae": float("nan"),
            "decomp_residual_max": float("nan"),
            "T00_mean": float("nan"),
            "T10_mean": float("nan"),
            "T01_mean": float("nan"),
            "T11_mean": float("nan"),
            "D_op_sym_mae": float("nan"),
            "D_rep_sym_mae": float("nan"),
            "D_op_sym_rmse": float("nan"),
            "D_rep_sym_rmse": float("nan"),
            "signed_op_sym_mean": float("nan"),
            "signed_rep_sym_mean": float("nan"),
            "sym_decomp_residual_mae": float("nan"),
            "sym_decomp_residual_max": float("nan"),
        }

    def _get_lrs(self, model: nn.Module) -> Tuple[float, float]:
        head_lr = float("nan")
        backbone_lr = float("nan")
        trainer = getattr(model, "trainer", None)
        optimizers = getattr(trainer, "optimizers", []) if trainer is not None else []
        if not optimizers:
            return head_lr, backbone_lr
        opt = optimizers[0]
        for group in opt.param_groups:
            lr = _safe_float(group.get("lr", float("nan")))
            name = group.get("group_name", "other")
            if name == "history_backbone":
                backbone_lr = lr
            else:
                head_lr = lr
        if math.isnan(backbone_lr):
            backbone_lr = head_lr
        return head_lr, backbone_lr

    @staticmethod
    def _append_rows(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists() and path.stat().st_size > 0
        df = pd.DataFrame(rows)
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[columns]
        df.to_csv(path, mode="a", header=not exists, index=False)

    def record_performance(self, split: str, rmse: float, **metrics: float) -> None:
        if not self.enabled:
            return
        row = {
            "run_id": self.run_id,
            "model": self.model_label,
            "mechanism_kind": self.mechanism_kind,
            "strategy": self.strategy,
            "seed": self.seed,
            "split": str(split),
            "rmse": _safe_float(rmse),
        }
        row.update({str(k): _safe_float(v) for k, v in metrics.items()})
        self._append_rows(self.performance_path, [row], list(row.keys()))
        self._write_summary()

    def finalize(self) -> None:
        if not self.enabled:
            return
        self._write_summary()
        self._write_metadata(initial=False)

    def _write_summary(self) -> None:
        if not self.trace_path.exists():
            return
        try:
            trace = pd.read_csv(self.trace_path)
        except Exception:
            return
        if trace.empty:
            return

        drift = trace[trace["D_total_mae"].notna()].copy()
        if drift.empty:
            return

        group_cols = ["run_id", "model", "mechanism_kind", "strategy", "seed", "horizon", "q_j"]
        rows: List[Dict[str, Any]] = []
        for key, g in drift.groupby(group_cols, dropna=False):
            row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
            row.update(
                {
                    "V_op": float(g["D_op_mae"].sum()),
                    "V_rep": float(g["D_rep_mae"].sum()),
                    "V_total": float(g["D_total_mae"].sum()),
                    "mean_D_op": float(g["D_op_mae"].mean()),
                    "mean_D_rep": float(g["D_rep_mae"].mean()),
                    "mean_D_total": float(g["D_total_mae"].mean()),
                    "max_D_total": float(g["D_total_mae"].max()),
                    "mean_D_op_sym": float(g["D_op_sym_mae"].mean())
                    if "D_op_sym_mae" in g else float("nan"),
                    "mean_D_rep_sym": float(g["D_rep_sym_mae"].mean())
                    if "D_rep_sym_mae" in g else float("nan"),
                    "mean_student_target_rmse": float(g["student_target_rmse"].mean()),
                    "max_decomp_residual": float(g["decomp_residual_mae"].max()),
                    "max_sym_decomp_residual": float(g["sym_decomp_residual_mae"].max())
                    if "sym_decomp_residual_mae" in g else float("nan"),
                    "max_teacher_head_delta_rms": float(g["teacher_head_param_delta_rms"].max()),
                    "max_teacher_repr_delta_rms": float(g["teacher_repr_param_delta_rms"].max()),
                    "mean_student_repr_delta_rms": float(g["student_repr_param_delta_rms"].mean()),
                    "cf_integrity_pass": int(bool((g["cf_integrity_pass"].fillna(0) > 0).all())),
                    "n_intervals": int(len(g)),
                    "last_global_step": int(g["global_step"].max()),
                }
            )
            rows.append(row)
        summary = pd.DataFrame(rows)
        summary.to_csv(self.summary_path, index=False)

    def _write_metadata(self, initial: bool) -> None:
        payload = {
            "run_id": self.run_id,
            "model": self.model_label,
            "mechanism_kind": self.mechanism_kind,
            "strategy": self.strategy,
            "seed": self.seed,
            "every_n_steps": self.every_n_steps,
            "probe_size": self.probe_size,
            "probe_count": self.probe_count,
            "probe_ready": self.probe_ready,
            "probe_cache_path": str(self.probe_cache_path) if self.probe_cache_path else None,
            "trace_path": str(self.trace_path),
            "performance_path": str(self.performance_path),
            "summary_path": str(self.summary_path),
            "decomposition": {
                "ordered": "Y00->Y10->Y11",
                "four_corners": ["Y00", "Y10", "Y01", "Y11"],
                "symmetric_two_factor": True,
            },
            "initial": bool(initial),
        }
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _is_global_zero(model: nn.Module) -> bool:
        trainer = getattr(model, "trainer", None)
        if trainer is None:
            return True
        return bool(getattr(trainer, "is_global_zero", True))
