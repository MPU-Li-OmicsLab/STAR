import pandas as pd
import numpy as np
import logging
import hashlib
import json
from scipy.stats import beta as beta_dist
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
from typing import List, Optional, Sequence
from sklearn.model_selection import train_test_split
import itertools

from src import ROOT_PATH
from build_h5_fio22 import TREATMENT_LIST, OUTCOME_LIST, VITAL_LIST, STATIC_LIST
from src.data.mimic_iv.load_data import (
    load_mimic4_data_raw,
    fit_static_preprocessor,
    apply_static_preprocessor,
)
from src.data.mimic_iv.utils import sigmoid, SplineTrendsMixture
from src.data.mimic_iv.real_dataset import MIMIC4RealDataset
from src.data.dataset_collection import SyntheticDatasetCollection
from utils.dose_trajectory_generator import DoseTrajectoryGenerator

logger = logging.getLogger(__name__)

LEGACY_MODE = 'legacy_sigmoid_beta'
COPULA_MODE = 'controlled_copula'
LEGACY_CF = 'legacy_clinical'
FIXED_CF = 'fixed_benchmark'

BETA_DOSE_MARGINAL = 'beta'
DISEASE_REAL_DOSE_MARGINAL = 'disease_real_hist'

DOSE_SCALE: float = 1.0


def set_dose_scale(scale: float):
    global DOSE_SCALE
    DOSE_SCALE = float(scale)
    logger.info(f'[dose-norm] DOSE_SCALE set to {DOSE_SCALE:.4f}')


def normalize_dose(raw_dose: float) -> float:
    return float(np.clip(raw_dose / max(DOSE_SCALE, 1e-8), 0.0, 1.0))


def denormalize_dose(norm_dose: float) -> float:
    return float(norm_dose) * DOSE_SCALE


class SyntheticOutcomeGenerator:

    def __init__(self,
                 exogeneous_vars: List[str],
                 exog_dependency: callable,
                 exog_weight: float,
                 endo_dependency: callable,
                 endo_rand_weight: float,
                 endo_spline_weight: float,
                 outcome_name: str):
        self.exogeneous_vars = exogeneous_vars
        self.exog_dependency = exog_dependency
        self.exog_weight = exog_weight
        self.endo_rand_weight = endo_rand_weight
        self.endo_spline_weight = endo_spline_weight
        self.endo_dependency = endo_dependency
        self.outcome_name = outcome_name

    def simulate_untreated(self, all_vitals: pd.DataFrame, static_features: pd.DataFrame):
        logger.info(f'Simulating untreated outcome {self.outcome_name}')
        user_sizes = all_vitals.groupby(level='subject_id', sort=False).size()

        all_vitals[f'{self.outcome_name}_exog'] = self.exog_weight * self.exog_dependency(
            all_vitals[self.exogeneous_vars].values
        )

        time_range = np.arange(0, user_sizes.max())
        y_endo_rand = self.endo_dependency(time_range, len(user_sizes))
        y_endo_splines = SplineTrendsMixture(
            n_patients=len(user_sizes), max_time=user_sizes.max()
        )(time_range)
        y_endo_full = self.endo_rand_weight * y_endo_rand + self.endo_spline_weight * y_endo_splines

        all_vitals[f'{self.outcome_name}_endo'] = np.array([
            value for (i, l) in enumerate(user_sizes) for value in y_endo_full[i, :l]
        ]).reshape(-1, 1)

        all_vitals[f'{self.outcome_name}_untreated'] = (
            all_vitals[f'{self.outcome_name}_exog'] + all_vitals[f'{self.outcome_name}_endo']
        )

        all_vitals[f'{self.outcome_name}'] = all_vitals[f'{self.outcome_name}_untreated'].copy()


class SyntheticTreatment:
    def __init__(self,
                 confounding_vars: List[str],
                 confounder_outcomes: List[str],
                 confounding_dependency: callable,
                 window: float,
                 conf_outcome_weight: float,
                 conf_vars_weight: float,
                 bias: float,
                 full_effect: float,
                 effect_window: float,
                 treatment_name: str,
                 post_nonlinearity: callable = None,
                 dose_concentration: float = 4.0,
                 dose_eps: float = 1e-3,
                 dose_bias: float = None,
                 dose_outcome_weight: float = None,
                 dose_vars_weight: float = None,
                 dose_response_shape: str = 'saturating',
                 dr_params: dict = None,
                 benchmark_response_mode: str = 'linear',
                 treatment_assignment_mode: str = COPULA_MODE,
                 target_treat_rate: float = 0.50,
                 positive_dose_mean: float = 0.45,
                 treatment_rho: float = 0.0,
                 dose_rho: float = 0.0,
                 history_rank_eps: float = 1e-4,
                 dose_marginal_conditional: bool = True,
                 history_outcome_source: str = 'untreated',
                 dose_marginal_mode: str = DISEASE_REAL_DOSE_MARGINAL,
                 inherit_real_treat_rate: bool = True,
                 real_marginal_max_bins: int = 160,
                 real_marginal_smooth_sigma_bins: float = 1.0,
                 real_marginal_pseudocount: float = 1e-6,
                 real_marginal_zero_eps: float = 1e-12,
                 real_marginal_min_positive_n: int = 50):
        assert treatment_assignment_mode in (LEGACY_MODE, COPULA_MODE)
        assert history_outcome_source in ('untreated', 'observed_feedback')
        assert 0.0 <= treatment_rho <= 0.95
        assert 0.0 <= dose_rho <= 0.95
        assert 0.0 < target_treat_rate < 1.0
        assert 0.0 < positive_dose_mean < 1.0
        assert dose_concentration > 0.0
        if dose_marginal_mode not in (BETA_DOSE_MARGINAL, DISEASE_REAL_DOSE_MARGINAL):
            raise ValueError(
                'dose_marginal_mode must be one of '
                f'{{{BETA_DOSE_MARGINAL!r}, {DISEASE_REAL_DOSE_MARGINAL!r}}}; '
                f'got {dose_marginal_mode!r}.'
            )
        if int(real_marginal_max_bins) < 8:
            raise ValueError('real_marginal_max_bins must be >= 8.')
        if float(real_marginal_smooth_sigma_bins) < 0.0:
            raise ValueError('real_marginal_smooth_sigma_bins must be >= 0.')
        if float(real_marginal_pseudocount) < 0.0:
            raise ValueError('real_marginal_pseudocount must be >= 0.')
        if float(real_marginal_zero_eps) < 0.0:
            raise ValueError('real_marginal_zero_eps must be >= 0.')
        if int(real_marginal_min_positive_n) < 10:
            raise ValueError('real_marginal_min_positive_n must be >= 10.')

        self.confounding_vars = confounding_vars
        self.confounder_outcomes = confounder_outcomes
        self.confounding_dependency = confounding_dependency
        self.treatment_name = treatment_name
        self.post_nonlinearity = post_nonlinearity

        self.window = window
        self.conf_outcome_weight = conf_outcome_weight
        self.conf_vars_weight = conf_vars_weight
        self.bias = bias

        self.full_effect = full_effect
        self.effect_window = effect_window

        self.dose_concentration = float(dose_concentration)
        self.dose_eps = float(dose_eps)

        self.dose_bias = float(dose_bias) if dose_bias is not None else float(bias)
        self.dose_outcome_weight = (
            float(dose_outcome_weight) if dose_outcome_weight is not None else float(conf_outcome_weight)
        )
        self.dose_vars_weight = (
            float(dose_vars_weight) if dose_vars_weight is not None else float(conf_vars_weight)
        )

        self.benchmark_response_mode = benchmark_response_mode
        if benchmark_response_mode == 'linear':
            if dose_response_shape != 'linear':
                logger.warning(
                    f'[benchmark] dose_response_shape={dose_response_shape} overridden '
                    'to "linear" (benchmark_response_mode="linear").'
                )
            dose_response_shape = 'linear'
        self.dose_response_shape = dose_response_shape
        self.dr_params = dr_params or {}

        self.saturating_heterogeneity_mode = str(self.dr_params.get('heterogeneity_mode', 'none'))
        self.saturating_heterogeneity_source = str(
            self.dr_params.get('heterogeneity_source', 'baseline_history_score')
        )
        self.saturating_heterogeneity_strength = float(
            self.dr_params.get('heterogeneity_strength', 0.0)
        )
        self.saturating_heterogeneity_z_clip = float(self.dr_params.get('heterogeneity_z_clip', 2.5))
        self.saturating_ed50_min = float(self.dr_params.get('ed50_min', 0.05))
        self.saturating_ed50_max = float(self.dr_params.get('ed50_max', 1.50))

        if self.saturating_heterogeneity_mode not in ('none', 'ed50_loglinear'):
            raise ValueError(
                'dr_params.heterogeneity_mode must be one of {"none", "ed50_loglinear"}; '
                f'got {self.saturating_heterogeneity_mode!r}.'
            )
        if self.saturating_heterogeneity_source != 'baseline_history_score':
            raise ValueError(
                'Saturating-v2 currently requires '
                'dr_params.heterogeneity_source="baseline_history_score"; got '
                f'{self.saturating_heterogeneity_source!r}.'
            )
        if self.saturating_heterogeneity_z_clip <= 0:
            raise ValueError('dr_params.heterogeneity_z_clip must be > 0.')
        if not (0.0 < self.saturating_ed50_min < self.saturating_ed50_max):
            raise ValueError('Need 0 < dr_params.ed50_min < dr_params.ed50_max.')

        if self.saturating_heterogeneity_mode == 'ed50_loglinear':
            if self.dose_response_shape != 'saturating':
                logger.warning(
                    '[saturating-v2] heterogeneity_mode=ed50_loglinear is set but '
                    'dose_response_shape=%s; modifier will not affect non-saturating shapes.',
                    self.dose_response_shape,
                )
            if self.saturating_heterogeneity_strength <= 0:
                raise ValueError(
                    'dr_params.heterogeneity_strength must be > 0 when '
                    'heterogeneity_mode="ed50_loglinear".'
                )
            logger.info(
                '[saturating] %s: mode=%s source=%s eta=%.4f z_clip=%.3f '
                'ed50_base=%.4f ed50_bounds=(%.4f, %.4f)',
                self.treatment_name, self.saturating_heterogeneity_mode,
                self.saturating_heterogeneity_source, self.saturating_heterogeneity_strength,
                self.saturating_heterogeneity_z_clip, float(self.dr_params.get('ed50', 0.3)),
                self.saturating_ed50_min, self.saturating_ed50_max,
            )

            _e0 = float(self.dr_params.get('ed50', 0.3))
            _eta = float(self.saturating_heterogeneity_strength)
            _zc = float(self.saturating_heterogeneity_z_clip)
            _raw_lo = _e0 * np.exp(-_eta * _zc)
            _raw_hi = _e0 * np.exp(+_eta * _zc)
            self.saturating_ed50_raw_min_theoretical = float(_raw_lo)
            self.saturating_ed50_raw_max_theoretical = float(_raw_hi)
            self.saturating_ed50_clipping_possible = bool(
                (_raw_lo < self.saturating_ed50_min) or (_raw_hi > self.saturating_ed50_max)
            )
            logger.info(
                '[saturating] theoretical raw ED50 range after z-clip: [%.6f, %.6f]; '
                'configured bounds=[%.6f, %.6f]; clipping_possible=%s',
                _raw_lo, _raw_hi,
                self.saturating_ed50_min, self.saturating_ed50_max,
                self.saturating_ed50_clipping_possible,
            )
            if self.saturating_ed50_clipping_possible:
                logger.warning(
                    '[saturating] ED50 clipping is theoretically possible. For the main '
                    'benchmark prefer eta small enough that the raw ED50 range remains inside '
                    'the configured bounds (typically heterogeneity_strength <= 0.6 for the '
                    'current benchmark).'
                )

        if not hasattr(self, 'saturating_ed50_raw_min_theoretical'):
            _e0 = float(self.dr_params.get('ed50', 0.3))
            self.saturating_ed50_raw_min_theoretical = _e0
            self.saturating_ed50_raw_max_theoretical = _e0
            self.saturating_ed50_clipping_possible = False

        self.treatment_assignment_mode = treatment_assignment_mode
        self.target_treat_rate = float(target_treat_rate)
        self.positive_dose_mean = float(positive_dose_mean)
        self.treatment_rho = float(treatment_rho)
        self.dose_rho = float(dose_rho)
        self.history_rank_eps = float(history_rank_eps)
        self.dose_marginal_conditional = bool(dose_marginal_conditional)
        self.history_outcome_source = history_outcome_source

        self.dose_marginal_mode = str(dose_marginal_mode)
        self.inherit_real_treat_rate = bool(inherit_real_treat_rate)
        self.real_marginal_max_bins = int(real_marginal_max_bins)
        self.real_marginal_smooth_sigma_bins = float(real_marginal_smooth_sigma_bins)
        self.real_marginal_pseudocount = float(real_marginal_pseudocount)
        self.real_marginal_zero_eps = float(real_marginal_zero_eps)
        self.real_marginal_min_positive_n = int(real_marginal_min_positive_n)

        self.real_marginal_edges = None
        self.real_marginal_bin_probs = None
        self.real_marginal_cdf_edges = None
        self.real_marginal_meta = {}

        self._treat_threshold = float(norm.ppf(1.0 - self.target_treat_rate))
        self.history_score_sorted = None
        self.treated_z_sorted = None
        self.treated_cdf = None
        self.calibration_meta = {}

    def positive_dose_params(self):
        a0 = self.dose_concentration * self.positive_dose_mean
        b0 = self.dose_concentration * (1.0 - self.positive_dose_mean)
        return float(a0), float(b0)

    def fit_disease_real_dose_marginal(self, normalized_real_doses: np.ndarray):

        x = np.asarray(normalized_real_doses, dtype=float).reshape(-1)
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise RuntimeError(
                f'{self.treatment_name}: no finite real TRAIN doses available '
                'for disease-specific marginal fitting.'
            )

        x = np.clip(x, 0.0, 1.0)
        zero_eps = self.real_marginal_zero_eps
        real_positive = x > zero_eps
        real_treat_rate = float(np.mean(real_positive))

        if self.inherit_real_treat_rate:
            clipped_rate = float(np.clip(real_treat_rate, 1e-4, 1.0 - 1e-4))
            if abs(clipped_rate - real_treat_rate) > 1e-12:
                logger.warning(
                    '[disease-marginal] %s real treatment rate %.8f clipped to %.8f.',
                    self.treatment_name, real_treat_rate, clipped_rate,
                )
            self.target_treat_rate = clipped_rate
            self._treat_threshold = float(norm.ppf(1.0 - self.target_treat_rate))

        pos_raw = x[real_positive]
        if pos_raw.size < self.real_marginal_min_positive_n:
            raise RuntimeError(
                f'{self.treatment_name}: only {pos_raw.size} positive real TRAIN doses; '
                f'need >= {self.real_marginal_min_positive_n}.'
            )

        pos = np.clip(pos_raw, self.dose_eps, 1.0 - self.dose_eps)
        lower_clip_frac = float(np.mean(pos_raw < self.dose_eps))
        upper_clip_frac = float(np.mean(pos_raw > 1.0 - self.dose_eps))

        q = np.quantile(pos, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

        if self.dose_marginal_mode == DISEASE_REAL_DOSE_MARGINAL:
            lo = float(np.min(pos))
            hi = float(np.max(pos))
            if not np.isfinite(lo + hi) or hi <= lo + 1e-12:
                raise RuntimeError(
                    f'{self.treatment_name}: positive real TRAIN doses have '
                    f'degenerate support [{lo}, {hi}].'
                )

            n_bins = int(min(
                self.real_marginal_max_bins,
                max(32, int(np.ceil(np.sqrt(pos.size)))),
            ))
            edges = np.linspace(lo, hi, n_bins + 1, dtype=float)
            counts, _ = np.histogram(pos, bins=edges)
            counts = counts.astype(float)

            if self.real_marginal_smooth_sigma_bins > 0.0:
                counts = gaussian_filter1d(
                    counts, sigma=self.real_marginal_smooth_sigma_bins, mode='reflect',
                )

            if self.real_marginal_pseudocount > 0.0:
                counts = counts + self.real_marginal_pseudocount

            total = float(np.sum(counts))
            if not np.isfinite(total) or total <= 0.0:
                raise RuntimeError(f'{self.treatment_name}: invalid fitted real marginal mass.')
            probs = counts / total
            cdf_edges = np.concatenate([[0.0], np.cumsum(probs)])
            cdf_edges[-1] = 1.0

            self.real_marginal_edges = edges
            self.real_marginal_bin_probs = probs
            self.real_marginal_cdf_edges = cdf_edges

        self.real_marginal_meta = {
            'n_real_valid': int(x.size),
            'n_real_positive': int(pos_raw.size),
            'real_zero_fraction': float(1.0 - real_treat_rate),
            'real_positive_fraction': real_treat_rate,
            'inherited_target_treat_rate': float(self.target_treat_rate),
            'dose_marginal_mode': self.dose_marginal_mode,
            'positive_mean': float(np.mean(pos)),
            'positive_std': float(np.std(pos)),
            'positive_min': float(np.min(pos)),
            'positive_max': float(np.max(pos)),
            'positive_p01': float(q[0]),
            'positive_p05': float(q[1]),
            'positive_p10': float(q[2]),
            'positive_p25': float(q[3]),
            'positive_p50': float(q[4]),
            'positive_p75': float(q[5]),
            'positive_p90': float(q[6]),
            'positive_p95': float(q[7]),
            'positive_p99': float(q[8]),
            'positive_lower_clip_fraction': lower_clip_frac,
            'positive_upper_clip_fraction': upper_clip_frac,
            'n_bins': (
                int(len(self.real_marginal_bin_probs))
                if self.real_marginal_bin_probs is not None else None
            ),
            'smooth_sigma_bins': self.real_marginal_smooth_sigma_bins,
            'pseudocount': self.real_marginal_pseudocount,
        }

        logger.info(
            '[disease-marginal] %s fitted on current disease TRAIN: mode=%s, '
            'P(A>0)=%.6f, n_pos=%d, positive P05/P50/P95=(%.6f, %.6f, %.6f), '
            'support=[%.6f, %.6f], bins=%s.',
            self.treatment_name, self.dose_marginal_mode,
            self.target_treat_rate, pos_raw.size, q[1], q[4], q[7],
            float(np.min(pos)), float(np.max(pos)),
            self.real_marginal_meta['n_bins'],
        )

    def _require_real_marginal(self):
        if self.dose_marginal_mode != DISEASE_REAL_DOSE_MARGINAL:
            return
        if (self.real_marginal_edges is None
                or self.real_marginal_bin_probs is None
                or self.real_marginal_cdf_edges is None):
            raise RuntimeError(
                f'{self.treatment_name}: disease-specific real dose marginal was not fitted. '
                'Collection must call fit_disease_real_dose_marginal() on the current disease '
                'TRAIN split before simulation.'
            )

    def positive_marginal_ppf(self, u):
        u_arr = np.asarray(u, dtype=float)
        u_clip = np.clip(u_arr, self.dose_eps, 1.0 - self.dose_eps)

        if self.dose_marginal_mode == BETA_DOSE_MARGINAL:
            a0, b0 = self.positive_dose_params()
            ans = beta_dist.ppf(u_clip, a0, b0)
        else:
            self._require_real_marginal()
            cdf = self.real_marginal_cdf_edges
            probs = self.real_marginal_bin_probs
            edges = self.real_marginal_edges
            idx = np.searchsorted(cdf, u_clip, side='right') - 1
            idx = np.clip(idx, 0, len(probs) - 1)
            left_cdf = cdf[idx]
            mass = np.maximum(probs[idx], 1e-15)
            frac = np.clip((u_clip - left_cdf) / mass, 0.0, 1.0)
            ans = edges[idx] + frac * (edges[idx + 1] - edges[idx])

        ans = np.clip(ans, self.dose_eps, 1.0 - self.dose_eps)
        return float(ans) if np.ndim(u_arr) == 0 else np.asarray(ans, dtype=float)

    def positive_marginal_cdf(self, d):
        d_arr = np.asarray(d, dtype=float)

        if self.dose_marginal_mode == BETA_DOSE_MARGINAL:
            a0, b0 = self.positive_dose_params()
            x = np.clip(d_arr, self.dose_eps, 1.0 - self.dose_eps)
            ans = beta_dist.cdf(x, a0, b0)
        else:
            self._require_real_marginal()
            edges = self.real_marginal_edges
            probs = self.real_marginal_bin_probs
            cdf = self.real_marginal_cdf_edges
            flat = d_arr.reshape(-1)
            out = np.empty_like(flat, dtype=float)
            for k, val in enumerate(flat):
                if val <= edges[0]:
                    out[k] = 0.0
                elif val >= edges[-1]:
                    out[k] = 1.0
                else:
                    j = int(np.searchsorted(edges, val, side='right') - 1)
                    j = min(max(j, 0), len(probs) - 1)
                    width = edges[j + 1] - edges[j]
                    frac = (val - edges[j]) / max(width, 1e-15)
                    out[k] = cdf[j] + probs[j] * frac
            ans = out.reshape(d_arr.shape)

        ans = np.clip(ans, self.dose_eps, 1.0 - self.dose_eps)
        return float(ans) if np.ndim(d_arr) == 0 else np.asarray(ans, dtype=float)

    def positive_marginal_pdf(self, d):
        d_arr = np.asarray(d, dtype=float)

        if self.dose_marginal_mode == BETA_DOSE_MARGINAL:
            a0, b0 = self.positive_dose_params()
            x = np.clip(d_arr, self.dose_eps, 1.0 - self.dose_eps)
            ans = beta_dist.pdf(x, a0, b0)
        else:
            self._require_real_marginal()
            edges = self.real_marginal_edges
            probs = self.real_marginal_bin_probs
            flat = d_arr.reshape(-1)
            out = np.zeros_like(flat, dtype=float)
            for k, val in enumerate(flat):
                if edges[0] <= val <= edges[-1]:
                    j = int(np.searchsorted(edges, val, side='right') - 1)
                    j = min(max(j, 0), len(probs) - 1)
                    width = edges[j + 1] - edges[j]
                    out[k] = probs[j] / max(width, 1e-15)
            ans = out.reshape(d_arr.shape)

        return float(ans) if np.ndim(d_arr) == 0 else np.asarray(ans, dtype=float)

    def _eval_dependency(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        try:
            f = np.asarray(self.confounding_dependency(X), dtype=float).reshape(-1)
            if f.shape[0] == X.shape[0]:
                return f
        except Exception:
            pass
        return np.array([
            float(np.asarray(self.confounding_dependency(row.reshape(1, -1))).flatten()[0])
            for row in X
        ])

    def _score_outcome_cols(self, columns, strict: bool = False):
        if not (self.treatment_assignment_mode == COPULA_MODE
                and self.history_outcome_source == 'untreated'):
            return list(self.confounder_outcomes)
        cols = []
        for c in self.confounder_outcomes:
            u = f'{c}_untreated'
            if u in columns:
                cols.append(u)
            elif strict:
                raise RuntimeError(f'history_outcome_source="untreated" needs column {u}')
            else:
                cols.append(c)
        return cols

    def raw_history_score(self, patient_df, t) -> float:
        t_start = max(0, t - int(self.window))
        agr_range = np.arange(t_start, t + 1)
        cols = self._score_outcome_cols(patient_df.columns)
        avg_y = patient_df.loc[agr_range, cols].values.mean()
        x = patient_df.loc[t, self.confounding_vars].values.reshape(1, -1)
        f_x = float(np.asarray(self.confounding_dependency(x)).flatten()[0])
        return float(self.conf_outcome_weight * avg_y + self.conf_vars_weight * f_x)

    def fit_history_score_distribution(self, all_vitals_train: pd.DataFrame):
        cols = self._score_outcome_cols(all_vitals_train.columns, strict=True)
        missing = [c for c in cols if c not in all_vitals_train.columns]
        if missing:
            raise RuntimeError(f'calibration needs outcome columns, missing: {missing}')

        w = int(self.window) + 1
        scores = []
        for _, df in all_vitals_train.groupby(level='subject_id', sort=False):
            avg_y = df[cols].mean(axis=1).rolling(w, min_periods=1).mean().values
            f_x = self._eval_dependency(df[self.confounding_vars].values)
            scores.append(self.conf_outcome_weight * avg_y + self.conf_vars_weight * f_x)

        s = np.concatenate(scores)
        s = s[np.isfinite(s)]
        assert len(s) > 100, f'history score calibration sample too small: {len(s)}'
        self.history_score_sorted = np.sort(s)
        self._fit_treated_cdf()
        self.calibration_meta = {
            'n': int(len(s)),
            'source': f'train_{self.history_outcome_source}',
            'outcome_cols': cols,
            'score_mean': float(s.mean()),
            'score_std': float(s.std()),
            'implied_treat_rate': float(self._implied_treat_rate),
        }
        logger.info(
            f'[copula] {self.treatment_name}: TRAIN calibration fitted '
            f'(n={len(s)}, implied P(T=1)={self._implied_treat_rate:.4f}).'
        )

    def _fit_treated_cdf(self):
        n = len(self.history_score_sorted)
        ranks = np.clip(
            (np.arange(1, n + 1) + 0.5) / (n + 1.0),
            self.history_rank_eps, 1.0 - self.history_rank_eps,
        )
        z = norm.ppf(ranks)
        p = np.asarray(self._copula_treat_proba(z), dtype=float)
        self.treated_z_sorted = z
        self.treated_cdf = np.cumsum(p) / p.sum()
        self._implied_treat_rate = float(p.mean())

    def _history_normal_score(self, patient_df, t) -> float:
        if self.history_score_sorted is None:
            raise RuntimeError(
                f'{self.treatment_name}: history score CDF not fitted — '
                'call fit_history_score_distribution() on TRAIN data first.'
            )
        s = self.raw_history_score(patient_df, t)
        n = len(self.history_score_sorted)
        rank = (np.searchsorted(self.history_score_sorted, s, side='right') + 0.5) / (n + 1.0)
        rank = float(np.clip(rank, self.history_rank_eps, 1.0 - self.history_rank_eps))
        return float(norm.ppf(rank))

    def _copula_treat_proba(self, z):
        rho = self.treatment_rho
        sigma = np.sqrt(max(1.0 - rho ** 2, 1e-8))
        p = 1.0 - norm.cdf((self._treat_threshold - rho * np.asarray(z, dtype=float)) / sigma)
        return np.clip(p, 1e-6, 1.0 - 1e-6)

    def _treated_normal_score(self, z):
        if not self.dose_marginal_conditional:
            return float(z) if np.ndim(z) == 0 else np.asarray(z, dtype=float)
        if self.treated_cdf is None:
            raise RuntimeError(
                f'{self.treatment_name}: weighted treated-CDF not fitted — '
                'call fit_history_score_distribution() on TRAIN data first.'
            )
        k = np.searchsorted(self.treated_z_sorted, np.asarray(z, dtype=float), side='right')
        F = np.where(k > 0, self.treated_cdf[np.maximum(k - 1, 0)], 0.0)
        z_plus = norm.ppf(np.clip(F, self.history_rank_eps, 1.0 - self.history_rank_eps))
        return float(z_plus) if np.ndim(z) == 0 else z_plus

    def treatment_proba(self, patient_df, t):
        if self.treatment_assignment_mode == COPULA_MODE:
            return self._copula_treat_proba(self._history_normal_score(patient_df, t))
        t_start = max(0, t - self.window)
        agr_range = np.arange(t_start, t + 1)
        avg_y = patient_df.loc[agr_range, self.confounder_outcomes].values.mean()
        x = patient_df.loc[t, self.confounding_vars].values.reshape(1, -1)
        f_x = self.confounding_dependency(x)
        treat_proba = sigmoid(
            self.bias + self.conf_outcome_weight * avg_y + self.conf_vars_weight * f_x
        ).flatten()
        if self.post_nonlinearity is not None:
            treat_proba = self.post_nonlinearity(treat_proba)
        return treat_proba

    def dose_intensity_mean(self, patient_df, t):
        t_start = max(0, t - self.window)
        agr_range = np.arange(t_start, t + 1)
        avg_y = patient_df.loc[agr_range, self.confounder_outcomes].values.mean()
        x = patient_df.loc[t, self.confounding_vars].values.reshape(1, -1)
        f_x = self.confounding_dependency(x)
        return sigmoid(
            self.dose_bias + self.dose_outcome_weight * avg_y + self.dose_vars_weight * f_x
        ).flatten()

    def sample_dose(self, treat_proba, dose_mean, rng: np.random.RandomState, z=None) -> float:
        p = float(np.clip(np.asarray(treat_proba).flatten()[0], 0.0, 1.0))
        if int(rng.binomial(1, p)) == 0:
            return 0.0

        if self.treatment_assignment_mode == LEGACY_MODE:
            mu = float(np.clip(
                np.asarray(dose_mean).flatten()[0], self.dose_eps, 1.0 - self.dose_eps,
            ))
            return float(rng.beta(
                self.dose_concentration * mu, self.dose_concentration * (1.0 - mu),
            ))

        if z is None:
            raise ValueError('controlled_copula sample_dose requires the history score z.')
        z_plus = self._treated_normal_score(z)
        latent = self.dose_rho * z_plus + np.sqrt(max(1.0 - self.dose_rho ** 2, 1e-8)) * rng.normal()
        u = float(np.clip(norm.cdf(latent), self.dose_eps, 1.0 - self.dose_eps))
        return float(self.positive_marginal_ppf(u))

    def sample_treatment_dose(self, patient_df, t, rng: np.random.RandomState):
        if self.treatment_assignment_mode == LEGACY_MODE:
            p = self.treatment_proba(patient_df, t)
            mu = self.dose_intensity_mean(patient_df, t)
            return p, self.sample_dose(p, mu, rng)
        z = self._history_normal_score(patient_df, t)
        p = float(self._copula_treat_proba(z))
        return p, self.sample_dose(p, None, rng, z=z)

    def positive_dose_density(self, d: float, z: float) -> float:
        d = float(d)
        base_pdf = float(self.positive_marginal_pdf(d))
        if base_pdf <= 0.0 or not np.isfinite(base_pdf):
            return 0.0
        u = float(self.positive_marginal_cdf(d))
        l = float(norm.ppf(np.clip(u, self.dose_eps, 1.0 - self.dose_eps)))
        z_plus = self._treated_normal_score(z)
        sigma = np.sqrt(max(1.0 - self.dose_rho ** 2, 1e-8))
        copula_ratio = (
            norm.pdf((l - self.dose_rho * z_plus) / sigma)
            / (sigma * max(norm.pdf(l), 1e-12))
        )
        dens = float(copula_ratio) * base_pdf
        return float(dens) if np.isfinite(dens) else 0.0

    def _persistent_effect_modifier(self, patient_df) -> float:

        if self.saturating_heterogeneity_mode == 'none':
            return 0.0

        if self.saturating_heterogeneity_source != 'baseline_history_score':
            raise RuntimeError(
                'Unsupported saturating heterogeneity source: '
                f'{self.saturating_heterogeneity_source!r}.'
            )

        if self.history_score_sorted is None:
            raise RuntimeError(
                '[saturating-v2] baseline_history_score requires the TRAIN-fitted '
                'history-score CDF. Use treatment_assignment_mode="controlled_copula" '
                'and fit_history_score_distribution() before simulation.'
            )

        z = float(self._history_normal_score(patient_df, 0))
        if not np.isfinite(z):
            raise RuntimeError('[saturating-v2] non-finite baseline patient modifier.')

        c = float(self.saturating_heterogeneity_z_clip)
        return float(np.clip(z, -c, c))

    def _patient_specific_ed50_components(self, patient_df=None) -> dict:

        ed50_base = float(self.dr_params.get('ed50', 0.3))
        if ed50_base <= 0:
            raise ValueError('dr_params.ed50 must be > 0 for saturating response.')

        if self.saturating_heterogeneity_mode == 'none':
            return {
                'modifier_z': 0.0,
                'ed50_base': ed50_base,
                'ed50_raw': ed50_base,
                'ed50_i': ed50_base,
                'clipped': False,
                'clip_side': 'none',
            }

        if patient_df is None:
            raise ValueError('[saturating] patient_df is required for patient-specific ED50.')

        z_i = self._persistent_effect_modifier(patient_df)
        eta = float(self.saturating_heterogeneity_strength)
        raw = float(ed50_base * np.exp(eta * z_i))
        clipped = float(np.clip(raw, self.saturating_ed50_min, self.saturating_ed50_max))

        if raw < self.saturating_ed50_min:
            side = 'lower'
        elif raw > self.saturating_ed50_max:
            side = 'upper'
        else:
            side = 'none'

        return {
            'modifier_z': float(z_i),
            'ed50_base': ed50_base,
            'ed50_raw': raw,
            'ed50_i': clipped,
            'clipped': bool(side != 'none'),
            'clip_side': side,
        }

    def _patient_specific_ed50(self, patient_df=None) -> float:
        return float(self._patient_specific_ed50_components(patient_df)['ed50_i'])

    def saturating_patient_parameters(self, patient_df) -> dict:
        out = self._patient_specific_ed50_components(patient_df)
        out.update({
            'heterogeneity_strength': float(self.saturating_heterogeneity_strength),
            'heterogeneity_z_clip': float(self.saturating_heterogeneity_z_clip),
            'ed50_min': float(self.saturating_ed50_min),
            'ed50_max': float(self.saturating_ed50_max),
        })
        return out

    def _dose_response(self, dose, patient_df=None, ed50=None):
        shape = self.dose_response_shape
        dose = float(dose)

        if shape == 'linear':
            return dose

        elif shape == 'saturating':
            if ed50 is None:
                ed50 = self._patient_specific_ed50(patient_df)
            ed50 = float(ed50)
            response = dose / (ed50 + dose) * (ed50 + 1.0)
            return float(response)

        elif shape == 'threshold':
            thr = self.dr_params.get('thr', 0.4)
            return float(np.clip((dose - thr) / (1.0 - thr), 0.0, 1.0))

        elif shape == 'ushape':
            opt = self.dr_params.get('opt', 0.6)
            return float(1.0 - ((dose - opt) / max(opt, 1.0 - opt)) ** 2)

        else:
            raise ValueError(f'unknown dose_response_shape: {shape}')

    @staticmethod
    def _cumulative_weight(lag: int) -> float:
        return 1.0 / np.sqrt(lag + 1)

    def _cumulative_exposure(self, patient_df, s: int, effect_window: int = None):
        if effect_window is None:
            effect_window = int(self.effect_window)
        prev_col = f'{self.treatment_name}_prev'
        start = max(1, s - effect_window + 1)

        E = 0.0
        for h in range(start, s + 1):
            try:
                d = float(patient_df.loc[h, prev_col])
            except (KeyError, TypeError):
                d = 0.0
            if np.isnan(d):
                d = 0.0
            if d > 0:
                E += self._cumulative_weight(s - h) * d
        return E

    def _normalize_exposure(self, E: float, window_size: int) -> float:
        E_max = sum(self._cumulative_weight(lag) for lag in range(max(window_size, 1)))
        return min(E / max(E_max, 1e-8), 1.0)

    def get_treated_outcome(self, patient_df, t, outcome_name, dose: float = 1.0):
        t_max = max(patient_df.index.get_level_values('hours_in'))
        t_stop = min(t_max, t + self.effect_window)
        treatment_range = np.arange(t + 1, t_stop + 1)

        untreated_col = f'{outcome_name}_untreated'
        future_outcome = patient_df.loc[treatment_range, untreated_col].copy()

        ew = int(self.effect_window)

        patient_ed50 = (
            self._patient_specific_ed50(patient_df)
            if self.dose_response_shape == 'saturating' else None
        )

        for s in treatment_range:
            E_s = self._cumulative_exposure(patient_df, int(s), ew)
            if E_s > 0:
                window_size = min(int(s), ew)
                E_norm = self._normalize_exposure(E_s, window_size)
                effect = self.full_effect * self._dose_response(
                    E_norm, patient_df=patient_df, ed50=patient_ed50,
                )
                future_outcome.loc[s] = future_outcome.loc[s] + effect

        return treatment_range, future_outcome

    @staticmethod
    def combine_treatments(treatment_ranges, treated_future_outcomes, doses,
                           baseline_outcome: pd.Series):
        treated_future_outcomes_df = pd.concat(treated_future_outcomes, axis=1)
        doses = np.asarray(doses, dtype=float)
        treated_mask = doses > 0

        if treated_mask.any():
            common_treatment_range = [
                set(tr) for i, tr in enumerate(treatment_ranges) if treated_mask[i]
            ]
            common_treatment_range = sorted(list(set.union(*common_treatment_range)))
            treated_future_outcomes_df = treated_future_outcomes_df.loc[common_treatment_range]

            baseline_vals = baseline_outcome.loc[common_treatment_range].values
            treated_vals = treated_future_outcomes_df.iloc[:, treated_mask].values

            effects = treated_vals - baseline_vals[:, None]
            effects_sum = np.nansum(effects, axis=1)
            treated_future_outcomes_df['agg'] = baseline_vals + effects_sum
        else:
            common_treatment_range = list(treatment_ranges[0])
            treated_future_outcomes_df = treated_future_outcomes_df.loc[common_treatment_range]
            treated_future_outcomes_df['agg'] = treated_future_outcomes_df.iloc[:, 0]

        return common_treatment_range, treated_future_outcomes_df['agg']


class _ForbiddenTrajectoryGenerator:

    def fit(self, *args, **kwargs):
        raise RuntimeError(
            'cf_trajectory_mode="fixed_benchmark" forbids fitting CF trajectories '
            'on intervention_train doses. The formal benchmark is defined in TRAIN-'
            'positive-marginal percentile space and mapped through F_+^{-1}.'
        )

    def generate_all(self, *args, **kwargs):
        raise RuntimeError(
            'cf_trajectory_mode="fixed_benchmark" forbids DoseTrajectoryGenerator.'
        )


def _hash_update_array(hasher, value, *, float_decimals: int = 10):
    arr = np.asarray(value)
    hasher.update(str(tuple(arr.shape)).encode('utf-8'))
    hasher.update(str(arr.dtype).encode('utf-8'))
    if arr.dtype.kind in ('O', 'U', 'S'):
        payload = json.dumps(
            np.asarray(arr).astype(str).reshape(-1).tolist(),
            ensure_ascii=False, separators=(',', ':'),
        )
        hasher.update(payload.encode('utf-8'))
        return
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.round(arr.astype(np.float64), float_decimals).astype('<f8')
    elif np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype('<i8')
    elif np.issubdtype(arr.dtype, np.bool_):
        arr = arr.astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    hasher.update(arr.tobytes())


def _sha1_named_arrays(**items) -> str:
    h = hashlib.sha1()
    for key in sorted(items):
        h.update(str(key).encode('utf-8'))
        _hash_update_array(h, items[key])
    return h.hexdigest()


def _sha1_json_dict(payload: dict) -> str:
    txt = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'), default=str,
    )
    return hashlib.sha1(txt.encode('utf-8')).hexdigest()


def _treatment_positive_marginal_fingerprint(
        synthetic_treatments: Sequence[SyntheticTreatment]) -> str:
    h = hashlib.sha1()
    for j, tr in enumerate(synthetic_treatments):
        meta = {
            'channel': int(j),
            'treatment_name': str(tr.treatment_name),
            'dose_marginal_mode': str(tr.dose_marginal_mode),
            'dose_eps': float(tr.dose_eps),
            'target_treat_rate': float(tr.target_treat_rate),
            'positive_dose_mean_beta_legacy': float(tr.positive_dose_mean),
            'dose_concentration_beta_legacy': float(tr.dose_concentration),
        }
        h.update(_sha1_json_dict(meta).encode('utf-8'))
        if tr.dose_marginal_mode == DISEASE_REAL_DOSE_MARGINAL:
            tr._require_real_marginal()
            _hash_update_array(h, tr.real_marginal_edges)
            _hash_update_array(h, tr.real_marginal_bin_probs)
            _hash_update_array(h, tr.real_marginal_cdf_edges)
        else:
            _hash_update_array(h, np.asarray(tr.positive_dose_params()))
    return h.hexdigest()


def build_master_cf_percentile_sequences(
        projection_horizon: int,
        n_treatments: int,
        fixed_cf_percentile_sequences: Optional[list] = None,
        policy_names: Optional[Sequence[str]] = None):
    K, D = int(projection_horizon), int(n_treatments)
    if K <= 0:
        raise ValueError('fixed_benchmark requires projection_horizon >= 1.')
    if D <= 0:
        raise ValueError('fixed_benchmark requires at least one treatment channel.')
    L = K + 1

    if fixed_cf_percentile_sequences is None:
        base = np.stack([
            np.full(L, 0.50, dtype=np.float64),
            np.linspace(0.50, 0.90, L, dtype=np.float64),
            np.linspace(0.50, 0.10, L, dtype=np.float64),
        ], axis=0)
        names = ['maintenance', 'escalation', 'de_escalation']
    else:
        base = np.asarray(fixed_cf_percentile_sequences, dtype=np.float64)
        if policy_names is None:
            names = [f'custom_percentile_{i}' for i in range(base.shape[0])]
        else:
            names = [str(x) for x in policy_names]

    if base.ndim == 2:
        base = np.repeat(base[:, :, None], D, axis=2)
    if base.ndim != 3 or base.shape[1:] != (L, D):
        raise RuntimeError(f'Percentile CF policies must be (Ntraj,{L},{D}); got {base.shape}.')
    if len(names) != base.shape[0]:
        raise RuntimeError(
            f'policy_names length {len(names)} != number of policies {base.shape[0]}.'
        )
    if not np.all(np.isfinite(base)):
        raise RuntimeError('Percentile CF policies contain non-finite values.')
    if np.any(base <= 0.0) or np.any(base >= 1.0):
        raise RuntimeError(
            'Formal positive-dose percentile policies must lie strictly inside (0,1).'
        )

    base = np.ascontiguousarray(np.round(base, 6), dtype=np.float64)
    percentile_hash = _sha1_named_arrays(
        policy_percentiles=base, policy_names=np.asarray(names, dtype='U'),
    )
    return base, names, percentile_hash


def build_fixed_cf_sequences(
        projection_horizon: int,
        synthetic_treatments: Sequence[SyntheticTreatment],
        fixed_cf_percentile_sequences: Optional[list] = None,
        policy_names: Optional[Sequence[str]] = None,
        fixed_cf_sequences: Optional[list] = None):
    K = int(projection_horizon)
    D = int(len(synthetic_treatments))
    L = K + 1
    if D <= 0:
        raise RuntimeError('No synthetic treatment channel is available for CF mapping.')

    marginal_hash = _treatment_positive_marginal_fingerprint(synthetic_treatments)

    if fixed_cf_sequences is not None:
        if fixed_cf_percentile_sequences is not None:
            raise RuntimeError(
                'Specify only one of fixed_cf_sequences (legacy dose-space override) '
                'or fixed_cf_percentile_sequences (formal P-space benchmark).'
            )
        dose_seq = np.asarray(fixed_cf_sequences, dtype=np.float64)
        if dose_seq.ndim == 2:
            dose_seq = np.repeat(dose_seq[:, :, None], D, axis=2)
        if dose_seq.ndim == 3 and dose_seq.shape[1] == K:
            logger.warning(
                '[CF] legacy fixed_cf_sequences has K elements; prepending element 0 '
                f'to reach K+1={L}. No value is truncated.'
            )
            dose_seq = np.concatenate([dose_seq[:, :1, :], dose_seq], axis=1)
        if dose_seq.ndim != 3 or dose_seq.shape[1:] != (L, D):
            raise RuntimeError(
                f'Legacy dose-space CF sequences must be (Ntraj,{L},{D}); got {dose_seq.shape}.'
            )
        if np.any(~np.isfinite(dose_seq)) or np.any(dose_seq < 0.0) or np.any(dose_seq > 1.0):
            raise RuntimeError('Legacy dose-space CF sequences must lie in [0,1].')
        names = (
            [str(x) for x in policy_names] if policy_names is not None
            else [f'custom_dose_{i}' for i in range(dose_seq.shape[0])]
        )
        if len(names) != dose_seq.shape[0]:
            raise RuntimeError('policy_names length does not match dose-space policies.')
        percentile_seq = None
        percentile_hash = None
        policy_space = 'normalized_dose_legacy_override'
    else:
        percentile_seq, names, percentile_hash = build_master_cf_percentile_sequences(
            K, D,
            fixed_cf_percentile_sequences=fixed_cf_percentile_sequences,
            policy_names=policy_names,
        )
        dose_seq = np.empty_like(percentile_seq, dtype=np.float64)
        for d, tr in enumerate(synthetic_treatments):
            dose_seq[:, :, d] = np.asarray(
                tr.positive_marginal_ppf(percentile_seq[:, :, d]), dtype=np.float64,
            )
        policy_space = 'train_positive_empirical_percentile'

    dose_seq = np.ascontiguousarray(np.round(dose_seq, 8), dtype=np.float64)
    dose_hash = _sha1_named_arrays(
        policy_doses=dose_seq, policy_names=np.asarray(names, dtype='U'),
    )

    h = hashlib.sha1()
    h.update(str(policy_space).encode('utf-8'))
    h.update(str(marginal_hash).encode('utf-8'))
    h.update(str(dose_hash).encode('utf-8'))
    if percentile_hash is not None:
        h.update(str(percentile_hash).encode('utf-8'))
    h.update(f'dose_scale={float(DOSE_SCALE):.12g}'.encode('utf-8'))
    policy_hash = h.hexdigest()

    meta = {
        'policy_space': policy_space,
        'policy_names': tuple(names),
        'policy_percentiles': (None if percentile_seq is None else percentile_seq.copy()),
        'policy_doses': dose_seq.copy(),
        'positive_marginal_hash': marginal_hash,
        'percentile_policy_hash': percentile_hash,
        'dose_trajectory_hash': dose_hash,
        'policy_hash': policy_hash,
        'dose_scale': float(DOSE_SCALE),
        'prediction_horizon': K,
        'intervention_sequence_length': L,
    }
    return dose_seq, policy_hash, names, meta


def _ensure_intervention_length(seq: np.ndarray, projection_horizon: int,
                                n_treatments: int, origin: str) -> np.ndarray:
    K, L = int(projection_horizon), int(projection_horizon) + 1
    if seq.ndim == 2:
        seq = np.repeat(seq[:, :, None], n_treatments, axis=2)
    if seq.shape[1] == K:
        logger.warning(
            f'[CF] {origin} 给出 {K} 个 intervention (== prediction horizon); '
            f'前置复制 element 0 补足 K+1={L}.'
        )
        seq = np.concatenate([seq[:, :1, :], seq], axis=1)
    if seq.shape[1] != L:
        raise RuntimeError(
            f'[CF] {origin}: expected {L} interventions for prediction horizon {K}, '
            f'got {seq.shape[1]}.'
        )
    return seq


class MIMIC4SyntheticDataset(MIMIC4RealDataset):

    def __init__(self,
                 all_vitals: pd.DataFrame,
                 static_features: pd.DataFrame,
                 synthetic_outcomes: List[SyntheticOutcomeGenerator],
                 synthetic_treatments: List[SyntheticTreatment],
                 treatment_outcomes_influence: dict,
                 subset_name: str,
                 mode='factual',
                 projection_horizon: int = None,
                 treatments_seq: np.array = None,
                 n_treatments_seq: int = None,
                 dose_grid: Optional[Sequence[float]] = None,
                 dose_scale: float = None,
                 untreated_scaling: bool = False,
                 pre_simulated: bool = False,
                 vital_cols: Optional[Sequence[str]] = None,
                 trajectory_names: Optional[Sequence[str]] = None):

        self.subset_name = subset_name
        self.all_vitals = all_vitals.copy()
        vital_cols = list(vital_cols) if vital_cols is not None else list(all_vitals.columns)
        self.synthetic_outcomes = synthetic_outcomes
        self.synthetic_treatments = synthetic_treatments
        self.treatment_outcomes_influence = treatment_outcomes_influence
        self.patient_ids = None
        self.trajectory_ids = None
        self.policy_ids = None
        self.trajectory_names = (
            None if trajectory_names is None else tuple(str(x) for x in trajectory_names)
        )
        self.untreated_scaling = bool(untreated_scaling)

        for treatment in self.synthetic_treatments:
            if treatment.treatment_assignment_mode == COPULA_MODE:
                assert treatment.history_score_sorted is not None, (
                    f'{treatment.treatment_name}: TRAIN-fitted history score CDF missing '
                    f'on "{subset_name}".'
                )
                assert len(treatment.history_score_sorted) > 100, (
                    f'{treatment.treatment_name}: history_score_sorted too small.'
                )
                assert (not treatment.dose_marginal_conditional) or treatment.treated_cdf is not None, (
                    f'{treatment.treatment_name}: TRAIN-fitted treated-CDF missing '
                    f'on "{subset_name}".'
                )

        if dose_scale is not None:
            set_dose_scale(dose_scale)
        self.dose_scale = DOSE_SCALE

        self.projection_horizon = projection_horizon if projection_horizon is not None else 0

        if dose_grid is None:
            self.dose_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
        else:
            self.dose_grid = [float(d) for d in dose_grid]
        if 0.0 not in self.dose_grid:
            logger.warning(
                'dose_grid does not contain 0.0 — '
                '"untreated" cannot be represented in CF enumeration.'
            )

        prev_treatment_cols = [
            f'{treatment.treatment_name}_prev' for treatment in self.synthetic_treatments
        ]
        outcome_cols = [outcome.outcome_name for outcome in self.synthetic_outcomes]

        if pre_simulated:
            for outcome in self.synthetic_outcomes:
                col = f'{outcome.outcome_name}_untreated'
                if col not in self.all_vitals.columns:
                    raise RuntimeError(f'pre_simulated=True but {col} is missing.')
                self.all_vitals[outcome.outcome_name] = self.all_vitals[col].copy()
        else:
            for outcome in self.synthetic_outcomes:
                outcome.simulate_untreated(self.all_vitals, static_features)
        for treatment in self.synthetic_treatments:
            self.all_vitals[f'{treatment.treatment_name}_prev'] = 0.0
        self.all_vitals['fact'] = np.nan
        self.all_vitals.loc[(slice(None), 0), 'fact'] = 1.0
        user_sizes = self.all_vitals.groupby(level='subject_id', sort=False).size()

        seeds = np.random.randint(np.iinfo(np.int32).max, size=len(static_features))
        par = Parallel(n_jobs=max(1, multiprocessing.cpu_count() - 10), backend='loky')
        logger.info(
            f'Simulating {mode} treatments (zero-inflated continuous dose) and applying '
            'them to outcomes.'
        )

        if mode == 'factual':
            self.all_vitals = par(
                delayed(self.treat_patient_factually)(patient_ix, seed)
                for patient_ix, seed in tqdm(
                    zip(static_features.index, seeds), total=len(static_features)
                )
            )

        elif mode == 'counterfactual_treatment_seq' or mode == 'counterfactual_one_step':
            self.treatments_seq = treatments_seq
            if mode == 'counterfactual_one_step':
                if self.treatments_seq is None:
                    grid_combos = list(itertools.product(
                        *([self.dose_grid] * len(prev_treatment_cols))
                    ))
                    self.treatments_seq = np.array(grid_combos, dtype=float)
                    self.treatments_seq = self.treatments_seq[:, None, :]
                    logger.info(
                        f'counterfactual_one_step: enumerating '
                        f'{len(self.treatments_seq)} dose combos '
                        f'({len(self.dose_grid)} ** {len(prev_treatment_cols)}).'
                    )
                self.cf_start = 0
            else:
                self.cf_start = 1

            if mode == 'counterfactual_one_step':
                self.projection_horizon = int(projection_horizon) if projection_horizon else 1
                self.cf_lead_intervention = False
                self.intervention_sequence_length = int(self.treatments_seq.shape[1])
                self.n_treatments_seq = int(self.treatments_seq.shape[0])
                assert self.intervention_sequence_length == 1, (
                    f'counterfactual_one_step needs 1 intervention, got '
                    f'{self.intervention_sequence_length}'
                )
            else:
                if projection_horizon is None:
                    raise RuntimeError(
                        'counterfactual_treatment_seq requires an explicit '
                        'projection_horizon (= prediction horizon K).'
                    )
                self.projection_horizon = int(projection_horizon)
                self.cf_lead_intervention = True
                if self.treatments_seq is not None:
                    self.intervention_sequence_length = int(self.treatments_seq.shape[1])
                    self.n_treatments_seq = int(self.treatments_seq.shape[0])
                else:
                    assert n_treatments_seq is not None
                    self.intervention_sequence_length = self.projection_horizon + 1
                    self.n_treatments_seq = int(n_treatments_seq)
                assert self.intervention_sequence_length == self.projection_horizon + 1, (
                    f'intervention_sequence_length={self.intervention_sequence_length} '
                    f'must equal projection_horizon+1={self.projection_horizon + 1}. '
                    '不要截断 config 的 K+1 序列.'
                )
            logger.info(
                f'[CF] prediction_horizon={self.projection_horizon}, '
                f'intervention_sequence_length={self.intervention_sequence_length}'
            )

            self.all_vitals = par(
                delayed(self.treat_patient_counterfactually)(patient_ix, seed)
                for patient_ix, seed in tqdm(
                    zip(static_features.index, seeds), total=len(static_features)
                )
            )

            logger.info('Concatenating all the trajectories together.')
            self.all_vitals = [
                pd.concat(cf_patient_df, keys=range(len(cf_patient_df)), names=['traj'])
                for cf_patient_df in self.all_vitals
            ]
        else:
            raise NotImplementedError()

        self.all_vitals = pd.concat(self.all_vitals, keys=static_features.index)

        static_feature_columns = list(static_features.columns)
        if mode == 'factual':
            self.all_vitals = (
                self.all_vitals.unstack(fill_value=np.nan, level=0)
                .stack(dropna=False).swaplevel(0, 1).sort_index()
            )
            static_features = static_features.sort_index()
            self.patient_ids = np.asarray(static_features.index.values)
            self.trajectory_ids = np.full(len(self.patient_ids), -1, dtype=np.int64)
            self.policy_ids = self.trajectory_ids.copy()
            static_features = static_features[static_feature_columns].values

        elif mode in ('counterfactual_one_step', 'counterfactual_treatment_seq'):
            self.all_vitals = (
                self.all_vitals.unstack(fill_value=np.nan, level=2)
                .stack(dropna=False).sort_index()
            )

            pair_index = self.all_vitals.groupby(
                level=['subject_id', 'traj'], sort=True
            ).size().index
            self.patient_ids = np.asarray(
                pair_index.get_level_values('subject_id').to_numpy()
            )
            self.trajectory_ids = np.asarray(
                pair_index.get_level_values('traj').to_numpy(), dtype=np.int64,
            )
            self.policy_ids = self.trajectory_ids.copy()

            if self.trajectory_names is None:
                self.trajectory_names = tuple(
                    f'traj_{i}' for i in range(int(self.n_treatments_seq))
                )
            if len(self.trajectory_names) != int(self.n_treatments_seq):
                raise RuntimeError(
                    f'trajectory_names has {len(self.trajectory_names)} entries but '
                    f'n_treatments_seq={self.n_treatments_seq}.'
                )
            if np.any(self.trajectory_ids < 0) or np.any(
                    self.trajectory_ids >= len(self.trajectory_names)):
                raise RuntimeError('CF trajectory IDs are outside the declared policy bank.')

            static_features = static_features.loc[
                self.patient_ids, static_feature_columns
            ].values

        treatments = (
            self.all_vitals[prev_treatment_cols].fillna(0.0).values
            .reshape((-1, max(user_sizes), len(prev_treatment_cols)))
        )
        vitals = (
            self.all_vitals[vital_cols].fillna(0.0).values
            .reshape((-1, max(user_sizes), len(vital_cols)))
        )
        outcomes_unscaled = (
            self.all_vitals[outcome_cols].fillna(0.0).values
            .reshape((-1, max(user_sizes), len(outcome_cols)))
        )
        active_entries = (~self.all_vitals.isna().all(1)).astype(float)
        active_entries = active_entries.values.reshape((-1, max(user_sizes), 1))
        user_sizes = np.squeeze(active_entries.sum(1))

        logger.info(f'Shape of exploded vitals: {vitals.shape}.')
        logger.info(
            f'Dose stats on {self.subset_name}: mean={treatments.mean():.3f}, '
            f'frac(dose>0)={(treatments > 0).mean():.3f}, '
            f'frac(dose==0)={(treatments == 0).mean():.3f}, '
            f'frac(dose>=0.5)={(treatments >= 0.5).mean():.3f}'
        )

        self.data = {
            'sequence_lengths': user_sizes - 1,
            'prev_treatments': treatments[:, :-1, :],
            'vitals': vitals[:, :-1, :],
            'next_vitals': vitals[:, 1:, :],
            'current_treatments': treatments[:, 1:, :],
            'static_features': static_features,
            'active_entries': active_entries[:, 1:, :],
            'unscaled_outputs': outcomes_unscaled[:, 1:, :],
            'prev_unscaled_outputs': outcomes_unscaled[:, :-1, :],
            'patient_ids': self.patient_ids,
            'trajectory_ids': self.trajectory_ids,
            'policy_ids': self.policy_ids,
        }

        self.processed = False
        self.processed_sequential = False
        self.processed_autoregressive = False
        self.norm_const = 1.0

    def __getitem__(self, index):
        result = super().__getitem__(index)
        if isinstance(result, dict):
            if 'patient_ids' in self.data:
                pid_arr = np.asarray(self.data['patient_ids'])
                result['patient_ids'] = (
                    np.int64(pid_arr[index])
                    if np.issubdtype(pid_arr.dtype, np.integer) else pid_arr[index]
                )
            for key in ('trajectory_ids', 'policy_ids'):
                if key in self.data:
                    result[key] = np.int64(np.asarray(self.data[key])[index])
        return result

    def cf_time_index_map(self, t: int):
        offset = 1 if getattr(self, 'cf_lead_intervention', False) else 0
        interventions = [
            {'element': e,
             'panel_prev_time': t + e + 1 - offset,
             'model_row_index': t + e - offset}
            for e in range(self.intervention_sequence_length)
        ]
        horizons = [
            {'horizon': h, 'panel_outcome_time': t + h, 'model_output_row_index': t + h - 1}
            for h in range(1, self.projection_horizon + 1)
        ]
        return interventions, horizons

    def cf_elements_consumed_by_horizon(self, t: int, treatment=None):
        treatment = treatment or self.synthetic_treatments[0]
        W = int(treatment.effect_window)
        interventions, horizons = self.cf_time_index_map(t)
        out = []
        for hz in horizons:
            s = hz['panel_outcome_time']
            used = [
                iv['element'] for iv in interventions
                if max(1, s - W + 1) <= iv['panel_prev_time'] <= s
            ]
            out.append({'horizon': hz['horizon'], 'elements': used})
        return out

    def _sample_treatments_from_factuals(self, patient_df, t, rng=np.random.RandomState(None)):
        factual_patient_df = patient_df[patient_df.fact.astype(bool)]
        treat_probas, treatment_doses = {}, {}
        for treatment in self.synthetic_treatments:
            p, dose = treatment.sample_treatment_dose(factual_patient_df, t, rng)
            treat_probas[treatment.treatment_name] = p
            treatment_doses[treatment.treatment_name] = dose
        return treat_probas, treatment_doses

    def _combined_treating(self, patient_df, t, outcome: SyntheticOutcomeGenerator,
                           treat_probas: dict, treatment_doses: dict):
        treatment_ranges, treated_future_outcomes, doses = [], [], []
        influencing_treatments = self.treatment_outcomes_influence[outcome.outcome_name]
        influencing_treatments = [
            treatment for treatment in self.synthetic_treatments
            if treatment.treatment_name in influencing_treatments
        ]

        for treatment in influencing_treatments:
            dose = float(treatment_doses[treatment.treatment_name])
            treatment_range, treated_future_outcome = treatment.get_treated_outcome(
                patient_df, t, outcome.outcome_name, dose=dose,
            )
            treatment_ranges.append(treatment_range)
            treated_future_outcomes.append(treated_future_outcome)
            doses.append(dose)

        union_range = sorted(list(set.union(*[set(r) for r in treatment_ranges])))
        baseline_outcome = patient_df.loc[union_range, f'{outcome.outcome_name}_untreated'].copy()

        common_treatment_range, future_outcomes = SyntheticTreatment.combine_treatments(
            treatment_ranges, treated_future_outcomes,
            np.asarray(doses, dtype=float), baseline_outcome,
        )
        return common_treatment_range, future_outcomes

    def treat_patient_factually(self, patient_ix: int, seed: int = None):
        patient_df = self.all_vitals.loc[patient_ix].copy()
        rng = np.random.RandomState(seed)
        prev_treatment_cols = [
            f'{treatment.treatment_name}_prev' for treatment in self.synthetic_treatments
        ]

        for t in range(len(patient_df)):
            treat_probas, treatment_doses = self._sample_treatments_from_factuals(
                patient_df, t, rng,
            )
            if t < max(patient_df.index.get_level_values('hours_in')):
                patient_df.loc[t + 1, 'fact'] = 1.0
                patient_df.loc[t + 1, prev_treatment_cols] = {
                    f'{name}_prev': v for name, v in treatment_doses.items()
                }
                for outcome in self.synthetic_outcomes:
                    common_treatment_range, future_outcomes = self._combined_treating(
                        patient_df, t, outcome, treat_probas, treatment_doses,
                    )
                    patient_df.loc[
                        common_treatment_range, f'{outcome.outcome_name}'
                    ] = future_outcomes
        return patient_df

    def treat_patient_counterfactually(self, patient_ix: int, seed: int = None):
        patient_df = self.all_vitals.loc[patient_ix].copy()
        patient_ae = (~patient_df.copy().drop(columns=['fact']).isna().all(1)).astype(float)
        patient_cf_start = patient_ae.sum() - 1 - self.projection_horizon

        rng = np.random.RandomState(seed)
        prev_treatment_cols = [
            f'{treatment.treatment_name}_prev' for treatment in self.synthetic_treatments
        ]
        dose_grid = np.asarray(self.dose_grid, dtype=float)
        cf_patient_dfs = []

        for t in range(len(patient_df)):
            if self.treatments_seq is None:
                possible_treatments = rng.choice(
                    dose_grid,
                    (self.n_treatments_seq, self.intervention_sequence_length,
                     len(self.synthetic_treatments)),
                )
            else:
                possible_treatments = self.treatments_seq

            if t + self.projection_horizon <= max(
                    patient_df.index.get_level_values('hours_in')):
                if t >= patient_cf_start:
                    possible_patient_dfs = [
                        patient_df.copy().loc[:t + self.projection_horizon]
                        for _ in range(possible_treatments.shape[0])
                    ]
                    offset = 1 if self.cf_lead_intervention else 0
                    if offset and t >= 1:
                        for traj_ind in range(possible_treatments.shape[0]):
                            lead_doses = {
                                treatment.treatment_name:
                                    float(possible_treatments[traj_ind, 0, j])
                                for j, treatment in enumerate(self.synthetic_treatments)
                            }
                            df_lead = possible_patient_dfs[traj_ind]
                            df_lead.loc[t, prev_treatment_cols] = {
                                f'{name}_prev': v for name, v in lead_doses.items()
                            }
                            for outcome in self.synthetic_outcomes:
                                common_treatment_range, future_outcomes = self._combined_treating(
                                    df_lead, t - 1, outcome, {}, lead_doses,
                                )
                                df_lead.loc[
                                    common_treatment_range, outcome.outcome_name
                                ] = future_outcomes
                    for time_ind in range(self.projection_horizon):
                        for traj_ind, possible_treatment in enumerate(
                                possible_treatments[:, time_ind + offset, :]):
                            future_treat_probas, _ = self._sample_treatments_from_factuals(
                                possible_patient_dfs[traj_ind], t + time_ind, rng,
                            )
                            future_treatment_doses = {
                                treatment.treatment_name: float(possible_treatment[j])
                                for j, treatment in enumerate(self.synthetic_treatments)
                            }
                            possible_patient_dfs[traj_ind].loc[
                                t + 1 + time_ind, prev_treatment_cols
                            ] = {f'{name}_prev': v for name, v in future_treatment_doses.items()}
                            possible_patient_dfs[traj_ind].at[t + 1 + time_ind, 'fact'] = 1.0
                            for outcome in self.synthetic_outcomes:
                                common_treatment_range, future_outcomes = self._combined_treating(
                                    possible_patient_dfs[traj_ind], t + time_ind, outcome,
                                    future_treat_probas, future_treatment_doses,
                                )
                                possible_patient_dfs[traj_ind].loc[
                                    common_treatment_range, outcome.outcome_name
                                ] = future_outcomes
                    for possible_patient_df in possible_patient_dfs:
                        possible_patient_df.loc[t + 1:, 'fact'] = 0.0
                    cf_patient_dfs.extend(possible_patient_dfs)

                treat_probas, treatment_doses = self._sample_treatments_from_factuals(
                    patient_df, t, rng,
                )
                patient_df.loc[t + 1, 'fact'] = 1.0
                patient_df.loc[t + 1, prev_treatment_cols] = {
                    f'{name}_prev': v for name, v in treatment_doses.items()
                }
                for outcome in self.synthetic_outcomes:
                    common_treatment_range, future_outcomes = self._combined_treating(
                        patient_df, t, outcome, treat_probas, treatment_doses,
                    )
                    patient_df.loc[
                        common_treatment_range, outcome.outcome_name
                    ] = future_outcomes
        return cf_patient_dfs

    def _rollout_trajectory(self, patient_df, t, dose_seq, outcome_name: str):
        df = patient_df.copy()
        prev_treatment_cols = [f'{tr.treatment_name}_prev' for tr in self.synthetic_treatments]
        max_h = max(df.index.get_level_values('hours_in'))
        K = len(dose_seq)

        for time_ind in range(K):
            cur_t = t + time_ind
            if cur_t + 1 > max_h:
                break
            doses = dose_seq[time_ind]
            df.loc[cur_t + 1, prev_treatment_cols] = {
                f'{name}_prev': float(doses[name]) for name in doses
            }
            for outcome in self.synthetic_outcomes:
                if outcome.outcome_name != outcome_name:
                    continue
                common_range, fut = self._combined_treating(
                    df, cur_t, outcome, treat_probas={}, treatment_doses=doses,
                )
                df.loc[common_range, outcome_name] = fut

        target_h = min(t + K, max_h)
        return float(df.loc[target_h, outcome_name])

    def _rollout_capo(self, patient_df, t, doses: dict, outcome_name: str):
        K = self.projection_horizon
        df = patient_df.copy()
        prev_treatment_cols = [f'{tr.treatment_name}_prev' for tr in self.synthetic_treatments]
        max_h = max(df.index.get_level_values('hours_in'))

        for time_ind in range(K):
            cur_t = t + time_ind
            if cur_t + 1 > max_h:
                break
            df.loc[cur_t + 1, prev_treatment_cols] = {
                f'{name}_prev': float(doses[name]) for name in doses
            }
            for outcome in self.synthetic_outcomes:
                if outcome.outcome_name != outcome_name:
                    continue
                common_range, fut = self._combined_treating(
                    df, cur_t, outcome, treat_probas={}, treatment_doses=doses,
                )
                df.loc[common_range, outcome_name] = fut

        target_h = min(t + K, max_h)
        return float(df.loc[target_h, outcome_name])

    def generate_dose_response_ground_truth(self, patient_ix, t,
                                            outcome_name=None, dense_grid=None):
        if dense_grid is None:
            dense_grid = np.linspace(0.0, 1.0, 51)
        if outcome_name is None:
            outcome_name = self.synthetic_outcomes[0].outcome_name
        patient_df = self.all_vitals.loc[patient_ix].copy()

        m_true = []
        for a in dense_grid:
            doses = {tr.treatment_name: float(a) for tr in self.synthetic_treatments}
            m_true.append(self._rollout_capo(patient_df, t, doses, outcome_name))
        m_true = np.asarray(m_true)
        dm_da = np.gradient(m_true, dense_grid)
        return np.asarray(dense_grid), m_true, dm_da

    def true_gps(self, patient_ix, t, dose_vec):
        patient_df = self.all_vitals.loc[patient_ix].copy()
        factual_df = patient_df[patient_df.fact.astype(bool)]
        density = 1.0
        for j, tr in enumerate(self.synthetic_treatments):
            d = float(dose_vec[j])

            if tr.treatment_assignment_mode == LEGACY_MODE:
                p_treat = float(np.clip(
                    np.asarray(tr.treatment_proba(factual_df, t)).flatten()[0], 0.0, 1.0,
                ))
                if d == 0.0:
                    density *= (1.0 - p_treat)
                else:
                    mu = float(np.clip(
                        np.asarray(tr.dose_intensity_mean(factual_df, t)).flatten()[0],
                        tr.dose_eps, 1 - tr.dose_eps,
                    ))
                    d_clip = float(np.clip(d, tr.dose_eps, 1 - tr.dose_eps))
                    density *= p_treat * float(beta_dist.pdf(
                        d_clip,
                        tr.dose_concentration * mu,
                        tr.dose_concentration * (1 - mu),
                    ))
            else:
                z = tr._history_normal_score(factual_df, t)
                p_treat = float(np.clip(tr._copula_treat_proba(z), 1e-6, 1 - 1e-6))
                if d == 0.0:
                    density *= (1.0 - p_treat)
                else:
                    density *= p_treat * tr.positive_dose_density(d, z)

        return float(density) if np.isfinite(density) else 0.0

    def get_scaling_params(self):
        outcome_cols = [outcome.outcome_name for outcome in self.synthetic_outcomes]
        if self.untreated_scaling:
            outcome_cols = [f'{c}_untreated' for c in outcome_cols]

        return {
            'output_means': self.all_vitals[outcome_cols].mean(0).to_numpy(),
            'output_stds': self.all_vitals[outcome_cols].std(0).to_numpy(),
        }

    def process_data(self, scaling_params):
        if not self.processed:
            logger.info(f'Processing {self.subset_name} dataset before training')
            self.data['outputs'] = (
                (self.data['unscaled_outputs'] - scaling_params['output_means'])
                / scaling_params['output_stds']
            )
            self.data['prev_outputs'] = (
                (self.data['prev_unscaled_outputs'] - scaling_params['output_means'])
                / scaling_params['output_stds']
            )
            data_shapes = {k: v.shape for k, v in self.data.items() if hasattr(v, 'shape')}
            logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')
            self.scaling_params = scaling_params
            self.processed = True
        else:
            logger.info(f'{self.subset_name} Dataset already processed')
        return self.data


class MIMIC4SyntheticDatasetCollection(SyntheticDatasetCollection):

    def __init__(self,
                 path,
                 synth_outcomes_list: list,
                 synth_treatments_list: list,
                 treatment_outcomes_influence: dict,
                 min_seq_length: int = None,
                 max_seq_length: int = None,
                 max_number: int = None,
                 seed: int = 100,
                 data_seed: int = 42,
                 split: dict = {'val': 0.2, 'test': 0.2},
                 projection_horizon: int = 6,
                 autoregressive=True,
                 n_treatments_seq: int = None,
                 treatment_sequence: list = None,
                 dose_grid: Optional[Sequence[float]] = None,
                 dose_scale_override: float = None,
                 dose_scale_col: str = None,
                 use_clinical_cf: bool = True,
                 cf_trajectory_mode: str = FIXED_CF,
                 fixed_cf_sequences: Optional[list] = None,
                 fixed_cf_percentile_sequences: Optional[list] = None,
                 fixed_cf_policy_names: Optional[Sequence[str]] = None,
                 fixed_cf_master_horizon: int = 6,
                 fixed_cf_enforce_master_horizon: bool = True,
                 **kwargs):
        super(MIMIC4SyntheticDatasetCollection, self).__init__()
        self.seed = seed
        np.random.seed(seed)

        assert cf_trajectory_mode in (FIXED_CF, LEGACY_CF)
        self.cf_trajectory_mode = cf_trajectory_mode
        self.fixed_cf_master_horizon = int(fixed_cf_master_horizon)
        self.fixed_cf_enforce_master_horizon = bool(fixed_cf_enforce_master_horizon)
        if self.fixed_cf_master_horizon <= 0:
            raise ValueError("fixed_cf_master_horizon must be >= 1.")
        if (cf_trajectory_mode == FIXED_CF
                and self.fixed_cf_enforce_master_horizon
                and int(projection_horizon) != self.fixed_cf_master_horizon):
            raise RuntimeError(
                f"Formal fixed_benchmark must be generated ONCE at master H"
                f"{self.fixed_cf_master_horizon}; got dataset projection_horizon="
                f"{projection_horizon}. Evaluate H1..H{self.fixed_cf_master_horizon} "
                "as prefixes/readouts of this same master dataset, not by "
                "regenerating shorter-horizon CF cohorts."
            )
        controlled = all(
            tr.treatment_assignment_mode == COPULA_MODE for tr in synth_treatments_list
        )

        intervention, all_vitals, static_features = load_mimic4_data_raw(
            ROOT_PATH + '/' + path,
            treatment_list=TREATMENT_LIST,
            outcome_list=OUTCOME_LIST,
            vital_list=VITAL_LIST,
            static_list=STATIC_LIST,
            min_seq_length=min_seq_length,
            max_seq_length=max_seq_length,
            max_number=max_number,
            data_seed=data_seed,
            **kwargs,
        )

        static_features, static_features_test = train_test_split(
            static_features, test_size=split['test'], random_state=seed,
        )
        all_vitals, all_vitals_test = (
            all_vitals.loc[static_features.index],
            all_vitals.loc[static_features_test.index],
        )

        if split['val'] > 0.0:
            static_features_train, static_features_val = train_test_split(
                static_features,
                test_size=split['val'] / (1 - split['test']),
                random_state=2 * seed,
            )
            all_vitals_train = all_vitals.loc[static_features_train.index]
            all_vitals_val = all_vitals.loc[static_features_val.index]
        else:
            static_features_train = static_features
            static_features_val = None
            all_vitals_train = all_vitals
            all_vitals_val = None

        # TRAIN-only static preprocessing. load_mimic4_data_raw deliberately
        # returns unprocessed static variables so validation/test do not enter
        # means, standard deviations, categories, or imputation choices.
        self.static_preprocessor = fit_static_preprocessor(static_features_train)
        static_features_train = apply_static_preprocessor(
            static_features_train, self.static_preprocessor
        )
        static_features_test = apply_static_preprocessor(
            static_features_test, self.static_preprocessor
        )
        if static_features_val is not None:
            static_features_val = apply_static_preprocessor(
                static_features_val, self.static_preprocessor
            )

        # TRAIN-only dynamic normalization. The revised H5 remains NaN-free on
        # disk, but the loader restores only unresolved 0 storage placeholders to
        # NaN in memory. pandas mean/std therefore ignore those placeholders.
        self.vital_cols = all_vitals_train.columns
        self.train_mean = all_vitals_train[self.vital_cols].mean(axis=0)
        self.train_std = all_vitals_train[self.vital_cols].std(axis=0)

        all_missing_train = self.train_mean[self.train_mean.isna()].index.tolist()
        if all_missing_train:
            logger.warning(
                '[state-preproc] columns entirely missing in TRAIN are neutralized: %s',
                all_missing_train,
            )
            self.train_mean.loc[all_missing_train] = 0.0
            self.train_std.loc[all_missing_train] = 1.0

        self.train_std = self.train_std.replace(0, 1.0).fillna(1.0)

        all_vitals_train = self.normalize_vitals(all_vitals_train)
        all_vitals_val = self.normalize_vitals(all_vitals_val)
        all_vitals_test = self.normalize_vitals(all_vitals_test)

        # Only now convert unresolved states to 0. On the standardized scale,
        # 0 is the TRAIN mean. This happens BEFORE history-score calibration and
        # before synthetic treatment/outcome generation, so the DGP is finite.
        n_imp_train = int(all_vitals_train.isna().sum().sum())
        n_imp_val = 0 if all_vitals_val is None else int(all_vitals_val.isna().sum().sum())
        n_imp_test = int(all_vitals_test.isna().sum().sum())
        all_vitals_train = all_vitals_train.fillna(0.0)
        if all_vitals_val is not None:
            all_vitals_val = all_vitals_val.fillna(0.0)
        all_vitals_test = all_vitals_test.fillna(0.0)
        logger.info(
            '[state-preproc] standardized unresolved placeholders -> 0: '
            'train=%d val=%d test=%d.',
            n_imp_train, n_imp_val, n_imp_test,
        )

        intervention_train = intervention.loc[static_features_train.index]
        intervention_test = intervention.loc[static_features_test.index]

        if dose_scale_override is not None:
            computed_dose_scale = float(dose_scale_override)
            logger.info(f'[dose-norm] Using overridden dose_scale = {computed_dose_scale:.2f}')
        else:
            dose_col = dose_scale_col if dose_scale_col is not None else (
                TREATMENT_LIST[0] if len(TREATMENT_LIST) > 0 else None
            )
            if dose_col is not None and dose_col in intervention_train.columns:
                nonzero_doses = intervention_train[dose_col].dropna()
                nonzero_doses = nonzero_doses[nonzero_doses > 0]
                computed_dose_scale = (
                    float(np.percentile(nonzero_doses, 99)) if len(nonzero_doses) > 0 else 1.0
                )
            else:
                computed_dose_scale = 1.0
                logger.warning('[dose-norm] No dose column available; D_scale defaults to 1.0')
        set_dose_scale(computed_dose_scale)
        self.dose_scale = DOSE_SCALE

        _dose_col = dose_scale_col if dose_scale_col is not None else (
            TREATMENT_LIST[0] if len(TREATMENT_LIST) > 0 else None
        )
        if _dose_col is not None and _dose_col in intervention_train.columns:
            _raw = intervention_train[_dose_col].dropna().values
            self.train_raw_doses = np.asarray(_raw, dtype=float)
            self.train_normalized_doses = np.clip(
                self.train_raw_doses / max(self.dose_scale, 1e-8), 0.0, 1.0,
            )
        else:
            self.train_raw_doses = np.array([])
            self.train_normalized_doses = np.array([])
            logger.warning(
                f'[dose-norm] dose column not found in {list(intervention_train.columns)}'
            )

        if controlled:
            if len(synth_treatments_list) != 1:
                needs_real = any(
                    tr.dose_marginal_mode == DISEASE_REAL_DOSE_MARGINAL
                    or tr.inherit_real_treat_rate
                    for tr in synth_treatments_list
                )
                if needs_real:
                    raise RuntimeError(
                        'Disease-specific real marginal currently expects one '
                        'continuous treatment channel, matching the existing '
                        'single dose_scale_col pipeline.  For multiple treatments, '
                        'provide a per-treatment real-dose-column mapping first.'
                    )

            for treatment in synth_treatments_list:
                if (treatment.dose_marginal_mode == DISEASE_REAL_DOSE_MARGINAL
                        or treatment.inherit_real_treat_rate):
                    if self.train_normalized_doses.size == 0:
                        raise RuntimeError(
                            '[disease-marginal] real TRAIN dose column is required '
                            'but no normalized doses were found.'
                        )
                    treatment.fit_disease_real_dose_marginal(self.train_normalized_doses)

        if controlled:
            self._calibrate_history_scores(
                all_vitals_train, static_features_train,
                synth_outcomes_list, synth_treatments_list, seed,
            )

        self.train_f = MIMIC4SyntheticDataset(
            all_vitals_train, static_features_train,
            synth_outcomes_list, synth_treatments_list,
            treatment_outcomes_influence, 'train',
            dose_grid=dose_grid, dose_scale=self.dose_scale,
            untreated_scaling=controlled,
        )

        if split['val'] > 0.0:
            self.val_f = MIMIC4SyntheticDataset(
                all_vitals_val, static_features_val,
                synth_outcomes_list, synth_treatments_list,
                treatment_outcomes_influence, 'val',
                dose_grid=dose_grid, dose_scale=self.dose_scale,
                untreated_scaling=controlled,
            )

        cf_hash = None
        cf_names = None
        cf_policy_meta = None
        if cf_trajectory_mode == FIXED_CF:
            self.dose_traj_generator = _ForbiddenTrajectoryGenerator()

            direct_sequences = fixed_cf_sequences
            direct_percentiles = fixed_cf_percentile_sequences
            direct_names = fixed_cf_policy_names
            if treatment_sequence is not None:
                requested = _ensure_intervention_length(
                    np.asarray([treatment_sequence], dtype=float),
                    projection_horizon, len(synth_treatments_list),
                    'dataset.treatment_sequence',
                )
                direct_sequences = requested
                direct_percentiles = None
                if direct_names is None or len(direct_names) != 1:
                    direct_names = ['configured_sequence']
                if fixed_cf_sequences is not None or fixed_cf_percentile_sequences is not None:
                    logger.warning(
                        '[CF] dataset.treatment_sequence takes precedence over fixed CF '
                        'policy settings for this run.'
                    )
                logger.info(
                    '[CF] fixed_benchmark uses dataset.treatment_sequence exactly: %s',
                    np.asarray(requested[0]).tolist(),
                )
                print(
                    f'[CF] using configured treatment_sequence exactly: '
                    f'{np.asarray(requested[0]).tolist()}', flush=True,
                )

            cf_treatments_seq, cf_hash, cf_names, cf_policy_meta = build_fixed_cf_sequences(
                projection_horizon=projection_horizon,
                synthetic_treatments=synth_treatments_list,
                fixed_cf_percentile_sequences=direct_percentiles,
                policy_names=direct_names,
                fixed_cf_sequences=direct_sequences,
            )
            if treatment_sequence is not None:
                if cf_treatments_seq.shape[0] != 1:
                    raise RuntimeError(
                        'Configured treatment_sequence must materialize exactly one CF policy.'
                    )
                expected = _ensure_intervention_length(
                    np.asarray([treatment_sequence], dtype=float),
                    projection_horizon, len(synth_treatments_list),
                    'dataset.treatment_sequence',
                )
                gap = float(np.max(np.abs(cf_treatments_seq - expected)))
                if gap > 1e-8:
                    raise RuntimeError(
                        'Configured treatment_sequence was not materialized exactly: '
                        f'max_abs_gap={gap:.3e}.'
                    )
            logger.info(
                '[CF] fixed_benchmark master trajectories %s names=%s policy_sha1=%s '
                'dose_sha1=%s marginal_sha1=%s',
                cf_treatments_seq.shape, cf_names, cf_hash,
                cf_policy_meta['dose_trajectory_hash'],
                cf_policy_meta['positive_marginal_hash'],
            )
            print(
                f'[CF] formal fixed_benchmark master H{projection_horizon}: '
                f'shape={cf_treatments_seq.shape}, policies={cf_names}, '
                f'policy_sha1={cf_hash}', flush=True,
            )
        elif treatment_sequence is not None:
            cf_treatments_seq = _ensure_intervention_length(
                np.array([treatment_sequence], dtype=float), projection_horizon,
                len(synth_treatments_list), 'dataset.treatment_sequence',
            )
        elif use_clinical_cf:
            self.dose_traj_generator = DoseTrajectoryGenerator(
                projection_horizon=projection_horizon,
                n_treatments=len(synth_treatments_list),
            )
            if _dose_col is not None and _dose_col in intervention_train.columns:
                raw_train_doses = intervention_train[_dose_col].dropna().values
                train_normalized_doses = np.clip(
                    raw_train_doses / max(self.dose_scale, 1e-8), 0.0, 1.0,
                )
                self.dose_traj_generator.fit(train_normalized_doses)
                self.train_normalized_doses = train_normalized_doses
                self.train_raw_doses = raw_train_doses
                logger.info(
                    f'[CF] DoseTrajectoryGenerator fitted on train data:\n'
                    f'{self.dose_traj_generator.summary()}'
                )
            else:
                logger.warning('[CF] No real dose column — using fallback uniform fit')
                self.dose_traj_generator.fit(np.array([0.25, 0.50, 0.75]))
            cf_treatments_seq = _ensure_intervention_length(
                np.asarray(self.dose_traj_generator.generate_all(), dtype=float),
                projection_horizon, len(synth_treatments_list), 'DoseTrajectoryGenerator',
            )
            logger.info(
                f'[CF] Generated {cf_treatments_seq.shape[0]} clinical trajectories '
                f'(shape {cf_treatments_seq.shape})'
            )
        else:
            cf_treatments_seq = None

        self.cf_treatments_seq = cf_treatments_seq
        self.cf_trajectory_hash = cf_hash
        self.cf_policy_hash = cf_hash
        self.cf_policy_names = tuple(cf_names) if cf_names is not None else None
        self.cf_policy_metadata = cf_policy_meta
        self.cf_percentile_sequences = (
            None if cf_policy_meta is None else cf_policy_meta.get("policy_percentiles")
        )

        self.test_cf_one_step = MIMIC4SyntheticDataset(
            all_vitals_test, static_features_test,
            synth_outcomes_list, synth_treatments_list,
            treatment_outcomes_influence, 'test',
            mode='counterfactual_one_step',
            treatments_seq=None,
            dose_grid=dose_grid, dose_scale=self.dose_scale,
            untreated_scaling=controlled,
        )

        _oracle_rng_before_target = np.random.get_state()

        self.test_cf_treatment_seq = MIMIC4SyntheticDataset(
            all_vitals_test, static_features_test,
            synth_outcomes_list, synth_treatments_list,
            treatment_outcomes_influence, 'test',
            mode='counterfactual_treatment_seq',
            projection_horizon=projection_horizon,
            n_treatments_seq=(
                cf_treatments_seq.shape[0] if cf_treatments_seq is not None else n_treatments_seq
            ),
            treatments_seq=cf_treatments_seq,
            dose_grid=dose_grid, dose_scale=self.dose_scale,
            untreated_scaling=controlled,
            trajectory_names=cf_names,
        )

        _oracle_rng_after_target = np.random.get_state()
        self.test_cf_oracle_pair = None

        if cf_treatments_seq is not None and int(projection_horizon) > 0:
            cf_arr = np.asarray(cf_treatments_seq, dtype=float)
            if cf_arr.ndim != 3:
                raise RuntimeError(f'Cf_treatments_seq must be (J,K+1,D), got {cf_arr.shape}.')

            J = int(cf_arr.shape[0])
            zero = np.zeros_like(cf_arr[0])
            oracle_pair_seq = np.empty((2 * J,) + cf_arr.shape[1:], dtype=float)
            oracle_pair_names = []
            for j in range(J):
                oracle_pair_seq[2 * j] = cf_arr[j]
                oracle_pair_seq[2 * j + 1] = zero
                _nm = (cf_names[j] if cf_names is not None else f"policy_{j}")
                oracle_pair_names.extend([f"target:{_nm}", f"zero_ref:{_nm}"])

            try:
                np.random.set_state(_oracle_rng_before_target)
                self.test_cf_oracle_pair = MIMIC4SyntheticDataset(
                    all_vitals_test, static_features_test,
                    synth_outcomes_list, synth_treatments_list,
                    treatment_outcomes_influence, 'test_oracle_pair',
                    mode='counterfactual_treatment_seq',
                    projection_horizon=projection_horizon,
                    n_treatments_seq=int(oracle_pair_seq.shape[0]),
                    treatments_seq=oracle_pair_seq,
                    dose_grid=dose_grid, dose_scale=self.dose_scale,
                    untreated_scaling=controlled,
                    trajectory_names=oracle_pair_names,
                )
            finally:
                np.random.set_state(_oracle_rng_after_target)

            self.test_cf_oracle_pair.oracle_pair_layout = 'target_zero_interleaved'
            self.test_cf_oracle_pair.oracle_pair_num_targets = J
            self.test_cf_oracle_pair.oracle_pair_traj_count = 2 * J
            self.test_cf_oracle_pair.oracle_reference_type = (
                'zero_future_treatment_same_latent_history'
            )
            self.test_cf_oracle_pair.oracle_target_sequences = cf_arr.copy()
            self.test_cf_oracle_pair.oracle_pair_sequences = oracle_pair_seq.copy()

            self._validate_oracle_target_replication(
                self.test_cf_treatment_seq, self.test_cf_oracle_pair,
            )

            logger.info(
                'Built paired target/zero-reference test set: '
                f'J={J}, pair trajectories={2 * J}, N={len(self.test_cf_oracle_pair)}.'
            )
            print(
                'Paired test set ready: '
                f'J={J}, layout=target,zero interleaved, '
                f'N={len(self.test_cf_oracle_pair)}',
                flush=True,
            )
        else:
            logger.warning('Cf_treatments_seq is None; paired Oracle dataset not built.')

        self.projection_horizon = projection_horizon
        self.autoregressive = autoregressive
        self.has_vitals = True
        self.train_scaling_params = self.train_f.get_scaling_params()

        tr0 = synth_treatments_list[0]
        self.benchmark_metadata = {
            'treatment_assignment_mode': tr0.treatment_assignment_mode,
            'target_treat_rate': tr0.target_treat_rate,
            'dose_marginal_mode': tr0.dose_marginal_mode,
            'inherit_real_treat_rate': tr0.inherit_real_treat_rate,
            'real_marginal_meta': dict(tr0.real_marginal_meta),
            'positive_dose_mean_beta_legacy': tr0.positive_dose_mean,
            'dose_concentration_beta_legacy': tr0.dose_concentration,
            'dose_scale_current_disease_train_p99': float(self.dose_scale),
            'treatment_rho': tr0.treatment_rho,
            'dose_rho': tr0.dose_rho,
            'dose_marginal_conditional': tr0.dose_marginal_conditional,
            'history_outcome_source': tr0.history_outcome_source,
            'dose_response_shape': tr0.dose_response_shape,
            'benchmark_response_mode': tr0.benchmark_response_mode,
            'saturating_heterogeneity_mode': tr0.saturating_heterogeneity_mode,
            'saturating_heterogeneity_source': tr0.saturating_heterogeneity_source,
            'saturating_heterogeneity_strength': tr0.saturating_heterogeneity_strength,
            'saturating_heterogeneity_z_clip': tr0.saturating_heterogeneity_z_clip,
            'saturating_ed50_base': float(tr0.dr_params.get('ed50', 0.3)),
            'saturating_ed50_min': tr0.saturating_ed50_min,
            'saturating_ed50_max': tr0.saturating_ed50_max,
            'saturating_ed50_raw_min_theoretical': tr0.saturating_ed50_raw_min_theoretical,
            'saturating_ed50_raw_max_theoretical': tr0.saturating_ed50_raw_max_theoretical,
            'saturating_ed50_clipping_possible': tr0.saturating_ed50_clipping_possible,
            'saturating_endpoint_normalized_r0':
                0.0 if tr0.dose_response_shape == 'saturating' else None,
            'saturating_endpoint_normalized_r1':
                1.0 if tr0.dose_response_shape == 'saturating' else None,
            'effect_window': int(tr0.effect_window),
            'cf_trajectory_mode': cf_trajectory_mode,
            'cf_trajectory_hash': cf_hash,
            'cf_policy_hash': self.cf_policy_hash,
            'cf_policy_names': self.cf_policy_names,
            'cf_policy_space': (
                None if cf_policy_meta is None else cf_policy_meta.get('policy_space')
            ),
            'cf_percentile_policy_hash': (
                None if cf_policy_meta is None else cf_policy_meta.get('percentile_policy_hash')
            ),
            'cf_dose_trajectory_hash': (
                None if cf_policy_meta is None else cf_policy_meta.get('dose_trajectory_hash')
            ),
            'cf_positive_marginal_hash': (
                None if cf_policy_meta is None else cf_policy_meta.get('positive_marginal_hash')
            ),
            'cf_master_horizon': int(self.fixed_cf_master_horizon),
            'cf_horizon_semantics': 'Hh_is_prefix_of_single_master_HK_policy',
            'prediction_horizon': int(projection_horizon),
            'intervention_sequence_length': (
                int(cf_treatments_seq.shape[1]) if cf_treatments_seq is not None else None
            ),
            'history_score_n': (
                len(tr0.history_score_sorted) if tr0.history_score_sorted is not None else 0
            ),
            'implied_treat_rate': tr0.calibration_meta.get('implied_treat_rate'),
            'scaling_source': 'train_untreated' if controlled else 'train_observed',
        }
        if cf_trajectory_mode == FIXED_CF:
            self._finalize_cf_benchmark_contract()

    def get_cf_policy_prefix(self, horizon: int, *, percentile_space: bool = False):
        h = int(horizon)
        K = int(self.projection_horizon)
        if not (1 <= h <= K):
            raise ValueError(f'horizon must be in [1,{K}], got {h}.')
        source = self.cf_percentile_sequences if percentile_space else self.cf_treatments_seq
        if source is None:
            raise RuntimeError('Requested CF policy representation is unavailable.')
        return np.asarray(source)[:, :h + 1, :].copy()

    def _cf_subject_policy_view(self):
        ds = self.test_cf_treatment_seq
        if ds is None or self.cf_policy_names is None:
            raise RuntimeError('Formal CF benchmark dataset/policies are unavailable.')
        J = int(len(self.cf_policy_names))
        N = int(len(ds))
        if J <= 0 or N % J != 0:
            raise RuntimeError(f'CF dataset size {N} is incompatible with J={J} policies.')
        n_subjects = N // J

        pid = np.asarray(ds.data['patient_ids'])
        tid = np.asarray(ds.data['trajectory_ids'], dtype=np.int64)
        if pid.shape[0] != N or tid.shape[0] != N:
            raise RuntimeError('CF ID arrays do not match dataset length.')
        pid2 = pid.reshape(n_subjects, J)
        tid2 = tid.reshape(n_subjects, J)
        if not np.all(pid2 == pid2[:, :1]):
            raise RuntimeError('CF subject ordering is not subject-major across policies.')
        expected = np.arange(J, dtype=np.int64)[None, :]
        if not np.array_equal(tid2, np.repeat(expected, n_subjects, axis=0)):
            raise RuntimeError(
                'CF trajectory ordering is not [policy_0,...,policy_J-1] within subject.'
            )
        return ds, J, n_subjects, pid2, tid2

    def _finalize_cf_benchmark_contract(self):
        ds, J, n_subjects, pid2, tid2 = self._cf_subject_policy_view()
        K = int(self.projection_horizon)
        plans = np.asarray(self.cf_treatments_seq, dtype=np.float64)
        if plans.shape != (J, K + 1, len(ds.synthetic_treatments)):
            raise RuntimeError(
                f'CF plan tensor shape {plans.shape} != expected '
                f'({J},{K + 1},{len(ds.synthetic_treatments)}).'
            )

        seq = np.asarray(ds.data['sequence_lengths']).astype(np.int64)
        seq2 = seq.reshape(n_subjects, J)
        if not np.all(seq2 == seq2[:, :1]):
            raise RuntimeError('sequence_lengths differ across policies for a subject.')
        fixed_split = seq2[:, 0] - K
        pred_idx = fixed_split - 1
        if np.any(fixed_split <= 0):
            raise RuntimeError('At least one subject has no valid factual prefix before CF.')

        current = np.asarray(ds.data['current_treatments'])
        T = int(current.shape[1])
        current2 = current.reshape((n_subjects, J) + current.shape[1:])
        for i in range(n_subjects):
            s = int(pred_idx[i])
            if s < 0 or s + K >= T:
                raise RuntimeError(
                    f'Invalid query-plan window for subject row {i}: '
                    f'start={s}, K={K}, T={T}.'
                )
            actual = current2[i, :, s:s + K + 1, :]
            gap = float(np.max(np.abs(actual - plans)))
            if gap > 1e-8:
                raise RuntimeError(
                    f'CF intervention materialization mismatch for subject row {i}: '
                    f'max_abs_gap={gap:.3e}.'
                )

        history_fields = (
            'prev_treatments', 'vitals', 'prev_unscaled_outputs', 'active_entries',
        )
        history_worst = 0.0

        static = np.asarray(ds.data['static_features'])
        static2 = static.reshape((n_subjects, J) + static.shape[1:])
        for i in range(n_subjects):
            base_static = static2[i, 0, ...]
            for j in range(1, J):
                other_static = static2[i, j, ...]
                finite = np.isfinite(base_static) & np.isfinite(other_static)
                if finite.any():
                    gap = float(np.max(np.abs(base_static[finite] - other_static[finite])))
                    history_worst = max(history_worst, gap)
                    if gap > 1e-8:
                        raise RuntimeError(
                            f'Static features differ across policies: subject_row={i}, '
                            f'policy={j}, gap={gap:.3e}.'
                        )

        for key in history_fields:
            x = np.asarray(ds.data[key])
            x2 = x.reshape((n_subjects, J) + x.shape[1:])
            for i in range(n_subjects):
                stop = int(fixed_split[i])
                base = x2[i, 0, :stop, ...]
                for j in range(1, J):
                    other = x2[i, j, :stop, ...]
                    finite = np.isfinite(base) & np.isfinite(other)
                    if finite.any():
                        gap = float(np.max(np.abs(base[finite] - other[finite])))
                        history_worst = max(history_worst, gap)
                        if gap > 1e-8:
                            raise RuntimeError(
                                f'Factual history differs across policies: field={key}, '
                                f'subject_row={i}, policy={j}, gap={gap:.3e}.'
                            )

        outputs = np.asarray(ds.data['unscaled_outputs'])
        active = np.asarray(ds.data['active_entries'])
        outputs2 = outputs.reshape((n_subjects, J) + outputs.shape[1:])
        active2 = active.reshape((n_subjects, J) + active.shape[1:])
        Dout = int(outputs.shape[-1])
        oracle_y = np.full((n_subjects, J, K, Dout), np.nan, dtype=np.float64)
        valid = np.zeros((n_subjects, J, K, 1), dtype=bool)
        for i in range(n_subjects):
            for h in range(1, K + 1):
                idx = int(fixed_split[i] + h - 1)
                if idx < 0 or idx >= outputs.shape[1]:
                    continue
                oracle_y[i, :, h - 1, :] = outputs2[i, :, idx, :]
                valid[i, :, h - 1, :] = active2[i, :, idx, :] > 0.5

        subject_ids = pid2[:, 0]
        cohort_hash = _sha1_named_arrays(
            subject_ids=np.asarray(subject_ids).astype(str),
            decision_indices=pred_idx,
            sequence_lengths=seq2[:, 0],
        )
        dgp_hash = _sha1_json_dict(dict(self.benchmark_metadata))
        oracle_hash = _sha1_named_arrays(
            policy_hash=np.asarray([self.cf_policy_hash], dtype='U'),
            cohort_hash=np.asarray([cohort_hash], dtype='U'),
            dgp_hash=np.asarray([dgp_hash], dtype='U'),
            oracle_y=np.nan_to_num(oracle_y, nan=0.0),
            oracle_valid=valid,
        )

        self.cf_dgp_hash = dgp_hash
        self.cf_cohort_hash = cohort_hash
        self.cf_oracle_dataset_hash = oracle_hash
        self.cf_benchmark_subject_ids = subject_ids.copy()
        self.cf_benchmark_decision_indices = pred_idx.copy()
        self.cf_benchmark_oracle_y = oracle_y
        self.cf_benchmark_valid_mask = valid
        self.cf_benchmark_integrity_status = 'PASS'
        self.cf_benchmark_history_max_gap = float(history_worst)

        self.benchmark_metadata.update({
            'cf_dgp_hash': dgp_hash,
            'cf_cohort_hash': cohort_hash,
            'cf_oracle_dataset_hash': oracle_hash,
            'cf_benchmark_integrity_status': 'PASS',
            'cf_factual_history_max_policy_gap': float(history_worst),
            'cf_n_subjects': int(n_subjects),
            'cf_n_policies': int(J),
            'cf_policy_order': tuple(self.cf_policy_names),
        })
        print(
            '[CF integrity] PASS: '
            f'n_subjects={n_subjects}, policies={self.cf_policy_names}, '
            f'policy_hash={self.cf_policy_hash}, cohort_hash={cohort_hash}, '
            f'oracle_hash={oracle_hash}', flush=True,
        )

    @staticmethod
    def _validate_oracle_target_replication(main_cf, oracle_pair, atol=1e-8):
        J = int(getattr(oracle_pair, 'oracle_pair_num_targets', 0))
        P = int(getattr(oracle_pair, 'oracle_pair_traj_count', 0))
        if J <= 0 or P != 2 * J:
            raise RuntimeError(f'[Oracle] invalid pair metadata: J={J}, trajectories={P}.')

        n_main = len(main_cf)
        n_pair = len(oracle_pair)
        if n_main % J != 0 or n_pair % P != 0:
            raise RuntimeError(
                f'[Oracle] dataset sizes incompatible with trajectory layout: '
                f'main={n_main}, pair={n_pair}, J={J}, P={P}.'
            )
        n_subjects_main = n_main // J
        n_subjects_pair = n_pair // P
        if n_subjects_main != n_subjects_pair:
            raise RuntimeError(
                f'[Oracle] subject-count mismatch: main={n_subjects_main}, '
                f'pair={n_subjects_pair}.'
            )

        keys = [
            'sequence_lengths', 'prev_treatments', 'current_treatments',
            'vitals', 'static_features', 'active_entries',
            'unscaled_outputs', 'prev_unscaled_outputs',
        ]
        checked = []
        worst = 0.0
        for key in keys:
            if key not in main_cf.data or key not in oracle_pair.data:
                continue
            a = np.asarray(main_cf.data[key])
            b = np.asarray(oracle_pair.data[key])
            if a.shape[0] != n_main or b.shape[0] != n_pair:
                continue
            a3 = a.reshape((n_subjects_main, J) + a.shape[1:])
            b3 = b.reshape((n_subjects_pair, P) + b.shape[1:])
            b_target = b3[:, 0::2, ...]
            if a3.shape != b_target.shape:
                raise RuntimeError(
                    f'[Oracle] raw target shape mismatch for {key}: '
                    f'{a3.shape} vs {b_target.shape}.'
                )
            if np.issubdtype(a3.dtype, np.number):
                finite = np.isfinite(a3) & np.isfinite(b_target)
                if finite.any():
                    gap = float(np.max(np.abs(a3[finite] - b_target[finite])))
                    worst = max(worst, gap)
                    if gap > atol:
                        raise RuntimeError(
                            f'[Oracle] target replication FAILED for {key}: '
                            f'max_abs_gap={gap:.3e} > {atol:.1e}.'
                        )
                nan_same = (
                    np.array_equal(np.isnan(a3), np.isnan(b_target))
                    if np.issubdtype(a3.dtype, np.floating) else True
                )
                if not nan_same:
                    raise RuntimeError(
                        f'[Oracle] target replication FAILED for {key}: NaN mask differs.'
                    )
            else:
                if not np.array_equal(a3, b_target):
                    raise RuntimeError(
                        f'[Oracle] target replication FAILED for nonnumeric {key}.'
                    )
            checked.append(key)

        if not checked:
            raise RuntimeError('[Oracle] no raw fields were available for replication check.')

        oracle_pair.oracle_target_replication_status = 'PASS'
        oracle_pair.oracle_target_replication_max_gap = worst
        oracle_pair.oracle_target_replication_fields = tuple(checked)
        logger.info(
            '[Oracle] target replication PASS; fields=%s; max_abs_gap=%.3e', checked, worst,
        )

    def process_data_multi(self):
        super().process_data_multi()
        pair = getattr(self, 'test_cf_oracle_pair', None)
        if pair is not None:
            pair.process_data(self.train_scaling_params)
            pair.process_sequential_test(self.projection_horizon)
            pair.process_sequential_multi(self.projection_horizon)
            logger.info(
                '[Oracle] processed paired dataset for multi-step validation: N=%d', len(pair),
            )

    @staticmethod
    def _calibrate_history_scores(all_vitals_train, static_features_train,
                                  synth_outcomes_list, synth_treatments_list, seed):

        np.random.seed(seed)
        calib_vitals = all_vitals_train.copy()
        for outcome in synth_outcomes_list:
            outcome.simulate_untreated(calib_vitals, static_features_train)
        for treatment in synth_treatments_list:
            treatment.fit_history_score_distribution(calib_vitals)
        np.random.seed(seed)

    def normalize_vitals(self, df):
        if df is None:
            return None
        df = df.copy()
        df[self.vital_cols] = (df[self.vital_cols] - self.train_mean) / self.train_std
        return df