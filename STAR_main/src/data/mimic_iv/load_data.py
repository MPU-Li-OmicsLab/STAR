import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from omegaconf import ListConfig

from src import ROOT_PATH

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Static-feature preprocessing
# -----------------------------------------------------------------------------

def _is_continuous_static(series: pd.Series) -> bool:
    """Match the legacy behavior as closely as possible: float columns are continuous."""
    return pd.api.types.is_float_dtype(series.dtype)


def fit_static_preprocessor(static_features: pd.DataFrame, drop_first: bool = False) -> Dict:
    """
    Fit static-feature preprocessing on TRAIN only.

    Continuous (float) variables:
        TRAIN-mean imputation, then TRAIN mean/std standardization.
    Other variables:
        TRAIN-mode imputation, then one-hot encoding using TRAIN categories only.

    The returned object is deliberately plain-Python so it can be logged/serialized.
    """
    params = {
        "drop_first": bool(drop_first),
        "features": [],
        "output_columns": [],
    }

    for feature in static_features.columns:
        s = static_features[feature]
        if _is_continuous_static(s):
            mean = float(pd.to_numeric(s, errors="coerce").mean())
            if not np.isfinite(mean):
                mean = 0.0
            std = float(pd.to_numeric(s, errors="coerce").std(ddof=0))
            if not np.isfinite(std) or std <= 0.0:
                std = 1.0

            spec = {
                "name": feature,
                "kind": "continuous",
                "mean": mean,
                "std": std,
                "output_columns": [feature],
            }
        else:
            nonmissing = s.dropna()
            if len(nonmissing) == 0:
                mode = 0
                categories = [0]
            else:
                mode = nonmissing.mode(dropna=True)
                mode = mode.iloc[0] if len(mode) else nonmissing.iloc[0]
                # Deterministic order; fall back to appearance order if values
                # are not mutually sortable.
                unique_vals = list(pd.unique(nonmissing))
                try:
                    categories = sorted(unique_vals)
                except TypeError:
                    categories = unique_vals

            used_categories = categories[1:] if drop_first and len(categories) > 0 else categories
            output_columns = [f"{feature}__{cat}" for cat in used_categories]
            spec = {
                "name": feature,
                "kind": "categorical",
                "mode": mode,
                "categories": categories,
                "used_categories": used_categories,
                "output_columns": output_columns,
            }

        params["features"].append(spec)
        params["output_columns"].extend(spec["output_columns"])

    return params


def apply_static_preprocessor(static_features: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Apply a TRAIN-fitted static preprocessor to any split."""
    pieces = []

    for spec in params["features"]:
        feature = spec["name"]
        if feature not in static_features.columns:
            raise KeyError(f"Static feature {feature!r} is missing at transform time.")

        s = static_features[feature]
        if spec["kind"] == "continuous":
            x = pd.to_numeric(s, errors="coerce").fillna(spec["mean"])
            x = (x - spec["mean"]) / spec["std"]
            pieces.append(pd.DataFrame({feature: x.astype(float)}, index=static_features.index))
        else:
            x = s.fillna(spec["mode"])
            cat = pd.Categorical(x, categories=spec["categories"])
            dummies = pd.get_dummies(cat)
            dummies.index = static_features.index

            out = pd.DataFrame(index=static_features.index)
            for cat_value, col_name in zip(spec["used_categories"], spec["output_columns"]):
                if cat_value in dummies.columns:
                    out[col_name] = dummies[cat_value].astype(float)
                else:
                    out[col_name] = 0.0
            pieces.append(out)

    if not pieces:
        return pd.DataFrame(index=static_features.index)

    ans = pd.concat(pieces, axis=1)
    return ans.reindex(columns=params["output_columns"], fill_value=0.0)


def process_static_features(static_features: pd.DataFrame, drop_first: bool = False) -> pd.DataFrame:
    """
    Backward-compatible convenience wrapper.

    For leakage-safe experiments, fit on TRAIN with fit_static_preprocessor() and
    apply to validation/test with apply_static_preprocessor().
    """
    params = fit_static_preprocessor(static_features, drop_first=drop_first)
    return apply_static_preprocessor(static_features, params)


# -----------------------------------------------------------------------------
# H5 helpers
# -----------------------------------------------------------------------------

def _restore_state_placeholders(h5: pd.HDFStore,
                                all_vitals: pd.DataFrame,
                                requested_columns: List[str]) -> pd.DataFrame:
    """
    Restore ONLY unresolved H5 storage placeholders to NaN in memory.

    /state_valid_mask is written after within-stay past-only state completion and
    before the final 0.0 storage fill. Therefore mask==0 means the stored zero is
    only a placeholder (for example, a leading value with no previous observation
    or a variable absent throughout the stay), whereas valid observed/past-ffilled
    values are retained.

    The H5 itself remains completely NaN-free.
    """
    key = "/state_valid_mask"
    if key not in h5.keys():
        logger.warning(
            "%s not found in H5; unresolved zero placeholders cannot be distinguished "
            "from true zeros. Regenerate H5 with the revised build_h5_sepsis.py.",
            key,
        )
        return all_vitals

    mask = h5[key][requested_columns].copy()
    if not mask.index.equals(all_vitals.index):
        mask = mask.reindex(all_vitals.index)
    if mask.isna().any().any():
        raise RuntimeError("/state_valid_mask is misaligned with /vitals_labs_mean.")

    return all_vitals.mask(mask.astype(bool) == 0)


def _restore_static_placeholders(h5: pd.HDFStore,
                                 static_features: pd.DataFrame,
                                 requested_columns: List[str]) -> pd.DataFrame:
    key = "/static_valid_mask"
    if key not in h5.keys():
        return static_features

    mask = h5[key][requested_columns].copy()
    if not mask.index.equals(static_features.index):
        mask = mask.reindex(static_features.index)
    if mask.isna().any().any():
        raise RuntimeError("/static_valid_mask is misaligned with /patients.")

    return static_features.mask(mask.astype(bool) == 0)


def _restore_intervention_placeholders(h5: pd.HDFStore,
                                       intervention: pd.DataFrame,
                                       requested_columns: List[str]) -> pd.DataFrame:
    """Restore invalid intervention storage placeholders to NaN.

    The strict past-only H5 is deliberately NaN-free. True untreated action rows
    therefore contain a genuine dose of 0, while two kinds of non-actions/unknown
    doses are also stored as 0 only for serialization:

      * the terminal boundary (H24), for which no A24 exists;
      * NE-on rows whose dose cannot be recovered from an earlier observed dose
        in the same contiguous treatment run.

    /intervention_valid_mask distinguishes these cases. Only mask==0 entries are
    restored to NaN; genuine untreated zeros (mask==1) remain zero. This is
    essential before estimating TRAIN treatment prevalence, dose percentiles, or
    the disease-specific positive-dose marginal.

    The final STAR benchmark requires this mask. Failing fast here prevents an old
    H5 from silently treating terminal/unresolved placeholders as untreated doses.
    """
    key = "/intervention_valid_mask"
    if key not in h5.keys():
        raise RuntimeError(
            f"{key} not found in H5. Regenerate the H5 with the strict past-only "
            "build_h5_sepsis.py before running the final STAR benchmark."
        )

    mask_table = h5[key]
    missing_columns = [c for c in requested_columns if c not in mask_table.columns]
    if missing_columns:
        raise RuntimeError(
            f"{key} does not contain required treatment columns {missing_columns}. "
            f"Available columns: {list(mask_table.columns)}"
        )

    mask = mask_table[requested_columns].copy()
    if not mask.index.equals(intervention.index):
        mask = mask.reindex(intervention.index)
    if mask.isna().any().any():
        raise RuntimeError(
            "/intervention_valid_mask is misaligned with /interventions."
        )

    restored = intervention.mask(mask.astype(bool) == 0)

    n_invalid = int((mask.astype(bool) == 0).to_numpy().sum())
    logger.info(
        "Restored %d invalid intervention storage placeholders to NaN using %s.",
        n_invalid, key,
    )
    return restored


def _promote_stay_to_subject_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the 'subject_id' index level's VALUES with 'icustay_id', instead of
    dropping icustay_id (and hadm_id) outright via droplevel().

    Why: a single subject_id can span multiple hospital admissions / ICU stays.
    The previous `df.droplevel(["hadm_id", "icustay_id"])` kept subject_id as the
    sole outer key, so different stays belonging to the same subject_id collapse
    onto overlapping remaining index values (e.g. the same hours_in range
    appearing twice). That produces duplicated (subject_id, hours_in) rows.

    Everything downstream (semi_synthetic_dataset.py) treats the outer index
    level — named 'subject_id' — as a unique per-trajectory key: it groups by
    `level='subject_id'`, does `.loc[patient_ix]` to pull exactly one patient's
    rows, and indexes a single time point with `.loc[t, cols]` expecting exactly
    one row back. Duplicated rows there don't raise immediately — they silently
    inflate feature counts several call-frames later (e.g. an RBFSampler fit on
    8 features suddenly seeing 32).

    Rather than touching that downstream code, this function keeps its contract
    true at the source: icustay_id (a genuine 1:1 stay/trajectory key) becomes
    the value stored under the 'subject_id' index level name. The true
    subject_id is NOT retained as a data column here — several downstream call
    sites take a table's ENTIRE .columns as "the feature set" (e.g.
    `self.vital_cols = all_vitals_train.columns` before `.mean(axis=0)`, and
    `static_feature_columns = list(static_features.columns)`), so any extra
    column — especially a non-numeric ID — corrupts those aggregations or gets
    fed into the model as a spurious feature. Only the index changes; the set
    of data columns is left exactly as it was.

    Works whether df carries a time level (e.g. hours_in, for vitals/
    interventions) or not (e.g. static_features with one row per stay).

    Raises if duplicates remain after promotion — that would indicate a genuine
    data-quality issue upstream (duplicate icustay_id/time rows in the H5)
    rather than the multi-admission collision this function fixes.
    """
    names = list(df.index.names)
    if "subject_id" not in names or "icustay_id" not in names:
        raise KeyError(
            f"Expected 'subject_id' and 'icustay_id' in index levels, got {names}."
        )

    stay = df.index.get_level_values("icustay_id")
    other_levels = [lv for lv in names if lv not in ("subject_id", "hadm_id", "icustay_id")]

    out = df.reset_index(drop=True)
    out["subject_id"] = stay.values  # icustay_id now plays the 'subject_id' role
    for lv in other_levels:
        out[lv] = df.index.get_level_values(lv).values

    out = out.set_index(["subject_id"] + other_levels).sort_index()

    n_dupes = int(out.index.duplicated().sum())
    if n_dupes:
        raise RuntimeError(
            f"{n_dupes} duplicated index rows remain after promoting icustay_id "
            "to the subject_id level. This points to genuine duplicate rows in "
            "the source H5 for the same stay/time — investigate the H5 directly "
            "rather than the multi-admission subject_id collision this function "
            "guards against."
        )

    return out


# -----------------------------------------------------------------------------
# Real-data loader
# -----------------------------------------------------------------------------

def load_mimic4_data_processed(data_path: str,
                               min_seq_length: int = None,
                               max_seq_length: int = None,
                               treatment_list: List[str] = None,
                               outcome_list: List[str] = None,
                               vital_list: List[str] = None,
                               static_list: List[str] = None,
                               max_number: int = None,
                               data_seed: int = 42,
                               drop_first: bool = False,
                               **kwargs) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
                 pd.DataFrame, None]:
    """
    Load MIMIC-IV data for REAL-DATA experiments without cohort-wide scaling.

    Important: train/validation/test splitting and all fitted preprocessing
    statistics are performed later by MIMIC4RealDatasetCollection so that
    validation/test records do not enter means, standard deviations, static
    preprocessing parameters, or outcome scaling.

    Returns the legacy six-tuple for compatibility; scaling_params is None here
    and is fitted on TRAIN by the dataset collection.
    """
    del drop_first  # preprocessing is intentionally deferred until after split

    logger.info("Loading MIMIC-IV real dataset from %s.", data_path)
    h5 = pd.HDFStore(data_path, "r")

    if treatment_list is None:
        treatment_list = ["vaso", "vent"]
    if outcome_list is None:
        outcome_list = ["diastolic blood pressure", "oxygen saturation"]
    else:
        outcome_list = ListConfig(list(outcome_list))
    if vital_list is None:
        vital_list = [
            "heart rate", "red blood cell count", "sodium", "mean blood pressure",
            "systemic vascular resistance", "glucose", "chloride urine",
            "glascow coma scale total", "hematocrit",
            "positive end-expiratory pressure set", "respiratory rate",
            "prothrombin time pt", "cholesterol", "hemoglobin", "creatinine",
            "blood urea nitrogen", "bicarbonate", "calcium ionized",
            "partial pressure of carbon dioxide", "magnesium", "anion gap",
            "phosphorous", "venous pvo2", "platelets", "calcium urine",
        ]
    if static_list is None:
        static_list = ["gender", "ethnicity", "age"]

    state_cols = list(outcome_list) + list(vital_list)
    treatments = h5["/interventions"][treatment_list]
    all_vitals = h5["/vitals_labs_mean"][state_cols]
    static_features = h5["/patients"][static_list]

    # H5 is NaN-free on disk. Restore invalid serialization placeholders in
    # memory before any split, scaling, treatment selection, or loss masking.
    treatments = _restore_intervention_placeholders(
        h5, treatments, list(treatment_list)
    )
    all_vitals = _restore_state_placeholders(h5, all_vitals, state_cols)
    static_features = _restore_static_placeholders(h5, static_features, list(static_list))

    # NOTE: previously `df.droplevel(["hadm_id", "icustay_id"])`, which kept
    # subject_id as the sole outer key and silently collided rows for any
    # subject_id spanning multiple admissions/stays. Promote icustay_id to the
    # 'subject_id' slot instead — see _promote_stay_to_subject_level docstring.
    treatments = _promote_stay_to_subject_level(treatments)
    all_vitals = _promote_stay_to_subject_level(all_vitals)
    static_features = _promote_stay_to_subject_level(static_features)

    all_vitals.columns = [c if isinstance(c, str) else c[0] for c in all_vitals.columns]

    # No loader-level ffill/bfill here. H5 construction owns the imputation
    # semantics, and repeating fill after dropping stay identifiers could cross
    # patient/stay boundaries in less constrained cohorts.

    user_sizes = all_vitals.groupby("subject_id").size()
    filtered_users = (
        user_sizes.index[user_sizes >= min_seq_length]
        if min_seq_length is not None else user_sizes.index
    )
    if max_number is not None:
        np.random.seed(data_seed)
        filtered_users = np.random.choice(filtered_users, size=max_number, replace=False)

    treatments = treatments.loc[filtered_users]
    all_vitals = all_vitals.loc[filtered_users]
    static_features = static_features[static_features.index.isin(filtered_users)]

    if max_seq_length is not None:
        treatments = treatments.groupby("subject_id").head(max_seq_length)
        all_vitals = all_vitals.groupby("subject_id").head(max_seq_length)

    logger.info("Number of patients filtered: %d.", len(filtered_users))

    outcomes_unscaled = all_vitals[list(outcome_list)].copy()
    outcomes = outcomes_unscaled.copy()
    vitals = all_vitals[list(vital_list)].copy()

    h5.close()
    return treatments, outcomes, vitals, static_features, outcomes_unscaled, None


# -----------------------------------------------------------------------------
# Semi-synthetic loader
# -----------------------------------------------------------------------------

def load_mimic4_data_raw(data_path: str,
                         min_seq_length: int = None,
                         max_seq_length: int = None,
                         max_number: int = None,
                         treatment_list: List[str] = None,
                         vital_list: List[str] = None,
                         static_list: List[str] = None,
                         data_seed: int = 42,
                         drop_first: bool = False,
                         **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load MIMIC-IV covariate substrate for semi-synthetic experiments.

    No scaling and no static preprocessing is fitted here. The synthetic dataset
    collection first splits patients, then fits all preprocessing on TRAIN only.
    The H5 remains NaN-free; unresolved storage placeholders are restored to NaN
    only in memory and are neutralized after TRAIN-based standardization.
    """
    del drop_first

    logger.info("Loading MIMIC-IV semi-synthetic substrate from %s.", data_path)
    h5 = pd.HDFStore(data_path, "r")

    if treatment_list is None:
        treatment_list = ["a_ne_rate_hr_mean"]
    if vital_list is None:
        vital_list = [
            "heart rate", "red blood cell count", "sodium", "mean blood pressure",
            "systemic vascular resistance", "glucose", "chloride urine",
            "glascow coma scale total", "hematocrit",
            "positive end-expiratory pressure set", "respiratory rate",
            "prothrombin time pt", "cholesterol", "hemoglobin", "creatinine",
            "blood urea nitrogen", "bicarbonate", "calcium ionized",
            "partial pressure of carbon dioxide", "magnesium", "anion gap",
            "phosphorous", "venous pvo2", "platelets", "calcium urine",
        ]
    if static_list is None:
        static_list = ["gender", "ethnicity", "age"]

    intervention = h5["/interventions"][treatment_list]
    all_vitals = h5["/vitals_labs_mean"][vital_list]
    static_features = h5["/patients"][static_list]

    # Restore H24/unresolved treatment placeholders before TRAIN dose statistics.
    # Genuine untreated zeros remain 0 because their validity mask equals 1.
    intervention = _restore_intervention_placeholders(
        h5, intervention, list(treatment_list)
    )
    all_vitals = _restore_state_placeholders(h5, all_vitals, list(vital_list))
    static_features = _restore_static_placeholders(h5, static_features, list(static_list))

    # NOTE: previously `df.droplevel(["hadm_id", "icustay_id"])` — see
    # _promote_stay_to_subject_level docstring for why that silently produced
    # duplicated (subject_id, hours_in) rows for multi-admission subjects.
    intervention = _promote_stay_to_subject_level(intervention)
    all_vitals = _promote_stay_to_subject_level(all_vitals)
    static_features = _promote_stay_to_subject_level(static_features)

    all_vitals.columns = [c if isinstance(c, str) else c[0] for c in all_vitals.columns]

    user_sizes = all_vitals.groupby("subject_id").size()
    filtered_users = (
        user_sizes.index[user_sizes >= min_seq_length]
        if min_seq_length is not None else user_sizes.index
    )
    if max_number is not None:
        np.random.seed(data_seed)
        filtered_users = np.random.choice(filtered_users, size=max_number, replace=False)

    intervention = intervention.loc[filtered_users]
    all_vitals = all_vitals.loc[filtered_users]
    static_features = static_features.loc[filtered_users]

    if max_seq_length is not None:
        all_vitals = all_vitals.groupby("subject_id").head(max_seq_length)
        intervention = intervention.groupby("subject_id").head(max_seq_length)

    logger.info("Number of patients filtered: %d.", len(filtered_users))
    h5.close()
    return intervention, all_vitals, static_features


if __name__ == "__main__":
    data_path = ROOT_PATH + "/" + "data/processed/all_hourly_data.h5"
    treatments, outcomes, vitals, stat_features, outcomes_unscaled, scaling_params = \
        load_mimic4_data_processed(data_path, min_seq_length=100, max_seq_length=100)
    logger.info(
        "Loaded raw real-data tables: treatments=%s outcomes=%s vitals=%s static=%s; "
        "scaling_params=%s (fitted later on TRAIN).",
        treatments.shape, outcomes.shape, vitals.shape, stat_features.shape, scaling_params,
    )