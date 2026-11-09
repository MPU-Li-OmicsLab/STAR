import logging
import os

import hydra
import numpy as np
import pandas as pd
import setproctitle
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader

from src.models.utils import FilteringMlFlowLogger


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_default_dtype(torch.double)
torch.set_float32_matmul_precision("high")
setproctitle.setproctitle("IGCNET")


def _append_horizon_result(path: str, row: dict) -> None:
    """Append one horizon's final performance/budget metadata to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([row])
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    df.to_csv(path, mode="a", header=not exists, index=False)


def _remove_previous_run_csvs(run_output_path: str) -> None:
    """
    Avoid accidentally appending a fresh rerun to an old horizon_results.csv.
    Mechanism trace files live under model.igcnet.mechanism.output_dir/run_id
    and are managed by TargetDriftMonitor itself.
    """
    horizon_csv = os.path.join(run_output_path, "horizon_results.csv")
    if os.path.exists(horizon_csv):
        os.remove(horizon_csv)


@hydra.main(
    version_base=None,
    config_name="igcnet.yaml",
    config_path="./config/map/",
)
def main(args: DictConfig):

    results = {}

    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)

    phase_b_output_dir = getattr(args.exp, "phase_b_output_dir", None)
    if phase_b_output_dir is None:
        raise ValueError("args.exp.phase_b_output_dir must be set.")

    phase_b_output_dir = os.path.join(
        phase_b_output_dir,
        f"run_seed{args.exp.seed}",
    )
    os.makedirs(phase_b_output_dir, exist_ok=True)
    run_output_path = os.path.abspath(phase_b_output_dir)

    _remove_previous_run_csvs(run_output_path)

    args.exp.mlflow_uri = "file://" + os.path.abspath(
        os.path.join(run_output_path, "mlruns")
    )

    logger.info("=" * 80)
    logger.info("IGC-Net ORIGINAL mechanism-validation run")
    logger.info("dataset.seed=%s", args.dataset.seed)
    logger.info("exp.seed=%s", args.exp.seed)
    logger.info("max projection horizon=%s", args.dataset.projection_horizon)
    logger.info("epochs for every model=%s", args.exp.max_epochs)
    logger.info("model target=%s", args.model.igcnet._target_)
    logger.info("mechanism enabled=%s", args.model.igcnet.mechanism.enabled)
    logger.info("mechanism run_id=%s", args.model.igcnet.mechanism.run_id)
    logger.info("strategy=%s", args.model.igcnet.mechanism.strategy)
    logger.info("probe=%s", args.model.igcnet.mechanism.probe_cache_path)
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Dataset: keep dataset seed fixed across optimization seeds
    # ------------------------------------------------------------------
    seed_everything(args.dataset.seed)
    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()

    args.model.dim_outcomes = dataset_collection.train_f.data["outputs"].shape[-1]
    args.model.dim_treatments = dataset_collection.train_f.data[
        "current_treatments"
    ].shape[-1]
    args.model.dim_vitals = (
        dataset_collection.train_f.data["vitals"].shape[-1]
        if dataset_collection.has_vitals
        else 0
    )
    args.model.dim_static_features = dataset_collection.train_f.data[
        "static_features"
    ].shape[-1]

    # Model initialization / optimization seed.
    seed_everything(args.exp.seed)

    # ------------------------------------------------------------------
    # Logger / callbacks
    # ------------------------------------------------------------------
    igcnet_callbacks = []
    experiment_name = f"{args.model.name}/{args.dataset.name}"

    if args.exp.logging:
        mlf_logger = FilteringMlFlowLogger(
            filter_submodels=[],
            experiment_name=experiment_name,
            tracking_uri=args.exp.mlflow_uri,
            run_name="0",
        )
        igcnet_callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    else:
        mlf_logger = None

    # ==================================================================
    # 1) projection_horizon = 0 factual/one-step model
    # ==================================================================
    args.model.igcnet.projection_horizon = 0

    igcnetmodel = instantiate(
        args.model.igcnet,
        args,
        dataset_collection,
        path=run_output_path,
        _recursive_=False,
    )

    if args.model.igcnet.tune_hparams:
        igcnetmodel.finetune(
            resources_per_trial=args.model.igcnet.resources_per_trial
        )

    encoder_trainer = Trainer(
        devices=eval(str(args.exp.gpus)),
        logger=mlf_logger,
        max_epochs=int(args.exp.max_epochs),
        callbacks=igcnet_callbacks,
        gradient_clip_val=args.model.igcnet.max_grad_norm,
        gradient_clip_algorithm="norm",
    )
    encoder_trainer.fit(igcnetmodel)

    val_dataloader = DataLoader(
        dataset_collection.val_f,
        batch_size=args.dataset.val_batch_size,
        shuffle=False,
    )
    encoder_trainer.test(igcnetmodel, dataloaders=val_dataloader)

    val_rmse_orig, val_rmse_all = igcnetmodel.get_normalised_masked_rmse(
        dataset_collection.val_f
    )
    logger.info(
        "Val normalised RMSE (all): %s; Val normalised RMSE (orig): %s",
        val_rmse_all,
        val_rmse_orig,
    )

    encoder_results = {}
    if hasattr(dataset_collection, "test_cf_one_step"):
        test_rmse_orig, test_rmse_all, test_rmse_last = (
            igcnetmodel.get_normalised_masked_rmse(
                dataset_collection.test_cf_one_step,
                one_step_counterfactual=True,
            )
        )
        logger.info(
            "Test normalised RMSE (all): %s; Test normalised RMSE (orig): %s; "
            "Test normalised RMSE (only counterfactual): %s",
            test_rmse_all,
            test_rmse_orig,
            test_rmse_last,
        )
        encoder_results = {
            "encoder_val_rmse_all": val_rmse_all,
            "encoder_val_rmse_orig": val_rmse_orig,
            "encoder_test_rmse_all": test_rmse_all,
            "encoder_test_rmse_orig": test_rmse_orig,
            "encoder_test_rmse_last": test_rmse_last,
        }
    elif hasattr(dataset_collection, "test_f"):
        test_rmse_orig, test_rmse_all = igcnetmodel.get_normalised_masked_rmse(
            dataset_collection.test_f
        )
        logger.info(
            "Test normalised RMSE (all): %s; Test normalised RMSE (orig): %s",
            test_rmse_all,
            test_rmse_orig,
        )
        encoder_results = {
            "encoder_val_rmse_all": val_rmse_all,
            "encoder_val_rmse_orig": val_rmse_orig,
            "encoder_test_rmse_all": test_rmse_all,
            "encoder_test_rmse_orig": test_rmse_orig,
        }

    if args.exp.logging:
        mlf_logger.log_metrics(encoder_results)
    results.update(encoder_results)

    # ==================================================================
    # 2) Original IGC decoders: K = 1 ... Kmax
    #    Every horizon uses the ORIGINAL exp.max_epochs.
    # ==================================================================
    max_horizon = int(args.dataset.projection_horizon)
    horizon_csv = os.path.join(run_output_path, "horizon_results.csv")

    for t in range(1, max_horizon + 1):
        seed_everything(args.exp.seed)

        logger.info("#" * 80)
        logger.info(
            "Training ORIGINAL IGC decoder K=%d for %d epochs",
            t,
            int(args.exp.max_epochs),
        )
        logger.info("#" * 80)

        if args.exp.logging:
            decoder_logger = FilteringMlFlowLogger(
                filter_submodels=[],
                experiment_name=experiment_name,
                tracking_uri=args.exp.mlflow_uri,
                run_name=str(t),
            )
        else:
            decoder_logger = None

        args.model.igcnet.projection_horizon = t

        igcnetmodel = instantiate(
            args.model.igcnet,
            args,
            dataset_collection,
            path=run_output_path,
            _recursive_=False,
        )

        decoder_callbacks = list(igcnet_callbacks)
        decoder_trainer = Trainer(
            devices=eval(str(args.exp.gpus)),
            logger=decoder_logger,
            max_epochs=int(args.exp.max_epochs),
            callbacks=decoder_callbacks,
            gradient_clip_val=args.model.igcnet.max_grad_norm,
            gradient_clip_algorithm="norm",
        )
        decoder_trainer.fit(igcnetmodel)

        # Mechanism validation does not require checkpoints.
        save_decoder_ckpt = bool(getattr(args.exp, "save_decoder_ckpt", False))
        if save_decoder_ckpt:
            decoder_ckpt = os.path.join(
                run_output_path,
                f"igcnet_decoder_h{t}.ckpt",
            )
            decoder_trainer.save_checkpoint(decoder_ckpt)

        # --------------------------------------------------------------
        # Counterfactual evaluation
        # --------------------------------------------------------------
        test_loader = DataLoader(
            dataset_collection.test_cf_treatment_seq,
            batch_size=args.dataset.val_batch_size,
            shuffle=False,
        )

        # decoder_trainer.validate(igcnetmodel)
        decoder_trainer.test(igcnetmodel, test_loader)

        if hasattr(dataset_collection, "test_cf_treatment_seq"):
            test_dataset = dataset_collection.test_cf_treatment_seq
        elif hasattr(dataset_collection, "test_f_multi"):
            test_dataset = dataset_collection.test_f_multi
        else:
            raise RuntimeError(
                "Neither test_cf_treatment_seq nor test_f_multi is available."
            )

        test_rmse = igcnetmodel.get_normalised_n_step_rmses(test_dataset)
        test_rmses = {f"{t + 1}-step": test_rmse}
        logger.info(
            "Test normalised RMSE (n-step prediction), K=%d: %s",
            t,
            test_rmses,
        )

        # --------------------------------------------------------------
        # Compact run-level CSV: performance + actual training budget
        # --------------------------------------------------------------
        mechanism_monitor = getattr(igcnetmodel, "mechanism_monitor", None)
        model_label = getattr(mechanism_monitor, "model_label", "IGC-Net")
        mechanism_kind = getattr(mechanism_monitor, "mechanism_kind", "igc")
        strategy_label = getattr(
            mechanism_monitor,
            "strategy",
            str(args.model.igcnet.mechanism.strategy),
        )

        _append_horizon_result(
            horizon_csv,
            {
                "model": model_label,
                "mechanism_kind": mechanism_kind,
                "training_protocol": "original_igc",
                "strategy": strategy_label,
                "seed": int(args.exp.seed),
                "dataset_seed": int(args.dataset.seed),
                "projection_horizon": int(t),
                "prediction_steps": int(t + 1),
                "test_rmse": float(test_rmse),
                "max_epochs": int(args.exp.max_epochs),
                "optimizer_steps": int(decoder_trainer.global_step),
                "decoder_ckpt_saved": int(save_decoder_ckpt),
                "model_target": str(args.model.igcnet._target_),
            },
        )

        decoder_results = {
            "decoder_val_rmse_all": val_rmse_all,
            "decoder_val_rmse_orig": val_rmse_orig,
        }
        decoder_results.update(
            {
                "decoder_test_rmse_" + key: value
                for key, value in test_rmses.items()
            }
        )

        if args.exp.logging:
            decoder_logger.log_metrics(decoder_results)
        results.update(decoder_results)

    logger.info("Completed original IGC-Net mechanism-validation run.")
    logger.info("Run outputs: %s", run_output_path)
    logger.info(
        "Mechanism outputs: %s/%s",
        args.model.igcnet.mechanism.output_dir,
        args.model.igcnet.mechanism.run_id,
    )

    return results


if __name__ == "__main__":
    main()
