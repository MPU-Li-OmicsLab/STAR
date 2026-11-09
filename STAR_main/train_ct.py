import logging
import numpy as np
import hydra
import torch
import os
import setproctitle
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

from src.models.utils import AlphaRise, FilteringMlFlowLogger
from utils.dose_trajectory_generator import DoseTrajectoryGenerator
from utils.trans_type import cast_dataset_to_float32

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

setproctitle.setproctitle("CT")


@hydra.main(config_name='ct.yaml', config_path='./config/fio22/')
def main(args: DictConfig):
    """
    Training / evaluation script for CT (Causal Transformer)
    """

    results = {}

    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)

    mlf_logger, multimodel_callbacks, experiment_name = None, [], None

    phase_b_output_dir = getattr(args.exp, 'phase_b_output_dir', None)
    if phase_b_output_dir is not None:
        phase_b_output_dir = os.path.join(
            phase_b_output_dir, f'run_seed{args.exp.seed}',
        )
        os.makedirs(phase_b_output_dir, exist_ok=True)
        args.exp.mlflow_uri = 'file://' + os.path.abspath(os.path.join(phase_b_output_dir, 'mlruns'))

    # Initialisation of data
    seed_everything(args.dataset.seed)
    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    cast_dataset_to_float32(dataset_collection)
    args.model.dim_outcomes        = dataset_collection.train_f.data['outputs'].shape[-1]
    args.model.dim_treatments      = dataset_collection.train_f.data['current_treatments'].shape[-1]
    args.model.dim_vitals          = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
    args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]

    max_projection_horizon = args.dataset.projection_horizon

    if args.exp.logging:
        experiment_name = f'{args.model.name}/{args.dataset.name}'

    # ============================== Training ==============================
    seed_everything(args.exp.seed)

    if args.exp.logging:
        mlf_logger = FilteringMlFlowLogger(
            filter_submodels=[], experiment_name=experiment_name,
            tracking_uri=args.exp.mlflow_uri, run_name='train'
        )
        multimodel_callbacks = [
            AlphaRise(rate=args.exp.alpha_rate),
            LearningRateMonitor(logging_interval='epoch'),
        ]
        artifacts_path = hydra.utils.to_absolute_path(
            mlf_logger.experiment.get_run(mlf_logger.run_id).info.artifact_uri)
    else:
        multimodel_callbacks = [AlphaRise(rate=args.exp.alpha_rate)]
        artifacts_path = None

    multimodel = instantiate(args.model.multi, args, dataset_collection, _recursive_=False)
    if args.model.multi.tune_hparams:
        multimodel.finetune(resources_per_trial=args.model.multi.resources_per_trial)

    multimodel_trainer = Trainer(
        devices=eval(str(args.exp.gpus)),
        logger=mlf_logger,
        max_epochs=args.exp.max_epochs,
        callbacks=multimodel_callbacks,
        gradient_clip_val=args.model.multi.max_grad_norm
    )
    multimodel_trainer.fit(multimodel)

    # ============================== One-step evaluation ==============================
    val_rmse_orig, val_rmse_all = multimodel.get_normalised_masked_rmse(dataset_collection.val_f)
    logger.info(f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}')

    encoder_results = {}
    if hasattr(dataset_collection, 'test_cf_one_step'):
        test_rmse_orig, test_rmse_all, test_rmse_last = multimodel.get_normalised_masked_rmse(
            dataset_collection.test_cf_one_step, one_step_counterfactual=True)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'(orig): {test_rmse_orig}; (only cf): {test_rmse_last}')
        encoder_results = {
            'encoder_val_rmse_all':   val_rmse_all,
            'encoder_val_rmse_orig':  val_rmse_orig,
            'encoder_test_rmse_all':  test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig,
            'encoder_test_rmse_last': test_rmse_last,
        }
    elif hasattr(dataset_collection, 'test_f'):
        test_rmse_orig, test_rmse_all = multimodel.get_normalised_masked_rmse(dataset_collection.test_f)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; (orig): {test_rmse_orig}.')
        encoder_results = {
            'encoder_val_rmse_all':   val_rmse_all,
            'encoder_val_rmse_orig':  val_rmse_orig,
            'encoder_test_rmse_all':  test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig,
        }

    if args.exp.logging and mlf_logger is not None:
        mlf_logger.log_metrics(encoder_results)
    results.update(encoder_results)

    if args.exp.logging and mlf_logger is not None:
        mlf_logger.experiment.set_terminated(mlf_logger.run_id)

    # ============================== N-step evaluation + per-horizon ckpt ==============================
    all_n_step_rmses = []
    if hasattr(dataset_collection, 'test_cf_treatment_seq'):
        all_n_step_rmses = multimodel.get_normalised_n_step_rmses(dataset_collection.test_cf_treatment_seq)
    elif hasattr(dataset_collection, 'test_f_multi'):
        all_n_step_rmses = multimodel.get_normalised_n_step_rmses(dataset_collection.test_f_multi)

    test_rmses = {f'{k+2}-step': v for k, v in enumerate(all_n_step_rmses)}
    logger.info(f'All n-step RMSEs (2..{max_projection_horizon+1}-step): {test_rmses}')

    # 每个 horizon 保存一个 ckpt，记录对应步的 metrics 到独立 MLflow run
    for t in range(1, max_projection_horizon + 1):
        if phase_b_output_dir is not None:
            ckpt_path = os.path.join(phase_b_output_dir, f'ct_h{t}.ckpt')
            multimodel_trainer.save_checkpoint(ckpt_path)
            logger.info(f'Saved checkpoint for horizon t={t}: {ckpt_path}')

        this_rmse = all_n_step_rmses[t - 1] if (t - 1) < len(all_n_step_rmses) else None
        logger.info(f'[horizon t={t}] {t+1}-step RMSE = {this_rmse}')

        horizon_results = {
            'decoder_val_rmse_all':          val_rmse_all,
            'decoder_val_rmse_orig':         val_rmse_orig,
            f'decoder_test_rmse_{t+1}-step': this_rmse,
        }

        if args.exp.logging:
            h_logger = FilteringMlFlowLogger(
                filter_submodels=[], experiment_name=experiment_name,
                tracking_uri=args.exp.mlflow_uri, run_name=str(t)
            )
            h_logger.log_metrics(horizon_results)
            h_logger.experiment.set_terminated(h_logger.run_id)

        results.update(horizon_results)

    return results


if __name__ == "__main__":
    main()