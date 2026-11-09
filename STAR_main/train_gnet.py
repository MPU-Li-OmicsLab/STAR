import logging
import os
import hydra
import torch
import numpy as np
import json
import time
import math
import setproctitle
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

from src.models.utils import FilteringMlFlowLogger
from utils.dose_trajectory_generator import DoseTrajectoryGenerator
from utils.trans_type import cast_dataset_to_float32

logging.basicConfig(level=logging.INFO)
torch.set_float32_matmul_precision('high')
setproctitle.setproctitle("GPS")
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_name=f'gnet.yaml', config_path='./config/fio22/')
def main(args: DictConfig):

    results = {}

    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)

    mlf_logger, decoder_train_logger, gnet_callbacks, experiment_name = None, None, [], None

    phase_b_output_dir = getattr(args.exp, 'phase_b_output_dir', None)
    if phase_b_output_dir is not None:
        phase_b_output_dir = os.path.join(
            phase_b_output_dir, f'run_seed{args.exp.seed}',
        )
        os.makedirs(phase_b_output_dir, exist_ok=True)
        args.exp.mlflow_uri = 'file://' + os.path.abspath(os.path.join(phase_b_output_dir, 'mlruns'))

    seed_everything(args.dataset.seed)
    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    cast_dataset_to_float32(dataset_collection)
    args.model.dim_outcomes       = dataset_collection.train_f.data['outputs'].shape[-1]
    args.model.dim_treatments     = dataset_collection.train_f.data['current_treatments'].shape[-1]
    args.model.dim_vitals         = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
    args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]
    args.model.gnet.comp_sizes = [(args.model.dim_outcomes + args.model.dim_vitals) // args.model.gnet.num_comp] * args.model.gnet.num_comp

    max_projection_horizon = args.dataset.projection_horizon

    if args.exp.logging:
        experiment_name = f'{args.model.name}/{args.dataset.name}'

    # ============================== Encoder (1-step) ==============================
    seed_everything(args.exp.seed)

    if args.exp.logging:
        mlf_logger = FilteringMlFlowLogger(
            filter_submodels=[], experiment_name=experiment_name,
            tracking_uri=args.exp.mlflow_uri, run_name='0')
        gnet_callbacks = [LearningRateMonitor(logging_interval='epoch')]
        artifacts_path = hydra.utils.to_absolute_path(
            mlf_logger.experiment.get_run(mlf_logger.run_id).info.artifact_uri)

    model = instantiate(args.model.gnet, args, dataset_collection, _recursive_=False)
    if args.model.gnet.tune_hparams:
        model.finetune(resources_per_trial=args.model.gnet.resources_per_trial)

    trainer = Trainer(devices=eval(str(args.exp.gpus)), logger=mlf_logger,
                      max_epochs=args.exp.max_epochs, callbacks=gnet_callbacks,
                      gradient_clip_val=args.model.gnet.max_grad_norm)
    trainer.fit(model)

    val_rmse_orig, val_rmse_all = model.get_normalised_masked_rmse(dataset_collection.val_f)
    logger.info(f'Val normalised RMSE (all): {val_rmse_all}; (orig): {val_rmse_orig}')

    encoder_results = {}
    if hasattr(dataset_collection, 'test_cf_one_step'):
        test_rmse_orig, test_rmse_all, test_rmse_last = model.get_normalised_masked_rmse(
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
        test_rmse_orig, test_rmse_all = model.get_normalised_masked_rmse(dataset_collection.test_f)
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

    # ============================== Decoder: 训练一次 + 按 horizon 存 ckpt ==============================
    seed_everything(args.exp.seed)

    if args.exp.logging:
        decoder_train_logger = FilteringMlFlowLogger(
            filter_submodels=[], experiment_name=experiment_name,
            tracking_uri=args.exp.mlflow_uri, run_name='decoder_train')

    model = instantiate(args.model.gnet, args, dataset_collection, _recursive_=False)
    trainer = Trainer(devices=eval(str(args.exp.gpus)), logger=decoder_train_logger,
                      max_epochs=args.exp.max_epochs, callbacks=gnet_callbacks,
                      gradient_clip_val=args.model.gnet.max_grad_norm)
    trainer.fit(model)

    if hasattr(dataset_collection, 'test_cf_treatment_seq_mc'):
        all_n_step_rmses = model.get_normalised_n_step_rmses(
            dataset_collection.test_cf_treatment_seq,
            dataset_collection.test_cf_treatment_seq_mc,
        )
    elif hasattr(dataset_collection, 'test_f_mc'):
        all_n_step_rmses = model.get_normalised_n_step_rmses(
            dataset_collection.test_f, dataset_collection.test_f_mc,
        )
    else:
        all_n_step_rmses = []

    logger.info(f'All n-step RMSEs (2..{max_projection_horizon + 1}-step): {list(all_n_step_rmses)}')

    if args.exp.logging and decoder_train_logger is not None:
        decoder_train_logger.experiment.set_terminated(decoder_train_logger.run_id)

    for t in range(1, max_projection_horizon + 1):

        if phase_b_output_dir is not None:
            decoder_ckpt = os.path.join(phase_b_output_dir, f'gnet_decoder_h{t}.ckpt')
            trainer.save_checkpoint(decoder_ckpt)
            logger.info(f'Saved decoder checkpoint for horizon {t}: {decoder_ckpt}')

        this_rmse = all_n_step_rmses[t - 1] if (t - 1) < len(all_n_step_rmses) else None

        decoder_results = {
            'decoder_val_rmse_all':  val_rmse_all,
            'decoder_val_rmse_orig': val_rmse_orig,
            f'decoder_test_rmse_{t + 1}-step': this_rmse,
        }
        logger.info(f'[horizon t={t}] {t + 1}-step RMSE = {this_rmse}')

        if args.exp.logging:
            h_logger = FilteringMlFlowLogger(
                filter_submodels=[], experiment_name=experiment_name,
                tracking_uri=args.exp.mlflow_uri, run_name=str(t))
            h_logger.log_metrics(decoder_results)
            h_logger.experiment.set_terminated(h_logger.run_id)

        results.update(decoder_results)

    return results


if __name__ == "__main__":
    main()