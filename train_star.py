import logging
import os
import hydra
import torch
import numpy as np
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
from utils.saturating_heterogeneity import S0HeterogeneityConfig, run_saturating_oracle_heterogeneity_gate
from utils.trans_type import cast_dataset_to_float32

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')
setproctitle.setproctitle("STAR")


@hydra.main(version_base=None, config_name=f'star.yaml', config_path='./config/map2/')
def main(args: DictConfig):
    """
    Training / evaluation script for STAR
    """

    results = {}

    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)

    phase_b_output_dir = getattr(args.exp, 'phase_b_output_dir', None)
    if phase_b_output_dir is None:
        raise ValueError(
            'args.exp.phase_b_output_dir must be set for the V3 benchmark/audit run.'
        )
    phase_b_output_dir = os.path.join(
        phase_b_output_dir, f'run_seed{args.exp.seed}',
    )
    os.makedirs(phase_b_output_dir, exist_ok=True)
    run_output_path = os.path.abspath(phase_b_output_dir)
    args.exp.mlflow_uri = 'file://' + os.path.abspath(
        os.path.join(run_output_path, 'mlruns')
    )

    # Initialisation of data
    seed_everything(args.dataset.seed)
    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    cast_dataset_to_float32(dataset_collection)
    args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
    args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
    args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
    args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]

    seed_everything(args.exp.seed)

    igcnet_callbacks = []
    if args.exp.logging:
        experiment_name = f'{args.model.name}/{args.dataset.name}'
        mlf_logger = FilteringMlFlowLogger(filter_submodels=[], experiment_name=experiment_name,
                                           tracking_uri=args.exp.mlflow_uri,
                                           run_name='0')

        igcnet_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        artifacts_path = hydra.utils.to_absolute_path(
            mlf_logger.experiment.get_run(mlf_logger.run_id).info.artifact_uri)
    else:
        mlf_logger = None
        artifacts_path = None

    args.model.star.projection_horizon = 0
    igcnetmodel = instantiate(
        args.model.star,
        args,
        dataset_collection,
        path=run_output_path,
        _recursive_=False,
    )
    if args.model.star.tune_hparams:
        igcnetmodel.finetune(resources_per_trial=args.model.star.resources_per_trial)

    igcnetmodel_trainer = Trainer(devices=eval(str(args.exp.gpus)), logger=mlf_logger, max_epochs=args.exp.max_epochs,
                                  callbacks=igcnet_callbacks,
                                  gradient_clip_val=args.model.star.max_grad_norm, gradient_clip_algorithm="norm")
    igcnetmodel_trainer.fit(igcnetmodel)

    val_dataloader = DataLoader(dataset_collection.val_f, batch_size=args.dataset.val_batch_size, shuffle=False)
    igcnetmodel_trainer.test(igcnetmodel, dataloaders=val_dataloader)
    val_rmse_orig, val_rmse_all = igcnetmodel.get_normalised_masked_rmse(dataset_collection.val_f)
    logger.info(f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}')

    encoder_results = {}
    if hasattr(dataset_collection, 'test_cf_one_step'):
        test_rmse_orig, test_rmse_all, test_rmse_last = igcnetmodel.get_normalised_masked_rmse(
            dataset_collection.test_cf_one_step, one_step_counterfactual=True)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'Test normalised RMSE (orig): {test_rmse_orig}; '
                    f'Test normalised RMSE (only counterfactual): {test_rmse_last}')
        encoder_results = {
            'encoder_val_rmse_all': val_rmse_all,
            'encoder_val_rmse_orig': val_rmse_orig,
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig,
            'encoder_test_rmse_last': test_rmse_last
        }
    elif hasattr(dataset_collection, 'test_f'):
        test_rmse_orig, test_rmse_all = igcnetmodel.get_normalised_masked_rmse(dataset_collection.test_f)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'Test normalised RMSE (orig): {test_rmse_orig}.')
        encoder_results = {
            'encoder_val_rmse_all': val_rmse_all,
            'encoder_val_rmse_orig': val_rmse_orig,
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig
        }

    mlf_logger.log_metrics(encoder_results) if args.exp.logging else None
    results.update(encoder_results)

    for t in range(1, args.dataset.projection_horizon + 1):
        seed_everything(args.exp.seed)
        test_rmses = {}
        decoder_results = {
            'decoder_val_rmse_all': val_rmse_all,
            'decoder_val_rmse_orig': val_rmse_orig
        }
        if args.exp.logging:
            mlf_logger = FilteringMlFlowLogger(filter_submodels=[], experiment_name=experiment_name,
                                               tracking_uri=args.exp.mlflow_uri, run_name=str(t))

        args.model.star.projection_horizon = t
        igcnetmodel = instantiate(
            args.model.star,
            args,
            dataset_collection,
            path=run_output_path,
            _recursive_=False,
        )

        decoder_callbacks = list(igcnet_callbacks)
        igcnetmodel_trainer = Trainer(
            devices=eval(str(args.exp.gpus)),
            logger=mlf_logger,
            max_epochs=args.exp.max_epochs,
            callbacks=decoder_callbacks,
            gradient_clip_val=args.model.star.max_grad_norm,
        )
        igcnetmodel_trainer.fit(igcnetmodel)

        decoder_ckpt = os.path.join(run_output_path, f'igcnet_decoder_h{t}.ckpt')
        igcnetmodel_trainer.save_checkpoint(decoder_ckpt)

        test_loader = DataLoader(dataset_collection.test_cf_treatment_seq, batch_size=args.dataset.val_batch_size,
                                 shuffle=False)
        igcnetmodel_trainer.validate(igcnetmodel)
        igcnetmodel_trainer.test(igcnetmodel, test_loader)

        if hasattr(dataset_collection, 'test_cf_treatment_seq'):
            test_rmse = igcnetmodel.get_normalised_n_step_rmses(dataset_collection.test_cf_treatment_seq)
        elif hasattr(dataset_collection, 'test_f_multi'):
            test_rmse = igcnetmodel.get_normalised_n_step_rmses(dataset_collection.test_f_multi)
        test_rmses = {f'{t + 1}-step': test_rmse}
        logger.info(f'Test normalised RMSE (n-step prediction): {test_rmses}')

        decoder_results.update({('decoder_test_rmse_' + k): v for (k, v) in test_rmses.items()})

        mlf_logger.log_metrics(decoder_results) if args.exp.logging else None
        results.update(decoder_results)

    mlf_logger.experiment.set_terminated(mlf_logger.run_id) if args.exp.logging else None

    return results


if __name__ == "__main__":
    main()
