import logging
import os
import hydra
import torch
import numpy as np
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
torch.set_float32_matmul_precision('high')
setproctitle.setproctitle("CRN")


@hydra.main(config_name='crn.yaml', config_path='./config/fio22/')
def main(args: DictConfig):
    """
    Training / evaluation script for CRN (Counterfactual Recurrent Network)
    """

    results = {}

    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)

    mlf_logger, experiment_name = None, None

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
    dataset_collection.process_data_encoder()
    cast_dataset_to_float32(dataset_collection)
    args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
    args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
    args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
    args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]

    max_projection_horizon = args.dataset.projection_horizon

    if args.exp.logging:
        experiment_name = f'{args.model.name}/{args.dataset.name}'

    seed_everything(args.exp.seed)

    encoder_callbacks = [AlphaRise(rate=args.exp.alpha_rate)]

    if args.exp.logging:
        mlf_logger = FilteringMlFlowLogger(
            filter_submodels=['encoder', 'decoder'],
            experiment_name=experiment_name,
            tracking_uri=args.exp.mlflow_uri,
            run_name='encoder'
        )
        encoder_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        artifacts_path = hydra.utils.to_absolute_path(
            mlf_logger.experiment.get_run(mlf_logger.run_id).info.artifact_uri)
    else:
        artifacts_path = None

    encoder = instantiate(args.model.encoder, args, dataset_collection, _recursive_=False)
    if args.model.encoder.tune_hparams:
        encoder.finetune(resources_per_trial=args.model.encoder.resources_per_trial)

    encoder_trainer = Trainer(
        devices=eval(str(args.exp.gpus)),
        logger=mlf_logger,
        max_epochs=args.exp.max_epochs,
        callbacks=encoder_callbacks
    )
    encoder_trainer.fit(encoder)

    # Validation factual rmse
    val_dataloader = DataLoader(dataset_collection.val_f, batch_size=args.dataset.val_batch_size, shuffle=False)
    encoder_trainer.test(encoder, dataloaders=val_dataloader)
    val_rmse_orig, val_rmse_all = encoder.get_normalised_masked_rmse(dataset_collection.val_f)
    logger.info(f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}')

    encoder_results = {}
    if hasattr(dataset_collection, 'test_cf_one_step'):
        test_rmse_orig, test_rmse_all, test_rmse_last = encoder.get_normalised_masked_rmse(
            dataset_collection.test_cf_one_step, one_step_counterfactual=True)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'(orig): {test_rmse_orig}; (only cf): {test_rmse_last}')
        encoder_results = {
            'encoder_val_rmse_all': val_rmse_all,
            'encoder_val_rmse_orig': val_rmse_orig,
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig,
            'encoder_test_rmse_last': test_rmse_last,
        }
    elif hasattr(dataset_collection, 'test_f'):
        test_rmse_orig, test_rmse_all = encoder.get_normalised_masked_rmse(dataset_collection.test_f)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; (orig): {test_rmse_orig}.')
        encoder_results = {
            'encoder_val_rmse_all': val_rmse_all,
            'encoder_val_rmse_orig': val_rmse_orig,
            'encoder_test_rmse_all': test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig,
        }

    if args.exp.logging and mlf_logger is not None:
        mlf_logger.log_metrics(encoder_results)
        mlf_logger.experiment.set_terminated(mlf_logger.run_id)
    results.update(encoder_results)

    if args.model.train_decoder:
        seed_everything(args.exp.seed)

        decoder_callbacks = [AlphaRise(rate=args.exp.alpha_rate)]

        if args.exp.logging:
            decoder_train_logger = FilteringMlFlowLogger(
                filter_submodels=['encoder', 'decoder'],
                experiment_name=experiment_name,
                tracking_uri=args.exp.mlflow_uri,
                run_name='decoder_train'
            )
            decoder_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        else:
            decoder_train_logger = None

        decoder = instantiate(args.model.decoder, args, encoder, dataset_collection, _recursive_=False)
        if args.model.decoder.tune_hparams:
            decoder.finetune(resources_per_trial=args.model.decoder.resources_per_trial)

        decoder_trainer = Trainer(
            devices=eval(str(args.exp.gpus)),
            logger=decoder_train_logger,
            max_epochs=args.exp.max_epochs,
            callbacks=decoder_callbacks
        )
        decoder_trainer.fit(decoder)

        # Validation factual rmse
        val_dataloader = DataLoader(
            dataset_collection.val_f,
            batch_size=10 * args.dataset.val_batch_size,
            shuffle=False
        )
        decoder_trainer.test(decoder, dataloaders=val_dataloader)
        val_rmse_orig, val_rmse_all = decoder.get_normalised_masked_rmse(dataset_collection.val_f)
        logger.info(f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}')

        all_n_step_rmses = []
        if hasattr(dataset_collection, 'test_cf_treatment_seq'):
            all_n_step_rmses = decoder.get_normalised_n_step_rmses(dataset_collection.test_cf_treatment_seq)
        elif hasattr(dataset_collection, 'test_f'):
            all_n_step_rmses = decoder.get_normalised_n_step_rmses(dataset_collection.test_f)

        test_rmses = {f'{k + 2}-step': v for k, v in enumerate(all_n_step_rmses)}
        logger.info(f'Test normalised RMSE (n-step prediction): {test_rmses}')

        decoder_results = {
            'decoder_val_rmse_all': val_rmse_all,
            'decoder_val_rmse_orig': val_rmse_orig,
        }
        decoder_results.update({f'decoder_test_rmse_{k}': v for k, v in test_rmses.items()})

        if args.exp.logging and decoder_train_logger is not None:
            decoder_train_logger.log_metrics(decoder_results)
            decoder_train_logger.experiment.set_terminated(decoder_train_logger.run_id)

        results.update(decoder_results)

        for t in range(1, max_projection_horizon + 1):
            if phase_b_output_dir is not None:
                ckpt_path = os.path.join(phase_b_output_dir, f'crn_decoder_h{t}.ckpt')
                decoder_trainer.save_checkpoint(ckpt_path)
                logger.info(f'Saved decoder checkpoint for horizon t={t}: {ckpt_path}')

            this_rmse = all_n_step_rmses[t - 1] if (t - 1) < len(all_n_step_rmses) else None
            logger.info(f'[horizon t={t}] {t + 1}-step RMSE = {this_rmse}')

            horizon_results = {
                'decoder_val_rmse_all': val_rmse_all,
                'decoder_val_rmse_orig': val_rmse_orig,
                f'decoder_test_rmse_{t + 1}-step': this_rmse,
            }

            if args.exp.logging:
                h_logger = FilteringMlFlowLogger(
                    filter_submodels=['encoder', 'decoder'],
                    experiment_name=experiment_name,
                    tracking_uri=args.exp.mlflow_uri,
                    run_name=str(t)
                )
                h_logger.log_metrics(horizon_results)
                h_logger.experiment.set_terminated(h_logger.run_id)

            results.update(horizon_results)

    return results


if __name__ == "__main__":
    main()
