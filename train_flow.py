import logging
import os
import hydra
import torch
import numpy as np
import setproctitle
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

from src.models.utils import FilteringMlFlowLogger
from utils.dose_trajectory_generator import DoseTrajectoryGenerator
from utils.trans_type import cast_dataset_to_float32

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')
setproctitle.setproctitle("FLOW")


@hydra.main(version_base=None, config_name='flow_IPW.yaml', config_path='./config/fio22/')
def main(args: DictConfig):
    """
    Training / evaluation script for FLOW (Conditional Normalizing Flow)
    """

    results = {}

    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)

    phase_b_output_dir = getattr(args.exp, 'phase_b_output_dir', None)
    if phase_b_output_dir is not None:
        phase_b_output_dir = os.path.join(
            phase_b_output_dir, f'run_seed{args.exp.seed}',
        )
        os.makedirs(phase_b_output_dir, exist_ok=True)
        args.exp.mlflow_uri = 'file://' + os.path.abspath(
            os.path.join(phase_b_output_dir, 'mlruns')
        )

    seed_everything(args.dataset.seed)
    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    cast_dataset_to_float32(dataset_collection)
    args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
    args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
    args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
    args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]

    seed_everything(args.exp.seed)

    flow_callbacks = []

    if args.exp.logging:
        experiment_name = f'{args.model.name}/{args.dataset.name}'
        mlf_logger = FilteringMlFlowLogger(
            filter_submodels=[],
            experiment_name=experiment_name,
            tracking_uri=args.exp.mlflow_uri,
            run_name='0',
        )
        flow_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        artifacts_path = hydra.utils.to_absolute_path(
            mlf_logger.experiment.get_run(mlf_logger.run_id).info.artifact_uri
        )
    else:
        mlf_logger = None
        artifacts_path = None

    # ============================== 1-step ahead prediction =======================
    args.model.flow.projection_horizon = 0
    flowmodel = instantiate(args.model.flow, args, dataset_collection, _recursive_=False)
    if args.model.flow.tune_hparams:
        flowmodel.finetune(resources_per_trial=args.model.flow.resources_per_trial)

    flowmodel_trainer = Trainer(
        devices=eval(str(args.exp.gpus)),
        logger=mlf_logger,
        max_epochs=args.exp.max_epochs,
        callbacks=flow_callbacks,
        gradient_clip_val=args.model.flow.max_grad_norm,
        gradient_clip_algorithm="norm",
    )
    flowmodel_trainer.fit(flowmodel)

    val_rmse_orig, val_rmse_all = flowmodel.get_normalised_masked_rmse(
        dataset_collection.val_f
    )
    val_rmse_orig = float(val_rmse_orig)
    val_rmse_all  = float(val_rmse_all)
    logger.info(
        f'Val normalised RMSE (all): {val_rmse_all}; '
        f'Val normalised RMSE (orig): {val_rmse_orig}'
    )

    encoder_results = {}
    if hasattr(dataset_collection, 'test_cf_one_step'):
        test_rmse_orig, test_rmse_all, test_rmse_last = (
            flowmodel.get_normalised_masked_rmse(
                dataset_collection.test_cf_one_step,
                one_step_counterfactual=True,
            )
        )
        test_rmse_orig = float(test_rmse_orig)
        test_rmse_all  = float(test_rmse_all)
        test_rmse_last = float(test_rmse_last)
        logger.info(
            f'Test normalised RMSE (all): {test_rmse_all}; '
            f'Test normalised RMSE (orig): {test_rmse_orig}; '
            f'Test normalised RMSE (only counterfactual): {test_rmse_last}'
        )
        encoder_results = {
            'encoder_val_rmse_all':   val_rmse_all,
            'encoder_val_rmse_orig':  val_rmse_orig,
            'encoder_test_rmse_all':  test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig,
            'encoder_test_rmse_last': test_rmse_last,
        }
    elif hasattr(dataset_collection, 'test_f'):
        test_rmse_orig, test_rmse_all = flowmodel.get_normalised_masked_rmse(
            dataset_collection.test_f
        )
        test_rmse_orig = float(test_rmse_orig)
        test_rmse_all  = float(test_rmse_all)
        logger.info(
            f'Test normalised RMSE (all): {test_rmse_all}; '
            f'Test normalised RMSE (orig): {test_rmse_orig}.'
        )
        encoder_results = {
            'encoder_val_rmse_all':   val_rmse_all,
            'encoder_val_rmse_orig':  val_rmse_orig,
            'encoder_test_rmse_all':  test_rmse_all,
            'encoder_test_rmse_orig': test_rmse_orig,
        }

    mlf_logger.log_metrics(encoder_results) if args.exp.logging else None
    results.update(encoder_results)

    # ============================== multi-step ahead prediction ===================
    for t in range(1, args.dataset.projection_horizon + 1):
        seed_everything(args.exp.seed)

        decoder_results = {
            'decoder_val_rmse_all':  val_rmse_all,
            'decoder_val_rmse_orig': val_rmse_orig,
        }

        if args.exp.logging:
            mlf_logger = FilteringMlFlowLogger(
                filter_submodels=[],
                experiment_name=experiment_name,
                tracking_uri=args.exp.mlflow_uri,
                run_name=str(t),
            )

        args.model.flow.projection_horizon = t
        flowmodel = instantiate(args.model.flow, args, dataset_collection, _recursive_=False)
        flowmodel_trainer = Trainer(
            devices=eval(str(args.exp.gpus)),
            logger=mlf_logger,
            max_epochs=args.exp.max_epochs,
            callbacks=flow_callbacks,
            gradient_clip_val=args.model.flow.max_grad_norm,
            gradient_clip_algorithm="norm",
        )
        flowmodel_trainer.fit(flowmodel)

        # 每个 horizon 单独保存 ckpt
        if phase_b_output_dir is not None:
            decoder_ckpt = os.path.join(phase_b_output_dir, f'flow_decoder_h{t}.ckpt')
            flowmodel_trainer.save_checkpoint(decoder_ckpt)
            logger.info(f'Saved checkpoint: {decoder_ckpt}')

        if hasattr(dataset_collection, 'test_cf_treatment_seq'):
            test_rmse_arr = flowmodel.get_normalised_n_step_rmses(
                dataset_collection.test_cf_treatment_seq
            )
        elif hasattr(dataset_collection, 'test_f_multi'):
            test_rmse_arr = flowmodel.get_normalised_n_step_rmses(
                dataset_collection.test_f_multi
            )
        else:
            test_rmse_arr = np.array([0.0])

        test_rmses = {f'{t + 1}-step': float(test_rmse_arr[t - 1])}
        logger.info(f'Test normalised RMSE (n-step prediction): {test_rmses}')

        decoder_results.update(
            {('decoder_test_rmse_' + k): v for k, v in test_rmses.items()}
        )

        mlf_logger.log_metrics(decoder_results) if args.exp.logging else None
        results.update(decoder_results)

    mlf_logger.experiment.set_terminated(mlf_logger.run_id) if args.exp.logging else None

    return results


if __name__ == "__main__":
    main()