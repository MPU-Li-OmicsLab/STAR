import logging
import os
import json
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
setproctitle.setproctitle("G-Transformer")


def get_mc_datasets(dataset_collection, name: str):

    mc_datasets = getattr(dataset_collection, f'{name}_mc', None)
    if mc_datasets is None:
        mc_datasets = getattr(dataset_collection, name, None)
    if not isinstance(mc_datasets, list):
        raise AttributeError(f'没有找到 {name}_mc 这样的 MC 数据集列表，'
                             f'请确认 explode_cf_treatment_seq 在 dataset_collection 上挂的属性名')
    return mc_datasets


@hydra.main(config_name=f'g_transformer.yaml', config_path='./config/fio22/')
def main(args: DictConfig):

    results = {}

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)

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
    args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
    args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
    args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
    args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]

    seed_everything(args.exp.seed)
    callbacks = []

    # MlFlow Logger
    if args.exp.logging:
        experiment_name = f'{args.model.name}/{args.dataset.name}_'
        mlf_logger = FilteringMlFlowLogger(filter_submodels=[],
                                           experiment_name=experiment_name, tracking_uri=args.exp.mlflow_uri)
        callbacks += [LearningRateMonitor(logging_interval='epoch')]
        artifacts_path = hydra.utils.to_absolute_path(mlf_logger.experiment.get_run(mlf_logger.run_id).info.artifact_uri)
    else:
        mlf_logger = None
        artifacts_path = None

    # ============================== Initialisation & Training of G-Transformer ==============================
    model = instantiate(args.model.g_transformer, args, dataset_collection, _recursive_=False)
    if args.model.g_transformer.tune_hparams:
        model.finetune(resources_per_trial=args.model.g_transformer.resources_per_trial)

    trainer = Trainer(devices=eval(str(args.exp.gpus)), logger=mlf_logger, max_epochs=args.exp.max_epochs,
                      gradient_clip_val=args.model.g_transformer.max_grad_norm, callbacks=callbacks)
    trainer.fit(model)

    if phase_b_output_dir is not None:
        trainer.save_checkpoint(os.path.join(phase_b_output_dir, 'g_transformer.ckpt'))
        if getattr(model, 'holdout_resid', None) is not None:
            np.savez(os.path.join(phase_b_output_dir, 'holdout_resid.npz'),
                     resid=model.holdout_resid, resid_len=model.holdout_resid_len)

    # ============================== Validation / one-step ==============================

    val_rmse_orig, val_rmse_all = model.get_normalised_masked_rmse(dataset_collection.val_f)
    logger.info(f'Val normalised RMSE (all): {val_rmse_all}; Val normalised RMSE (orig): {val_rmse_orig}')

    model_results = {
        'val_rmse_all': val_rmse_all,
        'val_rmse_orig': val_rmse_orig,
    }

    if hasattr(dataset_collection, 'test_cf_one_step'):  # Test one_step_counterfactual rmse
        test_rmse_orig, test_rmse_all, test_rmse_last = \
            model.get_normalised_masked_rmse(dataset_collection.test_cf_one_step, one_step_counterfactual=True)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'Test normalised RMSE (orig): {test_rmse_orig}; '
                    f'Test normalised RMSE (only counterfactual): {test_rmse_last}')
        model_results.update({
            'test_rmse_all': test_rmse_all,
            'test_rmse_orig': test_rmse_orig,
            'test_rmse_last': test_rmse_last,
        })
    elif hasattr(dataset_collection, 'test_f'):  # Test factual rmse
        test_rmse_orig, test_rmse_all = model.get_normalised_masked_rmse(dataset_collection.test_f)
        logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                    f'Test normalised RMSE (orig): {test_rmse_orig}.')
        model_results.update({
            'test_rmse_all': test_rmse_all,
            'test_rmse_orig': test_rmse_orig,
        })

    mlf_logger.log_metrics(model_results) if args.exp.logging else None
    results.update(model_results)

    # ============================== Multi-step g-computation rollout ==============================

    test_rmses = {}
    if hasattr(dataset_collection, 'test_cf_treatment_seq'):  # Test n_step_counterfactual rmse
        test_rmses = model.get_normalised_n_step_rmses(dataset_collection.test_cf_treatment_seq,
                                                       get_mc_datasets(dataset_collection, 'test_cf_treatment_seq'))
    elif hasattr(dataset_collection, 'test_f_multi'):  # Test n_step_factual rmse
        test_rmses = model.get_normalised_n_step_rmses(dataset_collection.test_f_multi,
                                                       get_mc_datasets(dataset_collection, 'test_f_multi'))
    test_rmses = {f'{k + 2}-step': v for (k, v) in enumerate(test_rmses)}

    logger.info(f'Test normalised RMSE (n-step prediction): {test_rmses}')
    n_step_results = {('test_rmse_' + k): v for (k, v) in test_rmses.items()}

    mlf_logger.log_metrics(n_step_results) if args.exp.logging else None
    results.update(n_step_results)

    if phase_b_output_dir is not None:
        with open(os.path.join(phase_b_output_dir, 'results.json'), 'w') as f:
            json.dump({k: float(v) for k, v in results.items()}, f, indent=2)

    if args.exp.logging:
        mlf_logger.experiment.set_terminated(mlf_logger.run_id)

    return results


if __name__ == "__main__":
    main()