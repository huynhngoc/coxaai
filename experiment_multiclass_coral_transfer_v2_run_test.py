"""
Run the experiments pipeline
"""
import numpy as np
import customize_obj
# import h5py
# from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from deoxys.experiment import DefaultExperimentPipeline
# from deoxys.model.callbacks import PredictionCheckpoint
# from deoxys.utils import read_file
import argparse
import os
from deoxys.utils import read_csv
# import numpy as np
# from pathlib import Path
# from comet_ml import Experiment as CometEx
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef
import json
import pandas as pd


class Matthews_corrcoef_scorer:
    def __call__(self, *args, **kwargs):
        return matthews_corrcoef(*args, **kwargs)

    def _score_func(self, *args, **kwargs):
        return matthews_corrcoef(*args, **kwargs)


try:
    metrics.SCORERS['mcc'] = Matthews_corrcoef_scorer()
except:
    pass
try:
    metrics._scorer._SCORERS['mcc'] = Matthews_corrcoef_scorer()
except:
    pass


def metric_avg_score(res_df, postprocessor):
    cols = ['accuracy', 'mcc', 'A-BCDE', 'AB-CDE', 'ABC-DE', 'ABCD-E']
    weights = [1, 1, 1.2, 1.3, 0.8, 0.7]
    res_df['avg_score'] = res_df[cols].mul(weights).sum(axis=1) / sum(weights)
    return res_df


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("GPU Unavailable")
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("log_folder")
    parser.add_argument("--temp_folder", default='', type=str)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--model_checkpoint_period", default=1, type=int)
    parser.add_argument("--prediction_checkpoint_period", default=1, type=int)
    parser.add_argument("--meta", default='patient_idx', type=str)
    parser.add_argument(
        "--monitor", default='avg_score', type=str)
    parser.add_argument(
        "--monitor_mode", default='max', type=str)
    parser.add_argument("--memory_limit", default=0, type=int)
    args, unknown = parser.parse_known_args()
    if args.memory_limit:
        # Restrict TensorFlow to only allocate X-GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(
                    memory_limit=1024 * args.memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    if '2d' in args.log_folder:
        meta = args.meta
    else:
        meta = args.meta.split(',')[0]
    print('training from configuration', args.config_file,
          'and saving log files to', args.log_folder)
    print('Unprocesssed prediction are saved to', args.temp_folder)

    def binarize(targets, predictions):
        return targets, (predictions > 0.5).astype(targets.dtype)

    def flip(targets, predictions):
        return 1 - targets, 1 - (predictions > 0.5).astype(targets.dtype)

    def decode(targets, predictions): # coral decode
        binarize_prediction = (predictions > 0.5).astype(targets.dtype)
        # find the index of the first zero in each row
        decoded_prediction = binarize_prediction.argmin(axis=1) + binarize_prediction.min(axis=1) * binarize_prediction.shape[1]
        return targets.sum(axis=1), decoded_prediction

    def select_class(class_idx):
        def select(targets, predictions):
            return targets[..., class_idx: class_idx + 1], (predictions[..., class_idx: class_idx + 1] > 0.5).astype(targets.dtype)
        return select

    # rename old test folder
    if os.path.exists(args.log_folder + '/test'):
        os.rename(args.log_folder + '/test', args.log_folder + '/test_old')
    if os.path.exists(args.log_folder + '/info.txt'):
        os.rename(args.log_folder + '/info.txt', args.log_folder + '/info_old.txt')

    # logs_df = pd.read_csv(args.log_folder + '/logs.csv')
    # best_epoch = logs_df['epoch'][logs_df['val_loss'].idxmin()] + 1
    # weights_file = args.log_folder + f'/model/model.{best_epoch:03d}.h5'
    # with open(args.log_folder + '/new_info.txt', 'w') as f:
    #     f.write(f'Best epoch: {best_epoch:03d} by val_loss\n')

    # exp = DefaultExperimentPipeline(
    #     log_base_path=args.log_folder,
    #     temp_base_path=args.temp_folder
    # ).from_file(weights_file).run_test().apply_post_processors(
    #     map_meta_data=meta, run_test=True,
    #     metrics=['AUC', 'roc_auc', 'roc_auc', 'CategoricalCrossentropy',
    #              'BinaryAccuracy', 'mcc', 'accuracy'],
    #     metrics_sources=['tf', 'sklearn', 'sklearn',
    #                      'tf', 'tf', 'sklearn', 'sklearn'],
    #     process_functions=[None, None, None, None, None, decode, decode],
    #     metrics_kwargs=[{}, {'metric_name': 'roc_auc_ovr', 'multi_class': 'ovr'},
    #                     {}, {}, {}, {}, {}]
    # )

    exp = DefaultExperimentPipeline(
        log_base_path=args.log_folder,
        temp_base_path=args.temp_folder
    ).from_file(args.log_folder + f'/model/model.070.h5').apply_post_processors(
        map_meta_data=meta, run_test=False,
        metrics=['AUC', 'roc_auc', 'roc_auc', 'BinaryCrossentropy',
                 'BinaryAccuracy', 'mcc', 'accuracy', 'accuracy', 'accuracy', 'accuracy', 'accuracy'],
        metrics_sources=['tf', 'sklearn', 'sklearn',
                         'tf', 'tf', 'sklearn', 'sklearn', 'sklearn', 'sklearn', 'sklearn', 'sklearn'],
        process_functions=[None, None, None, None, None, decode, decode, select_class(0), select_class(1), select_class(2), select_class(3)],
        metrics_kwargs=[{}, {'metric_name': 'roc_auc_ovr', 'multi_class': 'ovr'},
                        {}, {}, {}, {}, {}, {'metric_name': 'A-BCDE'},
                        {'metric_name': 'AB-CDE'},
                        {'metric_name': 'ABC-DE'},
                        {'metric_name': 'ABCD-E'}]
    ).load_best_model(
        monitor=args.monitor,
        use_raw_log=False,
        mode=args.monitor_mode,
        custom_modifier_fn=metric_avg_score
    ).run_test(
    ).apply_post_processors(
        map_meta_data=meta, run_test=True,
        metrics=['AUC', 'roc_auc', 'roc_auc', 'BinaryCrossentropy',
                 'BinaryAccuracy', 'mcc', 'accuracy', 'accuracy', 'accuracy', 'accuracy', 'accuracy'],
        metrics_sources=['tf', 'sklearn', 'sklearn',
                         'tf', 'tf', 'sklearn', 'sklearn', 'sklearn', 'sklearn', 'sklearn', 'sklearn'],
        process_functions=[None, None, None, None, None, decode, decode, select_class(0), select_class(1), select_class(2), select_class(3)],
        metrics_kwargs=[{}, {'metric_name': 'roc_auc_ovr', 'multi_class': 'ovr'},
                        {}, {}, {}, {}, {}, {'metric_name': 'A-BCDE'},
                        {'metric_name': 'AB-CDE'},
                        {'metric_name': 'ABC-DE'},
                        {'metric_name': 'ABCD-E'}]
    )
