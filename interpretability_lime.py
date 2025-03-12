"""
LIME Interpretability Script for Deoxys Pipeline

This script runs LIME interpretability analysis on a trained model using Deoxys.
It loads the best model from the provided `log_folder` and applies LIME to explain its predictions.
"""

import customize_obj
import tensorflow as tf
from deoxys.experiment import DefaultExperimentPipeline
import argparse
import numpy as np
import h5py
import gc
import pandas as pd
from lime import lime_image
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef


# Custom scorer for Matthews Correlation Coefficient
class Matthews_corrcoef_scorer:
    def __call__(self, *args, **kwargs):
        return matthews_corrcoef(*args, **kwargs)

    def _score_func(self, *args, **kwargs):
        return matthews_corrcoef(*args, **kwargs)


# Register MCC as a scoring metric in sklearn
try:
    metrics.SCORERS['mcc'] = Matthews_corrcoef_scorer()
except:
    pass
try:
    metrics._scorer._SCORERS['mcc'] = Matthews_corrcoef_scorer()
except:
    pass


# Function to compute average score
def metric_avg_score(res_df, postprocessor):
    res_df['avg_score'] = res_df[['AUC', 'roc_auc', 'f1', 'f1_0',
                                  'BinaryAccuracy', 'mcc']].mean(axis=1)
    return res_df


# Main function
if __name__ == '__main__':
    ## Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("GPU Unavailable")

    ## Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("log_folder")  # Removed dataset_file argument
    parser.add_argument("--temp_folder", default='', type=str)
    parser.add_argument("--model_checkpoint_period", default=1, type=int)
    parser.add_argument("--prediction_checkpoint_period", default=1, type=int)
    parser.add_argument("--meta", default='patient_idx', type=str)  # Default to patient_idx only
    parser.add_argument("--monitor", default='avg_score', type=str)
    parser.add_argument("--monitor_mode", default='max', type=str)
    parser.add_argument("--memory_limit", default=0, type=int)

    args, unknown = parser.parse_known_args()

    ## Set GPU memory limit if specified
    if args.memory_limit:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(
                    memory_limit=1024 * args.memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)  # Virtual devices must be set before GPUs are initialized

    print('Explaining models in', args.log_folder)
    print('Unprocessed predictions are saved to', args.temp_folder)

    # Load best model from Deoxys pipeline
    exp = DefaultExperimentPipeline(
        log_base_path=args.log_folder,
        temp_base_path=args.temp_folder
    ).load_best_model(
        monitor=args.monitor,
        use_raw_log=False,
        mode=args.monitor_mode,
        custom_modifier_fn=metric_avg_score
    )

    seed = 1
    model = exp.model.model  # Extract trained model
    dr = exp.model.data_reader  # Load dataset via model
    test_gen = dr.test_generator  # Test data generator
    steps_per_epoch = test_gen.total_batch  # Total batches in the test set

    # Extract patient IDs
    pids = []
    with h5py.File(exp.post_processors.dataset_filename, 'r') as f:
        print("Available keys in dataset:", list(f.keys()))  # Debugging line
        for fold in test_gen.folds:
            pids.append(f[fold][args.meta][:])  # Load patient IDs
    pids = np.concatenate(pids)

    # Create output HDF5 file for LIME results
    lime_file_path = f'{args.log_folder}/test_lime.h5'
    with h5py.File(lime_file_path, 'w') as f:
        print('Created file', lime_file_path)
        f.create_dataset(args.meta, data=pids)
        f.create_dataset('lime', shape=(len(pids), 800, 800))

    # Initialize LIME explainer
    data_gen = test_gen.generate()
    explainer = lime_image.LimeImageExplainer()
    i, sub_idx = 0, 0

    # Process test batches
    for x, _ in data_gen:
        print(f'Batch {i+1}/{steps_per_epoch}')
        for image in x:
            explanation = explainer.explain_instance(
                image.astype('double'), model.predict,
                top_labels=1, hide_color=0, num_samples=1000)
            
            ind = explanation.top_labels[0]
            dict_heatmap = dict(explanation.local_exp[ind])
            lime_heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

            # Save LIME results to HDF5 file
            with h5py.File(lime_file_path, 'a') as f:
                f['lime'][sub_idx] = lime_heatmap
            sub_idx += 1

        i += 1
        gc.collect()

        if i == steps_per_epoch:
            break
