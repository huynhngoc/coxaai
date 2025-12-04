import customize_obj
import tensorflow as tf
from deoxys.experiment import DefaultExperimentPipeline
from deoxys.model import load_model
import argparse
import numpy as np
import os
import h5py
import pandas as pd
from deoxys.data.preprocessor import preprocessor_from_config
import json
import gc
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef

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
    parser.add_argument("log_folder")
    parser.add_argument("--iter", default=40, type=int)
    ### Temporary folder for intermediate results
    parser.add_argument("--temp_folder", default='', type=str)
    ### Model saving frequency
    parser.add_argument("--model_checkpoint_period", default=1, type=int)
    ### Prediction saving frequency
    parser.add_argument("--prediction_checkpoint_period", default=1, type=int)
    ### Metadata identifier (e.g, patient ID)
    parser.add_argument("--meta", default='patient_idx', type=str)
    ### Metric to monitor for best model selection
    parser.add_argument("--monitor", default='avg_score', type=str)
    ### Maximize the monitered metrics
    parser.add_argument("--monitor_mode", default='max', type=str)
    ### Optional GPU memory limit
    parser.add_argument("--memory_limit", default=0, type=int)

    args, unknown = parser.parse_known_args()

    ### Set base path for results based on experiment name
    base_path = '../vargrad/' + args.log_folder.split('/')[-1]
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    ### Number of iterations
    n_iter = args.iter

    if '2d' in args.log_folder:
        meta = args.meta
    else:
        meta = args.meta.split(',')[0]

    print(f"log_folder: {args.log_folder}")

    # Load the best model based on the chosen metric
    ### Load the best model based on validation performance
    with open(f'{args.log_folder}/info.txt', 'r') as f:
        best_epoch = f.readlines()[0][-25:-22]
    best_model = load_model(args.log_folder + f'/model/model.{best_epoch}.h5')


    # Get the trained model and test data generator
    model = best_model.model  # Extract the TensorFlow model from Deoxys pipeline
    dr = best_model.data_reader  # Data reader (handles dataset loading)
    val_gen = dr.val_generator  # Validation data generator
    val_steps_per_epoch = val_gen.total_batch  # Total number of batches in the validation set
    val_batch_size = val_gen.batch_size  # Batch size used during validation
    test_gen = dr.test_generator  # Test data generator
    test_steps_per_epoch = test_gen.total_batch  # Total number of batches in the test set
    test_batch_size = test_gen.batch_size  # Batch size used during testing


    seed = 1
    # Load patient IDs from the dataset
    pids = []
    diagnosis = []
    with h5py.File('../datasets/hips_800_sort_7_clean.h5') as f:
        for fold in val_gen.folds:
            pids.append(f[fold][args.meta][:])  # Extract patient IDs from each val fold
            diagnosis.append(f[fold]['diagnosis'][:])  # Extract diagnosis from each val fold
    diagnosis = np.concatenate(diagnosis)
    pids = np.concatenate(pids)  # Combine IDs from all folds

    with h5py.File(base_path + f'/val_vargrad_05.h5', 'w') as f:
        print('created file', base_path + f'/val_vargrad_05.h5')
        f.create_dataset('pid', data=pids)
        f.create_dataset('diagnosis', data=diagnosis)
        f.create_dataset('tta_pred', shape=(len(pids), 4, n_iter))
        f.create_dataset('vargrad_A', shape=(len(pids), 800, 800))
        f.create_dataset('vargrad_B', shape=(len(pids), 800, 800))
        f.create_dataset('vargrad_C', shape=(len(pids), 800, 800))
        f.create_dataset('vargrad_D', shape=(len(pids), 800, 800))
        f.create_dataset('vargrad_E', shape=(len(pids), 800, 800))
        f.create_dataset('vargrad', shape=(len(pids), 800, 800))

    i = 0 # Initialize batch index
    sub_idx = 0
    # mc_preds = []
    keys = ['B', 'C', 'D', 'E']

    # Loop through val dataset batches
    for x, _ in val_gen.generate():
        print('MC results ....')
        tf.random.set_seed(seed)
        # mc_pred = model(x).numpy().flatten()
        # mc_preds.append(mc_pred)

        print(f'Processing batch {i+1}/{val_steps_per_epoch}...')

        np_random_gen = np.random.default_rng(1123)
        new_shape = list(x.shape) + [n_iter]
        var_grad = {key: np.zeros(new_shape) for key in ['A', 'B', 'C', 'D', 'E', 'all']}
        tta_pred = np.zeros((x.shape[0], 4, n_iter))

        # Generate n_iter noisy samples for each image in the batch
        for trial in range(n_iter):
            print(f'Trial {trial+1}/{n_iter}')
            noise = np_random_gen.normal(
                loc=0.0, scale=.05, size=x.shape[:-1]) * 255
            x_noised = x + np.stack([noise]*3, axis=-1)
            x_noised = tf.Variable(x_noised)
            tf.random.set_seed(seed)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x_noised)
                pred = model(x_noised)
            tta_pred[..., trial] = pred.numpy()
            var_grad['A'][..., trial] = tape.gradient(1 - pred[..., 0], x_noised).numpy()
            for i in range(pred.shape[1]):
                var_grad[keys[i]][..., trial] = tape.gradient(pred[..., i], x_noised).numpy()
            var_grad['all'][..., trial] = tape.gradient(tf.reduce_sum(pred, axis=-1), x_noised).numpy()
            del tape  # Free resources

        final_var_grad = {key: (data.std(axis=-1)**2).mean(axis=-1) for key, data in var_grad.items()}
        with h5py.File(args.log_folder + f'/val_vargrad_05.h5', 'a') as f:
            f['tta_pred'][sub_idx:sub_idx + len(x)] = tta_pred
            f['vargrad_A'][sub_idx:sub_idx + len(x)] = final_var_grad['A']
            f['vargrad_B'][sub_idx:sub_idx + len(x)] = final_var_grad['B']
            f['vargrad_C'][sub_idx:sub_idx + len(x)] = final_var_grad['C']
            f['vargrad_D'][sub_idx:sub_idx + len(x)] = final_var_grad['D']
            f['vargrad_E'][sub_idx:sub_idx + len(x)] = final_var_grad['E']
            f['vargrad'][sub_idx:sub_idx + len(x)] = final_var_grad['all']

        sub_idx += x.shape[0]
        i += 1
        gc.collect()

        # Stop early if we reach the end of the val set
        if i == val_steps_per_epoch:
            break

    # Load patient IDs from the dataset
    pids = []
    diagnosis = []
    with h5py.File('../datasets/hips_800_sort_7_clean.h5') as f:
        for fold in test_gen.folds:
            pids.append(f[fold][args.meta][:])  # Extract patient IDs from each test fold
            diagnosis.append(f[fold]['diagnosis'][:])  # Extract diagnosis from each test fold
    diagnosis = np.concatenate(diagnosis)
    pids = np.concatenate(pids)  # Combine IDs from all folds


    with h5py.File(base_path + f'/val_vargrad_05.h5', 'w') as f:
        print('created file', base_path + f'/val_vargrad_05.h5')
        f.create_dataset('pid', data=pids)
        f.create_dataset('diagnosis', data=diagnosis)
        f.create_dataset('tta_pred', shape=(len(pids), 4, n_iter))
        f.create_dataset('vargrad_A', shape=(len(pids), 800, 800))
        f.create_dataset('vargrad_B', shape=(len(pids), 800, 800))
        f.create_dataset('vargrad_C', shape=(len(pids), 800, 800))
        f.create_dataset('vargrad_D', shape=(len(pids), 800, 800))
        f.create_dataset('vargrad_E', shape=(len(pids), 800, 800))
        f.create_dataset('vargrad', shape=(len(pids), 800, 800))

    i = 0 # Initialize batch index
    sub_idx = 0
    # mc_preds = []
    tta_preds = []
    keys = ['B', 'C', 'D', 'E']

    # Loop through test dataset batches
    for x, _ in test_gen.generate():
        print('MC results ....')
        # tf.random.set_seed(seed)
        # mc_pred = model(x).numpy().flatten()
        # mc_preds.append(mc_pred)

        print(f'Processing batch {i+1}/{test_steps_per_epoch}...')

        np_random_gen = np.random.default_rng(1123)
        new_shape = list(x.shape) + [n_iter]
        var_grad = {key: np.zeros(new_shape) for key in ['A', 'B', 'C', 'D', 'E', 'all']}
        tta = np.zeros((x.shape[0], 4, n_iter))

        # Generate n_iter noisy samples for each image in the batch
        for trial in range(n_iter):
            print(f'Trial {trial+1}/{n_iter}')
            noise = np_random_gen.normal(
                loc=0.0, scale=.05, size=x.shape[:-1]) * 255
            x_noised = x + np.stack([noise]*3, axis=-1)
            x_noised = tf.Variable(x_noised)
            tf.random.set_seed(seed)
            with tf.GradientTape() as tape:
                tape.watch(x_noised)
                pred = model(x_noised)
            tta[..., trial] = pred.numpy()
            var_grad['A'][..., trial] = tape.gradient(1 - pred[..., 0], x_noised).numpy()
            for i in range(pred.shape[1]):
                var_grad[keys[i]][..., trial] = tape.gradient(pred[..., i], x_noised).numpy()
            var_grad['all'][..., trial] = tape.gradient(tf.reduce_sum(pred, axis=-1), x_noised).numpy()


        final_var_grad = {key: (data.std(axis=-1)**2).mean(axis=-1) for key, data in var_grad.items()}
        with h5py.File(args.log_folder + f'/val_vargrad_05.h5', 'a') as f:
            f['tta_pred'][sub_idx:sub_idx + len(x)] = tta
            f['vargrad_A'][sub_idx:sub_idx + len(x)] = final_var_grad['A']
            f['vargrad_B'][sub_idx:sub_idx + len(x)] = final_var_grad['B']
            f['vargrad_C'][sub_idx:sub_idx + len(x)] = final_var_grad['C']
            f['vargrad_D'][sub_idx:sub_idx + len(x)] = final_var_grad['D']
            f['vargrad_E'][sub_idx:sub_idx + len(x)] = final_var_grad['E']
            f['vargrad'][sub_idx:sub_idx + len(x)] = final_var_grad['all']

        sub_idx += x.shape[0]
        i += 1
        gc.collect()

        # Stop early if we reach the end of the test set
        if i == test_steps_per_epoch:
            break
