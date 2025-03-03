import customize_obj
import tensorflow as tf
from deoxys.experiment import DefaultExperimentPipeline
import argparse
import numpy as np
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
    base_path = '../results/' + args.log_folder.split('/')[-1]
    ### Number of iterations
    iter = args.iter

    if '2d' in args.log_folder:
        meta = args.meta
    else:
        meta = args.meta.split(',')[0]

    print(f"log_folder: {args.log_folder}")

    # Load the best model based on the chosen metric
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
    # Get the trained model and test data generator
    model = exp.model.model  # Extract the TensorFlow model from Deoxys pipeline
    dr = exp.model.data_reader  # Data reader (handles dataset loading)
    test_gen = dr.test_generator  # Test data generator
    steps_per_epoch = test_gen.total_batch  # Total number of batches in the test set
    batch_size = test_gen.batch_size  # Batch size used during testing

    # Load patient IDs from the dataset
    pids = []
    with h5py.File(exp.post_processors.dataset_filename) as f:
        for fold in test_gen.folds:
            pids.append(f[fold][meta][:])  # Extract patient IDs from each test fold
    pids = np.concatenate(pids)  # Combine IDs from all folds


    with h5py.File(args.log_folder + f'/test_vargrad_02.h5', 'w') as f:
        print('created file', args.log_folder + f'/test_vargrad_02.h5')
        f.create_dataset(meta, data=pids)
        f.create_dataset('vargrad', shape=(len(pids), 800, 800))

    
    i = 0 # Initialize batch index
    sub_idx = 0
    mc_preds = []

    # Loop through test dataset batches
    for x, _ in test_gen.generate():
        print('MC results ....')
        tf.random.set_seed(seed)
        mc_pred = model(x).numpy().flatten()
        mc_preds.append(mc_pred)

        print(f'Processing batch {i+1}/{steps_per_epoch}...')

        np_random_gen = np.random.default_rng(1123)
        new_shape = list(x.shape) + [40]
        var_grad = np.zeros(new_shape)

        # Apply Test-Time Augmentation (TTA) 40 times per sample
        for trial in range(40):
            print(f'Trial {trial+1}/40')
            noise = np_random_gen.normal(
                loc=0.0, scale=.02, size=x.shape[:-1]) * 255
            x_noised = x + np.stack([noise]*3, axis=-1)
            x_noised = tf.Variable(x_noised)
            tf.random.set_seed(seed)
            with tf.GradientTape() as tape:
                tape.watch(x_noised)
                pred = model(x_noised)

            grads = tape.gradient(pred, x_noised).numpy()
            var_grad[..., trial] = grads


        final_var_grad = (var_grad.std(axis=-1)**2).mean(axis=-1)
        with h5py.File(args.log_folder + f'/test_vargrad_02.h5', 'a') as f:
            f['vargrad'][sub_idx:sub_idx + len(x)] = final_var_grad
        sub_idx += x.shape[0]
        i += 1
        gc.collect()

        # Stop early if we reach the end of the test set
        if i == steps_per_epoch:
            break

