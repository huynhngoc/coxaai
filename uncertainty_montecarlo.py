import customize_obj
import tensorflow as tf
from deoxys.experiment import DefaultExperimentPipeline
import argparse
import numpy as np
import h5py
import pandas as pd
from deoxys.data.preprocessor import preprocessor_from_config
import json
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
            pids.append(f[fold][args.meta][:])  # Extract patient IDs from each test fold
    pids = np.concatenate(pids)  # Combine IDs from all folds

    # List to store mc predictions
    mc_preds = []
    i = 0 # Initialize batch index


    # Loop through test dataset batches
    for x, _ in test_gen.generate():
        print('Running MC...')
        print(f'Processing batch {i+1}/{steps_per_epoch}...')

        mc_pred = np.zeros((x.shape[0], 40))  # Placeholder for MC predictions (40 trials per sample)

        for trial in range(40):
            print(f'Trial {trial+1}/40')
            #mc_pred[..., trial] = model(x).numpy().flatten()  # this will run the model with dropout and store the results
            mc_pred[..., trial] = model(x, training=True).numpy().flatten() # Set training=True to enable dropout during inference mode

        mc_preds.append(mc_pred)  # Store predictions for this batch

        i += 1 # increment batch index

        # Stop early if we reach the end of the test set
        if i == steps_per_epoch:
            break

    # Convert results into a DataFrame
    df = pd.DataFrame({'pid': pids})  # Start with patient IDs
    mc_preds = np.concatenate(mc_preds)  # Combine predictions across batches

    # Add MC predictions to the DataFrame
    for trial in range(40):
        df[f'mc_pred_{trial}'] = mc_preds[..., trial]  # Store each MC trial result in a separate column

    # Save the final results as a CSV file
    df.to_csv(args.log_folder + f'/mc_predicted.csv', index=False)