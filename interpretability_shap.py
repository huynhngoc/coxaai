import customize_obj
import tensorflow as tf
from deoxys.experiment import DefaultExperimentPipeline
import argparse
import numpy as np
import h5py
import gc
import shap
from tensorflow.keras.models import Model
from skimage.transform import resize
from sklearn.metrics import matthews_corrcoef
from sklearn import metrics

# Define Matthews Correlation Coefficient scorer
class Matthews_corrcoef_scorer:
    def __call__(self, *args, **kwargs):
        return matthews_corrcoef(*args, **kwargs)

    def _score_func(self, *args, **kwargs):
        return matthews_corrcoef(*args, **kwargs)

# Register MCC scorer in sklearn
try:
    metrics.SCORERS['mcc'] = Matthews_corrcoef_scorer()
except:
    pass
try:
    metrics._scorer._SCORERS['mcc'] = Matthews_corrcoef_scorer()
except:
    pass

# Compute average metric score for best model selection
def metric_avg_score(res_df, postprocessor):
    """Compute the average score for model selection"""
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
    parser.add_argument("log_folder", type=str, help="Path to the experiment log folder")
    parser.add_argument("--iter", default=40, type=int, help="Number of iterations")
    parser.add_argument("--temp_folder", default='', type=str, help="Temporary folder for intermediate results")
    parser.add_argument("--model_checkpoint_period", default=1, type=int)
    parser.add_argument("--prediction_checkpoint_period", default=1, type=int)
    parser.add_argument("--meta", default="patient_idx", type=str, help="Meta identifier (e.g., patient ID)")
    parser.add_argument("--monitor", default="avg_score", type=str, help="Metric for best model selection")
    parser.add_argument("--monitor_mode", default="max", type=str, help="Optimization direction for monitored metric")
    parser.add_argument("--memory_limit", default=0, type=int)
    parser.add_argument("--background_samples", default=50, type=int, help="Number of background samples for SHAP")

    args, unknown = parser.parse_known_args()

    print(f"Using log folder: {args.log_folder}")

    # Load the best model using avg_score
    exp = DefaultExperimentPipeline(
        log_base_path=args.log_folder,
        temp_base_path=args.temp_folder
    ).load_best_model(
        monitor=args.monitor,
        use_raw_log=False,
        mode=args.monitor_mode,
        custom_modifier_fn=metric_avg_score  # Apply avg_score function
    )

    # Extract model & test generator
    model = exp.model.model  
    dr = exp.model.data_reader  
    test_gen = dr.test_generator  
    steps_per_epoch = test_gen.total_batch  
    batch_size = test_gen.batch_size  

    # Load patient IDs
    pids = []
    with h5py.File(exp.post_processors.dataset_filename, 'r') as f:
        for fold in test_gen.folds:
            pids.append(f[fold][args.meta][:])
    pids = np.concatenate(pids)

    # Create HDF5 file to store SHAP results
    shap_filename = args.log_folder + f'/test_shap.h5'
    with h5py.File(shap_filename, 'w') as f:
        print(f'Created file: {shap_filename}')
        f.create_dataset(args.meta, data=pids)
        f.create_dataset('shap_values', shape=(len(pids), 800, 800), dtype=np.float32)

    i = 0  # Batch index
    sub_idx = 0  # Track processed images

    # Select a small subset of the dataset to use as a "background" set for SHAP
    background_data = []
    for x, _ in test_gen.generate():
        background_data.append(x)
        i += 1
        if i >= args.background_samples:  # Limit number of background samples
            break

    background_data = np.concatenate(background_data)  # Combine all background images
    background_data = background_data[:args.background_samples]  # Use only required samples

    # Initialize SHAP explainer with the background dataset
    print("Initializing SHAP explainer...")
    explainer = shap.GradientExplainer(model, background_data)

    i = 0  # Batch index
    sub_idx = 0  # Track processed images

    # Process each batch in the test set
    for x, _ in test_gen.generate():
        print(f'Processing batch {i+1}/{steps_per_epoch}...')

        # Compute SHAP values
        shap_values = explainer.shap_values(x)  # Get SHAP values for each pixel
        shap_map = np.array(shap_values).mean(axis=0)  # Average across multiple channels

        # Normalize SHAP values (optional, but helps with visualization)
        shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min())

        # Resize SHAP maps to match the image size
        resized_shap_maps = np.array([resize(img, (800, 800)) for img in shap_map])

        # Save to HDF5 file
        with h5py.File(shap_filename, 'a') as f:
            f['shap_values'][sub_idx:sub_idx + len(x)] = resized_shap_maps

        sub_idx += x.shape[0]
        i += 1
        gc.collect()

        if i == steps_per_epoch:
            break

    print(f"SHAP processing completed. Results saved to {shap_filename}")

