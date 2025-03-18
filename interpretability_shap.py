import customize_obj
import tensorflow as tf
from deoxys.experiment import DefaultExperimentPipeline
import argparse
import numpy as np
import h5py
import gc
import shap
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Lambda
from skimage.transform import resize
from sklearn.metrics import matthews_corrcoef
from sklearn import metrics
import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
import tensorflow as tf

# Function to replace BatchNorm layers with identity function
def remove_batchnorm_layers(model):
    inputs = model.input
    x = inputs
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            x = Lambda(lambda y: tf.identity(y), name=layer.name + "_removed")(x)
        else:
            x = layer(x)
    return Model(inputs, x)

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

    # Enable memory growth to avoid OOM errors
    tf.config.experimental.set_memory_growth(gpus[0], True)

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
    parser.add_argument("--background_samples", default=5, type=int, help="Number of background samples for SHAP")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size for SHAP processing")

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
    batch_size = min(args.batch_size, test_gen.batch_size)  # Use a very small batch size

    # Remove BatchNorm layers to avoid SHAP errors
    modified_model = remove_batchnorm_layers(model)

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

    # Select a small subset of the dataset to use as a "background" set for SHAP
    print("Selecting background samples for SHAP...")
    background_data = []
    for x, _ in test_gen.generate():
        background_data.append(x)
        if len(background_data) >= args.background_samples:  # Limit background samples
            break

    background_data = np.concatenate(background_data, axis=0)[:args.background_samples]  # Restrict number of samples

    # Initialize SHAP explainer using DeepExplainer on the modified model
    print("Initializing SHAP explainer...")
    explainer = shap.DeepExplainer(modified_model, background_data)

    i = 0  # Batch index
    sub_idx = 0  # Track processed images

    # Process each batch in the test set
    for x, _ in test_gen.generate():
        print(f'Processing batch {i+1}/{steps_per_epoch}...')

        # Compute SHAP values for each image individually to reduce memory usage
        shap_maps = []
        for img in x:
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            shap_values = explainer.shap_values(img)  # Get SHAP values
            shap_map = np.array(shap_values).mean(axis=0)  # Average across channels
            shap_map = np.squeeze(shap_map)  # Remove batch dimension

            # Normalize SHAP values
            if shap_map.max() > shap_map.min():
                shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min())

            # Resize to match original image size
            shap_map = resize(shap_map, (800, 800))
            shap_maps.append(shap_map)

        # Convert to NumPy array
        shap_maps = np.array(shap_maps)

        # Save to HDF5 file
        with h5py.File(shap_filename, 'a') as f:
            f['shap_values'][sub_idx:sub_idx + len(x)] = shap_maps

        sub_idx += len(x)
        i += 1
        gc.collect()

        if i == steps_per_epoch:
            break

    print(f"SHAP processing completed. Results saved to {shap_filename}")
