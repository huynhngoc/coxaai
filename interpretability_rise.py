import numpy as np
import tensorflow as tf
import h5py
import argparse
import gc
from deoxys.experiment import DefaultExperimentPipeline
from tensorflow.keras.models import Model
from sklearn.metrics import matthews_corrcoef
from sklearn import metrics

# Define Matthews Correlation Coefficient (MCC) scorer
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

# Compute average metric score for model selection
def metric_avg_score(res_df, postprocessor):
    res_df['avg_score'] = res_df[['AUC', 'roc_auc', 'f1', 'f1_0', 'BinaryAccuracy', 'mcc']].mean(axis=1)
    return res_df

def generate_random_masks(img_size, num_masks=1000, mask_size=8, p=0.5):
    """
    Generate random binary masks using NumPy and TensorFlow.
    - img_size: (H, W) -> Size of input image
    - num_masks: Number of masks to generate
    - mask_size: Downsampled mask resolution (e.g., 8x8)
    - p: Probability of keeping pixels
    """
    small_masks = np.random.choice([0, 1], size=(num_masks, mask_size, mask_size, 1), p=[1-p, p])
    
    # Resize masks to image size using TensorFlow instead of OpenCV
    masks = np.array([tf.image.resize(m.astype(np.float32), img_size).numpy() for m in small_masks])
    
    return np.expand_dims(masks, axis=-1)  # Add channel dimension

def compute_rise_heatmap(model, image, num_masks=1000):
    """
    Compute RISE importance heatmap for a single image.
    - model: Trained model
    - image: Single input image (H, W, C)
    - num_masks: Number of random masks
    """
    img_size = image.shape[:2]
    masks = generate_random_masks(img_size, num_masks)

    masked_images = np.multiply(image, masks)  # Apply masks to the image
    preds = model.predict(masked_images, batch_size=32)  # Run inference

    class_idx = np.argmax(model.predict(np.expand_dims(image, axis=0)))  # Get predicted class
    scores = preds[:, class_idx]  # Extract scores for the predicted class

    heatmap = np.tensordot(scores, masks, axes=(0, 0))  # Weighted sum of masks
    heatmap /= np.max(heatmap)  # Normalize to [0,1]

    return heatmap

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
    parser.add_argument("--layer_name", default="top_conv", type=str, help="Target convolutional layer for Grad-CAM")
    parser.add_argument("--num_masks", default=1000, type=int, help="Number of random masks")
    parser.add_argument("--mask_size", default=8, type=int, help="Resolution of downsampled masks")

    args, unknown = parser.parse_known_args()

    ### Set base path for results based on experiment name
    base_path = '../results/' + args.log_folder.split('/')[-1]

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

    # Create HDF5 file to store RISE results
    rise_filename = args.log_folder + f'/test_rise.h5'
    with h5py.File(rise_filename, 'w') as f:
        print(f'Created file: {rise_filename}')
        f.create_dataset(args.meta, data=pids)
        f.create_dataset('rise_heatmap', shape=(len(pids), 800, 800), dtype=np.float32)

    i = 0  # Batch index
    sub_idx = 0  # Track processed images

    # Process each batch in the test set
    for x, _ in test_gen.generate():
        print(f'Processing batch {i+1}/{steps_per_epoch}...')

        rise_maps = []

        for j in range(len(x)):  # Process images one by one
            image = x[j]  # Single image
            heatmap = compute_rise_heatmap(model, image, num_masks=args.num_masks)  # Compute RISE map
            
            # Resize to match original image size using TensorFlow
            heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (800, 800)).numpy().squeeze()
            rise_maps.append(heatmap_resized)

        # Save to HDF5 file
        with h5py.File(rise_filename, 'a') as f:
            f['rise_heatmap'][sub_idx:sub_idx + len(x)] = rise_maps

        sub_idx += x.shape[0]
        i += 1
        gc.collect()

        if i == steps_per_epoch:
            break

    print(f"RISE processing completed. Results saved to {rise_filename}")
