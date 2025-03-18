import numpy as np
import tensorflow as tf
import h5py
import argparse
import gc
import cv2
from deoxys.experiment import DefaultExperimentPipeline
from tensorflow.keras.models import Model
from sklearn.metrics import matthews_corrcoef
from sklearn import metrics

# Define MCC scorer
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

# Compute avg_score for best model selection
def metric_avg_score(res_df, postprocessor):
    res_df['avg_score'] = res_df[['AUC', 'roc_auc', 'f1', 'f1_0', 'BinaryAccuracy', 'mcc']].mean(axis=1)
    return res_df

def generate_random_masks(img_size, num_masks=1000, mask_size=8, p=0.5):
    """
    Generate random binary masks.
    - img_size: (H, W)
    - num_masks: Number of masks
    - mask_size: Downsampled mask resolution
    - p: Probability of keeping pixels
    """
    small_masks = np.random.choice([0, 1], size=(num_masks, mask_size, mask_size, 1), p=[1-p, p])
    masks = np.array([cv2.resize(m, img_size, interpolation=cv2.INTER_LINEAR) for m in small_masks])
    return np.expand_dims(masks, axis=-1)

def compute_rise_heatmap(model, image, num_masks=1000):
    """
    Compute RISE importance heatmap for a single image.
    - model: Trained model
    - image: Single image (H, W, C)
    - num_masks: Number of random masks
    """
    img_size = image.shape[:2]
    masks = generate_random_masks(img_size, num_masks)

    masked_images = np.multiply(image, masks)  # Apply masks to the image
    preds = model.predict(masked_images, batch_size=32)

    class_idx = np.argmax(model.predict(np.expand_dims(image, axis=0)))  # Get predicted class
    scores = preds[:, class_idx]  # Extract scores for the predicted class

    heatmap = np.tensordot(scores, masks, axes=(0, 0))  # Weighted sum
    heatmap /= np.max(heatmap)  # Normalize

    return heatmap

# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("log_folder", type=str, help="Path to experiment log folder")
    parser.add_argument("--num_masks", default=1000, type=int, help="Number of random masks")
    parser.add_argument("--mask_size", default=8, type=int, help="Resolution of masks")
    parser.add_argument("--meta", default="patient_idx", type=str, help="Meta identifier")
    parser.add_argument("--monitor", default="avg_score", type=str, help="Metric for best model selection")
    parser.add_argument("--monitor_mode", default="max", type=str, help="Optimization direction")
    args = parser.parse_args()

    # Load best model using avg_score
    exp = DefaultExperimentPipeline(
        log_base_path=args.log_folder
    ).load_best_model(
        monitor=args.monitor,
        use_raw_log=False,
        mode=args.monitor_mode,
        custom_modifier_fn=metric_avg_score  # Apply avg_score function
    )

    model = exp.model.model
    dr = exp.model.data_reader
    test_gen = dr.test_generator
    steps_per_epoch = test_gen.total_batch
    batch_size = test_gen.batch_size

    pids = np.concatenate([f[args.meta][:] for fold in test_gen.folds])

    # Create HDF5 file to store RISE heatmaps
    rise_filename = args.log_folder + f'/test_rise.h5'
    with h5py.File(rise_filename, 'w') as f:
        f.create_dataset(args.meta, data=pids)
        f.create_dataset('rise_heatmap', shape=(len(pids), 800, 800), dtype=np.float32)

    i, sub_idx = 0, 0
    for x, _ in test_gen.generate():
        print(f'Processing batch {i+1}/{steps_per_epoch}...')
        rise_maps = []

        for j in range(len(x)):  # Process each image
            image = x[j]  # Single image
            heatmap = compute_rise_heatmap(model, image, num_masks=args.num_masks)  # Compute RISE map
            heatmap_resized = cv2.resize(heatmap, (800, 800))  # Resize to match output
            rise_maps.append(heatmap_resized)

        with h5py.File(rise_filename, 'a') as f:
            f['rise_heatmap'][sub_idx:sub_idx + len(x)] = rise_maps

        sub_idx += x.shape[0]
        i += 1
        gc.collect()

        if i == steps_per_epoch:
            break

    print(f"RISE processing completed. Results saved to {rise_filename}")
