import customize_obj
import tensorflow as tf
from deoxys.experiment import DefaultExperimentPipeline
import argparse
import numpy as np
import h5py
import gc
import cv2
from tensorflow.keras.models import Model
from sklearn.metrics import matthews_corrcoef
from sklearn import metrics

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
    parser.add_argument("--layer_name", default="top_conv", type=str, help="Target convolutional layer for Grad-CAM")

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

    model = exp.model.model
    print("Model layers:")
    for i, layer in enumerate(model.layers):
        print(f"{i}: {layer.name} ({layer.__class__.__name__})")

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

    # Create HDF5 file to store Grad-CAM results
    gradcam_filename = args.log_folder + f'/test_gradcam.h5'
    with h5py.File(gradcam_filename, 'w') as f:
        print(f'Created file: {gradcam_filename}')
        f.create_dataset(args.meta, data=pids)
        f.create_dataset('gradcam', shape=(len(pids), 800, 800), dtype=np.float32)

    i = 0  # Batch index
    sub_idx = 0  # Track processed images

    # Process each batch in the test set
    for x, _ in test_gen.generate():
        print(f'Processing batch {i+1}/{steps_per_epoch}...')

        gradcam_maps = []
        y = model.predict(x)

        for j in range(len(x)):  # Process images one by one
            image = np.expand_dims(x[j], axis=0)
            image_tensor = tf.cast(tf.convert_to_tensor(image), tf.float32)

            class_index = np.argmax(y[j])  # Predicted class

            # Build gradient model
            grad_model = Model(
                inputs=model.input,
                outputs=[
                    model.get_layer(args.layer_name).output,
                    model.output
                ]
            )

            # Grad-CAM computation using guided gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image_tensor)
                loss = predictions[:, class_index]

            grads = tape.gradient(loss, conv_outputs)

            # Guided backpropagation masking
            cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
            cast_grads = tf.cast(grads > 0, "float32")
            guided_grads = cast_conv_outputs * cast_grads * grads

            conv_outputs = conv_outputs[0]  # Remove batch
            guided_grads = guided_grads[0]

            # Compute weights and CAM
            weights = tf.reduce_mean(guided_grads, axis=(0, 1))
            cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

            heatmap = cam.numpy()

            # Resize using OpenCV to (800x800)
            heatmap = cv2.resize(heatmap, (800, 800))

            # Normalize to [0, 255]
            numer = heatmap - np.min(heatmap)
            denom = (np.max(heatmap) - np.min(heatmap)) + 1e-8
            heatmap = numer / denom
            heatmap = (heatmap * 255).astype("uint8")

            gradcam_maps.append(heatmap)

        # Save to HDF5 file
        with h5py.File(gradcam_filename, 'a') as f:
            f['gradcam'][sub_idx:sub_idx + len(x)] = gradcam_maps

        sub_idx += x.shape[0]
        i += 1
        gc.collect()

        if i == steps_per_epoch:
            break
