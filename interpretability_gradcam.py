from sklearn.metrics import matthews_corrcoef
from sklearn import metrics
import customize_obj
import tensorflow as tf
from deoxys.experiment import DefaultExperimentPipeline
import argparse
import numpy as np
import h5py
import gc
from tensorflow.keras.models import Model

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
    parser.add_argument("--layer_name", default="conv5_block3_out", type=str)  # Change this to the last conv layer
    parser.add_argument("--meta", default='patient_idx', type=str)
    
    args, unknown = parser.parse_known_args()
    
    print(f"log_folder: {args.log_folder}")

    # Load the best model from Deoxys
    exp = DefaultExperimentPipeline(
        log_base_path=args.log_folder
    ).load_best_model(
        monitor='avg_score',
        use_raw_log=False,
        mode='max'
    )

    # Get model and test data
    model = exp.model.model  
    dr = exp.model.data_reader  
    test_gen = dr.test_generator  
    steps_per_epoch = test_gen.total_batch  
    batch_size = test_gen.batch_size  

    # Load patient IDs
    pids = []
    with h5py.File(exp.post_processors.dataset_filename) as f:
        for fold in test_gen.folds:
            pids.append(f[fold][args.meta][:])
    pids = np.concatenate(pids)

    # Create an output HDF5 file for Grad-CAM results
    with h5py.File(args.log_folder + f'/test_gradcam.h5', 'w') as f:
        print('Created file', args.log_folder + f'/test_gradcam.h5')
        f.create_dataset(args.meta, data=pids)
        f.create_dataset('gradcam', shape=(len(pids), 800, 800))

    i = 0  # Batch index
    sub_idx = 0  # Keeps track of processed images

    # Create Grad-CAM model
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(args.layer_name).output, model.output]
    )

    # Loop through the test set
    for x, y in test_gen.generate():
        print(f'Processing batch {i+1}/{steps_per_epoch}...')

        gradcam_maps = []

        for j in range(len(x)):  # Process images one by one
            image = np.expand_dims(x[j], axis=0)  # Add batch dimension
            class_index = np.argmax(y[j])  # Get the predicted class

            # Compute Grad-CAM
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(image)
                loss = predictions[:, class_index]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # Compute weighted activation map
            conv_outputs = conv_outputs[0].numpy()
            pooled_grads = pooled_grads.numpy()
            for k in range(pooled_grads.shape[-1]):
                conv_outputs[..., k] *= pooled_grads[k]

            # Compute heatmap
            gradcam_map = np.mean(conv_outputs, axis=-1)
            gradcam_map = np.maximum(gradcam_map, 0)  # Apply ReLU
            gradcam_map /= np.max(gradcam_map)  # Normalize to [0,1]

            # Resize to match original image size
            gradcam_map = tf.image.resize(gradcam_map[..., np.newaxis], (800, 800)).numpy().squeeze()

            gradcam_maps.append(gradcam_map)

        gradcam_maps = np.array(gradcam_maps)

        # Save to HDF5 file
        with h5py.File(args.log_folder + f'/test_gradcam.h5', 'a') as f:
            f['gradcam'][sub_idx:sub_idx + len(x)] = gradcam_maps

        sub_idx += x.shape[0]
        i += 1
        gc.collect()

        if i == steps_per_epoch:
            break
