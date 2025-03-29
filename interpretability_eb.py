import customize_obj
import tensorflow as tf
from deoxys.experiment import DefaultExperimentPipeline
import argparse
import numpy as np
import h5py
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
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("GPU Unavailable")

    parser = argparse.ArgumentParser()
    parser.add_argument("log_folder")
    parser.add_argument("--temp_folder", default='', type=str)
    parser.add_argument("--model_checkpoint_period", default=1, type=int)
    parser.add_argument("--prediction_checkpoint_period", default=1, type=int)
    parser.add_argument("--meta", default='patient_idx', type=str)
    parser.add_argument("--monitor", default='avg_score', type=str)
    parser.add_argument("--monitor_mode", default='max', type=str)
    parser.add_argument("--memory_limit", default=0, type=int)
    args, unknown = parser.parse_known_args()

    print(f'Running Excitation Backprop for: {args.log_folder}')

    exp = DefaultExperimentPipeline(
        log_base_path=args.log_folder,
        temp_base_path=args.temp_folder
    ).load_best_model(
        monitor=args.monitor,
        use_raw_log=False,
        mode=args.monitor_mode,
        custom_modifier_fn=metric_avg_score
    )

    model = exp.model.model
    dr = exp.model.data_reader
    test_gen = dr.test_generator
    steps_per_epoch = test_gen.total_batch

    # Get patient IDs
    pids = []
    with h5py.File(exp.post_processors.dataset_filename, 'r') as f:
        for fold in test_gen.folds:
            pids.append(f[fold][args.meta][:])
    pids = np.concatenate(pids)

    # Create output file
    eb_file_path = f'{args.log_folder}/test_excitation_backprop.h5'
    with h5py.File(eb_file_path, 'w') as f:
        print('Created file', eb_file_path)
        f.create_dataset(args.meta, data=pids)
        f.create_dataset('excitation_backprop', shape=(len(pids), 800, 800))

    i, sub_idx = 0, 0

    for x, _ in test_gen.generate():
        print(f'Processing batch {i+1}/{steps_per_epoch}...')

        for j in range(len(x)):
            image = x[j]
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)

            input_tensor = tf.cast(tf.convert_to_tensor(image[np.newaxis, ...]), tf.float32)

            # Predict class and compute gradients
            with tf.GradientTape() as tape:
                tape.watch(input_tensor)
                predictions = model(input_tensor)
                class_idx = tf.argmax(predictions[0])
                loss = predictions[:, class_idx]

            grads = tape.gradient(loss, input_tensor)[0].numpy()

            # Excitation approximation: keep only positive gradients
            excitation_map = np.maximum(grads, 0)

            # Average over channels if multi-channel
            if excitation_map.ndim == 3:
                excitation_map = excitation_map.mean(axis=-1)

            # Normalize to [0, 1]
            excitation_map -= excitation_map.min()
            excitation_map /= (excitation_map.max() + 1e-8)

            # Resize to 800x800 if needed
            if excitation_map.shape != (800, 800):
                excitation_map = tf.image.resize(excitation_map[..., np.newaxis], (800, 800)).numpy().squeeze()

            # Save
            with h5py.File(eb_file_path, 'a') as f:
                f['excitation_backprop'][sub_idx] = excitation_map
                sub_idx += 1

        i += 1
        gc.collect()
        if i == steps_per_epoch:
            break
