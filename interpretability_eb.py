import customize_obj
import tensorflow as tf
from deoxys.experiment import DefaultExperimentPipeline
import argparse
import numpy as np
import h5py
import gc
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
    res_df['avg_score'] = res_df[['AUC', 'roc_auc', 'f1', 'f1_0',
                                  'BinaryAccuracy', 'mcc']].mean(axis=1)
    return res_df

def excitation_backprop_tf(model, image, class_idx=None):
    image = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)
    image = tf.Variable(image)

    with tf.GradientTape() as tape:
        tape.watch(image)
        preds = model(image)
        if class_idx is None:
            class_idx = 0  # Only one output unit in binary case
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, image)[0].numpy()

    # ReLU-style masking (positive gradients only)
    grads = np.maximum(grads, 0)

    # Element-wise multiplication with input
    eb_map = grads * image.numpy()[0]

    # Average channels
    if eb_map.ndim == 3:
        eb_map = eb_map.mean(axis=-1)

    # Normalize to [0, 1]
    eb_map -= eb_map.min()
    eb_map /= (eb_map.max() + 1e-8)

    return eb_map


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

    pids = []
    with h5py.File(exp.post_processors.dataset_filename, 'r') as f:
        for fold in test_gen.folds:
            pids.append(f[fold][args.meta][:])
    pids = np.concatenate(pids)

    eb_file_path = f'{args.log_folder}/test_eb.h5'
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
            saliency = excitation_backprop_tf(model, image)

            # Save saliency map
            with h5py.File(eb_file_path, 'a') as f:
                f['excitation_backprop'][sub_idx] = saliency
                sub_idx += 1

        i += 1
        gc.collect()
        if i == steps_per_epoch:
            break
