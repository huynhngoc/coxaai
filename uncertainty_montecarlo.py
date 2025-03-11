import matplotlib.pyplot as plt
from deoxys.customize import custom_layer
from deoxys.model import load_model
from deoxys.customize import custom_layer
from deoxys.model.model import model_from_full_config
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.python.keras.backend import dropout
import tensorflow as tf
import argparse
import os
import h5py
import pandas as pd
from deoxys.experiment import DefaultExperimentPipeline


@custom_layer
class MonteCarloDropout(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("GPU Unavailable")

# Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config")  # Model config file
    parser.add_argument("log_folder") # Output folder
    parser.add_argument("--iter", default=40, type=int) # Number of Monte Carlo iterations
    parser.add_argument("--dropout_rate", default=0.1, type=float) # Dropout rate
    parser.add_argument('--temp_folder', default ='', type = str)


    args, unknown = parser.parse_known_args()

    # Set base path for results
    base_path = f'../results/{args.log_folder.split('/')[-1]}'
    os.makedirs(base_path, exist_ok=True)

    print(f"Saving Monte Carlo predictions to: {base_path}")
    print(f"Running {args.iter} Monte Carlo trials with Dropout Rate {args.dropout_rate}...")
    
     # Load the trained model with Dropout
    exp = DefaultExperimentPipeline(
        log_base_path=args.log_folder,
        temp_base_path=args.temp_folder
    ).load_best_model()

    model = exp.model.model  # Extract the TensorFlow model

    # Ensure dropout rate is applied dynamically
    for layer in model.layers:
        if isinstance(layer, Dropout):
            layer.rate = args.dropout_rate  # Apply the dropout rate

    dr = exp.model.data_reader  # Data reader
    test_gen = dr.test_generator  # Test data generator
    steps_per_epoch = test_gen.total_batch  # Number of test batches
    batch_size = test_gen.batch_size  # Batch size

    # Extract patient IDs from dataset
    pids = []
    with h5py.File(exp.post_processors.dataset_filename) as f:
        for fold in test_gen.folds:
            pids.append(f[fold]['patient_idx'][:])  # Extract patient IDs
    pids = np.concatenate(pids)  # Combine IDs from all folds

    # Monte Carlo predictions storage
    mc_preds = []
    i = 0  # Batch index

    for x, _ in test_gen.generate():
        print(f"Processing batch {i+1}/{steps_per_epoch}...")

        # Run Monte Carlo Dropout `args.iter` times
        preds = np.array([model.predict(x) for _ in range(args.iter)])  # Shape: (iter, batch_size, num_classes)
        mc_preds.append(preds)  # Append batch predictions

        i += 1
        if i == steps_per_epoch:
            break  # Stop when all batches are processed

    # Convert Monte Carlo predictions into the same format as TTA
    mc_preds = np.concatenate(mc_preds)  # Shape: (iter, total_samples, num_classes)

    # Convert results into a DataFrame (same as TTA)
    df = pd.DataFrame({'pid': pids})  # Start with patient IDs
    for trial in range(args.iter):
        df[f'mc_pred_{trial}'] = mc_preds[trial].flatten()  # Store each MC trial result in a separate column

    # Save the final results as a CSV file
    df.to_csv(os.path.join(args.log_folder, f'mc_predicted.csv'), index=False)

    print(f"Monte Carlo predictions saved in {args.log_folder}/mc_predicted.csv")

 