import h5py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import tensorflow as tf


base_path = 'P:/CoxaAI/preprocess_data/'
info_path = base_path + 'csv_detection_info_clean/'
included_files = ['AA.csv', 'BB.csv', 'CC.csv',
                  'DD.csv', 'EE.csv']
dfs = []
for file in included_files:
    dfs.append(pd.read_csv(info_path + file))

df = pd.concat(dfs)
# df.to_csv(base_path + 'csv_train_info/Jan2024.csv', index_label='pid')


df = pd.read_csv(base_path + 'csv_train_info/Jan2024.csv')

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=23)
folds = []
for train_idx, test_idx in skf.split(df, df.diagnosis, groups=df.parent_name):
    folds.append(test_idx)
fold_name = np.zeros(df.shape[0], dtype=int)
for i in range(5):
    fold_name[folds[i]] = i


df['fold'] = fold_name

# df.to_csv(base_path + 'csv_train_info/Jan2024_split.csv', index=False)


df = pd.read_csv(base_path + 'csv_train_info/Jan2024_split.csv')


h5_filename = 'P:/CoxaAI/datasets/hips_800.h5'
resize_shape = 800

# create the dataset
with h5py.File(h5_filename, 'w') as f:
    for i in range(len(folds)):
        f.create_group(f'fold_{i}')

for i, fold in enumerate(folds):
    print('writing fold', i)
    images = []
    selected_df = df.iloc[fold]
    real_diagnosis = df.diagnosis[fold].copy()
    target = real_diagnosis.copy()
    # A&B are normal
    target[target > 1] = 1
    for _, item in selected_df.iterrows():
        # year = int(item['year'])
        diagnosis_raw = item['diagnosis_raw']
        crop_name = item['crop_name']
        cropped_fn = f'{base_path}cropped/{diagnosis_raw}/{crop_name}.npy'
        # add an additional dimension
        img = np.load(cropped_fn)[np.newaxis, ..., np.newaxis]
        # resize with bilinear (default)
        img = tf.image.resize_with_pad(img, resize_shape, resize_shape)
        images.append(img)
    images = np.concatenate(images)
    with h5py.File(h5_filename, 'a') as f:
        f[f'fold_{i}'].create_dataset('image', data=images, dtype='f4')
        f[f'fold_{i}'].create_dataset('target', data=target, dtype='f4')
        f[f'fold_{i}'].create_dataset(
            'diagnosis', data=real_diagnosis, dtype='f4')
        f[f'fold_{i}'].create_dataset(
            'patient_idx', data=df.pid[fold], dtype='i4')  # meta data for mapping

with h5py.File(h5_filename, 'r') as f:
    for k in f.keys():
        print(k)
        for ds in f[k].keys():
            print('--', f[k][ds])
