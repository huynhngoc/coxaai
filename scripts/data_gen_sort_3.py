import h5py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import tensorflow as tf
import random


base_path = 'P:/CoxaAI/preprocess_data/'
info_path = base_path + 'csv_detection_info_clean/'
included_files = ['AA.csv', 'BB.csv', 'CC.csv',
                  'DD.csv', 'EE.csv', 'sortering 2/AA.csv',
                  'sortering 2/BB.csv', 'sortering 2/CC.csv',
                  'sortering 2/DD.csv', 'sortering 2/EE.csv',
                  'sortering 3/AA.csv',
                  'sortering 3/BB.csv', 'sortering 3/CC.csv',
                  'sortering 3/DD.csv', 'sortering 3/EE.csv',]
dfs = []
for file in included_files:
    dfs.append(pd.read_csv(info_path + file))

df = pd.concat(dfs).reset_index(drop=True)
# df.to_csv(base_path + 'csv_train_info/Feb2024_sort3.csv', index_label='pid')


df = pd.read_csv(base_path + 'csv_train_info/Feb2024_sort3.csv')

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=23)
folds = []
for train_idx, test_idx in skf.split(df, df.diagnosis, groups=df.parent_name):
    folds.append(test_idx)
fold_name = np.zeros(df.shape[0], dtype=int)
for i in range(5):
    fold_name[folds[i]] = i


df['fold'] = fold_name

# df.to_csv(base_path + 'csv_train_info/Feb2024_sort3_split.csv', index=False)

df = pd.read_csv(base_path + 'csv_train_info/Feb2024_sort3_split.csv')


h5_filename = 'P:/CoxaAI/datasets/hips_800_sort_3.h5'
resize_shape = 800

# create the dataset
with h5py.File(h5_filename, 'w') as f:
    for i in range(5):
        f.create_group(f'fold_{i}')

random.seed(23)
for i in range(5):
    print('writing fold', i)
    images = []
    selected_indice = list(df[df.fold == i].index)
    # shuffle the indice
    random.shuffle(selected_indice)
    selected_df = df.iloc[selected_indice]
    real_diagnosis = selected_df.diagnosis.copy()
    target = real_diagnosis.copy()
    # A&B are normal
    target[target <= 1] = 0
    target[target > 1] = 1
    for _, item in selected_df.iterrows():
        # year = int(item['year'])
        diagnosis_raw = item['diagnosis_raw']
        crop_name = item['crop_name']
        if 'Sortering 2' in item['base_path']:
            sort_path = 'sortering 2/'
        elif 'Sortering 3' in item['base_path']:
            sort_path = 'sortering 3/'
        else:
            sort_path = ''
        cropped_fn = f'{base_path}cropped/{sort_path}{diagnosis_raw}/{crop_name}.npy'
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
            'patient_idx', data=selected_df.pid, dtype='i4')  # meta data for mapping

with h5py.File(h5_filename, 'r') as f:
    for k in f.keys():
        print(k)
        for ds in f[k].keys():
            print('--', f[k][ds])
        print('target', {int(k): v for k, v in zip(
            *(np.unique(f[k]['target'][:], return_counts=True)))})
        print('diagnosis', {int(k): v for k, v in zip(
            *(np.unique(f[k]['diagnosis'][:], return_counts=True)))})


with h5py.File(h5_filename, 'r') as f:
    print(f['fold_0']['target'][:])
    print(f['fold_0']['diagnosis'][:])
