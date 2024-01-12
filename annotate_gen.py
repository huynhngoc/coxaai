import pandas as pd
import numpy as np
import random


df = pd.read_csv('W:/annotations_export.csv')

# removing missing
df = df.dropna().reset_index(drop=True)

# remove wrong orientation
df = df[abs(df['hip1_center_x'] - df['hip2_center_x']) -
        abs(df['hip1_center_y'] - df['hip2_center_y']) > 0].reset_index(drop=True)


df.to_csv('csv_annotate/annotate_raw.csv', index=False)

df['bboxes_xywh'] = df.apply(lambda item: [
    [item['hip1_center_x'], item['hip1_center_y'],
        item['hip1_width'], item['hip1_height']],
    [item['hip2_center_x'], item['hip2_center_y'],
        item['hip2_width'], item['hip2_height']]
], axis=1)


df['bboxes'] = df.apply(lambda item: [
    [item['hip1_center_y'] - item['hip1_height'] / 2, item['hip1_center_x'] - item['hip1_width'] / 2,
        item['hip1_center_y'] + item['hip1_height'] / 2, item['hip1_center_x'] + item['hip1_width'] / 2],
    [item['hip2_center_y'] - item['hip2_height'] / 2, item['hip2_center_x'] - item['hip2_width'] / 2,
        item['hip2_center_y'] + item['hip2_height'] / 2, item['hip2_center_x'] + item['hip2_width'] / 2],
], axis=1)

df[['filename', 'bboxes_xywh', 'bboxes']].to_csv(
    'csv_annotate/annotate.csv', index=False)
random.sample(list(np.arange(309)), 59)


df = pd.read_csv('csv_annotate/annotate.csv')
df['bboxes'] = df['bboxes'].map(eval).map(np.array)
