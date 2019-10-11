"""
Create training and validation sets for ResNet and IQM hybrid model training

author: Ashish Gupta
email: ashishagupta@gmail.com
date: 09-16-2019
"""

import pandas as pd
import numpy as np
import os

iqm_data_path = 'D:/Repos/MI-DQA/x_abide.csv'
iqm_label_path = 'D:/Repos/MI-DQA/y_abide.csv'

iqm_data = pd.read_csv(iqm_data_path, delimiter=',', index_col=0)
iqm_label = pd.read_csv(iqm_label_path, delimiter=',', index_col=0)
#
# print(iqm_data.head())
# print(iqm_label.head())

# we hold 2/17 sites for validation
val_sites = ['NYU', 'LEUVEN']

train_iqm_label = iqm_label[~iqm_label.site.isin(val_sites)]
val_iqm_label = iqm_label[iqm_label.site.isin(val_sites)]

# print(iqm_label.shape)
# print(train_iqm_label.shape)
# print(val_iqm_label.shape)

train_iqm_label = train_iqm_label.sort_index()
val_iqm_label = val_iqm_label.sort_index()

train_iqm_data = iqm_data[iqm_data.index.isin(train_iqm_label.index)]
val_iqm_data = iqm_data[iqm_data.index.isin(val_iqm_label.index)]

# print(train_iqm_data.shape)
# print(val_iqm_data.shape)

train_iqm_label = train_iqm_label[train_iqm_label.index.isin(train_iqm_data.index)]
val_iqm_label = val_iqm_label[val_iqm_label.index.isin(val_iqm_data.index)]

# print(train_iqm_data.shape)
# print(val_iqm_data.shape)

train_iqm_data = train_iqm_data.sort_index()
val_iqm_data = val_iqm_data.sort_index()
train_iqm_label = train_iqm_label.sort_index()
val_iqm_label = val_iqm_label.sort_index()

# print(train_iqm_data.head())
# print(val_iqm_data.head())
# print(train_iqm_label.head())
# print(val_iqm_label.head())

# train_iqm_label = train_iqm_label.replace(-1, 0)
# val_iqm_label = val_iqm_label.replace(-1,0)

# print(train_iqm_label.head())
# print(val_iqm_label.head())

train_iqm_label['mos'] = train_iqm_label[['rater_1', 'rater_2', 'rater_3']].mean(axis=1, skipna=True, numeric_only=True)
val_iqm_label['mos'] = val_iqm_label[['rater_1', 'rater_2', 'rater_3']].mean(axis=1, skipna=True, numeric_only=True)

train_iqm_label['label'] = train_iqm_label['mos'].apply(lambda x : 1 if x > 0.0 else 0)
val_iqm_label['label'] = val_iqm_label['mos'].apply(lambda x: 1 if x > 0.0 else 0)


train_iqm_data = train_iqm_data.drop(columns=['size_x', 'size_y', 'size_z'])
val_iqm_data = val_iqm_data.drop(columns=['size_x', 'size_y', 'size_z'])

train_iqm = pd.merge(train_iqm_data, pd.DataFrame(train_iqm_label[['site', 'label']]), left_on='subject_id', right_on='subject_id')
val_iqm = pd.merge(val_iqm_data, pd.DataFrame(val_iqm_label[['site', 'label']]), left_on='subject_id', right_on='subject_id')

# print(train_iqm.head())
# print(val_iqm.head())
# print(train_iqm.shape)
# print(val_iqm.shape)

train_iqm['mri_path'] = train_iqm.apply(lambda row: f'/mnt/data/home/aag106/ABIDE1/{row.site}/sub-00{row.name}/anat/sub-00{row.name}_T1w.nii.gz', axis=1)

val_iqm['mri_path'] = val_iqm.apply(lambda row: f'/mnt/data/home/aag106/ABIDE1/{row.site}/sub-00{row.name}/anat/sub-00{row.name}_T1w.nii.gz', axis=1)

train_iqm_csv_path = 'train-hybrid-tesla.csv'
val_iqm_csv_path = 'val-hybrid-tesla.csv'

# train_iqm.to_csv(train_iqm_csv_path)
# val_iqm.to_csv(val_iqm_csv_path)

# print(train_iqm_data.columns)
# print(train_iqm_data.iloc[0])

train_0 = train_iqm_label[train_iqm_label.mos == -1.0]
val_0 = val_iqm_label[val_iqm_label.mos == -1.0]

# print(train_0.shape)
# print(train_0.head())
# print(val_0.shape)
# print(val_0.head())

train_cam_path = 'train-cam-office-0.csv'
val_cam_path = 'val-cam-office-0.csv'

# with open(train_cam_path, 'w') as f:
#     for idx in train_0.index:
#         site_id = train_0.loc[idx, 'site']
#         sub_path = f'D:/Datasets/ABIDE1/{site_id}/sub-00{idx}/anat/sub-00{idx}_T1w.nii.gz'
#         # print(sub_path, os.path.exists(sub_path))
#         f.write(sub_path+'\n')
#
# with open(val_cam_path, 'w') as f:
#     for idx in val_0.index:
#         site_it = val_0.loc[idx, 'site']
#         sub_path = f'D:/Datasets/ABIDE1/{site_id}/sub-00{idx}/anat/sub-00{idx}_T1w.nii.gz'
#         f.write(sub_path+'\n')

# --- Repeat for class 1 (consensus accept)
train_1 = train_iqm_label[train_iqm_label.mos == 1.0]
val_1 = val_iqm_label[val_iqm_label.mos == 1.0]

train_cam_path = 'train-cam-office-1.csv'
val_cam_path = 'val-cam-office-1.csv'

with open(train_cam_path, 'w') as f:
    for idx in train_1.index:
        site_id = train_1.loc[idx, 'site']
        sub_path = f'D:/Datasets/ABIDE1/{site_id}/sub-00{idx}/anat/sub-00{idx}_T1w.nii.gz'
        # print(sub_path, os.path.exists(sub_path))
        f.write(sub_path+'\n')

with open(val_cam_path, 'w') as f:
    for idx in val_1.index:
        site_it = val_1.loc[idx, 'site']
        sub_path = f'D:/Datasets/ABIDE1/{site_id}/sub-00{idx}/anat/sub-00{idx}_T1w.nii.gz'
        f.write(sub_path+'\n')



