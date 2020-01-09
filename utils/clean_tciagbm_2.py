"""
The TCIA-GBM dataset is structured extremely poorly.
Restructure the dataset.
"""
import pandas as pd
import os
import os.path
import glob
from pathlib import Path, PureWindowsPath
import shutil

orig_root = 'D:/Datasets/TCIA-GBM/'
dest_root = 'D:/Datasets/TCIA-GBM-2/'


xlsxfile_path = orig_root+'tcia-gbm-3.xlsx'


tciagmb_df = pd.read_excel(xlsxfile_path, header=None)

# print(tciagmb_df.info())

tciagmb_df.columns = ['name', 'date', 'a', 'u1', 'u2']

# drop subjects with missing slices


missing_idx = tciagmb_df[tciagmb_df['u1'].str.contains('missing', case=False, na=False)].index
tciagmb_df.drop(missing_idx, inplace=True)
print(tciagmb_df.shape)
missing_idx = tciagmb_df[tciagmb_df['u2'].str.contains('missing', case=False, na=False)].index
tciagmb_df.drop(missing_idx, inplace=True)
print(tciagmb_df.shape)

tciagmb_df['u1'][tciagmb_df['u1'].notna()] = 1.0
tciagmb_df['u2'][tciagmb_df['u2'].notna()] = 1.0
tciagmb_df = tciagmb_df.fillna(0.0)
tciagmb_df['u'] = tciagmb_df['u1']+tciagmb_df['u2']

tciagmb_df = tciagmb_df.drop(columns=['u1', 'u2'])
tciagmb_df.date = pd.to_datetime(tciagmb_df['date'], format='%m/%d/%Y')

tciagmb_df['s'] = tciagmb_df.a+tciagmb_df.u
idx = tciagmb_df.s != 0.0
tciagmb_df = tciagmb_df[idx]
# tciagmb_df.drop(idx, inplace=True)

tciagmb_df = tciagmb_df.drop(columns=['s'])
tciagmb_df.name = tciagmb_df['name'].astype(str)

# print(tciagmb_df.head())
# print(tciagmb_df.info())

dt = tciagmb_df.iloc[0,1]
# print(''.join(str(dt).split(' ')[0].split('-')))

tciagmb_df['dt'] = tciagmb_df['date'].apply(lambda x: ''.join(str(x).split(' ')[0].split('-')))
tciagmb_df['dtstr'] = tciagmb_df['date'].apply(lambda x: str(x).split(' ')[0])

tciagmb_df['full_name'] = tciagmb_df[['name','dtstr']].apply(lambda x: '-'.join(x), axis=1)
tciagmb_2 = tciagmb_df[['full_name', 'a']]
tciagmb_2.columns = ['name', 'label']

# Remove subjects with empty dicom content
empty_idx_list = []
for idx, row in tciagmb_df.iterrows():
    name = row['name']
    date_str = row['dtstr']
    out_dir_name = f'{name}-{date_str}'
    out_dir_name = dest_root + out_dir_name +'/'
    if len(os.listdir(out_dir_name))==0:
        empty_idx_list.append(idx)

print(empty_idx_list)

print(tciagmb_2.label.value_counts())
tciagmb_2 = tciagmb_2.drop(empty_idx_list)
print(tciagmb_2.label.value_counts())


# print(tciagmb_df.head())
# print(tciagmb_2.label.value_counts())
tciagmb_2.to_csv(dest_root+'tcia-gbm-2.csv', header=False, index=False)



