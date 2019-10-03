"""
Pre-process the ds030 testset data
"""

import pandas as pd
import numpy as np
import os
import shutil

if os.name == 'nt':
    test_csv = 'test-office.csv'
    data_root = 'D:/Datasets/DS030/'
else:
    test_csv = '../test-tesla.csv'
    data_root = '/mnt/data/home/aag106/DS030/'


def gen_data_csv():

    if os.name=='nt':
        iqm_data_path = 'D:/Repos/MI-DQA/x_ds030.csv'
        iqm_label_path = 'D:/Repos/MI-DQA/y_ds030.csv'
    else:
        iqm_data_path = '../x_ds030.csv'
        iqm_label_path = '../y_ds030.csv'

    iqm_data = pd.read_csv(iqm_data_path, delimiter=',', index_col=0)
    iqm_label = pd.read_csv(iqm_label_path, delimiter=',', index_col=0)

    # print(iqm_data.head())
    # print(iqm_label.head())

    iqm_data = iqm_data.sort_index()
    iqm_label = iqm_label.sort_index()
    iqm_label = iqm_label[iqm_label.rater_1 != 0]

    iqm_label.columns = ['site', 'label']

    iqm_data = iqm_data[iqm_data.index.isin(iqm_label.index)]
    iqm_label = iqm_label[iqm_label.index.isin(iqm_data.index)]

    test_iqm = pd.merge(iqm_data, iqm_label, left_on='subject_id', right_on='subject_id')
    # convert the -1 to 0
    test_iqm.label = test_iqm.label.apply(lambda x: int((x+1)/2))

    test_iqm['mri_path'] = test_iqm.apply(lambda  row: f'{data_root}sub-{row.name}_T1w.nii.gz' , axis=1)

    test_iqm = test_iqm[['mri_path', 'label']]
    print(test_iqm.head())
    print(test_iqm.shape)
    test_iqm.to_csv(test_csv, header=False, index=False)


def move_nii():
    data_origin_root = 'D:/Datasets/ds000030/'
    data_dest_root = 'D:/Datasets/DS030/'
    files = []
    for r,_,filenames in os.walk(data_origin_root):
            for f in filenames:
                if f.endswith('_T1w.nii.gz'):
                    path = str(os.path.join(r,f))
                    path = path.replace('\\','/')
                    files.append(path)
    # print(len(files))
    # print(files[:10])
    for src_f in files:
        filename = src_f.split('/')[-1]
        dest_f = data_dest_root+filename
        shutil.copyfile(src_f,dest_f)
        print(f'copied {src_f} to {dest_f}')


if __name__=='__main__':
    # move_nii()
    gen_data_csv()

