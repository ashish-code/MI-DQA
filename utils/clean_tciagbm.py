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

# sublist = os.listdir(orig_root)
# print(len(sublist))
# sublist = [itm for itm in sublist if os.path.isdir(orig_root+itm)]
#
# new_sublist = []
# for dir_name in sublist:
#     itms = dir_name.split('-')
#     out = f'{itms[1]}-{str(int(itms[2])).zfill(4)}'
#     new_sublist.append(out)
#
# print(new_sublist)
# for itm in new_sublist:
#     dir_name = dest_root+itm
#     if not os.path.exists(dir_name):
#         os.mkdir(dir_name)

xlsxfile_path = dest_root+'tcia-gbm.xlsx'
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
tciagmb_2.to_csv(dest_root+'tcia-gbm.csv', header=False, index=False)




"""
for idx, row in tciagmb_df.iterrows():
    name = row['name']
    date_str = row['dtstr']
    out_dir_name = f'{name}-{date_str}'
    out_dir_name = dest_root + out_dir_name +'/'
    try:
        id1, id2 = name.split('-')
    except:
        print(idx, name)
    id2 = '00'+str(int(id2))
    name = id1 + '-' + id2
    root_path = f'{orig_root}TCGA-{name}/'
    # print(root_path)
    all_paths = []
    for filename in glob.iglob(root_path + '**/*.dcm', recursive=True):
        filename = Path(filename)
        all_paths.append(filename)
    # print(len(all_paths))
    select_paths = []
    pruned_paths = []
    for _path in all_paths:
        s_path = _path.as_posix().lower()
        if 'ax' in s_path and 't1' in s_path:
            select_paths.append(_path)
    # print(len(select_paths))
    preLabel = False
    postLabel = False
    noLabel = False
    for _path in select_paths:
        s_path = _path.as_posix().lower()
        if 'pre' in s_path:
            preLabel = True
        elif 'post' in s_path:
            postLabel = True
        else:
            noLabel = True
    if preLabel:
        for _path in select_paths:
            if 'pre' in _path.as_posix().lower():
                pruned_paths.append(_path)
    elif postLabel:
        for _path in select_paths:
            if 'post' in _path.as_posix().lower():
                pruned_paths.append(_path)
    else:
        pruned_paths = select_paths
    # print(len(pruned_paths))
    if not os.path.exists(out_dir_name):
        os.mkdir(out_dir_name)
    for itm in pruned_paths:
        out_path = out_dir_name+itm.name
        if not os.path.exists(out_path):
            try:
                shutil.copyfile(itm.as_posix(), out_path)
                print(itm.as_posix())
            except:
                print(f'error copying {itm.as_posix()}')

"""