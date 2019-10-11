"""
Copy the volumes with artifacts
"""

import shutil
import os
import pandas as pd

src_dir = 'D:/Datasets/ABIDE1'
dst_dir = 'D:/Datasets/ABIDE1_Artifacts'

# artifact_file = 'D:/Repos/MI-DQA/utils/train-cam-office-0.csv'
artifact_file = 'D:/Repos/MI-DQA/utils/val-cam-office-0.csv'

df = pd.read_csv(artifact_file, header=None)
df.columns = ['src_path']
df['dest_path'] = df['src_path'].str.replace('ABIDE1', 'ABIDE1_Artifacts')

for idx, row in df.iterrows():
    dest_path = row['dest_path'].replace(row['dest_path'].split('/')[-1],'')
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    try:
        shutil.copyfile(row['src_path'], row['dest_path'])
    except Exception:
        print(f"error copying {row['src_path']}")
    if not os.path.exists(dest_path):
        print(f"success: {row['dest_path']}")