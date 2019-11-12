"""
create training and validation set for DS030

author: ashish gupta
email: ashishagupta@gmail.com
"""

import pandas as pd
import numpy as np

ds030_data_path = 'D:/Repos/MI-DQA/test-office.csv'
ds030_train_path = 'D:/Repos/MI-DQA/ds030-train.csv'
ds030_val_path = 'D:/Repos/MI-DQA/ds030-val.csv'

ds030_data = pd.read_csv(ds030_data_path, header=None)

print(ds030_data.head())
ds030_data.columns = ['path', 'label']
print(ds030_data.label.value_counts())

# size = 80        # sample size
# replace = True  # with replacement
# fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace),:]
# ds030_data.groupby('label', as_index=False).apply(fn)

rnd_sample_train = ds030_data.groupby('label').apply(lambda x: x.sample(50, replace=True)).reset_index(drop=True)
rnd_sample_val = ds030_data.groupby('label').apply(lambda x: x.sample(50, replace=True)).reset_index(drop=True)

rnd_sample_train.to_csv(ds030_train_path, header=False, index=False)
rnd_sample_val.to_csv(ds030_val_path, header=False, index=False)




