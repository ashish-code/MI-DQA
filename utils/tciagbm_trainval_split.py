
import pandas as pd
import numpy as np

tcia_data_path = 'D:/Repos/MI-DQA/tcia-gbm.csv'
tcia_train_path = 'D:/Repos/MI-DQA/tcia-train.csv'
tcia_val_path = 'D:/Repos/MI-DQA/tcia-val.csv'

tcia_data = pd.read_csv(tcia_data_path, header=None)

print(tcia_data.head())
tcia_data.columns = ['path', 'label']
print(tcia_data.label.value_counts())

# size = 80        # sample size
# replace = True  # with replacement
# fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace),:]
# ds030_data.groupby('label', as_index=False).apply(fn)

rnd_sample_train = tcia_data.groupby('label').apply(lambda x: x.sample(50, replace=True)).reset_index(drop=True)
rnd_sample_val = tcia_data.groupby('label').apply(lambda x: x.sample(30, replace=False)).reset_index(drop=True)

rnd_sample_train.to_csv(tcia_train_path, header=False, index=False)
rnd_sample_val.to_csv(tcia_val_path, header=False, index=False)
