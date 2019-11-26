
import pandas as pd
import numpy as np

tcia_data_path = 'D:/Repos/MI-DQA/tcia-gbm.csv'
tcia_train_path = 'D:/Repos/MI-DQA/tcia-train.csv'
tcia_val_path = 'D:/Repos/MI-DQA/tcia-val.csv'

tcia_data = pd.read_csv(tcia_data_path, header=None)

print(tcia_data.head())
tcia_data.columns = ['path', 'label']
print(tcia_data.label.value_counts())

# random shuffle the dataframe
tcia_data = tcia_data.sample(frac=1).reset_index(drop=True)
tcia_data_sorted = tcia_data.sort_values(by=['label'])

tcia_0 = tcia_data_sorted[tcia_data_sorted.label == 0.0]
tcia_1 = tcia_data_sorted[tcia_data_sorted.label == 1.0]

tcia_0 = tcia_0.sample(frac=1).reset_index(drop=True)
tcia_1 = tcia_1.sample(frac=1).reset_index(drop=True)

print(tcia_0.shape)
print(tcia_1.shape)

tcia_0_train = tcia_0.iloc[:59, :]
tcia_0_val = tcia_0.iloc[59:, :]
tcia_1_train = tcia_1.iloc[:26, :]
tcia_1_val = tcia_1.iloc[26:, :]

tcia_1_train = tcia_1_train.sample(n=59, replace=True) # oversample
tcia_train = pd.concat((tcia_0_train, tcia_1_train), axis=0)
# print(tcia_train.shape)

tcia_val = pd.concat((tcia_0_val, tcia_1_val), axis=0)

tcia_train.to_csv(tcia_train_path, header=False, index=False)
tcia_val.to_csv(tcia_val_path, header=False, index=False)

# size = 80        # sample size
# replace = True  # with replacement
# fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace),:]
# ds030_data.groupby('label', as_index=False).apply(fn)

# rnd_sample_train = tcia_data.groupby('label').apply(lambda x: x.sample(60, replace=True)).reset_index(drop=True)
# rnd_sample_val = tcia_data.groupby('label').apply(lambda x: x.sample(30, replace=False)).reset_index(drop=True)

# rnd_sample_train.to_csv(tcia_train_path, header=False, index=False)
# rnd_sample_val.to_csv(tcia_val_path, header=False, index=False)


