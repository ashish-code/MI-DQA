"""
Analyze TCIA-GBM annotations
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

data_root = 'D:/Datasets/TCIA-GBM/'
xl_file_name = 'TCIA-GBM-2.xlsx'

xl_file_path = data_root+xl_file_name
df1 = pd.read_excel(xl_file_path, sheet_name='Reader 1', usecols=[0,1,2,3,4])
df2 = pd.read_excel(xl_file_path, sheet_name='Reader 2', usecols=[0,1,2,3,4])
df3 = pd.read_excel(xl_file_path, sheet_name='Reader 3', usecols=[0,1,2,3,4])

# print(df1.shape)
# print(df2.shape)
# print(df3.shape)

def homogeneous(df):
    df[['good', 'ugly', 'bad']].loc[~df[['good', 'ugly', 'bad']].isnull()] = 1
    df[['good', 'ugly', 'bad']].loc[df[['good', 'ugly', 'bad']].isnull()] = 0


df1[['good', 'ugly', 'bad']] = df1[['good', 'ugly', 'bad']].notnull().astype('int')
df2[['good', 'ugly', 'bad']] = df2[['good', 'ugly', 'bad']].notnull().astype('int')
df3[['good', 'ugly', 'bad']] = df3[['good', 'ugly', 'bad']].notnull().astype('int')

df3['id'] = df3['id'].apply(lambda x: ''.join(x[5:]))
# print(df3.shape)
# print(df3.tail())
df1 = df1.set_index('id')
df2 = df2.set_index('id')
df3 = df3.set_index('id')

df3 = df3.reindex(df1.index)

print(df1.good.value_counts())
print(df1.ugly.value_counts())
print(df1.bad.value_counts())

print(df2.good.value_counts())
print(df2.ugly.value_counts())
print(df2.bad.value_counts())

print(df3.good.value_counts())
print(df3.ugly.value_counts())
print(df3.bad.value_counts())




