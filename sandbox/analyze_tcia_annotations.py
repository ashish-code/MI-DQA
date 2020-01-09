"""
Analyze TCIA-GBM annotations
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data_root = 'D:/Datasets/TCIA-GBM/'


def homogeneous(df):
    df[['good', 'ugly', 'bad']].loc[~df[['good', 'ugly', 'bad']].isnull()] = 1
    df[['good', 'ugly', 'bad']].loc[df[['good', 'ugly', 'bad']].isnull()] = 0

def explore_tcia_annotations(xl_file_name):
    xl_file_name = 'TCIA-GBM-2.xlsx'

    xl_file_path = data_root+xl_file_name
    df1 = pd.read_excel(xl_file_path, sheet_name='Reader 1', usecols=[0,1,2,3,4])
    df2 = pd.read_excel(xl_file_path, sheet_name='Reader 2', usecols=[0,1,2,3,4])
    df3 = pd.read_excel(xl_file_path, sheet_name='Reader 3', usecols=[0,1,2,3,4])

    # print(df1.shape)
    # print(df2.shape)
    # print(df3.shape)

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

def explore_tcia_2(xl_file_name):
    xl_file_path = data_root + xl_file_name
    df1 = pd.read_excel(xl_file_path, sheet_name='Reader 1', usecols=[0, 1, 2, 3])
    df2 = pd.read_excel(xl_file_path, sheet_name='Reader 2', usecols=[0, 1, 2, 3])

    df1[['good', 'ugly', 'bad']] = df1[['good', 'ugly', 'bad']].notnull().astype('int')
    df2[['good', 'ugly', 'bad']] = df2[['good', 'ugly', 'bad']].notnull().astype('int')

    df1 = df1.set_index('id')
    df2 = df2.set_index('id')

    print(df1.good.value_counts())
    print(df1.ugly.value_counts())
    print(df1.bad.value_counts())

    print(df2.good.value_counts())
    print(df2.ugly.value_counts())
    print(df2.bad.value_counts())

    rater1 = []
    rater1.append(df1.good.value_counts()[1])
    rater1.append(df1.ugly.value_counts()[1])
    rater1.append(df1.bad.value_counts()[1])

    rater2 = []
    rater2.append(df2.good.value_counts()[1])
    rater2.append(df2.ugly.value_counts()[1])
    rater2.append(df2.bad.value_counts()[1])

    ratings = pd.DataFrame([rater1, rater2])
    ratings.columns = ['Acceptable', 'Issues', 'Unacceptable']
    ratings.index = ['Rater 1', 'Rater 2']
    # ratings['raters'] = ['Rater1', 'Rater2']
    print(ratings)
    ratings_t = ratings.T
    ratings2 = pd.melt(ratings)
    # print(ratings2)
    ax = ratings_t.plot.bar(rot=0)
    ax.set_xlabel('Quality Rating Category')
    ax.set_ylabel('Number of subjects')
    ax.set_title('Rater QA for TCIA-GBM')
    plt.show()




if __name__=='__main__':
    xl_file_name = 'TCIA-GBM-3.xlsx'
    explore_tcia_2(xl_file_name)