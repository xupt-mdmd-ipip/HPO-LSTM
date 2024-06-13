import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from config import v


def loder_data():  # 年度报表
    data = pd.read_excel(os.path.join(v.absolute_path, "data", "2018年站点空气质量指数实时报(审核).xls"))
    data.fillna(method='ffill', inplace=True)
    data = data.loc[:,
           ['PM10浓度(μg/m3)', 'PM2.5浓度(μg/m3)', 'SO2浓度(μg/m3)', 'NO2浓度(μg/m3)',
            'CO浓度(mg/m3)', 'O3浓度(μg/m3)']]
    columns = []
    column = []
    temp = data.copy()
    column.extend(temp.columns)
    for i in range(v.split.num_times):
        temp.index = temp.index - 1
        data = pd.concat([data, temp], axis=1)
    for i in range(1, v.split.num_times + 2):
        for c in column:
            columns.append(f"day{i}_" + c)
    data = data.dropna()
    data.columns = columns

    x = data.loc[:, data.columns[:-6]]
    y = data.loc[:, data.columns[-6:]]

    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=5)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1, random_state=5)
    return train_X, test_X, valid_X, train_y, test_y, valid_y


def loder_data1():  # 地区报表
    data = pd.read_excel(os.path.join(v.absolute_path, "data", "高新西区站点空气质量指数实时报(审核).xls"))
    # data = pd.read_excel(os.path.join(v.absolute_path, "data", "草堂实时.xls"))
    data.fillna(method='ffill', inplace=True)  # replace the nan
    data = data.loc[:,
           ['PM10', 'PM2.5', 'SO2', 'NO2', 'CO', 'O3']]
    columns = []
    column = []
    temp = data.copy()
    column.extend(temp.columns)
    for i in range(v.split.num_times):
        temp.index = temp.index - 1
        data = pd.concat([data, temp], axis=1)
    for i in range(1, v.split.num_times + 2):
        for c in column:
            columns.append(f"day{i}_" + c)
    data = data.dropna()
    data.columns = columns

    # data.to_excel(os.path.join(v.absolute_path, "data", "空气质量指数实时报.xls"))

    x = data.loc[:, data.columns[:-6]]
    y = data.loc[:, data.columns[-6:]]

    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=5)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1, random_state=5)
    return train_X, test_X, valid_X, train_y, test_y, valid_y


if __name__ == '__main__':
    train_X, test_X, valid_X, train_y, test_y, valid_y = loder_data1()

    test_X = test_X.astype(float)
    print(type(test_X))
    test_X = torch.from_numpy(test_X)
    # train_X, test_X, valid_X, train_y, test_y, valid_y = loder_data()

    print(train_y.shape, valid_y.shape, test_y.shape)
