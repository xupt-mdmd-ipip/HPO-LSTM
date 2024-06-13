import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import v

data = pd.read_excel(os.path.join(v.absolute_path, "data", "2018年站点空气质量指数实时报(审核).xls"))
data.fillna(method='ffill', inplace=True)
data = data.loc[:,
       ['PM10浓度(μg/m3)', 'PM2.5浓度(μg/m3)', 'SO2浓度(μg/m3)', 'NO2浓度(μg/m3)',
        'CO浓度(mg/m3)', 'O3浓度(μg/m3)']]


def loader_PM_10():
    # PM-10
    pm_10 = np.array(data["PM10浓度(μg/m3)"])
    for i in range(len(pm_10)):
        if 0 <= pm_10[i] <= 50:
            pm_10[i] = 0
        if 50 < pm_10[i] <= 150:
            pm_10[i] = 1
        if 150 < pm_10[i] <= 250:
            pm_10[i] = 2
        if 250 < pm_10[i] <= 350:
            pm_10[i] = 3
        if 350 < pm_10[i] <= 420:
            pm_10[i] = 4
        if 420 < pm_10[i] <= 500:
            pm_10[i] = 5
        if 500 < pm_10[i] <= 600:
            pm_10[i] = 6
        if pm_10[i] > 600:
            pm_10[i] = 7
    pm_10 = pd.DataFrame(pm_10)
    temp = pm_10.copy()
    for i in range(v.split.num_times):
        temp.index = temp.index - 1
        pm_10 = pd.concat([pm_10, temp], axis=1)
    pm_10 = pm_10.dropna()
    pm_10_x = pm_10.iloc[:, :24]
    pm_10_y = pm_10.iloc[:, 24]
    np.save("PM_10", pm_10)
    train_X, test_X, train_y, test_y = train_test_split(pm_10_x, pm_10_y, test_size=0.3, random_state=5)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1, random_state=5)
    return train_X, test_X, valid_X, train_y, test_y, valid_y


def loader_PM_25():
    # PM-2.5
    pm_25 = np.array(data["PM2.5浓度(μg/m3)"])
    for i in range(len(pm_25)):
        if 0 <= pm_25[i] <= 35:
            pm_25[i] = 0
        if 35 < pm_25[i] <= 75:
            pm_25[i] = 1
        if 75 < pm_25[i] <= 115:
            pm_25[i] = 2
        if 115 < pm_25[i] <= 150:
            pm_25[i] = 3
        if 150 < pm_25[i] <= 250:
            pm_25[i] = 4
        if 250 < pm_25[i] <= 350:
            pm_25[i] = 5
        if 350 < pm_25[i] <= 500:
            pm_25[i] = 6
    pm_25 = pd.DataFrame(pm_25)
    temp = pm_25.copy()
    for i in range(v.split.num_times):
        temp.index = temp.index - 1
        pm_25 = pd.concat([pm_25, temp], axis=1)
    pm_25 = pm_25.dropna()
    np.save("pm_25", pm_25)
    pm_25_x = pm_25.iloc[:, :24]
    pm_25_y = pm_25.iloc[:, 24]
    train_X, test_X, train_y, test_y = train_test_split(pm_25_x, pm_25_y, test_size=0.3, random_state=5)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1, random_state=5)
    return train_X, test_X, valid_X, train_y, test_y, valid_y


# SO2
def loader_SO2():
    SO2 = np.array(data["SO2浓度(μg/m3)"])
    for i in range(len(SO2)):
        if 0 <= SO2[i] <= 5:
            SO2[i] = 0
        if 5 < SO2[i] <= 10:
            SO2[i] = 1
        if 10 < SO2[i] <= 15:
            SO2[i] = 2
        if 15 < SO2[i] <= 150:
            SO2[i] = 3
        if 150 < SO2[i] <= 500:
            SO2[i] = 4
    SO2 = pd.DataFrame(SO2)
    temp = SO2.copy()
    for i in range(v.split.num_times):
        temp.index = temp.index - 1
        SO2 = pd.concat([SO2, temp], axis=1)
    SO2 = SO2.dropna()
    np.save("SO2", SO2)
    x = SO2.iloc[:, :24]
    y = SO2.iloc[:, 24]
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=5)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1, random_state=5)
    return train_X, test_X, valid_X, train_y, test_y, valid_y


# NO2
def loader_NO2():
    NO2 = np.array(data["NO2浓度(μg/m3)"])
    for i in range(len(NO2)):
        if 0 <= NO2[i] <= 30:
            NO2[i] = 0
        if 30 < NO2[i] <= 50:
            NO2[i] = 1
        if 50 < NO2[i] <= 100:
            NO2[i] = 2
        if 100 < NO2[i] <= 200:
            NO2[i] = 3
        if 200 < NO2[i] <= 700:
            NO2[i] = 4
    NO2 = pd.DataFrame(NO2)
    temp = NO2.copy()
    for i in range(v.split.num_times):
        temp.index = temp.index - 1
        NO2 = pd.concat([NO2, temp], axis=1)
    NO2 = NO2.dropna()
    np.save("NO2", NO2)
    x = NO2.iloc[:, :24]
    y = NO2.iloc[:, 24]
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=5)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1, random_state=5)
    return train_X, test_X, valid_X, train_y, test_y, valid_y


# CO
def loader_CO():
    CO = np.array(data["CO浓度(mg/m3)"])
    for i in range(len(CO)):
        if 0 <= CO[i] <= 0.6:
            CO[i] = 0
        if 0.6 < CO[i] <= 1:
            CO[i] = 1
        if 1 < CO[i] <= 1.5:
            CO[i] = 2
        if 1.5 < CO[i] <= 5:
            CO[i] = 3
    CO = pd.DataFrame(CO)
    temp = CO.copy()
    for i in range(v.split.num_times):
        temp.index = temp.index - 1
        CO = pd.concat([CO, temp], axis=1)
    CO = CO.dropna()
    np.save("CO", CO)
    x = CO.iloc[:, :24]
    y = CO.iloc[:, 24]
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=5)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1, random_state=5)
    return train_X, test_X, valid_X, train_y, test_y, valid_y


# O3
def loader_O3():
    O3 = np.array(data["O3浓度(μg/m3)"])
    for i in range(len(O3)):
        if 0 <= O3[i] <= 25:
            O3[i] = 0
        if 25 < O3[i] <= 50:
            O3[i] = 1
        if 50 < O3[i] <= 100:
            O3[i] = 2
        if 100 < O3[i] <= 160:
            O3[i] = 3
        if 160 < O3[i] <= 200:
            O3[i] = 4
        if 200 < O3[i] <= 300:
            O3[i] = 5
        if 300 < O3[i] <= 400:
            O3[i] = 6
    O3 = pd.DataFrame(O3)
    temp = O3.copy()
    for i in range(v.split.num_times):
        temp.index = temp.index - 1
        O3 = pd.concat([O3, temp], axis=1)
    O3 = O3.dropna()
    np.save("O3", O3)
    x = O3.iloc[:, :24]
    y = O3.iloc[:, 24]
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=5)
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.1, random_state=5)
    return train_X, test_X, valid_X, train_y, test_y, valid_y


if __name__ == '__main__':
    train_X, test_X, valid_X, train_y, test_y, valid_y = loader_PM_10()
    print(valid_X.shape)
