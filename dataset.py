import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data():
    # read data
    data = pd.read_csv('Iris.csv')
    print(f'data head: {data.head()}')
    print(f'data info: {data.info()}')

    # drop id column
    print('dropping Id column')
    data.drop('Id',axis=1,inplace=True)
    print(f'data info: {data.info()}')

    # encode label
    print('label encode')
    data['Species'] = LabelEncoder().fit_transform(data['Species'])
    print(data.iloc[[0, -1],:])

    return data


def data_split(data):
    X, y = data.values[:, 0:4], data.values[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 11)
    print(f'X_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'y_test shape: {y_test.shape}')
    return X_train, X_test, y_train, y_test
    

