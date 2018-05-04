import xgboost as xgb
import numpy as np
import pandas


def reading_data():
    train = pandas.read_csv('train.csv', index_col=0)
    test = pandas.read_csv('test.csv', index_col=0)
    return train,test
    
if __name__ == "__main__":
    train,test=reading_data()
    a=training(train,test)
    
