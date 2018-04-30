import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor


train_ = pd.read_csv("train.csv")
train_df, output_df= getCleanData(train_)
test_ = pd.read_csv("test.csv")
test_df = getTestDataframe(test_)
