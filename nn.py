## Authors : 
##  1. Kedar Anavardekar
##  2. Harsh Mahiswari
## Subject : EE769-Intoduction To Machine Learning 

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def getTestDataframe(inputDataFrame):
    inputDataFrame.drop('Id', axis=1, inplace=True)
    ## replace missing values with mean of that column
    testDataFrameTemp = inputDataFrame.fillna(inputDataFrame.mean())
    return testDataFrameTemp


def preProcessData(inputDataFrame):
    inputDataFrame.drop('Id', axis=1, inplace=True)
    ## replace missing values with mean of that column
    df = inputDataFrame.fillna( inputDataFrame.mean() )
    trainDataFrameTemp = df.iloc[:, :146]
    output_DataFrame = df.iloc[:, 146:208]
    return trainDataFrameTemp, output_DataFrame

def writeToCSV(prediction):
        file = open(outputFile, "w")
        noOfsamples = 120001
        noOfFeatures = 63
        columnNames = "Id,Predicted\n"

        file.write(columnNames)
        counter = 0
        for i in range(1, noOfsamples):
            for j in range(1, noOfFeatures ):
                x = str(i) + "_" + str(j)
                toWrite = x + "," + str(prediction[k]) + "\n"
                file.write( toWrite )
                counter = counter + 1
        file.close()

trainDataFile = "train.csv"
testDataFile = "test.csv"
outputFile = "output.csv"

trainCSV = pd.read_csv(trainDataFile)
testCSV = pd.read_csv(testDataFile)

trainDataFrame, output_df = preProcessData(trainCSV)
testDataFrame = getTestDataframe(testCSV)

nn = MLPRegressor(solver='adam', alpha=0.0001, hidden_layer_sizes=(100, 100), random_state=1, max_iter=200)
nn.fit(trainDataFrame, output_df)
predictions = nn.predict(testDataFrame)
prediction = predictions.ravel()
writeToCSV(prediction)

