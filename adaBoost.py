import numpy as np, pandas, xgboost as xgb

def trainModel(train, test):
    counter = 0
    # from 1 to 60 every minut we are predicting value
    for counter in range(1, 62 + 1):  
        # predicting  next day value
        if counter == 61:
            returns_name = 'Ret_PlusOne'
            return_weight = 'Weight_Daily'
        # predicting next to next day value
        elif counter == 62:
            returns_name = 'Ret_PlusTwo'
            return_weight = 'Weight_Daily'
        else:
            returns_name = 'Ret_' + str(counter + 120)
            return_weight = 'Weight_Intraday'

        train_targets = train[returns_name].values
        train_weights = train[return_weight].values
        
        training_data = train.drop(train.columns[range(146, 210)], axis=1)
        training_data = training_data.values
        
        testing_data = test.values

        data_train = xgb.DMatrix(training_data, label=train_targets, missing=np.NaN, weight=train_weights)
        data_test = xgb.DMatrix(testing_data, missing=np.NaN)

        model_parameters = {'max_depth': 10, 'eta': 0.1, 'silent': 1, 'gamma': 0, 'lambda': 500, 'alpha': 400}
        number_of_rounds = 200

        watchlist = [(data_train, 'train')]
        bst = xgb.train(model_parameters, data_train, number_of_rounds, watchlist, early_stopping_rounds=10)

        predictions = bst.predict(data_test)

        finalOutPut.append(predictions.ravel())
    a = np.array(finalOutPut).ravel()
    a = list(a)
    return a


def writeToCSV(prediction):
    file = open(outputFile, "w")
    prediction = np.array(prediction)
    noOfsamples = 120001
    noOfFeatures = 63

    columnNames = "Id,Predicted\n"
    file.write(columnNames)

    counter = 0
    for i in range(1, noOfsamples):
        for j in range(1, noOfFeatures):
            x = str(i) + "_" + str(j)
            toWrite = x + "," + str(prediction[k]) + "\n"
            file.write(toWrite)
            counter = counter + 1
    file.close()

def getData():
    train = pandas.read_csv( trainDataFile , index_col=0 )
    test = pandas.read_csv( testDataFile, index_col=0 )
    return train, test


trainDataFile = "train.csv"
testDataFile = "test.csv"
outputFile = "output.csv"
finalOutPut = []
trainCSV, testCSV = getData()
prediction = trainModel(trainCSV, testCSV)
writeToCSV(prediction)

