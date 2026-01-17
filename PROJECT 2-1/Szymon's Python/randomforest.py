import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

path = [] #EDIT THIS IF YOU WANT TO RUN THE CODE

def aggregate(data):
    memory = []

    for i in range (len(data)):
        df = pd.read_csv(data[i])
        memory.append(df)
    
    frame = pd.concat(memory, ignore_index=True)

    return frame

def getData(data):
    if isinstance(data, list):
        df = aggregate(data)
    else:
        df = pd.read_csv(data)

    df = df.select_dtypes(include=[np.number, float, bool])

    df['benchmark_reached'] = df['benchmark_reached'].astype(int)

    print(df)

    X = df.iloc[:, :-1]
    Y = df.iloc[:,-1]
    return X, Y

def bestForest():
    X, Y = getData(path)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    base = RandomForestClassifier()

    parameter_grid = {
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],#, None],
        'n_estimators': [i for i in range(100, 501, 100)]
    }

    tunedForest = GridSearchCV(base, parameter_grid, cv=2)
    tunedForest.fit(X_train, Y_train)

    bestModel = tunedForest.best_estimator_

    print(Y_test)

    print(f"Best params: {tunedForest.best_params_}")
    print(f"Acc score: {bestModel.score(X_test, Y_test)}")



    return tunedForest.best_estimator_


def main():
    getData(path)
    bestForest()

if __name__ == "__main__":
    main()
