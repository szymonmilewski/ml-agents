import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn.inspection import permutation_importance
import os

path = [] #EDIT THIS IF YOU WANT TO RUN THE CODE

def aggregateList(data):
    memory = []

    for i in range (len(data)):
        df = pd.read_csv(data[i])
        memory.append(df)
    
    frame = pd.concat(memory, ignore_index=True)

    return frame

def aggregateDir(data):
    memory = []
    for file in os.listdir(data):
        if file.endswith(".csv"):
            fullPath = os.path.join(data, file)
            df = pd.read_csv(fullPath)
            memory.append(df)

    frame = pd.concat(memory, ignore_index=True)

def getFeatureScores(X: list, Y: list,forest: RandomForestClassifier):
    df = pd.DataFrame(X)
    feature_names = df.columns.tolist()
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    #Mean decrease in impurity
    forest_importances = pd.Series(importances, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using mean decrease in impurity")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()


    #Feature permutation
    result = permutation_importance(forest, X, Y, n_repeats=10, random_state=42, n_jobs=2)
    figures, axis = plt.subplots()
    permutationImportances = pd.Series(result.importances_mean, index=feature_names)
    permutationImportances.plot.bar(yerr=result.importances_std, ax=axis)
    axis.set_title("Feature importances using permutation on full model")
    axis.set_ylabel("Mean accuracy decrease")
    figures.tight_layout()

    
    plt.show()

def getData(data):
    if isinstance(data, list):
        df = aggregateList(data)
    elif os.path.isdir(data):
        df = aggregateDir(data)
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
        'max_features': ['sqrt', 'log2', None],
        'n_estimators': [i for i in range(100, 501, 100)]
    }

    tunedForest = GridSearchCV(base, parameter_grid, cv=2)
    tunedForest.fit(X_train, Y_train)

    bestModel = tunedForest.best_estimator_

    print(Y_test)

    print(f"Best params: {tunedForest.best_params_}")
    print(f"Acc score: {bestModel.score(X_test, Y_test)}")

    getFeatureScores(X_train, Y_train, bestModel)

    return bestModel

def main():
    getData(path)
    bestForest()


if __name__ == "__main__":
    main()
