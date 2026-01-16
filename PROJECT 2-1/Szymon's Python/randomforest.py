import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt 

filename = ''
data = pd.read_csv(filename)

def forest(data):
    y = data['output']
    X = data.drop(['output'], axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num=10)]
    max_features = ['sqrt', 'int', None]
    max_depth = [2, 4, 6, 8]
    min_samples_split = [2, 4, 5]
    min_samples_leaf = [1, 2, 3]
    bootstrap = [True, False]


    parameter_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}

    baseForest = RandomForestClassifier()

    gridForest = GridSearchCV(estimator=baseForest, param_grid=parameter_grid, cv=8, verbose=2, n_jobs=4)

    gridForest.fit(X_train, Y_train)

    testAccuracy = gridForest.best_estimator_.score(X_test, Y_test)
    print(f"Final test accuracy of gridsearch model: " + str(testAccuracy))

    return gridForest.best_estimator_



def calculateFeatureImportance(data):
    
    randomForest = forest(data)
    importance = randomForest.feature_importances_
    X = data.drop(['output'], axis=1)

    plt.bar(range(X.shape[1]), importance)
    plt.xticks(range(X.shape[1]), data.feature_names, rotation=90)
    plt.show()







