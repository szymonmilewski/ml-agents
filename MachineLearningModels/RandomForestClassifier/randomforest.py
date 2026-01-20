import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import sys
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import PrecisionRecallDisplay
import os

path = sys.argv[1:]

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
    return frame

def getFeatureScores(X: list, Y: list, forest: RandomForestClassifier):
    df = pd.DataFrame(X)
    feature_names = df.columns.tolist()
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    #Mean decrease in impurity (Gini)
    forest_importances = pd.Series(importances, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using Gini importance")
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
    if len(data) == 1:
        dir = data[0]
        if os.path.isdir(dir):
            df = aggregateDir(dir)
    elif isinstance(data, list):
        df = aggregateList(data)
    else:
        df = pd.read_csv(data)

    df = df.select_dtypes(include=[np.number, float, bool])

    df['benchmark_reached'] = df['benchmark_reached'].astype(int)

    print(df)

    X = df.iloc[:, :-1]
    Y = df.iloc[:,-1]

    dropCols = ['benchmark_mean', 'benchmark_steps', 'ram_mb_used', 'ram_usage_percent', 'ram_avg_percent']
    X = X.drop(columns=dropCols, errors='ignore')
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
        'n_estimators': [i for i in range(100, 501, 100)],
        'class_weight': ['balanced', None]
    }

    tunedForest = GridSearchCV(base, parameter_grid, cv=2)
    tunedForest.fit(X_train, Y_train)

    bestModel = tunedForest.best_estimator_

    print(Y_test)

    print(f"Best params: {tunedForest.best_params_}")
    print(f"Acc score: {bestModel.score(X_test, Y_test)}")

    getFeatureScores(X_train, Y_train, bestModel)
    calculateROC(bestModel, X_test, Y_test)
    calculatePR(bestModel, X_test, Y_test)

    return bestModel

def calculateROC(forest: RandomForestClassifier, X_test:pd.DataFrame, Y_test:pd.Series):
    yProb = forest.predict_proba(X_test)[:, 1]
    fpr, tpr, threshs = roc_curve(Y_test, yProb, pos_label=1)
    roc_auc = roc_auc_score(Y_test, yProb)
    print("ROC AUC = " + str(roc_auc))
    plt.plot(fpr, tpr, label='ROC Curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0,1], [0,1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def calculatePR(forest: RandomForestClassifier, X_test: pd.DataFrame, Y_test: pd.Series):
    display = PrecisionRecallDisplay.from_estimator(forest, X_test, Y_test, name="RandomForestClassifier", plot_chance_level=True, despine=True)
    display.ax_.set_title("2-class Precision-Recall curve")
    plt.show()

def main():
    getData(path)
    bestForest()


if __name__ == "__main__":
    main()
