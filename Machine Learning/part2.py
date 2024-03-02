import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import csv
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
h = .02  # step size in the mesh

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV # Hyperparameter tuning import
# You can add classifiers to this dictionary, don't forget to import them!
classifiers = {"Logistic Regression": LogisticRegression(),
               "Decision Tree": DecisionTreeClassifier(),
               "Random Forest": RandomForestClassifier(),
               "SVC": SVC(),
               "Naive Bayes": GaussianNB(),
               "Nearest Neighbors": KNeighborsClassifier(),
               "Neural Network": MLPClassifier()
               }

datasets = [
            ]
Label = "Credit"
Features = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19"]

def saveBestModel(clf):
    pickle.dump(clf, open("bestModel.model", 'wb'))

def readData(file):
    df = pd.read_csv(file)
    return df

def trainOnAllData(df, clf):
    #Use this function for part 4, once you have selected the best model
    print("Train on All Data")
    ### Code Begin
    ds = datasets[di]
    # preprocess dataset, split into training and test part
    dsX, y = ds[Features], ds[Label]
    X = StandardScaler().fit_transform(dsX)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1)
    yPred = sklearn.model_selection.cross_val_predict(clf,X,y, cv=KFold(n_splits=10))
    # clf.fit(X, y)
    # yPred = clf.predict(X)

    

    saveBestModel(clf)

df = readData("credit_train.csv")
datasets.append(df)
datasets_names = ["Credit Train"]


### Code Begin

print(len(datasets), len(datasets_names))
figure = plt.figure(figsize=(27, 9))
bestTest = []
bestPred = []
bestGround = []
bestScore = 0
bestName = ""
bestParams = None

for di in range(0, len(datasets), 1):
    print(di, datasets_names[di])
    ds = datasets[di]
    # preprocess dataset, split into training and test part
    dsX, y = ds[Features], ds[Label]
    # X, y = ds[["A1","A2"]], ds[Label]
    X = StandardScaler().fit_transform(dsX)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
    avg_table = []
    std_table = []
    class_table = []
    # iterate over classifiers
    for name, clf in zip(classifiers.keys(), classifiers.values()):
        class_table.append(name)
        print(name)
        k_folds = KFold(n_splits=10)
        # score = cross_val_score(clf,X_test,y_test,cv=k_folds)
        score = cross_val_score(clf, X, y, cv=k_folds, scoring="roc_auc")
        yPred = sklearn.model_selection.cross_val_predict(clf,X,y, cv=k_folds)
        print("K-Fold Score: \n"+str(score))
        avg = 0
        for i in score:
            avg += i
        avg = avg/len(score)
        print("\tAvg AUROC: "+str(avg))
        std_score = np.std(score)
        print("\tStandard Deviation: "+str(std_score))
        avg_table.append(avg)
        std_table.append(std_score)
        if avg > bestScore:
            bestScore = avg
            bestTest = X
            bestGround = y
            bestPred = yPred
            bestName = name
            
    table = []
    for i in range(len(avg_table)):
        table.append([avg_table[i],std_table[i]])
    df = pd.DataFrame(table, columns = ["Avg AUROC","Std Dev"], index=class_table)
    print(df)

##Hyper Parameter Tuning
print("\nHyperparameter Tuning Start")
param_grid_SVC = {'C':[1,10,100], 'gamma':['scale','auto']}
param_grid_RF = {'max_depth':[100,200,400],'n_estimators':[200,250,350,425]}
classifiers = {"Random Forest": RandomForestClassifier(),
                "SVC": SVC()
               }
avg_table = []
std_table = []
class_table = []


estRF = None
estSVC = None
for name, clf in zip(classifiers.keys(), classifiers.values()):
    class_table.append(name)
    print(name)
    if name == "Random Forest":
        paramGrid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid_RF)
        # paramGrid.fit(X_train, y_train)
        
        yPred = sklearn.model_selection.cross_val_predict(paramGrid,X,y, cv=KFold(n_splits=10))
        s = cross_val_score(paramGrid, X, y, cv=KFold(n_splits=10), scoring="roc_auc")
        # yPred = paramGrid.predict(X_test)
        paramGrid.fit(X, y)
        temp = paramGrid.predict(X)
        print("\tRF Best Estimator: "+str(paramGrid.best_estimator_))
        estRF = paramGrid.best_estimator_
        avg = 0
        for i in s:
            avg += i
        avg = avg/len(s)
        print("\tAvg AUROC: "+str(avg))
    else:
        paramGrid = GridSearchCV(SVC(), param_grid=param_grid_SVC)
        
        
        # yPred = paramGrid.predict(X_test)
        yPred = sklearn.model_selection.cross_val_predict(paramGrid,X,y, cv=KFold(n_splits=10))
        s = cross_val_score(paramGrid, X, y, cv=KFold(n_splits=10), scoring="roc_auc")
        paramGrid.fit(X, y)
        temp = paramGrid.predict(X)
        print("\tSVC Best Estimator: "+str(paramGrid.best_estimator_))
        estSVC = paramGrid.best_estimator_
        avg = 0
        for i in s:
            avg += i
        avg = avg/len(s)
        print("\tAvg AUROC: "+str(avg))
    if avg > bestScore:
        bestScore = avg
        bestTest = X
        bestGround = y
        bestPred = yPred
        bestName = name
        if bestName == "SVC":
            bestParams = estSVC
        else:
            bestParams = estRF
    std_score = np.std(score)
    avg_table.append(avg)
    std_table.append(std_score)
    
print("Hyperparameter Tuning AVG Values")
table = []
for i in range(len(avg_table)):
    table.append([avg_table[i],std_table[i]])
df = pd.DataFrame(table, columns = ["Avg AUROC","Std Dev"], index=class_table)
print(df)

print("Best model CSV: "+str(bestName))
print("Best AUROC: "+str(bestScore))
print("Best Params: "+str(bestParams))
print("Confusion Matrix: "+str(metrics.confusion_matrix(y,yPred)))
tn, fp, fn, tp = metrics.confusion_matrix(y,yPred).ravel()
print(tn, fp, fn, tp)
print("Accuracy: "+str((tp+tn)/(tn+tp+fn+tp)))
print("Precision: "+str(tp/(tp+fp)))
print("Recall: "+str(tp/(fn+tp)))
# fieldnames = []
with open('bestModel.output.csv', 'w', newline='') as csvfile:
    # for k in Features:
    #     fieldnames.append(k)
    fieldnames = Features.copy()
    fieldnames.append('Ground Truth')
    fieldnames.append('Prediction')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in range(len(bestPred)):
        row = {}
        
        j = 0
        for f in Features:
            # row[f] = dsX[r][j]
            # row[f] = X[r][j]
            row[f] = bestTest[r][j]
            j += 1
        # row['Ground Truth'] = bestGround[r]
        row['Ground Truth'] = y[r]
        # row['Prediction'] = bestPred[r]
        row['Prediction'] = bestPred[r]
        writer.writerow(row)

trainOnAllData(df, RandomForestClassifier())