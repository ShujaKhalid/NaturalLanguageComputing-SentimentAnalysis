from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import argparse
import sys
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import json
from sklearn.metrics import accuracy_score
import pickle

def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    # Extract an object from the npz file
    out = np.load(args.input)
    data = out['feats']
    X = np.zeros((40000,173))
    y = np.zeros((40000))

    for i in range(data.shape[0]):
        X[i,:] = np.array(data[i][:173])
        y[i] = np.array(data[i][173])

    # Remove nan and inf from data
    X[np.isfinite(X)==False] = 0
    y[np.isfinite(y)==False] = 0
    # print(X)
    # print(y)
    # Split the data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=0)
    
    # Train an SVC (Linear)
    svc_linear = SVC(C=1.0, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

    svc_linear.fit(X_train, y_train)

    # save the classifier
    with open('svc_linear.pkl', 'wb') as fid:
        pickle.dump(svc_linear, fid) 

    y_pred_svc_linear = svc_linear.predict(X_test)

    accuracy_linear = accuracy_score(y_test,y_pred_svc_linear)

    print()
    print('Accuracy for the linear SVC: ' + str(accuracy_linear))
    print()

    # Train an SVC (radial basis function)
    svc_rbf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

    svc_rbf.fit(X_train, y_train)

    # save the classifier
    with open('svc_rbf.pkl', 'wb') as fid:
        pickle.dump(svc_rbf, fid) 

    y_pred_svc_rbf = svc_rbf.predict(X_test)

    accuracy_rbf = accuracy_score(y_test,y_pred_svc_rbf)

    print()
    print('Accuracy for the RBF SVC: ' + str(accuracy_rbf))
    print()

    # Train a Random Forest Classifier
    rfc = RandomForestClassifier(max_depth=5, random_state=0, verbose=False)

    rfc.fit(X_train, y_train)

    # save the classifier
    with open('svc_rbf.pkl', 'wb') as fid:
        pickle.dump(svc_rbf, fid) 

    y_pred_rfc = rfc.predict(X_test)

    accuracy_rfc = accuracy_score(y_test,y_pred_rfc)

    print()
    print('Accuracy for the AdaBoostClassifier: ' + str(accuracy_rfc))
    print()

    # Train a Multi Layer Perceptron
    mlp = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, verbose=False)

    mlp.fit(X_train, y_train)

    # save the classifier
    with open('mlp.pkl', 'wb') as fid:
        pickle.dump(mlp, fid) 

    y_pred_mlp = mlp.predict(X_test)

    accuracy_mlp = accuracy_score(y_test,y_pred_mlp)

    print()
    print('Accuracy for the Multi Layer Perceptron: ' + str(accuracy_mlp))
    print()

    # Train an Adaboost Classifier
    adb = AdaBoostClassifier(verbose=False)

    adb.fit(X_train, y_train)

    # save the classifier
    with open('adb.pkl', 'wb') as fid:
        pickle.dump(adb, fid) 

    y_pred_adb = adb.predict(X_test)

    accuracy_adb = accuracy_score(y_test,y_pred_adb)

    print()
    print('Accuracy for the AdaBoostClassifier: ' + str(accuracy_adb))
    print()

    return (X_train, X_test, y_train, y_test)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    # 3.1 - Extract data and run classifier
    output_31 = class31(args.input)