#!/usr/bin/env python

""""
    File name: train_model.py
    Author: Leo Stanislas
    Date created: 2018/07/11
    Python Version: 2.7
"""

import sys
import rospy
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from config import cfg
import os

sys.path.insert(0, '../model')
sys.path.insert(0, '../utils')

from LidarDatasetHC import LidarDatasetHC


def display_scores(scores):
    print('Scores: ', scores)
    print('Mean: ', scores.mean())
    print('Std Dev: ', scores.std())


class Classifier(object):
    def __init__(self, model=None, search_params=None):
        self.model = model
        self.search_params = search_params


if __name__ == '__main__':
    rospy.init_node('train_model', anonymous=True)



    root_dir = cfg.ROOT_DIR
    data_dir = os.path.join(root_dir, 'data')

    input_model = 'int_rough_slope'
    # input_model = None
    # output_model = 'cccccccccccccccccccccc2'
    output_model = None

    # Control command
    smoke = True
    dust = True
    feature_selection = False
    features = [
        'int_mean',
        'int_var',
        # 'echo',
        'roughness',
        'slope'
    ]
    shuffle = False
    mu = None
    sigma = None

    X_train = np.array([])
    y_train = np.array([])
    X_test = np.array([])
    y_test = np.array([])
    clf = Classifier()
    if input_model:
        input_path = os.path.join(root_dir, 'model/saved_models', input_model + '.pkl')
        print('Loading model from %s' % input_path)
        clf.model, smoke, dust, features, mu, sigma = joblib.load(input_path)
    else:
        print('Training new model')

        trainset = LidarDatasetHC(features=features, root_dir=data_dir, train=True, val=False, smoke=smoke, dust=dust, shuffle=42)

        if feature_selection:
            trainset.select_features('TreeBased')
        features = trainset.features
        mu, sigma = trainset.mu, trainset.sigma

        X_train = trainset.data
        y_train = trainset.labels

        clf = Classifier(RandomForestClassifier(max_features=len(features), n_estimators=100, max_depth=10))
        # clf = Classifier(SVC())  # Linear SVM
        # clf = Classifier(GaussianNB())  # Naive Bayes

        # Uncomment this to perform random search
        # classifiers['random_forest'].search_params = {'n_estimators': [3, 10, 30, 100, 200],
        #                                               'max_features': range(1, len(features)+1),
        #                                               'max_depth': [1, 5, 10, 20]}
        #
        # classifiers['SVM'].search_params = {'kernel': ['rbf', 'linear'], 'C': [0.025, 0.1, 0.5, 1],
        #                                     'gamma': ['auto', 2]}

        # Model fine-tuning
        if clf.search_params:
            param_search = RandomizedSearchCV(clf.model, clf.search_params, cv=5, scoring='neg_mean_squared_error')
            param_search.fit(X_train, y_train)

            final_model = param_search.best_estimator_  # Get full best estimator (if refit=true it retains it on the whole training set)

            print('Grid search evaluation')
            cvres = param_search.cv_results_  # Get evaluation score

            for mean_score, params in sorted(zip(cvres["mean_test_score"], cvres["params"]), reverse=True):
                print(np.sqrt(-mean_score), params)
            clf.model = param_search.best_estimator_  # Get full best estimator (if refit=true it retains it on the whole training set)
        else:
            clf.model.fit(X_train, y_train)  # Train classifier

    print("Testing classifier...")

    testset = LidarDatasetHC(features=features, root_dir=data_dir, train=False, test=True, smoke=smoke, dust=dust,
                             mu=mu, sigma=sigma)

    X_test = testset.data
    y_test = testset.labels

    y_pred = clf.model.predict(X_test)  # Perform prediction

    scores = cross_val_score(clf.model, X_test, y_test, scoring='neg_mean_squared_error', cv=10)
    rmse_scores = np.sqrt(-scores)
    # score = clf.model.score(X_test, y_test)  # Compute score (without cross-validation, won't see overfitting)
    print('Confusion Matrix')
    cnf_matrix = confusion_matrix(y_test, y_pred)  # Compute confusion matrix
    print(cnf_matrix)

    accuracy = float(sum(y_pred == y_test)) / float(len(y_test)) * 100
    print("Accuracy: %s%%" % '{:0.1f}'.format(accuracy))

    print('RMSE scores over 10-fold cross-validation')
    display_scores(rmse_scores)

    print('Precision: %f' % precision_score(y_test, y_pred))

    print('Recall: %f' % recall_score(y_test, y_pred))

    print('F1 score: %f' % f1_score(y_test, y_pred))

    print('Classification Report:')
    print(classification_report(y_test, y_pred))

    if output_model:
        output_path = os.path.join(root_dir, 'model/saved_models', output_model + '.pkl')
        print('Saving model at %s' % output_path)
        joblib.dump([clf.model, smoke, dust, features, mu, sigma],
                    output_path)
