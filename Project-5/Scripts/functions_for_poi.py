#!/usr/bin/python

import sys
sys.path.append("../tools/")
import pickle
import os
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.cross_validation import StratifiedShuffleSplit
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.feature_selection import SelectKBest

def check_NaN_values(dict_object, list_object):
    """
    Input data_dict and features_list.
    Inspects the data_dict and returns the name of any employees
    in the dataset for whom we have no information.
    """
    for name, value in dict_object.items():
        name_list = []
        for e in list_object:
            if math.isnan(float(value[e])):
                name_list.append("1")
        if len(name_list) == len(list_object):
            print "For {}, we have no information.".format(name)

def kbestfunction(feature_object, label_object, features_list_object):
    """
    Uses sklearn selectkbest to select features
    according to the k highest scores.
    """
    selector = SelectKBest(k = 'all')
    selector.fit_transform(feature_object, label_object)
    scores = selector.scores_

    unsorted_score_list = zip(features_list_object, scores)
    sorted_score_list = sorted(unsorted_score_list,\
    key=lambda x: x[1], reverse = True)

    features_by_importance = []
    for e in range(len(features_list_object)):
        print sorted_score_list[e]
        features_by_importance.append(sorted_score_list[e][0])
    return features_by_importance

def test_classifier(clf, dataset, feature_list, folds = 1000):
    """
    Take dataset and return the precision and recall of the classifier.

    Modified the test_classifier from tester.py to return
    precision and recall and enable optimal feature selection.
    """
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv:
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )

        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        return precision, recall
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."
