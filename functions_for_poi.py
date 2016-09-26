#!/usr/bin/python

import sys
sys.path.append("../tools/")
import pickle
import os
import numpy as np
import math
import matplotlib.pyplot as plt

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.feature_selection import SelectKBest

def check_NaN_values(dict_object, list_object):
    """
    Input data_dict and features_list.
    Inspects the data_dict and returns the name of any employees in the dataset for whom we have no information.
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
    Uses sklearn selectkbest to select features according to the k highest scores.
    """
    selector = SelectKBest()
    selector.fit_transform(feature_object, label_object)
    scores = selector.scores_
    unsorted_score_list = zip(features_list_object, scores)
    sorted_score_list = sorted(unsorted_score_list, key=lambda x: x[1], reverse = True)
    print "\nFeatures listed according to their k score:"
    features_by_importance = []
    for e in range(len(features_list_object)):
        print sorted_score_list[e]
        features_by_importance.append(sorted_score_list[e][0])
    return features_by_importance

def check_new_features(dict_object, newfeature1):
    counter_feature_1 = 0
    for key, value in dict_object.items():
        if value [newfeature1] == 0:
            counter_feature_1 = counter_feature_1 + 1

    print "\nFor the variable " + newfeature1 + ", {} percent of the values were zero".format(100*float(counter_feature_1)/ len(dict_object))

def create_plot(data_dict, feature_x, feature_y):
    """
    creates a scatterplot. Poi value is used as colour.
    """
    data = featureFormat(data_dict, [feature_x, feature_y, 'poi'])
    for point in data:
        x = point[0]
        y = point[1]
        poi = point[2]
        if poi:
            color = "green"
        else:
            color = "red"
        plt.scatter(x, y, color = color)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()
