#!/usr/bin/python

import sys
sys.path.append("../tools/")
import pickle
import os
import numpy as np
import math
import matplotlib.pyplot as plt

import functions_for_poi

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import train_test_split

# Begin by loading in all 21 features in the dataset
# target_list contains the target label. features_list contains the features
# email address not included in features_list below. So only 19 features.

target_list = ['poi']

email_and_financial_list =  ['from_messages',
                             'from_poi_to_this_person',
                             'from_this_person_to_poi',
                             'shared_receipt_with_poi',
                             'to_messages',
                             'bonus',
                             'deferral_payments',
                             'deferred_income',
                             'director_fees',
                             'exercised_stock_options',
                             'expenses',
                             'loan_advances',
                             'long_term_incentive',
                             'other',
                             'restricted_stock',
                             'restricted_stock_deferred',
                             'salary',
                             'total_payments',
                             'total_stock_value']

features_list = target_list + email_and_financial_list

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Processes the data_dict to returns labels, features.
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

########### Task 1: Remove outliers ###########
# As the dataset is small, inspect the original source (PDF) for outliers.
# Will remove "TOTAL" (a spreadsheet anomaly) and
# "THE TRAVEL AGENCY IN THE PARK"(not a person)
# Check if we have any employees with no information

functions_for_poi.check_NaN_values(data_dict, email_and_financial_list)

# Remove the three entries identified above
data_dict.pop("TOTAL", 0 )
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0 )
data_dict.pop("LOCKHART EUGENE E", 0 )

print "\nPrint out important characteristics of the cleaned dataset"
print "There are {} samples in the dataset".format(len(data_dict))
no_pois = 0
for key, value in data_dict.items():
    if value['poi']:
        no_pois = no_pois + 1

no_non_pois = len(data_dict) - no_pois

print "{} are POI and {} are non-POIs.".format(no_pois, no_non_pois)

print "The number of features used is {}".format(len(email_and_financial_list))

## Check numbers of valid (non-NaN) data points each feature has.
# Initialise dictionary to hold counts

d = {}
for name, value in data_dict.items():
    del(value['email_address']) # remove this entry as non-numeric and can't check if nan.
    for e in value.keys():
        d[e] = 0

for name, value in data_dict.items():
    for e,v in value.items():
        if math.isnan(float(v)):
            d[e] = d[e] + 1

# print out the number of valid points for each feature
print "\nThe number of valid points for each feature is:"
for k,v in d.items():
    print k, ':', len(data_dict) - v

########### Task 2: Select features to use ###########

# Use univariate feature selection to rank features

# Redefine data now that we have removed three samples.
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print "\nOriginal Features listed according to their k score:"
feature_list_by_kscore = \
functions_for_poi.kbestfunction(features, labels, email_and_financial_list)

########### Task 3: Create new feature(s) ###########
# I will create some new features
# 1. Email: Look at the number of emails from_poi_to_this_person as a
#    fraction of the total number of emails received by this person
# 2. Email: Similarly, look at the number of emails from_this_person_to_poi
#    as a fraction of the total number of emails sent by this person
# 1. Financial: A POI committing fraud would have not placed much faith in
#    the long term future of ENRON. They would have wanted to get as much
#    money out as quickly as possible.
#    deferred_income "reflects voluntary executive deferrals of salary,
#    annual cash incentives, and long-term cash incentives"
#    deferred_income as a fraction of the sum of salary, bonus and
#    long_term_incentive might be lower for pois than for non-pois.

# create features 1,2 and 3 in data_dict
# if the value of any key is NaN, assign a value of zero to the new feature.

for key, value in data_dict.items():
    # create fraction_from_poi_to_this_person feature
    if math.isnan(float(value["from_poi_to_this_person"])) or math.isnan(float(value["to_messages"])):
        value["fraction_from_poi_to_this_person"] = 0.0 #  set to 0 and not NaN to help make selectkbest work
    else:
        value["fraction_from_poi_to_this_person"] = float(value["from_poi_to_this_person"])/float(value["to_messages"])

    # create fraction_from_this_person_to_poi feature
    if math.isnan(float(value["from_this_person_to_poi"])) or math.isnan(float(value["from_messages"])):
        value["fraction_from_this_person_to_poi"] = 0.0 #  set to 0 and not NaN to help make selectkbest work
    else:
        value["fraction_from_this_person_to_poi"] = float(value["from_this_person_to_poi"])/value["from_messages"]

    # create fraction_deferred_income feature
    if math.isnan(float(value["deferred_income"])) or math.isnan(float(value["salary"])) or math.isnan(float(value["bonus"])) or math.isnan(float(value["long_term_incentive"])) :
        value["fraction_deferred_income"] = 0.0 #  set to 0 and not NaN to help make selectkbest work
    else:
        value["fraction_deferred_income"] = float(value["deferred_income"])/(float(value["salary"]) + float(value["bonus"]) + float(value["long_term_incentive"]))

# Let's add these new engineered features to our feature list,
# create a new list containing all the features and
# recompute the k score for each one

# Redefine data, labels and features to include new features

features_list = features_list + \
['fraction_from_poi_to_this_person'] + \
['fraction_from_this_person_to_poi'] + \
['fraction_deferred_income']

email_and_financial_and_engineered_list = email_and_financial_list + \
['fraction_from_poi_to_this_person'] + \
['fraction_from_this_person_to_poi'] + \
['fraction_deferred_income']

data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print "\nOriginal and Engineered Features listed according to their k score:"
feature_list_by_kscore = functions_for_poi.kbestfunction(features, labels,\
                         email_and_financial_and_engineered_list)

# set dataset
features_list = ['poi'] + feature_list_by_kscore
data = featureFormat(data_dict, features_list , sort_keys = True)
labels, features = targetFeatureSplit(data)


########## Task 4: Try a variety of classifiers ###########
# Initually, test a variety of classifiers with 5, 10 and 15 features.
# This identified the naive bayes classifier as the best
# For this classifier, see how performance varies as we change
# the number of features.

# Split our data into training and testing sets
# As the dataset is small and imbalanced with only 18 POI out
# of a total of 146, cannot use train_test_split.

from sklearn.cross_validation import StratifiedShuffleSplit
shuffle = StratifiedShuffleSplit(labels, n_iter=10, test_size=0.3, random_state=42)


for train_idx, test_idx in shuffle:
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

# try a NaiveBayes classifier
from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()
clf_nb.fit(features_train, labels_train)
# test performance using tester.py file

# try a decision tree
from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier(random_state = 42)
clf_dt.fit(features_train, labels_train)
# test performance using tester.py file

# Let's try gridsearch on the decision tree classifier
# As the data is imbalanced use the StratifiedShuffleSplit as cv
# Use f1 scoring

from sklearn.grid_search import GridSearchCV
# create a dictionary with all the parameters we want to search through

param_grid = {"min_samples_split": [1,2,3,4,5],
              "max_depth": [2,3,4,5]
             }

# Redefine the shuffle for use in the crossvalidation in the grid search
shuffle_cv = StratifiedShuffleSplit(y = labels_train, n_iter = 10,\
                                    test_size = 0.3, random_state = 42)

# create the gridsearch object
grid_search_object = GridSearchCV(
    estimator = DecisionTreeClassifier(random_state = 42), # cannot use clf_dt again as it's been fitted
    param_grid = param_grid,
    cv = shuffle_cv,
    scoring = 'f1')

# fit classifier to the data and assign classifier
grid_search_object.fit(features_train, labels_train)
clf_dt_optimised = grid_search_object.best_estimator_

# try a random forest
# use same grid_search_object as above

from sklearn.ensemble import RandomForestClassifier

grid_search_object = GridSearchCV(estimator = RandomForestClassifier(random_state = 42),
                                  param_grid = param_grid,
                                  cv = shuffle_cv,
                                  scoring = 'f1'
                                  )

# fit classifier to the data and assign classifier
grid_search_object.fit(features_train, labels_train)
clf_rf = grid_search_object.best_estimator_

# Having identified the naive bayes classifier as the most performant
# let's test how its performance varies with the number of features used

# set my_dataset for export
my_dataset = data_dict

dict_precision = dict()
dict_recall = dict()

for i in range(1, len(feature_list_by_kscore) + 1):
    dict_precision[i], dict_recall[i] = functions_for_poi.test_classifier(
            clf = GaussianNB(), dataset = my_dataset,
            feature_list = ['poi'] + feature_list_by_kscore [:i])

x_values = dict_precision.keys()
precision_values = dict_precision.values()
recall_values = dict_recall.values()

# plot precision and recall scores vs number of features selected.
plt.plot(x_values, precision_values,
            label = "Precision", color = 'blue', marker = 'o')
plt.plot(x_values, recall_values,
            label = "Recall", color = 'green', marker = 'o')
plt.axhline(0.3, linestyle = '--', color = 'red')
plt.grid()
plt.legend(loc = 'best')
plt.title('Precision and Recall vs Number of Features')
plt.xlabel('K Best Features')
plt.ylabel('Score')
plt.show()

### Task 5: Dump classifier, dataset, and features_list

# Naive Bayes classifier selected
# From plot, select top 8 features for final classifier.
clf = GaussianNB()
features_list = ['poi'] + feature_list_by_kscore[:8]

dump_classifier_and_data(clf, my_dataset, features_list)
