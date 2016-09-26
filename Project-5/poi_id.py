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

features_list_full = target_list + email_and_financial_list

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Processes the data_dict to returns labels, features.
data = featureFormat(data_dict, features_list_full, sort_keys = True)
labels, features = targetFeatureSplit(data)

########### Task 1: Remove outliers ###########
# As the dataset is small, inspect the original source (PDF) for outliers.
# Will remove "TOTAL" (a spreadsheet anomaly) and "THE TRAVEL AGENCY IN THE PARK"(not a person)
# Check if we have any employees with no information

functions_for_poi.check_NaN_values(data_dict, email_and_financial_list)

# Remove the three entries identified above
data_dict.pop("TOTAL", 0 )
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0 )
data_dict.pop("LOCKHART EUGENE E", 0 )

# Now have 143 samples in our data.

########### Task 2: Select features to use ###########

# Use univariate feature selection to automatically select features in sklearn.
# use default k value = 10

# Redefine data now that we have removed three samples.
data = featureFormat(data_dict, features_list_full, sort_keys = True)
labels, features = targetFeatureSplit(data)

feature_list_by_kscore = functions_for_poi.kbestfunction(features, labels, email_and_financial_list)

########### Task 3: Create new feature(s) ###########
# I will create some new features
# 1. Email: Look at the number of emails from_poi_to_this_person as a fraction of the total number of emails received by this person
# 2. Email: Similarly, look at the number of emails from_this_person_to_poi as a fraction of the total number of emails sent by this person
# Possibly combine these two new features in a weighted average. This can be considered to be the interaction of a given person with a poi.
# 1. Financial: A POI committing fraud would have not placed much faith in the long term future of ENRON. They would have wanted to get as much money out as quickly as possible.
# deferred_income "reflects voluntary executive deferrals of salary, annual cash incentives, and long-term cash incentives"
# deferred_income as a fraction of the sum of salary, bonus and long_term_incentive might be lower for pois than for non-pois.

# create features 1,2 and 3 in data_dict
# if the value of any key is NaN, assign a value of zero to the new feature.
# Will need to consider how many non-zero values I get when it comes to using the feature

for key, value in data_dict.items():
    # create fraction_from_poi_to_this_person feature
    if math.isnan(float(value["from_poi_to_this_person"])) and math.isnan(float(value["to_messages"])):
        value["fraction_from_poi_to_this_person"] = 0
    else:
        value["fraction_from_poi_to_this_person"] = float(value["from_poi_to_this_person"])/float(value["to_messages"])

    # create fraction_from_this_person_to_poi feature
    if math.isnan(float(value["from_this_person_to_poi"])) and math.isnan(float(value["from_messages"])):
        value["fraction_from_this_person_to_poi"] = 0
    else:
        value["fraction_from_this_person_to_poi"] = float(value["from_this_person_to_poi"])/value["from_messages"]

    # create fraction_deferred_income feature
    if math.isnan(float(value["deferred_income"])) and math.isnan(float(value["salary"])) and math.isnan(float(value["bonus"])) and math.isnan(float(value["long_term_incentive"])) :
        value["fraction_deferred_income"] = 0
    else:
        value["fraction_deferred_income"] = float(value["deferred_income"])/(float(value["salary"]) + float(value["bonus"]) + float(value["long_term_incentive"]))

# check for how many people we actually have non-zero values for the new features
functions_for_poi.check_new_features(data_dict, "fraction_from_poi_to_this_person")
functions_for_poi.check_new_features(data_dict, "fraction_from_this_person_to_poi")
functions_for_poi.check_new_features(data_dict, "fraction_deferred_income")

# Let's visualise some of these new features to see if they allow us to discriminate between poi and non-pois.
functions_for_poi.create_plot(data_dict, "fraction_from_poi_to_this_person", "fraction_from_this_person_to_poi")
functions_for_poi.create_plot(data_dict, "deferred_income", "fraction_deferred_income")

# The new features do not help discriminate between pois and non-pois.
# They will therefore not be included in our classifier

########## Task 4: Try a variety of classifiers ###########
## Please name your classifier clf for easy export below.
## Note that if you want to do PCA or other multi-stage operations,
## you'll need to use Pipelines. For more info:
## http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# Split our data into training and testing sets
# I will try the GaussianNB Classifier with all features, ten features, five features and three features

features_list_top_ten = target_list + feature_list_by_kscore[:10]
features_list_top_five = target_list + feature_list_by_kscore[:5]
features_list_top_three = target_list + feature_list_by_kscore[:3]

data_top_ten = featureFormat(data_dict, features_list_top_ten, sort_keys = True)
labels_top_ten, features_top_ten = targetFeatureSplit(data_top_ten)

data_top_five = featureFormat(data_dict, features_list_top_five, sort_keys = True)
labels_top_five, features_top_five = targetFeatureSplit(data_top_five)

data_top_three = featureFormat(data_dict, features_list_top_three, sort_keys = True)
labels_top_three, features_top_three = targetFeatureSplit(data_top_three)

# As the dataset is small and imbalanced with only 18 POI out of a total of 146, cannot use train_test_split.
# Even with a test_size of 0.25, with train_test_split, I'd only be trying to predict 4 to 5 POIs in the dataset
# Need to change this in future. But for now, get one version up and working.

labels_train_all_features, labels_test_all_features, features_train_all_features,features_test_all_features = train_test_split(labels, features, test_size = 0.1)
labels_train_ten_features, labels_test_ten_features, features_train_ten_features,features_test_ten_features = train_test_split(labels_top_ten, features_top_ten, test_size = 0.1)
labels_train_five_features, labels_test_five_features, features_train_five_features,features_test_five_features = train_test_split(labels_top_five, features_top_five, test_size = 0.1)
labels_train_three_features, labels_test_three_features, features_train_three_features,features_test_three_features = train_test_split(labels_top_three, features_top_three, test_size = 0.1)

# try a NaiveBayes classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

clf_NB_all = GaussianNB()
clf_NB_all.fit(features_train_all_features, labels_train_all_features)
pred_NB_all = clf_NB_all.predict(features_test_all_features)
accuracy_NB_all = accuracy_score(pred_NB_all, labels_test_all_features)
precision_NB_all = precision_score(pred_NB_all, labels_test_all_features)
recall_NB_all = recall_score(pred_NB_all, labels_test_all_features)
print "\nThe NB classifier using all features has accuracy {}, precision {} and recall {}.".format(accuracy_NB_all, precision_NB_all, recall_NB_all)

from sklearn import metrics
print metrics.classification_report(labels_test_all_features, pred_NB_all)
print metrics.confusion_matrix(labels_test_all_features, pred_NB_all)

# Print the averaged f1-score - a measure of the overall performance of an algorithm.
print metrics.f1_score(labels_test_all_features, pred_NB_all)
# Quantify any over-fitting by omputing the f1-score on the training data itself:
print metrics.f1_score(labels_train_all_features, clf_NB_all.predict(features_train_all_features))




clf_NB_ten = GaussianNB()
clf_NB_ten.fit(features_train_ten_features, labels_train_ten_features)
pred_NB_ten = clf_NB_ten.predict(features_test_ten_features)
accuracy_NB_ten = accuracy_score(pred_NB_ten, labels_test_ten_features)
precision_NB_ten = precision_score(pred_NB_ten, labels_test_ten_features)
recall_NB_ten = recall_score(pred_NB_ten, labels_test_ten_features)
print "\nThe NB classifier using ten features has accuracy {}, precision {} and recall {}.".format(accuracy_NB_ten, precision_NB_ten, recall_NB_ten)

clf_NB_five = GaussianNB()
clf_NB_five.fit(features_train_five_features, labels_train_five_features)
pred_NB_five = clf_NB_five.predict(features_test_five_features)
accuracy_NB_five = accuracy_score(pred_NB_five, labels_test_five_features)
precision_NB_five = precision_score(pred_NB_five, labels_test_five_features)
recall_NB_five = recall_score(pred_NB_five, labels_test_five_features)
print "\nThe NB classifier using five features has accuracy {}, precision {} and recall {}.".format(accuracy_NB_five, precision_NB_five, recall_NB_five)

# # ### Task 5: Tune your classifier to achieve better than .3 precision and recall
# # ### using our testing script. Check the tester.py script in the final project
# # ### folder for details on the evaluation method, especially the test_classifier
# # ### function. Because of the small size of the dataset, the script uses
# # ### stratified shuffle split cross validation. For more info:
# # ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# #
# # # Example starting point. Try investigating other evaluation techniques!
# # from sklearn.cross_validation import train_test_split
# # features_train, features_test, labels_train, labels_test = \
# #     train_test_split(features, labels, test_size=0.3, random_state=42)
# #
# # ### Task 6: Dump your classifier, dataset, and features_list so anyone can
# # ### check your results. You do not need to change anything below, but make sure
# # ### that the version of poi_id.py that you submit can be run on its own and
# # ### generates the necessary .pkl files for validating your results.
# #
# # ### Store to my_dataset for easy export below.
# # my_dataset = data_dict
# #
# # dump_classifier_and_data(clf, my_dataset, features_list)
