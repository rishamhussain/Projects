### ------------ Libraries
import time
import pytest
import os
# Class Decision tree classifier requires Pandas and Numpy
import pandas as pd
import numpy as np
import ucimlrepo
from ucimlrepo import fetch_ucirepo
from memory_profiler import profile
from memory_profiler import memory_usage
import matplotlib.pyplot as plt
from itertools import combinations

### ----------- Importing classes of classifiers

# Importing created decision tree
from DecisionTreeClassifier_v8 import Node
from DecisionTreeClassifier_v8 import DecisionTreeClassifier

# Testing against sklearn-decision tree classifier
from sklearn.tree import DecisionTreeClassifier as DTC
# Importing training and testing set creater from sklearn
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef



### ------------ Test Functions --------------------
# Compare accuracy 'X' with accuracy of SKlearn library 'SK' allowing an error range
def test_compareAccuracy(Y_test,SK_pred,Y_pred,error = 0.2):
    """
    Function to compare the accuracies of custom tree predictions with SK decision tree
    :param error: error range allowed
    :return: empty
    """

    X = accuracy_score(Y_test, Y_pred)
    SK = accuracy_score(Y_test, SK_pred)
    classification_acc_self.append(X)
    classification_acc_SK.append(SK)
    print("\n Accuracy:",round(X,3),"is close to SKlearn: ", round(SK,3),"±",error)
    try:
        assert abs(X - SK) <= error
    except AssertionError:
        global total_failed
        total_failed += 1

# Compare Predictions from decisiontree and SKlearn classification
def test_comparePrediction(SK_pred, Y_pred, thresh = 0.9):
    """
    Function to compare the predicted values of custom decision tree with SK learn
    :param thresh: minimum accuracy threshold
    :return: empty
    """
    acc = accuracy_score(SK_pred, Y_pred)
    prediction_acc.append(acc)
    print("\n Prediction Accuracy:",round(acc,3))
    try:
        assert acc >= thresh
    except AssertionError:
        global total_failed
        total_failed += 1
        print("Failed test accuracy less than threshold")

def test_comparePrecision(Y_test,SK_pred,Y_pred,error = 0.2):
    """
    Function to compare the accuracies of custom tree predictions with SK decision tree
    :param error: error range allowed
    :return: empty
    """

    X = precision_score(Y_test, Y_pred, average= 'weighted', zero_division=1)
    SK = precision_score(Y_test, SK_pred, average='weighted', zero_division=1)
    precision_self.append(X)
    precision_SK.append(SK)
    print("\n Precision:",round(X,3),"is close to SKlearn: ", round(SK,3)," ±0.5 ")
    try:
        assert abs(X - SK) <= error
    except AssertionError:
        global total_failed
        total_failed += 1

def test_compareRecall(Y_test,SK_pred,Y_pred,error = 0.2):
    """
    Function to compare the recall of custom tree predictions with SK decision tree
    :param error: error range allowed
    :return: empty
    """

    X = recall_score(Y_test, Y_pred, average= 'weighted', zero_division=1)
    SK = recall_score(Y_test, SK_pred, average='weighted', zero_division=1)
    recall_self.append(X)
    recall_SK.append(SK)
    print("\n Recall:",round(X,3),"is close to SKlearn: ", round(SK,3),"±",error)
    try:
        assert abs(X - SK) <= error
    except AssertionError:
        global total_failed
        total_failed += 1

def test_compareF1(Y_test,SK_pred,Y_pred,error = 0.2):
    """
    Function to compare the F1 score of custom tree predictions with SK decision tree
    :param error: error range allowed
    :return: empty
    """

    X = f1_score(Y_test, Y_pred, average= 'weighted')
    SK = f1_score(Y_test, SK_pred, average='weighted')
    f1_self.append(X)
    f1_SK.append(SK)
    print("\n F1:",round(X,3),"is close to SKlearn: ", round(SK,3),"±",error)
    try:
        assert abs(X - SK) <= error
    except AssertionError:
        global total_failed
        total_failed += 1

def test_curve(Y_test,SK_pred,Y_pred,error = 0.2):
    """
    Function to compare the roc of custom tree predictions with SK decision tree
    :param error: error range allowed
    :return: empty
    """

    X = roc_auc_score(Y_test, Y_pred)
    SK = roc_auc_score(Y_test, SK_pred)
    auc_roc_self.append(X)
    auc_roc_SK.append(SK)
    print("\n roc_auc score:",round(X,3),"is close to SKlearn: ", round(SK,3)," ±0.5 ")
    try:
        assert abs(X - SK) <= error
    except AssertionError:
        global total_failed
        total_failed += 1

def test_MCC(y_true,Y_pred, error = 0.2):
    """
    Function to calculate the mcc of custom tree predictions with SK decision tree
    :param error: error range allowed
    :return: empty
    """
    # Identify unique classes
    classes = np.unique(y_true)

    # Initialize variables to store MCC values and confusion matrices
    mcc_values = []
    confusion_matrices = []

    # Calculate MCC for each pair of classes
    for class1, class2 in combinations(classes, 2):
        # Create binary labels for the current pair of classes
        binary_y_true = [1 if label == class1 else -1 if label == class2 else 0 for label in y_true]
        binary_y_pred = [1 if label == class1 else -1 if label == class2 else 0 for label in Y_pred]

        # Calculate MCC for the binary classification
        mcc = matthews_corrcoef(binary_y_true, binary_y_pred)
        mcc_values.append(mcc)

        # Create a confusion matrix for the binary classification
        confusion_matrix_binary = confusion_matrix(binary_y_true, binary_y_pred)
        confusion_matrices.append(confusion_matrix_binary)

    # Aggregate MCC values and confusion matrices
    overall_mcc = sum(mcc_values) / len(mcc_values)
    #overall_confusion_matrix = sum(confusion_matrices)
    return overall_mcc

# Test-3
def test_compareTime(start1,start2,end1,end2, decimal_value = 3):
    """
    Function to compare the time taken by custom decision tree with SK learn decision tree
    :param decimal_value: Round the time difference to decimal vlaue
    :return: empty

    """

    time_difference = round((end1 - start1) - (end2 - start2), decimal_value)
    timediff.append(time_difference)
    if time_difference > 0:
        print("\n",time_difference,"secs More time taken by Custom classifier")
    else:
        print("\n", time_difference,"secs Less time taken by classifier")

def calculate_split_index(df, split_percentage_rows, split_percentage_cols):
    """
    Calculate the row index for splitting a DataFrame based on a given percentage.

    Parameters:
    - df: DataFrame to be split.
    - split_percentage: Percentage at which to split the DataFrame.

    Returns:
    - Index where the split should occur.
    """
    total_rows = df.shape[0]
    split_index_rows = int(total_rows * split_percentage_rows / 100)
    total_cols = df.shape[1]
    split_index_cols = int(total_cols * split_percentage_cols / 100)
    return split_index_rows, split_index_cols




@profile
def test_Parameters(df2, percentages, method_name, min_samples_split, tree_levels):
    # Loop through size combinations
    for i in percentages:
        rows, cols = calculate_split_index(df2, i, 100)  # calculate_split_index(df, split_percentage_rows, split_percentage_cols):
        df = df2.iloc[0:rows, 0: cols]

        for i2 in method_name:

            for i3 in min_samples_split:

                for i4 in tree_levels:

                    print("Currently --------- ", i, i2, i3, i4)
                    # Appending variables
                    dataset.append(dataset_name)
                    variables.append(variable_types)
                    rows_percent.append(i)
                    method.append(i2)
                    min_samples.append(i3)
                    depth.append(i4)
                    length_data.append(df.shape[0])
                    features_data.append(df.shape[1])

                    # Data separated for training and testing
                    df.columns = column_names
                    X = df.iloc[:, :-1].values
                    Y = df.iloc[:, -1].values.reshape(-1, 1)
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

                    # CLassify with Custom Decision Tree
                    print("Fitting model / Training")
                    start1 = time.time()
                    mem_usage1 = memory_usage(-1, interval=.2, timeout=3, multiprocess=True)
                    x = [i[0] for i in mem_usage1]
                    mem_usage1 = sum(x) / len(mem_usage1)
                    classifier = DecisionTreeClassifier(method=i2, min_samples_split=i3, tree_levels=i4)
                    tree1 = classifier.fit(X_train, Y_train, df)
                    # classifier.print_tree()

                    print("Predicting")
                    Y_pred = classifier.predict(X_test)
                    acc_tree = accuracy_score(Y_test, Y_pred)
                    print("Accuracy Decision Tree: ", acc_tree)
                    end1 = time.time()

                    ### Classify with SKlearn Decison Tree Classifier
                    start2 = time.time()
                    mem_usage2 = memory_usage(-1, interval=.2, timeout=3, multiprocess=True)
                    x = [i[0] for i in mem_usage2]
                    mem_usage2 = sum(x)/len(mem_usage2)
                    clf = DTC(criterion=i2, splitter='best', min_samples_split=i3, max_depth=i4)
                    clf = clf.fit(X_train, Y_train)

                    SK_pred = clf.predict(X_test)
                    acc_sk = metrics.accuracy_score(Y_test, SK_pred)
                    print("Accuracy SK Learn:", acc_sk)
                    end2 = time.time()

                    ### Run Comparison tests
                    test_compareAccuracy(Y_test,SK_pred,Y_pred, error)
                    test_comparePrediction(SK_pred, Y_pred, similarity_thresh)
                    test_compareTime(start1,start2,end1,end2)
                    test_comparePrecision(Y_test,SK_pred,Y_pred, error)
                    test_compareRecall(Y_test,SK_pred,Y_pred, error)
                    test_compareF1(Y_test,SK_pred,Y_pred, error)
                    #test_curve(Y_test,SK_pred,Y_pred, error)
                    mcc_val = test_MCC(Y_test,Y_pred, error)
                    mcc_self.append(mcc_val)
                    mcc_val = test_MCC(Y_test, SK_pred, error)
                    mcc_SK.append(mcc_val)

                    # # Average Memory Use
                    mem_tree.append(mem_usage1)
                    mem_SK.append(mem_usage2)


    resultsA = pd.DataFrame({'Name': dataset, 'Sample_size': rows_percent,'rows': length_data, 'features': features_data,
                             'variables': variables, 'method': method, 'custom_acc': classification_acc_self,
                             'SK_acc': classification_acc_SK, 'prediction_acc': prediction_acc,
                             'precision_self': precision_self, 'precision_SK': precision_SK,
                             'recall_self': recall_self, 'recall_Sk': recall_SK, 'f1_self': f1_self,
                             'f1_SK': f1_SK, 'mcc_self': mcc_self,'mcc_SK': mcc_SK, 'time_diff': timediff, 'min_samples': min_samples, 'depth': depth,
                             'mem_tree': mem_tree, 'mem_SK': mem_SK})
    return resultsA

def convert_string_columns_to_numeric_categorical(df):
    """
    Convert string columns in a DataFrame to numeric categorical columns, for SK learn
    classification requirement.

    Parameters:
    - df: Input DataFrame.

    Returns:
    - DataFrame with string columns converted to numeric categorical columns.
    """
    for column in df.columns:
        if pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].astype('category').cat.codes

    return df

### Variables to store results for transfer to excel file
dataset = []
rows_percent = []
length_data = []
features_data = []
variables = []
method = []
classification_acc_self = []
classification_acc_SK = []
prediction_acc = []
precision_self = []
precision_SK = []
recall_self = []
recall_SK = []
f1_self = []
f1_SK = []
#auc_roc_self = []
#auc_roc_SK = []
mcc_self = []
mcc_SK = []


timediff = []
min_samples = []
depth = []

mem_tree = []
mem_SK = []


global total_failed
total_failed = 0
result = pd.DataFrame()


## ------------- Data-1 Wine data

## Loading data

column_names = pd.read_csv("winequalityred.csv", header=None, sep=';', nrows=1)
column_names = column_names.T
df1 = pd.read_csv("winequalityred.csv", header=None, sep=';', skiprows=1)
column_names  = df1.iloc[0,]
#df1 = df1.drop(0)
#df1.columns = column_names.T
df2 = pd.read_csv("winequalitywhite.csv", header=None, sep=';', skiprows=1)
df = pd.concat([df1, df2], axis=0, ignore_index=True)
# Using shuffle since two datasets were imported in order
df = shuffle(df)
df.reset_index(inplace=True, drop=True)
#df = df.iloc[0:400,:]
df.head

"""
## Check label balance
class_distribution = df.iloc[:,-1].value_counts()
print("Class Distribution:")
print(class_distribution)

# Plotting a bar chart to visualize class distribution
class_distribution.plot(kind='bar', rot=0)
plt.xlabel('Class Label')
plt.ylabel('Count')
plt.title('Class Distribution - Wine Quality')
plt.show()
"""

### Parameter Tuning

# Setting values of parameters for testing
dataset_name = "Wine"
variable_types = "numeric"

# Test function parameters
error = 0.1
similarity_thresh = 0.9

method_nameA = ["gini", "entropy"]
percentages_of_rows = [25, 50, 75, 100]
min_samplesize = [3, 15, 50, 100, 200, 500]     # minimum items in a subset to allow splitting or branching
depth_value = [2, 4, 8, 16]     # branching depth / tree levels allowed


# Testing Classification
resultsA = test_Parameters(df, percentages_of_rows, method_nameA, min_samplesize, depth_value)
#result = pd.concat([resultsA, result], ignore_index=True)



## ------------- Data-2 Customer Churn data

## Loading data

df = pd.read_csv("Customer Churn.csv")
column_names  = df.columns
df.head

### Parameter Tuning

# Setting values of parameters for testing
dataset_name = "CustomerChurn"
variable_types = "mixed"

# Test function parameters
error = 0.5
similarity_thresh = 0.9


method_nameA = ["gini", "entropy"]
percentages_of_rows = [25, 50]
min_samplesize = [3, 15, 50, 100, 200, 500]     # minimum items in a subset to allow splitting or branching
depth_value = [2, 4, 8, 16]     # branching depth / tree levels allowed

# Testing Classification
resultsA = test_Parameters(df, percentages_of_rows, method_nameA, min_samplesize, depth_value)
result = pd.concat([resultsA, result], ignore_index=True)

percentages_of_rows = [75]

# Testing Classification
resultsA = test_Parameters(df, percentages_of_rows, method_nameA, min_samplesize, depth_value)
result = pd.concat([resultsA, result], ignore_index=True)

percentages_of_rows = [100]

# Testing Classification
resultsA = test_Parameters(df, percentages_of_rows, method_nameA, min_samplesize, depth_value)
#result = pd.concat([resultsA, result], ignore_index=True)


## ------------- Data-3 HCV Prediction Data

## Loading data

df = pd.read_csv("HCV_Data.csv")
column_names  = df.columns
df.head
df.shape

### Parameter Tuning

# Setting values of parameters for testing
dataset_name = "HCV"
variable_types = "mixed"

# Test function parameters
error = 0.5
similarity_thresh = 0.8

method_nameA = ["gini", "entropy"]
percentages_of_rows = [25, 50]
min_samplesize = [3, 15, 50, 100, 200, 500]     # minimum items in a subset to allow splitting or branching
depth_value = [2, 4, 8, 16]     # branching depth / tree levels allowed

# Testing Classification
resultsA = test_Parameters(df, percentages_of_rows, method_nameA, min_samplesize, depth_value)
result = pd.concat([resultsA, result], ignore_index=True)

percentages_of_rows = [75, 100]

# Testing Classification
resultsA = test_Parameters(df, percentages_of_rows, method_nameA, min_samplesize, depth_value)
#result = pd.concat([resultsA, result], ignore_index=True)


## ------------- Data-4 Hypothyroid Patients and Medical Tests Data

## Loading data

df = pd.read_csv("hypothyroid.csv")
column_names  = df.columns
df.head
df.shape

# Check and correct missing values
(df.isna()).sum(axis=0)
# Impute with mean for numerical columns
df = df.fillna(df.mean())
(df.isna()).sum(axis=0)

# Convert string to numeric categorical for SKlearn requirements
df = convert_string_columns_to_numeric_categorical(df)

### Parameter Tuning

# Setting values of parameters for testing
dataset_name = "Hypothyroid"
variable_types = "mixed"

# Test function parameters
error = 0.5
similarity_thresh = 0.8

method_nameA = ["gini", "entropy"]
percentages_of_rows = [25, 50]
min_samplesize = [3, 15, 50, 100, 200, 500]     # minimum items in a subset to allow splitting or branching
depth_value = [2, 4, 8, 16]     # branching depth / tree levels allowed

# Testing Classification
result = test_Parameters(df, percentages_of_rows, method_nameA, min_samplesize, depth_value)
#result = pd.concat([resultsA, result], ignore_index=True)

percentages_of_rows = [75, 100]

# Testing Classification
result = test_Parameters(df, percentages_of_rows, method_nameA, min_samplesize, depth_value)
#result = pd.concat([resultsA, result], ignore_index=True)

# Check failed
print("\n Total Failed Tests: ", total_failed)


### --------- Save to Files

# Save to Excel file
result.to_excel("results_v8.xlsx", index=False)

# Save to text file (CSV in this case)
result.to_csv("results_v8.csv", index=False, sep=',')


### -----------------  End   --------------- ###
"""
    The code below was used to create simple decision tree figures for the final paper

## Loading data

df = pd.read_csv("hypothyroid.csv")
column_names = df.columns
df.head
df.shape

# Check and correct missing values
(df.isna()).sum(axis=0)
# Impute with mean for numerical columns
df = df.fillna(df.mean())
(df.isna()).sum(axis=0)

# Convert string to numeric categorical for SKlearn requirements
df = convert_string_columns_to_numeric_categorical(df)

# Train test splitting
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

### Parameter Tuning

# Setting values of parameters for testing
dataset_name = "Hypothyroid"
variable_types = "mixed"

# Test function parameters
error = 0.5
similarity_thresh = 0.8

print("Fitting model / Training")
classifier = DecisionTreeClassifier(method="gini", min_samples_split=6, tree_levels=3)
# Add in data set to collect feature names
# Parameters: classifier.fit (Training data, Labels of training data, Complete dataset)
tree1 = classifier.fit(X_train,Y_train, df)
classifier.print_tree()

#classifier.print_tree_to_file("TreePrinted_v8.txt")

print("Predicting")
Y_pred = classifier.predict(X_test)

#### ------------------- sklearn.decisiontree -----------
# Create Decision Tree classifer object
clf = DTC(criterion='gini', splitter='best', max_depth=3, min_samples_split=6)

# Train Decision Tree Classifer
clf = clf.fit(X_train,Y_train)

#Predict the response for test dataset
SK_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, SK_pred))

from sklearn import tree
tree.plot_tree(clf, feature_names=column_names[0:-1], )

from sklearn import tree
import matplotlib.pyplot as plt

resolution = (18, 14)  # Adjust the resolution as needed
image_filename = "SK_Treeprinted_v8.png"  # Specify the filename for the saved image

# Create a figure with custom resolution
fig, ax = plt.subplots(figsize=resolution)

# Plot the decision tree
tree.plot_tree(clf, feature_names=column_names[:-1], filled=True, rounded=True, ax=ax, fontsize=10, class_names=True)

# Save the figure
plt.savefig(image_filename, format='png', bbox_inches='tight')

# Show the plot (optional)
plt.show()

"""

