### ---------- 1. Libraries
import numpy as np
import pandas as pd
from memory_profiler import profile

### ---------- 2. Node and DecisonTree Classes

# Class node and its attributes created to define the nodes of the decision tree
class Node():
    def __init__(self, feature_name=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        """
                Initializing a decision tree node.

                Parameters:
                - feature_name: Feature used for branching.
                - left: Left branch of the node.
                - right: Right branch of the node.
                - threshold: Value from the feature used for branching.
                - info_gain: Information gain of labels explained through the split.
                - value: Label value of the leaf node.
        """
        # Attributes of the decision node
        self.feature_name = feature_name # Feature used for branching
        self.left = left # Left branch of node
        self.right = right # Right branch of node
        self.threshold = threshold # Value from feature used for branching
        self.info_gain = info_gain # Information of labels explained through split

        # Attribute of the leaf node
        self.value = value # Label value of leaf node

class DecisionTreeClassifier():
    def __init__(self, method="gini", min_samples_split=2, tree_levels=2):
        """
        Initializing a DecisionTreeClassifier.

        Parameters:
        - method: The method used for splitting nodes (default: "gini").
        - min_samples_split: The minimum number of samples required for a split to be made (default: 2).
        - tree_levels: The maximum levels of branching allowed before termination (default: 2).
        """
        # initializing the root of the tree when class is called
        self.method = method
        self.root = None
        self.names = None
        self.categorical = None

        # Classification Hyperparameters
        # minimum no. of samples required for a split to be made
        self.min_samples_split = min_samples_split
        # maximum levels of branching allowed before termination
        self.tree_levels = tree_levels

    def fit(self, data, labels, datafull):
        """
        Fits the DecisionTreeClassifier to the given data.

        Parameters:
        - data: The dataset without label column and column names
        - labels: Labels of the dataset.
        - datafull: Full dataset including features and labels. Used here to extract
                    column names.

        Returns:
        - The root node of the trained decision tree.
        """
        # Set column names for reference
        self.names = datafull.columns
        # Combine features and labels into a single dataset
        dataset = np.concatenate((data, labels), axis=1)
        # Convert the combined dataset to a Pandas DataFrame for functions ahead
        dataset = pd.DataFrame(dataset)
        # Convert object-type columns to numeric or string based on content
        dataset = self.convert_data_types(dataset)
        # Check and store categorical information for features
        self.categorical = self.check_string_columns(dataset)
        # Convert DataFrame back to NumPy array
        dataset = dataset.values
        # Train the decision tree and set the root node
        self.root = self.train(dataset, 1)

        return self.root

    def convert_data_types(self, df):
        """
        Function to convert 'object' columns to 'numeric' or 'string' based on content.

        Parameters:
        - file_path: Path to the CSV file.
        Returns:
        - DataFrame with converted data types.
        """

        # Iterate over columns and convert 'object' columns based on content
        for column in df.columns:
            if pd.api.types.is_object_dtype(df[column]):
                try:
                    # Attempt to convert to numeric
                    df[column] = pd.to_numeric(df[column])
                except ValueError:
                    # If conversion to numeric fails, keep it as 'object' (string)
                    df[column] = df[column].astype(str)

        return df

    def check_string_columns(self, df):
        """
        Check if each column in the DataFrame is of type string.

        Parameters:
        - df: DataFrame.

        Returns:
        - Dictionary with column names as keys and a boolean indicating if the column is of type string as values.
        """
        string_columns = {}

        for column in df.columns:
            is_string = pd.api.types.is_string_dtype(df[column])
            string_columns[column] = is_string

        return string_columns

    def train(self, dataset, current_level=0):
        """
        Recursively creates the classification tree after the root node.

        Parameters:
        - dataset: The concatenated dataset for the current node.
        - current_level: The current depth level in the tree.

        Returns:
        - A Node representing either a decision or leaf node.
        """
        data, labels = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(data)

        # Split until stopping conditions of hyper parameters are met
        if num_samples >= self.min_samples_split and current_level <= self.tree_levels:
            # Find the best split
            best_split = self.best_split(dataset, num_samples, num_features)

            # Check if information gain is positive and non-zero (zero indicates purity)
            # Else, the subset doesn't need splitting and can form a leaf node
            # an empty best_split occurs when there is just 1 unique value in a subset
            if not best_split == {}:
                if best_split["info_gain"] > 0:
                    # Recursively create the left branches of the root node left subset
                    left_subtree = self.train(best_split["dataset_left"], current_level + 1)
                    # Recursively create the right branches of the root node right subset
                    right_subtree = self.train(best_split["dataset_right"], current_level + 1)
                    # Join the left and right branches to the Decision node
                    # along with some additional parameters
                    return Node(
                        feature_name=best_split["feature_name"],
                        threshold=best_split["threshold"],
                        left=left_subtree,
                        right=right_subtree,
                        info_gain=best_split["info_gain"]
                    )

        # Compute the value of the leaf node (most frequent label in the subset)
        leaf_value = max(labels, key=labels.tolist().count)
        # Return leaf node
        return Node(value=leaf_value)

    def best_split(self, dataset, num_samples, num_features):
        """
        Finds the best split for the current dataset based on information gain.

        Parameters:
        - dataset: The dataset for which to find the best split.
        - num_samples: The number of samples in the dataset.
        - num_features: The number of features in the dataset.

        Returns:
        - A dictionary containing details of the best split.
        """
        # Dictionary to store details of the best threshold-based split
        # Attributes will be passed to self attributes to form a node
        best_split = {}
        max_info_gain = -float("inf")

        # Loop over all features
        for feature_name in range(num_features):
            feature_values = dataset[:, feature_name]
            possible_thresholds = np.unique(feature_values)

            # Loop over all feature values present in the data
            for threshold in possible_thresholds:
                # Get the current split
                dataset_left, dataset_right = self.split(dataset, feature_name, threshold)

                # Check if child datasets are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]

                    # Compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, self.method)

                    # Update the best split in each iteration of threshold if needed
                    if curr_info_gain > max_info_gain:
                        best_split["feature_name"] = feature_name
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        # Return the best split attributes

        return best_split

    def split(self, dataset, feature_name, threshold):
        """
        Splits the dataset based on a threshold for a given feature.
        Checks to see if the data is categorical or numeric
        Categorical data is used to make splits for each category

        Parameters:
        - dataset: The dataset to be split, subset of the original data.
        - feature_name: The index of the feature used for splitting.
        - threshold: The threshold value for the split.

        Returns:
        - Two datasets representing the left and right branches of the split.
        """
        feature_values = dataset[:, feature_name]

        if self.categorical[feature_name]:
            # Use boolean indexing to filter rows efficiently
            dataset_left = dataset[feature_values == threshold]
            dataset_right = dataset[feature_values != threshold]
        else:
            # Use boolean indexing for numeric data as well
            dataset_left = dataset[feature_values <= threshold]
            dataset_right = dataset[feature_values > threshold]

        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        """
        Compute information gain based on the provided mode.

        Parameters:
        - parent: The parent dataset.
        - l_child: The left child dataset.
        - r_child: The right child dataset.
        - mode: The mode for information gain calculation ("entropy" or "gini").

        Returns:
        - The calculated information gain.
        """
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)

        if mode == "gini":
            gini_parent = self.gini_index(parent)
            gain = gini_parent - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        else:
            entropy_parent = self.entropy(parent)
            gain = entropy_parent - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))

        return gain

    def entropy(self, y):
        """
        Compute entropy for a given dataset.

        Parameters:
        - y: The dataset.

        Returns:
        - The calculated entropy.
        """
        unique_values, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        entropy = -(probabilities * np.log2(probabilities)).sum()
        return entropy

    def gini_index(self, y):
        """
        Compute Gini index for a given dataset.

        Parameters:
        - y: The dataset.

        Returns:
        - The calculated Gini index.
        """
        unique_values, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)

        return gini

    def print_tree(self, tree=None, indent=" "):
        """
        Print the decision tree structure.

        Parameters:
        - tree: The tree to be printed. If not provided, the root of the tree is used.
        - indent: The indentation string.

        Returns:
        - None
        """
        if not tree:
            tree = self.root

        if tree.value is not None:  # Check for a leaf node
            print(tree.value)
        else:
            print(f"node_{self.names[tree.feature_name]} <= {tree.threshold} | Gain {round(tree.info_gain, 3)}")
            print(f"{indent}Left:", end="")
            self.print_tree(tree.left, indent + indent)  # Recursively print left
            print(f"{indent}Right:", end="")
            self.print_tree(tree.right, indent + indent)  # Recursively print right

    def save_tree_to_variable(self, tree=None, indent=""):
        """
        Save the decision tree structure to a variable.

        Parameters:
        - tree: The tree to be saved. If not provided, the root of the tree is used.
        - indent: The indentation string.

        Returns:
        - A string containing the decision tree structure.
        """
        result = ""
        if not tree:
            tree = self.root

        if tree.value is not None:  # Check for a leaf node
            result += str(tree.value) + '\n'
        else:
            result += f"{indent}node_{self.names[tree.feature_name]} <= {tree.threshold} | Gain {round(tree.info_gain, 3)}\n"
            result += f"{indent}Left:"
            result += self.save_tree_to_variable(tree.left, indent + "  ")  # Recursively save left
            result += f"{indent}Right:"
            result += self.save_tree_to_variable(tree.right, indent + "  ")  # Recursively save right

        return result

    def print_tree_to_file(self, filename):
        """
        Print the decision tree structure to a text file.

        Parameters:
        - filename: The name of the text file to write the output.

        Returns:
        - None
        """
        tree_structure = self.save_tree_to_variable()
        with open(filename, 'w') as file:
            file.write(tree_structure)

    def predict(self, X):
        """
        Predict the output for a new dataset.

        Parameters:
        - X: The new dataset.

        Returns:
        - A list of predictions for each data point in the dataset.
        """
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions

    def make_prediction(self, x, tree):
        """
        Make a prediction for a single data point.

        Parameters:
        - x: The data point to be predicted.
        - tree: The decision tree.

        Returns:
        - The predicted value.
        """
        if tree.value is not None:
            return tree.value

        feature_val = x[tree.feature_name]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)




