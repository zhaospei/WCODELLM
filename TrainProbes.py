import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from pathlib import Path

from sklearn.model_selection import train_test_split


def correct_str(str_arr):
    """
    Converts a string representation of a numpy array into a comma-separated string.

    Arguments:
    str_arr -- A string representation of a numpy array.

    Returns:
    val_to_ret -- A comma-separated string derived from 'str_arr'.

    Note:
    This function assumes that 'str_arr' is a string representation of a numpy array 
    with dtype=float32. It removes the array formatting as well as whitespace and 
    newlines, and it also replaces '],' with ']'.
    """
    val_to_ret = (str_arr.replace("[array(", "")
                        .replace("dtype=float32)]", "")
                        .replace("\n","")
                        .replace(" ","")
                        .replace("],","]")
                        .replace("[","")
                        .replace("]",""))
    return val_to_ret

def define_model(input_dim):
    """
    Defines and compiles a Sequential model in Keras.

    Arguments:
    input_dim -- The dimension of the input data (positive integer).

    Returns:
    model -- A compiled Sequential model.

    This function creates a Sequential model with three hidden layers, each followed 
    by a ReLU activation function. The output layer uses a sigmoid activation function.
    The model is compiled with the Adam optimizer, binary cross-entropy loss, and 
    accuracy as a metric.
    
    Raises:
    ValueError -- If input_dim is not a positive integer.
    """
    if not isinstance(input_dim, int) or input_dim <= 0:
        raise ValueError("Input dimension must be a positive integer.")
    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=input_dim))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_embeddings, train_labels):
    """
    Trains the input model on the provided embeddings and labels.
    
    Parameters:
    model (keras Model): The model to be trained.
    train_embeddings (numpy.ndarray): The embeddings used for training. Each embedding should correspond to a label in 'train_labels'.
    train_labels (Series or ndarray): The labels for each training embedding.

    Returns:
    model (keras Model): The trained model.

    Raises:
    ValueError: If the length of 'train_embeddings' and 'train_labels' does not match.
    """
    if len(train_embeddings) != len(train_labels):
        raise ValueError("Training embeddings and labels must have the same length.")
    
    model.fit(train_embeddings, train_labels, epochs=50, batch_size=32)
    return model

def evaluate_model(model, test_embeddings, test_labels):
    """
    Evaluates the performance of the trained model on the test data.

    Parameters:
    model (keras Model): The trained model to be evaluated.
    test_embeddings (numpy.ndarray): The embeddings used for testing. Each embedding should correspond to a label in 'test_labels'.
    test_labels (Series or ndarray): The labels for each test embedding.

    Returns:
    loss (float): The loss value calculated by the model on the test data.
    accuracy (float): The accuracy of the model on the test data, as a decimal.

    Raises:
    ValueError: If the length of 'test_embeddings' and 'test_labels' does not match.
    """
    if len(test_embeddings) != len(test_labels):
        raise ValueError("Test embeddings and labels must have the same length.")
    
    loss, accuracy = model.evaluate(test_embeddings, test_labels)
    return loss, accuracy


def compute_roc_curve(test_labels, test_pred_prob):
    """
    Computes the Receiver Operating Characteristic (ROC) curve and the area under the curve (AUC).

    Parameters:
    test_labels (Series or ndarray): The true labels for the test data.
    test_pred_prob (numpy.ndarray): The predicted probabilities for each data point in the test set.

    Returns:
    roc_auc (float): The area under the ROC curve, a single scalar value representing the model's overall performance.
    fpr (numpy.ndarray): The false positive rate at various decision thresholds.
    tpr (numpy.ndarray): The true positive rate at various decision thresholds.

    Note:
    This function assumes a binary classification task.
    """
    fpr, tpr, _ = roc_curve(test_labels, test_pred_prob)  # Assuming binary classification
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr

def find_optimal_threshold(X, y, model):
    """
    Finds the optimal threshold for a binary classification model by maximizing the accuracy score over the ROC curve thresholds.

    Parameters:
    X (numpy.ndarray): Input feature array.
    y (numpy.ndarray or list): True binary labels in range {0, 1} or {-1, 1}. If labels are not binary, pos_label should be explicitly given.
    model (keras.Model): The binary classification model.

    Returns:
    float: The optimal threshold value.

    Raises:
    ValueError: If the dimensions of X and y do not match.
    """
    # Predict probabilities for the data set
    y_pred_prob = model.predict(X)

    # Compute ROC curve to find the optimal threshold
    fpr_val, tpr_val, thresholds_val = roc_curve(y, y_pred_prob)
    optimal_threshold = thresholds_val[np.argmax([accuracy_score(y, y_pred_prob > thr) for thr in thresholds_val])]

    return optimal_threshold


def print_results(results, dataset_names, repeat_each, layer_num_from_end):
    """
    Prints the average accuracy, AUC, and optimal threshold for each dataset, and returns a list of these results.

    Parameters:
    results (list of tuples): Each tuple represents the results for a dataset and contains the dataset name, index, 
                              accuracy, AUC, optimal threshold, and test accuracy.
    dataset_names (list of str): The names of the datasets.
    repeat_each (int): The number of times each experiment is repeated.
    layer_num_from_end (int): The index of the layer from the end of the model.

    Returns:
    list of str: Each string contains the average results for a dataset.
    Raises:
    ValueError: If the length of the results list is not equal to the length of the dataset_names list multiplied by repeat_each.
    """
    if len(results) != len(dataset_names) * repeat_each:
        raise ValueError("Results array length should be equal to dataset_names length multiplied by repeat_each.")
    overall_res = []
    for ds in range(len(dataset_names)):
        relevant_results_portion = results[repeat_each*ds:repeat_each*(ds+1)]
        acc_list = [t[2] for t in relevant_results_portion]
        auc_list = [t[3] for t in relevant_results_portion]
        opt_thresh_list = [t[4] for t in relevant_results_portion]
        avg_acc = sum(acc_list) / len(acc_list)
        avg_auc = sum(auc_list) / len(auc_list)
        avg_thrsh = sum(opt_thresh_list) / len(opt_thresh_list)
        text_res = ("dataset: " + str(dataset_names[ds]) + " layer_num_from_end:" 
                    + str(layer_num_from_end) + " Avg_acc:" + str(avg_acc) 
                    + " Avg_AUC:" + str(avg_auc) + " Avg_threshold:" 
                    + str(avg_thrsh))
        print(text_res)
        overall_res.append(text_res)

    return overall_res

# df = pd.read_parquet('benchmark/MBPP/data/mbpp_data_ramdom_first_last_classify_dataset.parquet')
# # df = pd.read_pickle('benchmark/MBPP/data/mbpp_data_random_token_classify_dataset.pkl')

# # df_label_0 = df[df['label'] == 0]
# # df_label_1 = df[df['label'] == 1]
# # print("Number of label 0 samples:", len(df_label_0))
# # print("Number of label 1 samples:", len(df_label_1))

# # # Step 2: Split each label group separately with the same 9:1 ratio for task_id
# # train_task_ids_0, test_task_ids_0 = train_test_split(
# #     df_label_0['task_id'].unique(), test_size=0.1, random_state=42
# # )
# # train_task_ids_1, test_task_ids_1 = train_test_split(
# #     df_label_1['task_id'].unique(), test_size=0.1, random_state=42
# # )

# # # Step 3: Get the training and test sets by combining results for each label group
# # train_dataset = df[df['task_id'].isin(train_task_ids_0) | df['task_id'].isin(train_task_ids_1)]
# # test_dataset = df[df['task_id'].isin(test_task_ids_0) | df['task_id'].isin(test_task_ids_1)]

# # # Optional: Check the label distributions to confirm they're similar
# # print("Train T/F distribution:\n", train_dataset['label'].value_counts(normalize=True))
# # print("Test T/F distribution:\n", test_dataset['label'].value_counts(normalize=True))

# unique_task_ids = df['task_id'].unique()

# # Step 2: Shuffle and split the task_ids
# np.random.shuffle(unique_task_ids)
# split_index = int(0.9 * len(unique_task_ids))

# # Split task_ids into two groups for 9:1 ratio
# task_ids_90 = unique_task_ids[:split_index]
# task_ids_10 = unique_task_ids[split_index:]

# Step 3: Create sub-dataframes by filtering based on task_id
# train_dataset = df[df['task_id'].isin(task_ids_90)].reset_index(drop=True)
# test_dataset = df[df['task_id'].isin(task_ids_10)].reset_index(drop=True)

train_dataset = pd.read_parquet('benchmark/MBPP/data/mbpp_data_ramdom_first_last_classify_train_dataset.parquet')
test_dataset = pd.read_parquet('benchmark/MBPP/data/mbpp_data_ramdom_first_last_classify_test_dataset.parquet')

# df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle rows randomly

# # Step 2: Determine split index for 9:1 ratio
# split_index = int(0.9 * len(df))

# # Step 3: Split into two sub-DataFrames
# train_dataset = df.iloc[:split_index].reset_index(drop=True)  # First 90%
# test_dataset = df.iloc[split_index:].reset_index(drop=True)  # Last 10%

# print("Train labels:", train_dataset['label'].value_counts())
# print("Test labels:", test_dataset['label'].value_counts())

# train_dataset.to_parquet('benchmark/MBPP/data/mbpp_data_ramdom_first_last_classify_train_dataset.parquet')
# test_dataset.to_parquet('benchmark/MBPP/data/mbpp_data_ramdom_first_last_classify_test_dataset.parquet')

train_embeddings = np.array([embedding for embedding in train_dataset['input_token_middle_layer'].tolist()])
train_labels = train_dataset['label'].astype(int)
test_embeddings = np.array([embedding for embedding in test_dataset['input_token_middle_layer'].tolist()])
test_labels = test_dataset['label'].astype(int)

print(train_embeddings[0], train_labels[0])

# Repeat training and testing for specified number of times
best_accuracy = 0
min_accuracy = 1
best_model = None
all_probs_list = []

repeat_each = 5

results = []
for i in range(repeat_each):

    # Define the model
    model = define_model(train_embeddings.shape[1])

    # Train the model on full training data
    model = train_model(model, train_embeddings, train_labels)

    # Find the optimal threshold and compute validation set accuracy
    optimal_threshold = find_optimal_threshold(train_embeddings, train_labels, model)

    # Evaluate the model
    loss, accuracy = evaluate_model(model, test_embeddings, test_labels)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
    
    if accuracy < min_accuracy:
        min_accuracy = accuracy

    test_pred_prob = model.predict(test_embeddings)
    all_probs_list.append(deepcopy(test_pred_prob)) #Store probabilities

    # Compute ROC curve and ROC area
    roc_auc, fpr, tpr = compute_roc_curve(test_labels, model.predict(test_embeddings))

    # Compute test set accuracy using the optimal threshold
    test_accuracy = accuracy_score(test_labels, model.predict(test_embeddings) > optimal_threshold)

    results.append((i, accuracy, roc_auc, optimal_threshold, test_accuracy))

print("Results:", results)
print(f"[{min_accuracy}, {best_accuracy}]")