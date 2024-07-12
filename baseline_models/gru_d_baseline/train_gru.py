import pandas as pd
import numpy as np
import pickle
import torch.utils.data as utils
import torch


def open_pickle_data(pickle_path = ""):
    # open sequences
    with open(pickle_path, "rb") as f:
        return pickle.load(f)
train_sequences = open_pickle_data('/home/giesan/rnn_delir/data/sequences/sequences_sampling_3_train.pkl')
test_sequences = open_pickle_data('/home/giesan/rnn_delir/data/sequences/sequences_sampling_3_test.pkl')

def flat_tuple_seq(sequences_tuples):
    dfs = []
    for index in range(len(sequences_tuples)):
        dfs.append(sequences_tuples[index][0].assign(c_target = sequences_tuples[index][1]).assign(index = sequences_tuples[index][2]))
    return pd.concat(dfs)

def build_tuple_seq(df, gr_col, label_col):
    drop_columns = [gr_col, label_col]
    if "level_0" in df.columns:
        drop_columns = drop_columns + ["level_0"]
    sequences = []
    labels = []
    for id, group in df.groupby(gr_col):
        feature_space = group.drop(
            columns=drop_columns, axis=1
        )
        label = int(group.iloc[0][label_col])
        labels.append(label)
        sequences.append(feature_space.values.tolist()) #TODO
    return np.array(sequences), np.array(labels)


flatten_train_sequences = flat_tuple_seq(train_sequences).groupby('index').head(10)
flatten_test_sequences = flat_tuple_seq(test_sequences).groupby('index').head(10)
# remove miss indicator columns
miss_cols = [c for c in flatten_train_sequences.columns if "_miss" in c]
flatten_train_sequences = flatten_train_sequences.drop(columns=miss_cols)
flatten_test_sequences = flatten_test_sequences.drop(columns=miss_cols)
# count sequence lengths and omit non matching sequences
seq_counts = flatten_train_sequences.fillna(0).groupby("index")["nibp_map"].count().reset_index()
flatten_train_sequences = flatten_train_sequences.merge(seq_counts[seq_counts.nibp_map == 10][["index"]], on="index")

seq_counts = flatten_test_sequences.fillna(0).groupby("index")["nibp_map"].count().reset_index()
flatten_test_sequences = flatten_test_sequences.merge(seq_counts[seq_counts.nibp_map == 10][["index"]], on="index")
#train_mask = flatten_train_sequences.isna().astype(int)
test_mask = flatten_test_sequences.isna().astype(int)\
    .assign(index=flatten_test_sequences["index"])\
    .assign(c_target=flatten_test_sequences["c_target"])
train_mask = flatten_train_sequences.isna().astype(int)\
    .assign(index=flatten_train_sequences["index"])\
    .assign(c_target=flatten_train_sequences["c_target"])
# target dimensions are number of samples or sequences, sequences length, features (989, 10, 323)
def create_deltas_from_mask(mask, sampling, sequences):
    delta = (mask.groupby(["c_target", "index"]).cumsum() * sampling)
    delta = delta / delta.max()
    delta = delta\
        .assign(index=sequences["index"])\
        .assign(c_target=sequences["c_target"])
    return delta.fillna(0)/10 #TODO increase size
    
test_delta = create_deltas_from_mask(test_mask, 3, flatten_test_sequences)
train_delta = create_deltas_from_mask(train_mask, 3, flatten_train_sequences)
# retrieve the corresponding last observed value within one sequence
X_last_observed_test = flatten_test_sequences.groupby(["c_target", "index"]).ffill()\
    .assign(index=flatten_test_sequences["index"])\
    .assign(c_target=flatten_test_sequences["c_target"])
    
X_last_observed_train = flatten_train_sequences.groupby(["c_target", "index"]).ffill()\
    .assign(index=flatten_train_sequences["index"])\
    .assign(c_target=flatten_train_sequences["c_target"])
# standardize and fill nas
# imputing last observed and sequences
feature_cols = flatten_test_sequences.drop(columns=["index", "c_target"]).columns
last_X_mean = X_last_observed_train[feature_cols].mean()
last_X_min = X_last_observed_train[feature_cols].min()
last_X_max = X_last_observed_train[feature_cols].max()

# sequence train means are used to describe convergence of clinical variables 
seq_X_mean = flatten_train_sequences[feature_cols].mean() 
seq_X_min = flatten_train_sequences[feature_cols].min()
seq_X_max = flatten_train_sequences[feature_cols].max()


last_X_std = X_last_observed_train[feature_cols].std()
seq_X_std = flatten_train_sequences[feature_cols].std()

STANDARDIZATION = "MIN"
print("features ")
print(list(feature_cols))


if STANDARDIZATION == "Z":

    X_last_observed_train.loc[:, list(feature_cols)] = ((X_last_observed_train.loc[:, list(feature_cols)] - last_X_mean) / last_X_std).fillna(0)
    X_last_observed_test.loc[:, list(feature_cols)] = ((X_last_observed_test.loc[:, list(feature_cols)] - last_X_mean) / last_X_std).fillna(0)

    flatten_train_sequences.loc[:, list(feature_cols)] = ((flatten_train_sequences.loc[:, list(feature_cols)] - seq_X_mean) / seq_X_std).fillna(0)
    flatten_test_sequences.loc[:, list(feature_cols)] = ((flatten_test_sequences.loc[:, list(feature_cols)] - seq_X_mean) / seq_X_std).fillna(0)
    
if STANDARDIZATION == "MIN": #TODO
    
    X_last_observed_train.loc[:, list(feature_cols)] = ((X_last_observed_train.loc[:, list(feature_cols)].fillna(last_X_mean) - last_X_min) / (last_X_max - last_X_min)).fillna(0)
    X_last_observed_test.loc[:, list(feature_cols)] = ((X_last_observed_test.loc[:, list(feature_cols)].fillna(last_X_mean)- last_X_min) / (last_X_max - last_X_min)).fillna(0)

    flatten_train_sequences.loc[:, list(feature_cols)] = ((flatten_train_sequences.loc[:, list(feature_cols)].fillna(seq_X_mean)- seq_X_min) / (seq_X_max - seq_X_min)).fillna(0)
    flatten_test_sequences.loc[:, list(feature_cols)] = ((flatten_test_sequences.loc[:, list(feature_cols)].fillna(seq_X_mean)- seq_X_min) / (seq_X_max - seq_X_min)).fillna(0)
    
def concat_data(x_last, flatten_seq, mask, delta):
    X_last, labels = build_tuple_seq(df=x_last, gr_col="index", label_col="c_target")
    X, _ = build_tuple_seq(df=flatten_seq, gr_col="index", label_col="c_target")
    X_mask, _ = build_tuple_seq(df=mask, gr_col="index", label_col="c_target")
    X_delta, _ = build_tuple_seq(df=delta, gr_col="index", label_col="c_target")
    return np.concatenate((
        np.expand_dims(X, axis=1), 
        np.expand_dims(X_last, axis=1), 
        np.expand_dims(X_mask, axis=1), 
        np.expand_dims(X_delta, axis=1)),
        axis=1), labels

dataset_concat_train, train_labels = concat_data(X_last_observed_train, flatten_train_sequences, train_mask, train_delta)
dataset_concat_test, test_labels = concat_data(X_last_observed_test, flatten_test_sequences, test_mask, test_delta)
train_seq_tuple = build_tuple_seq(df=flatten_train_sequences, gr_col="index", label_col="c_target")
X_mean = np.mean(np. expand_dims(train_seq_tuple[0], axis=1), axis=0)
train_data, train_label = torch.Tensor(dataset_concat_train), torch.Tensor(train_labels)
valid_data, valid_label = torch.Tensor(dataset_concat_test), torch.Tensor(test_labels)
test_data, test_label = torch.Tensor(dataset_concat_test), torch.Tensor(test_labels)
BATCH_SIZE = 64

train_dataset = utils.TensorDataset(train_data, train_label)
valid_dataset = utils.TensorDataset(valid_data, valid_label)
test_dataset = utils.TensorDataset(test_data, test_label)

train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


from GRUD import *
# number of features determines the input and hidden dimensions
numb_feats = len(flatten_test_sequences.drop(columns=["c_target", "index"]).columns)
print("number of features: ", numb_feats)
input_dim = numb_feats
hidden_dim = numb_feats
# one dimensional output as labels
output_dim = numb_feats
# defining the GRUD model


grud = GRUD(input_dim, hidden_dim, output_dim, X_mean, output_last=True)
from main import Train_Model, Test_Model
best_grud, losses_grud = Train_Model(model = grud, 
                                    train_dataloader=train_dataloader, 
                                    valid_dataloader=valid_dataloader,
                                    num_epochs=1, patience=10, min_delta=0.00001,
                                    batch_size=64, learning_rate = 0.0001)


test_results = Test_Model(best_grud, test_dataloader)












