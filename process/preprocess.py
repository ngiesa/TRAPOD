import json
import pandas as pd
import numpy as np
import random
from data.variable_manager import var_conf
from utils import nan_percentile, non_nan_count, haar_coefficients, hjorth_parameters
from numpy import inf

NON_MISSING_VARIABLES = [x for x in var_conf.keys() if var_conf[x]["required"] == True]
TIME_VARIANT_VARIABLES = [x for x in var_conf.keys() if var_conf[x]["time_variant"] == True]
TIME_INVARIANT_VARIABLES = [x for x in var_conf.keys() if var_conf[x]["time_variant"] == False]
BIN_VARIABLES = [x for x in var_conf.keys() if var_conf[x]["binary"] == True]
VITAL_SIGNS = [x for x in var_conf.keys() if ("vital_" in var_conf[x]["path"]) or ("vital_" in var_conf[x]["path"][0])]

# length reduction of sequences holding time series data, drop out sequences that are shorter than length
def apply_length_reduction(sequences, length, remove="post", drop=True):
    if remove == "post":
        if drop:
            return [(x[0].iloc[:length], x[1], x[2]) for x in sequences if len(x[0]) >= length]
        else:
            return [(x[0].iloc[:length], x[1], x[2]) for x in sequences]
    if remove == "pre":
        if drop:
            return [(x[0].iloc[-length:], x[1], x[2]) for x in sequences if len(x[0]) >= length]
        else:
            return [(x[0].iloc[-length:], x[1], x[2]) for x in sequences]
    return sequences

# imputation and standardization of data calucaling mean, std, min, max for all parameters
def get_stat_values(train_sequences):
    # append data values from train set
    flat_data_dfs = []
    for _, train_sequence in enumerate(train_sequences):
            flat_data_dfs.append(train_sequence[0])
    df = pd.concat(flat_data_dfs)
    return df.describe().to_dict()

def calculate_values_per_hour(df):
    df = df.sort_values(["c_start_ts"])
    df["c_value"] = pd.to_numeric(df.c_value, errors="coerce")
    df["df_hour_timestamps"] = pd.to_datetime(
        [x[:-10] for x in df["c_start_ts"]], format="%Y-%m-%d %H"
    )
    df = df.groupby(["df_hour_timestamps"])["c_start_ts"].count().reset_index()
    return list(df["c_start_ts"])

def build_tuple_seq(df, gr_col, label_col):
    drop_columns = [gr_col, label_col]
    if "level_0" in df.columns:
        drop_columns = drop_columns + ["level_0"]
    sequences = []
    for id, group in df.groupby(gr_col):
        feature_space = group.drop(
            columns=drop_columns, axis=1
        )
        label = int(group.iloc[0][label_col])
        sequences.append((feature_space, label, id))
    return sequences

def get_values_for_period(df_value, df_master):
    # focus on intraoperative phase TL2
    df = df_master.merge(
        df_value[["c_pat_id", "c_case_id", "c_value", "c_start_ts"]],
        on=["c_pat_id", "c_case_id"],
    )
    df = df[(df.c_start_ts > df.c_an_start_ts) & (df.c_start_ts < df.c_an_end_ts)]
    return df

def apply_ffill_imputation(df, gr_col, cols):
    # forwardfill the values
    df.loc[:,cols] = df.groupby(gr_col)[cols].ffill()
    return df

def apply_avg_imputation(df, st_dict, cols, imputation_type = "mean"):
    # apply median imputation
    df.loc[:, cols] = df.loc[:, cols].fillna({k: st_dict[k][imputation_type] for k in st_dict})
    return df

def apply_zero_imputation(df, cols):
    # apply zero imputation
    df.loc[:, cols] = df.loc[:, cols].fillna(0)
    return df

def apply_std(df, st_dict, binary_vars, non_bin_vars):
    # apply standardization
    means = [st_dict[k]["mean"] for k in st_dict if k in non_bin_vars]
    stds = [st_dict[k]["std"] for k in st_dict if k in non_bin_vars]
    d =((df[non_bin_vars] - means) / stds)
    df.loc[:, non_bin_vars] = d
    df.loc[:, binary_vars] = df[binary_vars].replace(0, -1)
    df[df == -inf] = 0
    return df.fillna(0)

def apply_time_series_imputation(df, mean_dict, non_bin_vars, zero_imp_vars, imputation_type = "mean"):
    df = apply_ffill_imputation(df=df, gr_col = "index", cols = non_bin_vars)
    apply_avg_imputation(df = df, st_dict = mean_dict, cols = non_bin_vars, imputation_type = imputation_type)
    df = apply_zero_imputation(df=df, cols=zero_imp_vars).fillna(0)
    return df

def get_median_imp_vars(median_impute = True, var_conf = var_conf):
    return [k for k in var_conf.keys() 
            if var_conf[k]['median_impute'] == median_impute]

def get_binary_vars(var_conf = var_conf):
    return [k for k in var_conf.keys() 
            if var_conf[k]['binary'] == True]

def get_miss_ind_vars(train_sequences):
    return [v for v in list(train_sequences[0][0].columns) if "_miss_ind" in v]

def yield_preprocessing_status(index, len_seq, message = "preprocessing"):
    if index%100 == 0:
        print(message, index , " from ", len_seq, " ", np.round(index/len_seq, 3))

def flat_tuple_seq(sequences_tuples):
    dfs = []
    for index in range(len(sequences_tuples)):
        dfs.append(sequences_tuples[index][0].assign(c_target = sequences_tuples[index][1]).assign(index = sequences_tuples[index][2]))
    return pd.concat(dfs)

def transpose_time_variant_variables(cols, static_vars, df_train, df_test, df_train_stats, df_test_stats):
    # transpose per group feeding all time stamps in model
    df_map = {
        "train": [df_train, df_train_stats],
        "test": [df_test, df_test_stats]
    }
    res = []
    for df_type in ["train", "test"]:
        dfs = []
        for var in cols:
            if (var in TIME_VARIANT_VARIABLES):
                df = df_map[df_type][0].assign(time_index = df_map[df_type][0].groupby(['index']).cumcount())
                dfs.append(df.pivot(index='index', columns='time_index', values=var).add_prefix(var + "_"))
        df_conc = pd.concat(dfs, axis=1).reset_index()
        res.append(df_conc.merge(df_map[df_type][1][static_vars + ["index"]], on="index"))
    return res[0], res[1]

def remove_one_value_columns(df = None):
    # removes columns with just one single value in all rows
    col_removed = [col for col in list(df.columns) 
                    if (len(df[col].value_counts().reset_index()) <= 1)
                    and (not "hst" in col) 
                    and (not "curr" in col) 
                    and (not "durat" in col) 
                    and (not "target" in col)
                    and (not "anes" in col)]
    return df.drop(columns=col_removed), col_removed

def iterate_columns(col):
    return [haar_coefficients(x) for x in col]


def calculate_time_series_transform(df = None,  group_col = "index"): 
    # calculating haar and horjth parameters for high frequency vital signs
    vital_columns = [x for x in VITAL_SIGNS if (x in list(df.columns))]
    df.update(df.groupby(group_col).ffill().fillna(df.mean()))
    res_haar = []
    for feature in vital_columns:
        print("preprocess time feat for ", feature)
        group_haar = df.groupby("index", axis=0)[feature].apply(haar_coefficients).reset_index()
        group_haar = group_haar[feature].apply(lambda x: pd.Series(str(x).replace("]", "").replace("[", "").split(","))).loc[:,1:4].astype(float).assign(index = group_haar["index"])
        group_haar.columns = [c if str(c)=="index" else "haar_coef_{}_{}".format(str(c), feature) for c in group_haar.columns]
        res_haar.append(group_haar)
        group_haar = df.groupby("index", axis=0)[feature].apply(hjorth_parameters).reset_index()
        group_haar = group_haar[feature].apply(lambda x: pd.Series(str(x).replace("]", "").replace("[", "").split(","))).loc[:,1:4].astype(float).assign(index = group_haar["index"])
        group_haar.columns = [c if str(c)=="index" else "hjorth_coef_{}_{}".format(str(c), feature) for c in group_haar.columns]
        res_haar.append(group_haar)
    res = pd.concat(res_haar, axis=1)
    res = res.loc[:,~res.columns.duplicated()].copy()
    return res

def remove_time_features(sequences):
    columns_names = [f for f in list(sequences[0][0].columns) if ("haar" in f) | ("hjorth" in f)]
    return [(s[0].drop(columns=columns_names), s[1], s[2]) for s in sequences]

def apply_summary_statistic_preprocessing(train_sequences, test_sequences, max_seq_len = None, with_transpose=True, with_time_feat = True, remove_one_col=True):
    print("start imputation and standardization static variables")
    cols = list(train_sequences[0][0].columns) + ["c_target"]
    df_train = flat_tuple_seq(train_sequences)
    df_test = flat_tuple_seq(test_sequences)
    # calculation of summary statistics per op
    statistic_dict, bin_vars, static_vars = {}, [], []
    for var in cols:
        if (var in TIME_VARIANT_VARIABLES) and (not "durat" in var) and (not "curr_in_class" in var):
            statistic_dict[var] = [nan_percentile(10), nan_percentile(50), nan_percentile(90)]
        else:
            if ("durat" in var) or ("curr_in_class" in var):
                statistic_dict[var] = [non_nan_count()] 
            elif (("miss_" in var) | (var in BIN_VARIABLES)):
                bin_vars.append(var)
            elif ("_cumsum" in var) | ("hst_" in var) | ("prev_" in var) \
                | ("curr_" in var) | ("anes_" in var) \
                | ("miss_" in var) | ("c_target" in var):
                statistic_dict[var] = [np.nanmax]
            else:
                statistic_dict[var] = [nan_percentile(50)]
            static_vars.append(var)
    # apply aggregates on time series data 
    df_train_stats = df_train.groupby("index").agg(statistic_dict)
    df_train_stats.columns = ['_'.join(col).strip() for col in df_train_stats.columns.values]
    df_test_stats = df_test.groupby("index").agg(statistic_dict)
    df_test_stats.columns = ['_'.join(col).strip() for col in df_test_stats.columns.values]
    # apply haar transformed features
    df_train_transposed, df_test_transposed = None, None
    if max_seq_len:
        if with_time_feat:
            print("Process time series features")
            time_series_transf_train = calculate_time_series_transform(df = df_train)
            df_train_stats = df_train_stats\
                .merge(time_series_transf_train, on=["index"], how="inner")
            time_series_transf_test = calculate_time_series_transform(df = df_test)
            df_test_stats = df_test_stats\
                .merge(time_series_transf_test, on=["index"], how="inner")
        # apply time transposing
        static_vars_stats = [v for v in list(df_test_stats.columns) if any([k in v for k in static_vars])]
        if with_transpose:
            df_train_transposed, df_test_transposed = transpose_time_variant_variables(df_train=df_train, df_test=df_test, 
                                                                                df_train_stats=df_train_stats, df_test_stats=df_test_stats,
                                                                                cols=cols, static_vars=static_vars_stats)


    seq_return = []
    for set in [[df_train_stats, df_test_stats], [df_train_transposed, df_test_transposed]]:
        df_train, df_test = set[0], set[1]
        if (df_train is None) or (df_test is None):
            seq_return.append((None, None))
            continue
        print("Len train set: ", len(df_train))
        print("Len test set: ", len(df_test))
        # apply median imputation for replacing NaN values
        st_dict = df_train.describe().to_dict()
        df_train = apply_avg_imputation(df=df_train, st_dict=st_dict, cols=list(df_train.columns))
        df_test = apply_avg_imputation(df=df_test, st_dict=st_dict, cols=list(df_train.columns))
        # apply standardization leave target [0, 1] as it is 
        bin_vars = [x for x in list(df_train.columns) if any([x.startswith(y) for y in bin_vars])]
        non_bin_vars = [x for x in list(df_train.columns) if not ((x in bin_vars) | ("c_target" in x))]
        df_train = apply_std(df=df_train, st_dict=st_dict, binary_vars=bin_vars, non_bin_vars=non_bin_vars).reset_index()
        df_test = apply_std(df=df_test, st_dict=st_dict, binary_vars=bin_vars, non_bin_vars=non_bin_vars).reset_index()
        # build sequences 
        if remove_one_col:
            df_train, col_removed =  remove_one_value_columns(df = df_train)
            df_test = df_test.drop(columns=col_removed)
        #build sequencies 
        seq_train = build_tuple_seq(df=df_train, gr_col="index", label_col="c_target_nanmax")
        seq_test = build_tuple_seq(df=df_test, gr_col="index", label_col="c_target_nanmax")
        # shuffle data
        random.shuffle(seq_train)
        random.shuffle(seq_test)
        # get the number of features
        #print(" # of features: ", len(df_test.columns))
        #print("feats ", list(df_test.columns))
        seq_return.append((seq_train, seq_test))
    # return list train set stats, test set stats, train set transposed, test set transposed
    return seq_return

def preprocess_static_data(train_sequences = None, test_sequences = None, train_static_df = None, test_static_df= None):
    cols = [x for x in list(train_static_df.columns) if x != "c_op_id"]
    st_dict = train_static_df.describe().to_dict()
    train_static_df = apply_avg_imputation(df=train_static_df, st_dict=st_dict, cols=cols)
    test_static_df = apply_avg_imputation(df=test_static_df, st_dict=st_dict, cols=cols)
    binary_vars = [x for x in cols if ("_miss_ind" in x) | ("anes_" in x) | ("curr_" in x)]
    non_bin_vars = [v for v in cols if not (v in binary_vars)]
    train_static_df = apply_std(df=train_static_df, st_dict=st_dict, binary_vars=binary_vars, non_bin_vars=non_bin_vars)
    test_static_df = apply_std(df=test_static_df, st_dict=st_dict, binary_vars=binary_vars, non_bin_vars=non_bin_vars)
    # combining static and sequence data in one single data structure
    train_sequences = [(x[0], x[1], x[2], 
                        train_static_df[train_static_df.c_op_id == x[2]].drop(columns=["c_op_id", "c_target"])) 
                        for x in train_sequences]
    test_sequences = [(x[0], x[1], x[2], 
                        test_static_df[test_static_df.c_op_id == x[2]].drop(columns=["c_op_id", "c_target"])) 
                        for x in test_sequences]
    return train_sequences, test_sequences
    
    
def apply_time_series_preprocessing(train_sequences, test_sequences, imputation_type="mean", iscombined=False, remove_one_col=True):
    print("start imputation and standardization of time series data")
    # reduce feature set to time variant features only
    cut_cols = [x for x in train_sequences[0][0].columns if (x not in TIME_INVARIANT_VARIABLES) \
                & (x not in [v + "_miss_ind" for v in TIME_INVARIANT_VARIABLES])] + ["index", "c_target"]
    # apply standardization calculated from train applied to train and test
    st_dict_train = get_stat_values(train_sequences)
    st_dict_test = get_stat_values(test_sequences)
    mean_dict_train = {}
    # select the median and zero variables from the variable configuration
    median_imp_vars = get_median_imp_vars(True)
    zero_imp_vars = get_median_imp_vars(False)
    # get miss indicator variables and add them to binary vars
    binary_vars = get_binary_vars() + get_miss_ind_vars(train_sequences=train_sequences)
    binary_vars = [var for var in binary_vars if var in train_sequences[0][0].columns]
    non_bin_vars = [v for v in train_sequences[0][0].columns if not (v in binary_vars)]
    # selecting the variables which are included in the prepared sequences
    median_imp_vars = [k for k in median_imp_vars if k in list(st_dict_train.keys())]
    zero_imp_vars = [k for k in zero_imp_vars if k in list(st_dict_train.keys())]
    for k in median_imp_vars:
        mean_dict_train[k]=st_dict_train[k]
    json.dump(st_dict_train, open("./stats/res/statistics_train.json", "w"))
    json.dump(st_dict_test, open("./stats/res/statistics_test.json", "w"))
    # apply for train sequences
    df_train = flat_tuple_seq(train_sequences)
    #df_train.to_csv("./data/df_train_data_debug_time_series_unstandard_29_01_23.csv")
    df_train = apply_time_series_imputation(df = df_train, mean_dict = mean_dict_train, non_bin_vars = non_bin_vars, zero_imp_vars=zero_imp_vars, imputation_type = imputation_type)
    df_train = apply_std(df=df_train, st_dict=st_dict_train, binary_vars=binary_vars, non_bin_vars=non_bin_vars)
    if iscombined:
        df_train = df_train[cut_cols]
    # apply for test sequences with train summary stats
    df_test = flat_tuple_seq(test_sequences)
    df_test = apply_time_series_imputation(df = df_test, mean_dict = mean_dict_train, non_bin_vars = non_bin_vars, zero_imp_vars=zero_imp_vars, imputation_type = imputation_type)
    df_test = apply_std(df=df_test, st_dict=st_dict_train, binary_vars=binary_vars, non_bin_vars=non_bin_vars)
    if iscombined:
        df_test = df_test[cut_cols]
    if remove_one_col:
        df_train, col_removed =  remove_one_value_columns(df = df_train)
        #df_train.to_csv("./data/df_train_data_time_series_debug_standard_29_01_23.csv")
        df_test = df_test.drop(columns=col_removed)
    # build tuples
    seq_test = build_tuple_seq(df=df_test, gr_col="index", label_col="c_target")
    seq_train = build_tuple_seq(df=df_train, gr_col="index", label_col="c_target")
    random.shuffle(seq_test)
    random.shuffle(seq_train)
    return seq_train, seq_test

# remove preoperative data if needed and reduce to time invariant variables
def reduce_time_invariant(df):
    # get time invariant columnms and reduce df to these cols 
    l = [col for col in df.columns if any([t.lower() in col.lower() for t in TIME_INVARIANT_VARIABLES])]
    return remove_nudesc(df[l])

def remove_nudesc(df):
    # get time invariant columnms and reduce df to these cols 
    l = [col for col in df.columns if not "nudesc" in col]
    return df[l]

# cut ranges in separate function 
def perform_cut_valid_ranges(df):
    for col in list(df.columns):
        if col in list(var_conf.keys()):
            if not var_conf[col]['valid_range']:
                continue
            # filter condition with valid ranges 
            df.loc[:, col] = [x if (x >= var_conf[col]['valid_range'][0]) & 
                (x <= var_conf[col]['valid_range'][1]) else float('NaN') for x in df[col]]
    return df

# perform data cleansing remove outliers and constrain on available features
def perform_cleansing(sequences = None, static_df = None, cut_valid_range = True, reduce_missing = True, max_seq_len = 0):
    print("start cleansing")
    df_seq = flat_tuple_seq(sequences)
    if cut_valid_range:
        perform_cut_valid_ranges(df_seq)
        if (static_df is not None) and (not static_df.empty):
            perform_cut_valid_ranges(static_df)
    if reduce_missing:
        # filter condition if at least one value in sequence for non-missing variables is set
        df = df_seq.drop('index', 1).isna().groupby(df_seq["index"]).sum().reset_index()
        df.loc[:,NON_MISSING_VARIABLES] = (df[NON_MISSING_VARIABLES] != max_seq_len)
        # discarding sequences that do not fullfill the non-missing constraint
        df = df.assign(include=df[NON_MISSING_VARIABLES].all(axis=1))
        df_seq = df_seq[df_seq["index"].isin(list(df[df["include"] == True]["index"]))]
    sequences = build_tuple_seq(df=df_seq, gr_col="index", label_col="c_target")
    return sequences, static_df