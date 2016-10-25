# coding=utf-8
import sys
import time

sys.path += ['./']

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import p_g_error_detection as pg

# import sklearn
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
# from sklearn import tree   ### to print the tree i think...
"""
the Imputer function is for the missing values;
currently randomforest in sklearn does not support/handle missing values well yet, so u'll need the
Imputer functions to kind of replace the missing data with mean, mode, etc
"""

import kh_config as config

# import p_g_str_col_analysis as pg_str

# set global random seed to get reproducible results
np.random.seed(42)

# load error detection data handler
# col_names_filename = config.header_file
data_filename = config.data_file
columns_description_filename = config.column_description_file

ed = pg.ErrorDetection(data_filename, columns_description_filename)

"""
REMINDER!!!
self.df = self.df_complete[self.used_cols]
"""

###############################################################################
# STRING METRIC ANALYSIS
###############################################################################

# strAnalysis = pg_str.analyzeTable(ed.df, ['Maintenance status'])
# print(strAnalysis[:10])

###############################################################################
# GLOBAL HISTOGRAM ANALYSIS
###############################################################################
#
# if len(ed.df) > 10000:
#     # if there are less than < numCat >
#     numDiffCat = 50
#     # then a category is considered rare if it occurs less than or equal to <
#     numRareCat = 10
# else:
#     # if there are less than < numCat >
#     numDiffCat = 50
#     # then a category is considered rare if it occurs less than or equal to <
#     numRareCat = 5
#
# print('Global Histogram Analysis...')
# sys.stdout.flush()
#
# # provides the indices to be kept per column
# colIndex = {}
#
# # provides the values to be removed per column
# colDroppedValues = {}
#
# # initialize list of global histogram errors
# globalHistogramErrors = []  # (material, plant, )
#
# # analyze every column independently
# for col in ed.used_cols:
#     # print 'global analysis here! col: ', col
#
#     # skip special columns
#     if col in ['Material', 'Plant Description', 'Description', 'Follow-up matl', 'Maintenance status']:
#         continue
#
#     # analyze numeric columns
#     if ed.colTypes[col] in ['INT', 'FLOAT']:
#
#         values = ed.df[col]
#         # if values.max() > 10: # if the numerical values are larger than 10, round them...
#         #     values = round(values)
#         values_histogram_map, _ = ed.getValueCounter(values)
#         values_histogram, values_total = ed.getValueCounterList(values)
#
#         # if the total number of different values is not too high
#
#         if len(values.unique()) <= numDiffCat:
#             dropped_values = []
#             for vk in values_histogram:
#                 if vk[0] <= numRareCat:
#                     dropped_values += [(0, vk[1])]
#         else:
#             non_zero_values = values[values != 0]
#             q25 = np.percentile(non_zero_values, 25)
#             q75 = np.percentile(non_zero_values, 75)
#             iqr = q75 - q25
#             ub = q75 + 2.2 * iqr
#
#             dropped_values = []
#             dropped_values_without_score = values[values > ub]
#             for dropped_value in dropped_values_without_score:
#                 # if the value to be dropped occurs more than < numRareCat> times, don't drop it!
#                 if values_histogram_map[dropped_value] < numRareCat:
#                     score = ub / dropped_value
#                     dropped_values += [(score, dropped_value)]
#
#         dropped_values.sort()
#         colDroppedValues[col] = dropped_values
#         colIndex[col] = ed.df[col].apply(lambda x: not x in [v[1] for v in dropped_values])
#
#         continue
#
#     # analyze date columns
#     elif ed.colTypes[col] in ['DATE']:
#         # TODO
#         colIndex[col] = ed.df[col].apply(lambda x: True)
#         continue
#
#     # analyze categorical columns
#     values = ed.df[col]
#     valueCounterList, total = ed.getValueCounterList(values)
#
#     rareValues = []
#     x = []
#     y = []
#     for ck in valueCounterList:
#         count = ck[0]
#         key = ck[1]
#         x += [key]
#         y += [count]
#         if count <= numRareCat:
#             rareValues += [key]
#
#     colIndex[col] = ed.df[col].apply(lambda x: not x in rareValues)
#     dropped_values = []
#     colDroppedValues[col] = dropped_values
#
#     if len(rareValues) > 0:
#         x_ = np.array(range(len(y)))
#         y = np.array(y)
#         for rareValue in rareValues:
#             dropped_values += [(0, rareValue)]
#         #
#         plot = False
#         if plot:
#             plt.bar(x_ - 0.4, y / total)
#             plt.plot(x_, np.cumsum(y / total), 'r', linewidth=3)
#             plt.grid()
#             plt.xticks(x_, x[:len(y)])
#             plt.title(col)
#             plt.show()
#
#     for row in ed.df[colIndex[col] == False].iterrows():
#         score = 0
#         material = row[1]['Material']
#         plant = row[1]['Plant']
#         val = row[1][col]
#         comment = 'Global Histogram Analysis'
#         advice = 'currently unavailable'
#         globalHistogramErrors += [(material, plant, col, val, comment, advice)]
#         # globalHistogramErrors += [(material, plant, col, comment, value, recommendation)]
#
# new_columns = ['cell', 'cell value', 'cognitive score 2', 'comment', 'advice']
# all_columns = np.append(ed.df_complete.columns, new_columns)
# df_global_histogram_errors = pd.DataFrame(columns=all_columns)
#
# # transform error list to material -> plant -> list
# materialPlantErrorList = {}
# for error in globalHistogramErrors:
#     # (score, material, plant, col, comment, value, recommendation)
#     material = error[1]
#     plant = error[2]
#     errorList = materialPlantErrorList.setdefault(material, {}).setdefault(plant, [])
#     errorList += [error]
#     """
#     .setdefault():
#         built-in function; If key is in the dictionary, return its value. If not, insert key with a value of default
#         and return default. default defaults to None.
#     """
#
# for row in ed.df_complete.iterrows():
#     """
#     DataFrame.iterrows(): Iterate over DataFrame rows as (index, Series) pairs.
#         Returns:
#         it : generator; A generator that iterates over the rows of the frame.
#     """
#     material = row[1]['Material']
#     plant = row[1]['Plant']
#     errorList = materialPlantErrorList.get(material, {}).get(plant, [])
#
#     for error in errorList:
#         newRow = row[1]
#         newRow['cell name'] = error[2]
#         newRow['cell value'] = error[3]
#         newRow['cognitive score 2'] = int(round(9.0 * (1.0 - error[0]) + 1.0))
#         newRow['comment'] = error[4]
#         newRow['advice'] = error[6]
#
#         df_global_histogram_errors = df_global_histogram_errors.append(newRow)
#
#
# df_global_histogram_errors['cognitive score 2'] = df_global_histogram_errors['cognitive score 2'].astype(int)
# df_global_histogram_errors = df_global_histogram_errors.fillna('')
# # df_global_histogram_errors.to_csv(config.global_histogram_errors_file, index=False, columns=all_columns)
#
# print(' Done.')







###############################################################################
# DECISION TREE ANALYSIS
###############################################################################

if len(ed.df) > 10000:
    minSamplesPerLeave = 100
    maxLeafNodes = 25
    # requiredAccuracy = 0.99  # only consider decision trees with a minimal required accuracy
    requiredAccuracy = 0.9
    # minScore = 0.99  # only consider leaves that have a minimal confidence
    minScore = 0.9
    numSigmas = 4  # number of +/- sigmas to consider values as outliers
else:
    minSamplesPerLeave = 20
    maxLeafNodes = 10
    requiredAccuracy = 0.75  # only consider decision trees with a minimal required accuracy
    minScore = 0.8999  # only consider leaves that have a minimal confidence
    numSigmas = 3  # number of +/- sigmas to consider values as outliers

print('Decision Tree Analysis...')
sys.stdout.flush()


ed = pg.ErrorDetection(data_filename, columns_description_filename)
# print 'original columns_description_filenameols'
# print ed.used_cols

special_col = ['Material', 'Plant Description', 'Description', 'Follow-up matl', 'Valid from']
cols_ignore = special_col
"""
both  'Valid from' & 'Eff.-out' are Date objects, and they date obj are not treated in this code....
Plant Description & Description are both TEXTS, so exclude.
IDK about the followup matl and material type, but they were excluded in the original code. why? DONT KNOW.
i feel like they should be included... but anyway. will worry about that later. (10/23/2016)
"""

### check which column has only Null values (aka NaN), if so, must take out those columns
for col in ed.df_complete.columns.values:
    if ed.df_complete.ix[:, col].isnull().values.all():
        cols_ignore.append(col)

# maintenanceStatusFeatures = []
# colEncoder = {}
### i'm going to create the encoder outside of the function, so that i can use this global var later
encoder = preprocessing.LabelEncoder()

def relabeledDataFrame(df, colTypes):
    df_relabeled = pd.DataFrame()

    df_relabeled['Material'] = df['Material']
    df_relabeled['Plant'] = df['Plant']

    for col in df.columns:
        if col.strip() in cols_ignore:
            continue

    if colTypes[col] in ['PRIMARY_KEY', 'FOREIGN_KEY', 'INT', 'FLOAT']:
        df_relabeled[col] = df[col]
    if colTypes[col] == 'BOOLEAN':
        df_relabeled[col] = df[col].apply(lambda x: int(x == 'X' or x == 1))
    if colTypes[col] in ['ENUM']:
        """
        so okay... i was doing one hot encoding and label encoding, but seems like he already did it...
        i didnt include ENUM in my colTypes before, so i added that to take care of it.
        """

        """
        values = df[col].unique()
        colEncoder = []
        if len(values) <= 25:
            for value in values:
                df_relabeled[col + '_' + value] = df[col].apply(lambda x: int(x == value))
        else:
            # encoder = preprocessing.LabelEncoder().fit(values)
            encoder = preprocessing.LabelEncoder()

            # colEncoder[col] = encoder

            encoded_vals = encoder.fit_transform(df[col])
            # df_relabeled =+ [encoder.inverse_transform(temp)]
            encoded_vals = encoder.inverse_transform(encoded_vals)
            df_relabeled[col + '_' + value] = encoded_vals

            pass
        """

        # encoder = preprocessing.LabelEncoder().fit(values)

        encoded_vals = encoder.fit_transform(df[col])
        # encoded_vals = encoder.inverse_transform(encoded_vals)   ### do i need this line?!!?
        df_relabeled[col] = encoded_vals
        pass

    return df_relabeled

X = relabeledDataFrame(ed.df, ed.colTypes)


"""
below is a section of short code that i added to impute the missing values in the data frame
so that the random forest classifier could work without complaining...

sklearn.preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0, copy=True)
    strategy- string, optional (default = ”mean”)
        If “mean”, then replace missing values using the mean along the axis.
        If “median”, then replace missing values using the median along the axis.
        If “most_frequent”, then replace missing using the most frequent value along the axis.

    axis : integer, optional (default=0)
        If axis=0, then impute along columns.
        If axis=1, then impute along rows.
"""

imp = Imputer(missing_values="NaN", strategy='median', axis=0)

X = pd.DataFrame(imp.fit_transform(X), columns=X.columns.values)
# b = np.where(X.applymap(lambda x: pd.isnull(x)))
# print b
X = X.replace([np.inf, -np.inf], np.nan)

# new_columns = ['cell', 'cell name', 'cluster', 'cognitive score 1', 'cognitive score 2', 'comment', 'advice']
new_columns = ['cell name', 'cell value', 'cognitive score 2', 'comment', 'advice']
all_columns = np.append(ed.df_complete.columns, new_columns)

df_dec_tree_errors = pd.DataFrame(columns=all_columns)

# special_col = ['Material', 'Plant', 'Plant Description', 'Description', 'Follow-up matl', 'Valid from', 'Eff.-out']

# print ed.unused_cols
# print ed.used_cols
# i = 0
for col in ed.used_cols:

    if col.strip() in cols_ignore:
        continue

    # # get column index and columns
    # index = colIndex[col]
    # cols = [c for c in ed.used_cols if not col in c.split('_')[0]] ### original, but i expanded into following:
    cols = ed.used_cols

    if col not in cols_ignore:
        for val in cols_ignore:
            if val in ed.used_cols:
                cols.remove(val)
        cols.remove(col)
    # cols.remove('Plant')    ### there's only ONE plant value, so it got removed from "used col" list

    ### need to re-define index value
    # index = ed.df.columns.get_loc(col)
    # index = ed.used_cols.index(col)

    # only fit columns with more than one value
    values = ed.df[col].unique()
    if len(values) == 1:
        cols.remove(col)
        continue

    """

    for INT columns, some of them only have values of: 1,2,blank.
    in your improved verion of code, you may want to do one-hot-encoding on those columns.
    for now... save yourself some time from them first.

    """


    # analyze numeric columns
    if ed.colTypes[col] in ['INT', 'FLOAT']:

        clf = RandomForestRegressor(max_leaf_nodes=maxLeafNodes, min_samples_leaf=minSamplesPerLeave)

        Y_orig = X.ix[:, col]
        X = X.ix[:, cols].dropna(axis=1, how='all')
        # X.ix[:5, cols].dropna(axis=1, how='all')
        clf = clf.fit(X, Y_orig)
        Y = clf.predict(X)

        # sklearn.tree.export_graphviz(clf, out_file = 'tree.dot', feature_names=cols)
        ### above line doesnt fucking work

        errors = np.array(Y - Y_orig)
        mu = np.mean(errors)
        sigma = np.sqrt(np.var(errors))
        errors_scaled = (errors - mu) / sigma

        # 6 sigma bounds on scaled errors
        lb = -numSigmas
        ub = +numSigmas

        error_index = (errors_scaled < lb) | (errors_scaled > ub)

        if sum(error_index) == 0:
            continue

        plot = False
        # plot = True
        if plot:
            plt.hist(errors_scaled)
            ylim = plt.ylim()
            plt.plot([lb, lb], ylim)
            plt.plot([ub, ub], ylim)
            plt.ylim(ylim)
            plt.grid()
            plt.title(col)
            plt.show()

            print('num errors:', sum(error_index))

        # df_index = ed.df_complete[index]
        df_index = ed.df_complete  ### the original data set
        df_index_not_correct = df_index.ix[error_index,:].copy(deep=True).reset_index()   ### the original data w error index

        ### now we have to inverse encoder to Y so that Y will print its original output
        Y_not_correct = Y[error_index].copy()
        score = errors_scaled[error_index]

        # df_index_not_correct = ed.df_complete.ix[error_index, :]
        df_index_not_correct['cell name'] = col
        # df_index_not_correct['cell value'] = ed.df_complete[error_index, col]
        df_index_not_correct['cell value'] = df_index_not_correct[col]
        df_index_not_correct['comment'] = 'Decision Tree Analysis (numerical)'
        df_index_not_correct['advice'] = Y_not_correct
        if ed.colTypes[col] == 'FLOAT':
            df_index_not_correct['advice'] = df_index_not_correct['advice'].apply(lambda x: "%.3f" % x)
        else:
            df_index_not_correct['advice'] = df_index_not_correct['advice'].apply(lambda x: int(round(x)))
        divisor = max(score) - min(score) + 1
        df_index_not_correct['cognitive score 2'] = np.round(9.0 * (score - min(score) + 1) / divisor + 1)

        df_dec_tree_errors = df_dec_tree_errors.append(df_index_not_correct)
        # df_dec_tree_errors = df_dec_tree_errors.append(ed.df_complete.ix[error_index, :])

        continue
    #
    # analyze date columns
    if ed.colTypes[col] in ['DATE']:
        # TODO
        continue


    #########################################
    ### analyze categorical columns
    # elif ed.colTypes[col] in ['BOOLEAN', 'ENUM']:
    elif ed.colTypes[col] in ['ENUM']:
        clf = RandomForestClassifier(max_leaf_nodes=maxLeafNodes, min_samples_leaf=minSamplesPerLeave)

        # Y_orig = ed.df[index][col]
        Y_orig = X.ix[:, col]
        X = X.ix[:, cols].dropna(axis=1, how='all')

        # if ed.colTypes[col] == 'BOOLEAN':
        #     Y_orig = Y_orig.apply(lambda x: x == 'X' or x == '1' or x == 'TRUE')

        for col in X.columns:
            if X.ix[:, col].isnull().values.all():
                print col

        clf = clf.fit(X, Y_orig)
        Y = clf.predict(X)
        print Y

        Y = list(encoder.inverse_transform(Y))
        # P = clf.predict_proba(X.ix[:,cols])
        # ## predict_proba(X): Predict class probabilities for X
        # correct = np.array(Y_orig == Y)
        # hit_rate = sum(correct) / len(correct)
        #
        # # only consider columns that can be predicted with a certain quality
        # print(col, hit_rate)
        # if hit_rate < requiredAccuracy:
        #     # print 'less than required'
        #     continue
        # elif hit_rate == 1:
        #     continue
        #
        # print "======================================================================"



        # df_index = ed.df_complete.columns.values
        # print 'df index is: ', df_index
        # df_index_not_correct = df_index[correct == False].copy(deep=True).reset_index()
        # print "df index not correct: "
        # print df_index_not_correct
        #
        # for col in ed.df.columns.values:
        #     if col not in cols_ignore:
        # # for row in ed.df[col not in cols_ignore].iterrows():
        #         print col
        #
        # #### the index of rows we want to print
        # Y_not_correct = Y[correct == False].copy()
        # print 'y not correct'
        # print Y_not_correct
        # P_not_correct = P[correct == False].copy()


        # df_index_not_correct['cell name'] = col
        # df_index_not_correct['comment'] = 'Decision Tree Analysis (categorical)'
        # df_index_not_correct['advice'] = Y_not_correct
        #
        # score = []
        # for i in range(len(P_not_correct)):
        #     score += [P_not_correct[i][Y_not_correct_ids[i]]]
        # score = np.array(score)
        # divisor = max(score) - min(score) + 1
        # df_index_not_correct['cognitive score 2'] = (np.round(9.0 * (score - min(score) + 1) / divisor + 1))
        #
        # print 'df index not correct: '
        # print df_index_not_correct
        #
        # # filter on those where the cognitive score is above a certain threshold
        # df_index_not_correct = df_index_not_correct[score >= minScore]
        # print 'index not correct: ', df_index_not_correct
        # df_dec_tree_errors = df_dec_tree_errors.append(df_index_not_correct)

#
# try:
#     df_dec_tree_errors['cognitive score 2'] = df_dec_tree_errors['cognitive score 2'].astype(int)
# except:
#     print('Exception: cannot convert cognitive score to int!')
#
# df_dec_tree_errors = df_dec_tree_errors.fillna('')
# df_dec_tree_errors.to_csv(config.decision_tree_errors_file, index=False, columns=all_columns)
#
# print('    Done.')





###############################################################################
# MERGE RESULTS
###############################################################################

# print 'Merge Results...'
# sys.stdout.flush()
#
# df_all_errors = pd.DataFrame(columns = df_global_histogram_errors.columns)
# df_all_errors = df_all_errors.append(df_global_histogram_errors)
# df_all_errors = df_all_errors.append(df_dec_tree_errors)
#
# try:
#     df_all_errors['cognitive score 2'] = df_all_errors['cognitive score 2'].astype(int)
# except:
#     print('Exception: cannot convert cognitive score to int!')
#
#
# df_all_errors.to_csv(config.all_errors_file, index=False, header=None, columns=all_columns, encoding='latin-1')
#
# print '             Done.'

