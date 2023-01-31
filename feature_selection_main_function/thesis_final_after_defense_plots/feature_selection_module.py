# Requirements - base classifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
import warnings
warnings.filterwarnings('ignore')
from expttools import Experiment
from sklearn.feature_selection import f_classif
from sklearn.base import BaseEstimator, MetaEstimatorMixin

# Requirements - fair classifier
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.metrics import ClassificationMetric
from sklearn.metrics import accuracy_score
from aif360.datasets import AdultDataset
from sklearn.preprocessing import StandardScaler
from aif360.datasets import StructuredDataset, BinaryLabelDataset
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def categorical_feature_encoder(data,features):
    '''
    takes a data frame and returns numerical encoding for categorical features
    
    Parameters: 
    ----------- 
    dataset: csv file 
    features : list of categorical features

    Return 
    ------
    returns : dataframe with encoded features, encoding for categorical features  
    '''
    enc = {}
    
    for f in features:
        encoder = OrdinalEncoder()
        data[f] = encoder.fit_transform(data[[f]]).astype(int)
        enc[f] = encoder
    return data, enc


def df_manipulation(features_encoded_data,data,col_name,privileged_vals):
    '''
    takes encoded data frame and original data to substitute an original column from original df to 
    encoded df for further analysis
    
    Parameters: 
    ----------- 
    features_encoded_data: encoded dataset 
    data : original data
    col_name : column for manipulation. eg: if comparison is done between black and white, this piece 
    of code will remove all other races and return black as 0 white as 1 and 
    privileged_vals: list of value/values considered as priviliged in society
    Return 
    ------
    returns : dataframe with encoded features containing manipulated column 
    '''
    
    decoded_col_name = f'{col_name}_decoded'
    filtered_col_name = f'filtered_{col_name}'
    features_encoded_data[decoded_col_name] = data[col_name]
    features_encoded_data[filtered_col_name] = features_encoded_data[decoded_col_name]\
    .isin(privileged_vals).astype(int)

    features_encoded_data = features_encoded_data.drop([decoded_col_name], axis = 1)

    return features_encoded_data



def base_model(data,target,col_name):
    '''
    takes feature encoded data, splits for training and test set and returns the data frame with predictions.
    
    Parameters: 
    ----------- 
    data : encoded data
    target : proxy target
    col_name : column for manipulation. 

    Return 
    ------
    returns : original_output -- predicted set
    '''
    encoded_df = data.copy()
    x = encoded_df.drop([target], axis = 1)
    
    y = encoded_df[target]
    filtered_col_name = f'filtered_{col_name}'
    #print(filtered_col_name)
    
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state = 0)
    x_train[filtered_col_name] = x_train[filtered_col_name].apply(lambda x:1 if x>0 else 0)
    x_test[filtered_col_name] = x_test[filtered_col_name].apply(lambda x:1 if x>0 else 0)
    sc = StandardScaler()
    x_train = pd.DataFrame(sc.fit_transform(x_train),columns = x_train.columns)
    x_test = pd.DataFrame(sc.transform(x_test),columns = x_test.columns)
    x_train[filtered_col_name] = x_train[filtered_col_name].apply(lambda x:1 if x>0 else 0)
    x_test[filtered_col_name] = x_test[filtered_col_name].apply(lambda x:1 if x>0 else 0)
    classifier = LogisticRegression()
    print(x_train.columns)
    classifier.fit(x_train.drop(filtered_col_name,axis=1), y_train) # dropping here
    # We now need to add this array into x_test as a column for when we calculate the fairness metrics.
    y_pred = classifier.predict(x_test.drop(filtered_col_name,axis=1))
    x_test['target_predicted'] = y_pred
    original_output = x_test
    original_output['actual'] = y_test.values
    return original_output

    
def get_fairness_metrics_bc(original_output,col_name):
    '''
    takes prediction df and returns calculated fairness metrics 
    
    Parameters: 
    ----------- 
    original_output : original_output with prediction column
    col_name : column for manipulation. 

    Return 
    ------
    returns : fairness metrics
    '''
    filtered_col_name = f'filtered_{col_name}'

    male_df = original_output[original_output[filtered_col_name] == 1]
    num_of_priviliged = male_df.shape[0]
    female_df = original_output[original_output[filtered_col_name] == 0]
    num_of_unpriviliged = female_df.shape[0]

    unpriviliged_outcomes = female_df[female_df['target_predicted'] == 1].shape[0]
    unpriviliged = unpriviliged_outcomes/num_of_unpriviliged
    unpriviliged

    priviliged_outcomes = male_df[male_df['target_predicted'] == 1].shape[0]
    priviliged = priviliged_outcomes/num_of_priviliged
    priviliged

    #Disparate impact
    try:
        disparate_impact = unpriviliged / priviliged
    except:
        disparate_impact = np.inf

    #Statistical parity difference 
    statistical_parity_difference  = unpriviliged - priviliged
    
    #Equal opportunity difference
    eod = original_output.copy()
    eod = eod[eod['actual'] == 1] 
    eod ['true_positives'] = eod ['target_predicted'] == eod['actual']

    eod_other = eod[eod[filtered_col_name]== 0]['true_positives'].mean()

    eod_married_civ_absent = eod[eod[filtered_col_name] == 1]['true_positives'].mean()
    equal_opportunity_difference  = eod_other - eod_married_civ_absent
    
    #accuracy
    accuracy = (original_output['target_predicted']== original_output['actual']).mean()
    
    return ([disparate_impact,statistical_parity_difference,equal_opportunity_difference,accuracy])
    

def fair_model(data, subset_cols, target, p_att):
    
    '''
    takes encoded df and formats into a binary label df format to split and return test and train df.
    
    Parameters: 
    ----------- 
    dataset: original data 
    subset_cols : columns from dictionary generated from feature selection technique
    target : proxy target
    p_att: single protected attribute, key from dictionary generated from feature selection technique
    
    Return 
    ------
    returns : dataset_orig_train, dataset_orig_test -- test and trained datasets 
    '''
    
    encoded_df = data.copy()
    structured_data = BinaryLabelDataset(favorable_label=1.0, unfavorable_label=0.0, df = encoded_df[subset_cols]\
                                         .dropna(), label_names = [target], protected_attribute_names = [p_att], \
                                         instance_weights_name=None, scores_names=[], unprivileged_protected_attributes\
                                         =[[0]], privileged_protected_attributes=[[1]], metadata=None)
    dataset_orig = structured_data
    privileged_groups = [{p_att: 1}] #male 
    unprivileged_groups = [{p_att: 0}] #female

    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
    scaler = StandardScaler()
    dataset_orig_train.features = scaler.fit_transform(dataset_orig_train.features)
    dataset_orig_test.features = scaler.transform(dataset_orig_test.features) 
    return dataset_orig_train,dataset_orig_test


def get_fairness_metrics_fc(dataset_orig_train,dataset_orig_test,p_att,privileged_groups,unprivileged_groups,etas):
    
    '''
    takes train and test dataset runs using PrejudiceRemover classifier and returns fairness metrics
    
    Parameters: 
    ----------- 
    dataset_orig_train: training data set 
    dataset_orig_test: test dataset
    p_att: single protected attribute, chooses particular key from dictionary generated from feature selection technique
    privileged_groups : list of priviliged values in society eg. male priviliged = [1]
    unprivileged_groups : list of unpriviliged values in society eg. female unpriviliged = [0]
    etas : fairness penalty parameter
    
    Return 
    ------
    returns : fairness metrics 
    '''
    
    outputs = []
    #print(dataset_orig_train,dataset_orig_test,p_att,privileged_groups,unprivileged_groups,etas)
    
    for eta in etas:
        debiased_model = PrejudiceRemover(eta=eta, sensitive_attr = p_att, class_attr=dataset_orig_train.label_names[0])        
        model = debiased_model.fit(dataset_orig_train)
        pred = model.predict(dataset_orig_test)

        metric = ClassificationMetric(dataset_orig_test, pred, unprivileged_groups=unprivileged_groups, \
                                      privileged_groups=privileged_groups)
        
        outputs.append([eta,metric.disparate_impact(),metric.statistical_parity_difference(),\
                        metric.equal_opportunity_difference(),\
                        accuracy_score(pred.labels, dataset_orig_test.labels)])
    return outputs

#function for calculating mutual information score for each X (features) and Y (target)
def calculate_miscore_xy(data, y_col,a_col):
    '''
    takes a data frame and returns a data frame with mutual information score between X(features) and Y(target)
    
    Parameters: 
    ----------- 
    data : csv file 
    y_col : target column
    a_col : protected attributes 
    
    Return 
    ------
    returns : returns mutual information score between X(features) and Y(target) in a dataframe 
    '''
    mis_xy = []
    x_cols = []
    for x in data.columns:
        if not (x in a_col): # skipping the demographic features 
            mis = mutual_info_score(data[x], data[y_col], contingency=None) # mis calculation
            mis_xy.append(mis) 
            x_cols.append(x)


    adult_dataFrame_feature_target = pd.DataFrame({'I(Xi,Y)': mis_xy, 'X': x_cols}) # creating pandas dataframe
    adult_dataFrame_feature_target = adult_dataFrame_feature_target.loc[adult_dataFrame_feature_target['X'] != y_col]#loc : filtering dataframe based on index
    adult_dataFrame_feature_target['Y'] = y_col # adding y column
    return adult_dataFrame_feature_target

#function for calculating mutual information score for each X (features) and A (demographic variables)

def calculate_miscore_xa(data,protected_attributes):
    '''
    takes a data frame and returns a data frame with mutual information score between X(features) and A(protected attributes)
    
    Parameters: 
    ----------- 
    dataset: csv file 
    protected_attributes : protected attributes 
    
    Return 
    ------
    returns : returns mutual information score between X(features) and A(protected attributes) in a dataframe 
    '''
    
    mis_xa = []
    attribute_unfiltered = []
    feature_unfiltered = []
    for x in data.columns:
        for a in protected_attributes:
            if not (x in protected_attributes):
                mis = mutual_info_score(data[a], data[x], contingency=None)
                mis_xa.append(mis)
                attribute_unfiltered.append(a)  
                feature_unfiltered.append(x)

    unfiltered_mis_adult_dataFrame = pd.DataFrame({'X':feature_unfiltered, 'A':attribute_unfiltered, \
                                                   'I(Xi,A)': mis_xa})
    return unfiltered_mis_adult_dataFrame


#merging dataframes
def generate_xy_greater_xa(data1,data2):
    '''
    merges two data frames (in our case two df generated for I(X,Y) and I(X,A) and returns a dictionary with keys as protected att and values as features 
    (features = all features where XY>XA for a particular protected att)
    
    Parameters: 
    ----------- 
    data1: dataframe 
    data2: dataframe
    
    Return 
    ------
    returns : dictionary with key as p attributes and values as features
    '''
    merged_xiY_xiA = pd.merge(data1,data2, on=['X'], how = 'inner') 
    merged_xiY_xiA
    #merged_xiY_xiA.to_csv('merged_xiY_xiA.csv')
    #adding a new bool column, True if xi_A > xi_Y
    merged_xiY_xiA['X_sub_A = True'] = merged_xiY_xiA['I(Xi,A)'] > merged_xiY_xiA['I(Xi,Y)']

    # for each val of A pick list of features where XY>XA - X_sub_A¯
    dictonary_xy_greaterthan_xa = {} 
    for a_v, df_a in merged_xiY_xiA.groupby('A'):
        dictonary_xy_greaterthan_xa[a_v] = df_a[df_a['X_sub_A = True']== False]['X'].values

    return dictonary_xy_greaterthan_xa


# Criteria 1
def xy_greater_xa(data,features_cat,features_num,protected_attributes, target,feat_dictionary):
    '''
    takes data and returns encoded df and a dictionary with keys as protected att and values as features 
    (features = all features where XY>XA for a particular protected att) 
    
    Parameters: 
    ----------- 
    data : original dataframe
    features_cat: list of categorical features in data 
    protected_attributes : list of protected attributes in data
    target: proxy target
    feat_dictionary : dictionary of all the features listed as keys and bins as keys for respective features.
    Return 
    ------
    returns : encoded dataframe and a dictionary
    '''
    features_encoded_data,enc = categorical_feature_encoder(data.copy(),features_cat + protected_attributes)
    features_encoded_data = features_encoded_data[features_cat + features_num + protected_attributes + [target]]

    mi_Xi_Y = calculate_miscore_xy(features_encoded_data, target, protected_attributes)
    mi_Xi_A = calculate_miscore_xa(features_encoded_data,protected_attributes)
    dict_subsets_xy_greaterthan_xa = generate_xy_greater_xa(mi_Xi_Y,mi_Xi_A) 
    return features_encoded_data, dict_subsets_xy_greaterthan_xa

# no feature selection
def no_fselection(data,features_cat,features_num,protected_attributes, target,feat_dictionary):
    '''
    takes data and returns encoded df and a dictionary with keys as protected att and values as features 
    (features = all features where XY>XA for a particular protected att) 
    
    Parameters: 
    ----------- 
    data : original dataframe
    features_cat: list of categorical features in data 
    protected_attributes : list of protected attributes in data
    target: proxy target
    feat_dictionary : dictionary of all the features listed as keys and bins as keys for respective features.
    Return 
    ------
    returns : encoded dataframe and a dictionary
    '''
    features_encoded_data,enc = categorical_feature_encoder(data.copy(),features_cat + protected_attributes)
    features_encoded_data = features_encoded_data[features_cat + features_num + protected_attributes + [target]]

    # mi_Xi_Y = calculate_miscore_xy(features_encoded_data, target, protected_attributes)
    # mi_Xi_A = calculate_miscore_xa(features_encoded_data,protected_attributes)
    # dict_subsets_xy_greaterthan_xa = generate_xy_greater_xa(mi_Xi_Y,mi_Xi_A) 
    dict_subsets_xy_greaterthan_xa = {k:features_cat + features_num for k in protected_attributes}
    return features_encoded_data, dict_subsets_xy_greaterthan_xa

# Criteria 2

# class MiEstimator(BaseEstimator):
#     '''
#     MI estimator for calculating the mutual information score between XY and XA.

#     Parameters: 
#     ----------- 
#     X : data
#     protected_attributes (A/Y): demographic attributed in data or target variable, while calling the method A/Y can be used as per requirement.
#     bins :  list of bins for feature binning

#     Return 
#     ------
#     returns : returns a single value of MI XY or MI XA   
#     '''
    
#     def __init__(self,protected_attributes, bins):
#         self.protected_attributes = protected_attributes
#         self.bins = bins # will contain a dict for cutting into bins (dictionary)
#         self.count = 0
        
#     def fit(self, X):
#         #print('temp')
#         self.fitted_ = True
#         return self
        
#     def predict(self, X):
#         print('temp2')
#         return X
        
#     def score(self, X):
#         self.count+=1 # When MiEstimator object is being called, it does not enter score() because the count is not incrementing and if I remove the function it throws an error that score function is required. but the issue is when I added score function back it did not throw error but did not even pront the count. This is what I wanted to see if it's entering score function or not. I am confused !
#         mis_xa = []
#         target_bins = X[self.protected_attributes].nunique()
#         print('here',list(X.columns) + [self.protected_attributes])
#         xa_mat, bins = np.histogramdd(X[list(X.columns) + [self.protected_attributes]].values,[self.bins[j] for j in X.columns]+[target_bins]) 
#         xalist = xa_mat.reshape(-1,target_bins)
#         mis_a = mutual_info_score(None, None, contingency = xalist)
#         return mis_a 
    
#     def get_params(self,deep=True):
#         return {'protected_attributes':self.protected_attributes,'bins':self.bins}
    
#     def set_params(self, **parameters):
#         for parameter, value in parameters.items():
#             setattr(self, parameter, value)
#         return self

from sklearn.base import BaseEstimator

class MiEstimator(BaseEstimator):
    '''
    MI estimator for calculating the mutual information score between XY and XA.

    Parameters: 
    ----------- 
    X : data
    protected_attributes (A/Y): demographic attributed in data or target variable, while calling the method A/Y can be used as per requirement.
    bins :  list of bins for feature binning

    Return 
    ------
    returns : returns a single value of MI XY or MI XA   
    '''
    
    def __init__(self,protected_attributes, bins):
        self.protected_attributes = protected_attributes
        self.bins = bins # will contain a dict for cutting into bins (dictionary)
        
    def fit(self, X, y = None):
        return self
        
    def predict(self, X):
        return X
        
    def score(self, X, y = None):
        mis_xa = []
        target_bins = X[self.protected_attributes].nunique()
        xa_mat, bins = np.histogramdd(X[list(X.columns)].values,[self.bins[j] for j in X.columns if j != self.protected_attributes]+[target_bins]) 
        xalist = xa_mat.reshape(-1,target_bins)
        mis_a = mutual_info_score(None, None, contingency = xalist)
        return mis_a 
    
    def get_params(self,deep=True):
        return {'protected_attributes':self.protected_attributes,'bins':self.bins}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
class SFS:
    def __init__(self,estimator,direction, tolerance):
        self.estimator = estimator
        self.direction = direction
        self.tolerance = tolerance
    def fit(self,data):
        if self.direction == 'forward':
            cols = []
            num_cols = len(data.columns)//2
            while True:
                score = 0
                curr = ''
                for column in data.columns:
                    if column != self.estimator.protected_attributes and column not in cols:
                        mi = self.estimator.score(data[cols+[column,self.estimator.protected_attributes]])
                        if mi > score:
                            score = mi
                            curr = column
                cols.append(curr)
                if len(cols) == num_cols:
                    break
            return cols
        elif self.direction == 'backward':
            num_cols = len(data.columns)//2
            cols = [j for j in data.columns if j != self.estimator.protected_attributes]
            score = self.estimator.score(data[cols+[self.estimator.protected_attributes]])
            diff = np.inf
            curr = ''
            while True:
                for column in cols:
                    temp = [j for j in cols if j!= column]
                    mi = self.estimator.score(data[temp+[self.estimator.protected_attributes]])
                    temp_diff = score - mi
                    if temp_diff < diff:
                        diff = temp_diff
                        curr = column
                cols = [j for j in cols if j!= curr]
                diff = np.inf
                curr = ''
                score = self.estimator.score(data[cols+[self.estimator.protected_attributes]])
                if len(cols) == num_cols:
                        break
            return cols
    
    
#Forward sfs

def sequential_feature_selection_forward(data,features_cat,features_num,protected_attributes, target,feat_dictionary):  
    #return 
    '''
    this criteria works in a forward sequential manner to select relevant and fair sub sets of features based on mutual information calculation.
    
    Parameters: 
    ----------- 
    data : original dataframe
    features_cat: list of categorical features in data
    features_num: list of numerical features in data
    protected_attributes : list of protected attributes in data
    target: proxy target
    feat_dictionary : dictionary of all the features listed as keys and bins as keys for respective features.

    Return 
    ------
    returns : returns features selected by sfs as values of protected attribute(key) in use   
    '''
    
    # features_encoded_data,enc = categorical_feature_encoder(data.copy(),features_cat + protected_attributes)
    # dictionary_features = {}
    # for att in protected_attributes:
    #     mie = MiEstimator(att,feat_dictionary)
    #     #print(features_encoded_data[features_cat+features_num].info())
    #     sfs = SequentialFeatureSelector(mie, n_features_to_select= "auto")
    #     sfs.fit(features_encoded_data[features_cat+features_num])
    #     #print(mie.count)
    #     mask = sfs.get_support()
    #     dictionary_features[att] = [(features_cat+features_num)[i] for i in range(len(mask)) if not mask[i]] 
    #     #print('dictionary_features',dictionary_features)
    # return features_encoded_data, dictionary_features
        
    features_encoded_data,enc = categorical_feature_encoder(data.copy(),features_cat + protected_attributes)
    dictionary_features = {}
    for att in protected_attributes:
        mie = MiEstimator(att,feat_dictionary)
        sfs = SFS(mie,direction = 'forward',tolerance = 0)
        cols = sfs.fit(features_encoded_data[features_cat+features_num+[att]])
        dictionary_features[att] = [j for j in features_cat+features_num if j not in cols]
    return features_encoded_data, dictionary_features

# Backward sfs
def sequential_feature_selection_backward(data,features_cat,features_num,protected_attributes, target,feat_dictionary):
    '''
    this criteria works in a backward sequential manner to select relevant and fair sub sets of features based on mutual information calculation.
    
    Parameters: 
    ----------- 
    data : original dataframe
    features_cat: list of categorical features in data
    features_num: list of numerical features in data
    protected_attributes : list of protected attributes in data
    target: proxy target
    feat_dictionary : dictionary of all the features listed as keys and bins as keys for respective features.

    
    Return 
    ------
    returns : returns features selected by sfs backward as values of protected attribute(key) in use   
    '''
    
    features_encoded_data,enc = categorical_feature_encoder(data.copy(),features_cat + protected_attributes)
    dictionary_features = {}
    for att in protected_attributes:
        mie = MiEstimator(att,feat_dictionary)
        sfs = SFS(mie,direction = 'backward',tolerance = 0)
        cols = sfs.fit(features_encoded_data[features_cat+features_num+[att]])
        dictionary_features[att] = [j for j in features_cat+features_num if j not in cols]
    return features_encoded_data, dictionary_features


# off the shelf fs from sk learn - forward feature selection

def sequential_fs(data,features_cat,features_num,protected_attributes, target,feat_dictionary):
    features_encoded_data,enc = categorical_feature_encoder(data.copy(),features_cat + protected_attributes)
    dictionary_features = {}
    for att in protected_attributes:
        lr = LogisticRegression()
        #print(features_encoded_data[features_cat+features_num].info())
        sfs = SequentialFeatureSelector(lr, n_features_to_select= "auto")
        sfs.fit(features_encoded_data[features_cat+features_num],features_encoded_data[target])
        mask = sfs.get_support()
        dictionary_features[att] = [(features_cat+features_num)[i] for i in range(len(mask)) if mask[i]] 
        #print('dictionary_features',dictionary_features)
    return features_encoded_data, dictionary_features

# Criteria 3

def score(X, bins,protected_attributes): 
    '''
    Computes MI score
    
    Parameters: 
    ----------- 
    X : dataframe
    bins: dictionary of all the features listed as keys and bins as keys for respective features. 
    protected_attributes : list of protected attributes in data
    
    Return 
    ------
    returns : returns calculated MI column from data  
    '''
    mis_xa = []
    for feat, num_bins in bins.items():
        if protected_attributes!= 'age':
            target_bins = X[protected_attributes].nunique()
        else:
            target_bins = 5
        xy_mat, bins = np.histogramdd(X[[feat,protected_attributes]].values, [num_bins,target_bins])
        xylist = xy_mat.reshape(-1,target_bins)
        mis_a = mutual_info_score(None, None, contingency = xylist)
        mis_xa.append([feat,mis_a])
    dataframe = pd.DataFrame(mis_xa,columns = ['feat','mi']).set_index('feat')
    return dataframe['mi']
    
def maximally_predictive_minimally_demographic(data,features_cat,features_num,protected_attributes, target,feat_dictionary): # add feat_dictionary
    '''
    Computes MI using score function for both XA and XY and by taking the ration selects K relevant and fair features.
    
    Parameters: 
    ----------- 
    data : dataframe
    features_cat: list of categorical features in data
    features_num : list of numerical features in data
    protected_attributes : list of protected attributes in data
    target : proxy target
    feat_dictionary : dictionary of all the features listed as keys and bins as keys for respective features.
    
    Return 
    ------
    returns : returns dictionary with values as subset of features and keys as protected attribute in use  
    '''
# features_encoded_data = cat+p_att+num+y

    features_encoded_data,enc = categorical_feature_encoder(data.copy(),features_cat + protected_attributes)
    dictionary_features = {}
    for att in protected_attributes:
        # compute MI between XY and XA
        F = score(features_encoded_data, feat_dictionary,target) #X, bins, protected attributes
        FA = score(features_encoded_data,feat_dictionary,att)
        # dividing for higher scores of XY 
        F = F/FA
        F = F.sort_values(ascending = False)
        k = int(F.shape[0]/1.5) #size of features
        F = F.iloc[:k]
        dictionary_features[att] = list(F.index.values)
    return features_encoded_data, dictionary_features
        
    
# Model functions

def base_classifier(features_encoded_data,target,p_att_col,model_cols,filtered_col, etas): # etas not in use but passing for consistency of format
    '''
    takes encoded df, runs logistic regression on the data to return a df consiting of fairness metrics and model type column.
    
    Parameters: 
    ----------- 
    features_encoded_data: encoded dataframe from categorical_feature_encoder function 
    target: proxy target
    p_att_col : protected attributes in the data
    model_cols : specific subset used from the data for analysis
    filtered_col : used for filtering columns based on conditions eg. keeping black(0) and white(1) in race column and removing other races.
    etas : fairness penalty parameter, not used in base classifier
    
    Return 
    ------
    returns : fairness metric dataframe 
    '''
    original_output = base_model(features_encoded_data[model_cols].copy(),target,p_att_col)
    base_classifier_fairness_metrics = get_fairness_metrics_bc(original_output,p_att_col)
    return pd.DataFrame([['base_model']+base_classifier_fairness_metrics],columns = ['model_type',\
                                                                                     'disparate_impact',\
                                                                                     'statistical_parity_difference',\
                                                                                     'equal_opportunity_difference',\
                                                                                     'accuracy'])

def fair_classifier(features_encoded_data,target,p_att_col,model_cols, filtered_col, etas):
    '''
    takes encoded df, runs prejudice remover on the data to return a df consiting of fairness metrics and model type column.
    
    Parameters: 
    ----------- 
    features_encoded_data: encoded dataframe from categorical_feature_encoder function 
    target: proxy target
    p_att_col : protected attributes in the data
    model_cols : specific subset used from the data for analysis
    filtered_col : used for filtering columns based on conditions eg. keeping black(0) and white(1) in race column and removing other races.
    etas : fairness penalty parameter
    
    Return 
    ------
    returns : fairness metric dataframe 
    '''
    print(model_cols)
    dataset_orig_train,dataset_orig_test = fair_model(features_encoded_data, model_cols, target, filtered_col)
    fair_classifier_fairness_metrics = get_fairness_metrics_fc(dataset_orig_train,dataset_orig_test, \
                                                               filtered_col,privileged_groups = [{filtered_col:1}] ,\
                                                               unprivileged_groups = [{filtered_col:0}],etas=etas)
    result = pd.DataFrame(fair_classifier_fairness_metrics,columns = ['eta','disparate_impact',\
                                                                      'statistical_parity_difference',\
                                                                      'equal_opportunity_difference',\
                                                                      'accuracy'])
    result.insert(0,'model_type','fair_model')
    return result

# building model
def build_model(data,features_encoded_data, p_att_col, dict_subsets_xy_greaterthan_xa,target, privileged_vals, model_type,etas):
    '''
    build model takes encoded data and handles the data manipulation part prior to modeling phase (used for both fair and base classifier) and 
    finally runs the model and returns output dataframe with fairness metrics
    
    Parameters: 
    ----------- 
    data : original dataframe
    features_encoded_data: encoded dataframe from categorical_feature_encoder function 
    p_att_col : protected attributes in the data
    dict_subsets_xy_greaterthan_xa : generated subsets 
    target : proxy target
    privileged_vals : priviliged value in a particular protected att, which should be considered as priviliged group. eg. Whites in races can be 
    considered as priviliged
    model_type : fair or base models 
    etas : fairness penalty parameter for fair model - prejudice remover
    
    Return 
    ------
    returns : dataframe 
    '''
    filtered_col = f'filtered_{p_att_col}'
    model_cols = [filtered_col] + list(dict_subsets_xy_greaterthan_xa[p_att_col]) + [target]
    features_encoded_data = df_manipulation(features_encoded_data,data,p_att_col,privileged_vals)
    output = model_type(features_encoded_data,target,p_att_col,model_cols,filtered_col, etas)
    return output


#main function
def main_exp_bf_func(data,features_cat,features_num,protected_attributes, target,p_att_description, etas, feat_dictionary,\
                     technique = xy_greater_xa, model = fair_classifier):
   
    '''
    takes data and other arguments to return fairness metrics for a particular model and feature selection technique 
    
    Parameters: 
    ----------- 
    data : original dataframe
    features_cat: list of categorical features 
    features_num : list of numerical features
    protected_attributes : list of protected attributes
    target: proxy target
    p_att_col : single protected attribute we want an analysis for
    privileged_vals : privileged values for feature mentioned in p_att_col
    etas : fairness penalty parameter
    feat_dictionary: dictionary of all the features listed as keys and bins as keys for respective features.
    technique : feature selection technique used for subset generation
    model : estimator object, fair or base classifier
    
    Return 
    ------
    returns : dataframe with model type column and fairness metrics
    '''
    data_df = data.df
    p_att_col = p_att_description[0]
    privileged_vals = p_att_description[1]
    features_encoded_data, dict_subsets_xy_greaterthan_xa = technique(data_df,features_cat,features_num,protected_attributes, target,feat_dictionary)
    return build_model(data_df,features_encoded_data, p_att_col, dict_subsets_xy_greaterthan_xa,target, privileged_vals, model, etas)
     
    

