###############################################################################
########################## lIBRARIES NEEDED ###################################
################################################################################

#===================== Data preparation (cleansing) ==========================#
#Numerics and Data Analysis
import pandas as pd #Pandas
import numpy as np #Numpy 

#To store results as objects
from sklearn.datasets.base import Bunch #Dictionary

# Graphics and Vizualization
import seaborn as sns 
import matplotlib.pyplot as plt

# Categorization of data
from sklearn.preprocessing import LabelEncoder

#===================== Machine Learning algorithms ===========================#
#logistict regression. 
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm # statsmodel is chosen because it outputs descriptive stats for the model

# Decision Trees
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

# Accuracy validation
from scipy.stats import pointbiserialr, spearmanr
from sklearn.cross_validation import cross_val_score
import sklearn.cross_validation as cross_validation
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.cross_validation import StratifiedShuffleSplit # split data test and train

###############################################################################
############ FUNCTION DEFINITIONS FOR DATA CLEANSING AND PREPARATION ##########
###############################################################################

# Setting file Path 
filename = 'C:/Users/PC/PycharmProjects/PythonProgramers/adult.csv'

def load_csv(filename):
    data = pd.read_csv(filename, sep="\s*,", engine='python')
    return Bunch(
        rawdata = data.copy())

def checking_data(df):
    
    print (df.head())   # Printing how data was read it 
    print '{} {}'.format("\nAdult dataframe dimmensions are : ", df.shape) #will print 48842 Rows 15 Columns
    
    #Will print number unique values per column df checking data
    print '{}'.format("\nWe will check for each column, number of observations : \n")
    for i in df:
        print '{} {}'.format("\nColumn name : ", i)
        print df[i].value_counts()
    
    # Data Cleansing
    col_names_adultdf = df.columns
    num_data_adultdf = df.shape[0] 
        
    #The dataframe contains some values represented with a ? character
    #This loop will check each who has character "?"     
    print '{}'.format("\nThe following columns contains ? symbol as observations : \n")
  
    columns_With_NA = []
    for i in col_names_adultdf:
        num_NA_Values = df[i].isin(["?"]).sum()
        if num_NA_Values > 0:
            columns_With_NA.append(i)
            print '{} {}'.format("\nColumn name: ", i)  #will pring each column name which contains missing values
            print '{} {}'.format("Number of ? Values : ", num_NA_Values)
            print ("{0:.2f}%".format(float(num_NA_Values) / num_data_adultdf * 100))

def clean_data(df):
    # We are getting some categorical we can make them meaningful, 
    # We will replace this categories 6 categories into two married or not married
    df.replace(['Divorced', 'Married-AF-spouse', 
                  'Married-civ-spouse', 'Married-spouse-absent', 
                  'Never-married','Separated','Widowed'],
                 ['not married','married','married','married',
                  'not married','not married','not married'], inplace = True)  
    
    col_names_adultdf = df.columns
    
    columns_With_NA = []
    for i in col_names_adultdf:
        num_NA_Values = df[i].isin(["?"]).sum()
        if num_NA_Values > 0:
            columns_With_NA.append(i)
    
    # Deleting data with ? missing information
    for i in columns_With_NA:
        df = df[df[i] != "?"]
    
    # Printing comparison between old - and new dataframe dimensions
    #print '{} {}'.format("Adult dataframe original dimmensions were: ", df_copy.shape) #will print 48842 Rows 15 Columns 
    print '{} {}'.format("The new Adult dataframe dimmensions are : ", df.shape) #will print 48842 Rows 15 Columns 

    return Bunch(
    newdata = df.copy())
    
def visualize_data(data):
    #encoded_data, _ = number_encode_features(data)
    sns.heatmap(data.corr(), square=True)
    plt.show()

    sns.countplot(y='occupation', hue='income', data=data, )
    sns.plt.title('Occupation vs Income')
    sns.plt.show()

    sns.countplot(y='education', hue='income', data=data, )
    sns.plt.title('Education vs Income')
    sns.plt.show()

    # How years of education correlate to income, disaggregated by race.
    # More education does not result in the same gains in income
    # for Asian Americans/Pacific Islanders and Native Americans compared to Caucasians.
    g = sns.FacetGrid(data, col='race', size=4, aspect=.5)
    g = g.map(sns.boxplot, 'income', 'educational-num')
    #sns.plt.title('Years of Education vs Income, disaggregated by race')
    sns.plt.show()

    # How years of education correlate to income, disaggregated by sex.
    # More education also does not result in the same gains in income for women compared to men.
    g = sns.FacetGrid(data, col='gender', size=4, aspect=.5)
    g = g.map(sns.boxplot, 'income', 'educational-num')
    #sns.plt.title('Years of Education vs Income, disaggregated by sex')
    sns.plt.show()

    # How age correlates to income, disaggregated by race.
    # Generally older people make more, except for Asian Americans/Pacific Islanders.
    g = sns.FacetGrid(data, col='race', size=4, aspect=.5)
    g = g.map(sns.boxplot, 'income', 'age')
    #sns.plt.title('Age vs Income, disaggregated by race')
    sns.plt.show()

    # How hours worked per week correlates to income, disaggregated by marital status.
    g = sns.FacetGrid(data, col='marital-status', size=4, aspect=.5)
    g = g.map(sns.boxplot, 'income', 'hours-per-week')
    #sns.plt.title('Hours by week vs Income, disaggregated by marital status')
    sns.plt.show()

    sns.violinplot(x='gender', y='educational-num', hue='income', data=data, split=True, scale='count')
    sns.plt.title('Years of Education and Gender vs Income')
    sns.plt.show()

    sns.violinplot(x='gender', y='hours-per-week', hue='income', data=data, split=True, scale='count')
    sns.plt.title('Hours-per-week and Sex vs Income')
    sns.plt.show()

    sns.violinplot(x='gender', y='age', hue='income', data=data, split=True, scale='count')
    sns.plt.title('Age and Sex vs Income')
    sns.plt.show()

    g = sns.PairGrid(data,
                     x_vars=['income', 'gender'],
                     y_vars=['age'],
                     aspect=.75, size=3.5)
    g.map(sns.violinplot, palette='pastel')
    sns.plt.show()

    g = sns.PairGrid(data,
                     x_vars=['marital-status', 'race'],
                     y_vars=['educational-num'],
                     aspect=.75, size=3.5)
    g.map(sns.violinplot, palette='pastel')
    sns.plt.show()

def number_encode_features(data):
    result = data.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return Bunch(num_data = result.copy(),
                 dic_num = encoders.copy())
    
def prepare_data(data):
    data = data.copy()
    
    meta = {
        'target_names': list(data.income.unique()),
        'feature_names': list(data.columns),
        'categorical_features': {
            column: list(data[column].unique())
            for column in data.columns
            if data[column].dtype == 'object'
        },
    }

    names = meta['feature_names']
    #meta['categorical_features'].pop('income')

    train, test = cross_validation.train_test_split(data, test_size=0.25)

    # Return the bunch with the appropriate data chunked apart
    return Bunch(
        rawdata = data,
        data = data[names[:-1]], #All columns less target (income)
        data_target = data[names[-1]],
        train = train[names[:-1]],
        train_target = train[names[-1]],
        test = test[names[:-1]],
        target_test = test[names[-1]],
        target_names = meta['target_names'],
        feature_names = meta['feature_names'],
        categorical_features = meta['categorical_features'],
    )

###############################################################################
###################### MACHINE LEARNING DEFINITIONS ###########################
###############################################################################
def correlation (data):
    
    col_names = data.columns
    param=[]
    correlation=[]
    abs_corr=[]
    
    for c in col_names:
        #Check if binary or continuous
        if c != "income":
            if len(data[c].unique()) <= 2:
                corr = spearmanr(data['income'],data[c])[0]
            else:
                corr = pointbiserialr(data['income'],data[c])[0]
            param.append(c)
            correlation.append(corr)
            abs_corr.append(abs(corr))
    
    #Create dataframe for visualization
    param_df = pd.DataFrame({'correlation':correlation,'parameter':param, 'abs_corr':abs_corr})
    
    #Sort by absolute correlation
    param_df = param_df.sort_values(by=['abs_corr'], ascending=False)
    
    #Set parameter name as index
    param_df = param_df.set_index('parameter')
    
    scoresCV = []
    scores = []
    
    for i in range (1 , len(param_df)):
        new_df = data[param_df.index[0:i+1].values] #sorting DF by correlation importance
        X = new_df.iloc[:,1::]
        target = new_df.iloc[:,0]    
        clf = DecisionTreeClassifier()
        scoreCV = cross_val_score(clf, X, target, cv= 10)
        scores.append(np.mean(scoreCV))
        
    plt.figure(figsize= (15,5))
    plt.plot(range(1, len(scores)+1), scores, '.-')
    plt.axis("tight")
    plt.title('Feature Selection', fontsize=14)
    plt.xlabel('# Features', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.grid();
    
    new_df = data[param_df.index[1:i+1].values]
    new_df.shape
    X = new_df.iloc[:,1::]
    
    return Bunch(data_corr = correlation,
                 data_par = param,
                 data_df_corr = param_df.copy(),
                 target = target.copy())

#=============================================================================#
#  1. lOGISTIC REGRESSION using statsmodels
#-----------------------------------------------------------------------------#
def log_regre_statsmodels():    
    
    ############################### Training ##################################
    print("Training set result\n")
    logit_train = sm.Logit(datased_prepared_encoded.train_target, datased_prepared_encoded.train) 
    result_train = logit_train.fit()
    print("\n")
    print(result_train.summary())
    
    # Accuracy-Training
    y_train_pred = result_train.predict(datased_prepared_encoded.train) 
    y_train_pred = (y_train_pred > 0.5).astype(int)  #is neccesary round up to compare predictions
    
    print ("Model Evaluation Statistics Accuracy - Area under the curve AUC.\nTraining : ")
    acc_train = accuracy_score(datased_prepared_encoded.train_target, y_train_pred) 
    #print("ACC=%f" % (acc))
    print'{}{}%'.format('Accuraccy score: ', round((acc_train*100), 2))
    auc_train = roc_auc_score(datased_prepared_encoded.train_target, y_train_pred) 
    #print("AUC=%f" % (auc))
    print'{}{}%'.format('ROC AUC score: ', round((auc_train*100), 2))
    #-------------------------------------------------------------------------#
    
    ################################ Testing ##################################
    print("test set result\n")
    logit_test = sm.Logit(datased_prepared_encoded.target_test, datased_prepared_encoded.test) 
    result_test = logit_test.fit()
    print("\n")
    print(result_test.summary())
    
    # Model Evaluation Statistics Accuracy - Area under the curve AUC
    y_test_pred = result_test.predict(datased_prepared_encoded.test) 
    y_test_pred = (y_test_pred > 0.5).astype(int) 
    
    print ("Model Evaluation Statistics Accuracy - Area under the curve AUC.\nTesting : ")
    acc_test = accuracy_score(datased_prepared_encoded.target_test, y_test_pred)
    print'{}{}%'.format('Accuraccy score: ', round((acc_test*100), 2))
    #print("\n ACC=%f" % (acc))
    auc_test = roc_auc_score(datased_prepared_encoded.target_test, y_test_pred)
    print'{}{}%'.format('ROC AUC score: ', round((auc_test*100), 2))
    #print("\n AUC=%f" % (auc))

    return Bunch(train_results = y_train_pred.copy(),
                 ACC_train = acc_train.copy(),
                 AUC_train = auc_train.copy(),
                 test_results = y_test_pred.copy(), 
                 ACC_test = acc_test.copy(),
                 AUC_test = auc_test.copy()
                 )

#=============================================================================#
#  2. lOGISTIC REGRESSION using sklearn.linear_model
#-----------------------------------------------------------------------------#
def log_regre_sklearn():    
    logr = LogisticRegression()
    logr.fit( datased_prepared_encoded.train, datased_prepared_encoded.train_target )
    results_train = logr.predict(datased_prepared_encoded.train)
    
    print ("\nModel Evaluation Statistics Accuracy - Area under the curve AUC.\nTraining : ")
    acc_train = accuracy_score(datased_prepared_encoded.train_target, results_train)
    print'{}{}%'.format('Accuraccy score: ', round((acc_train*100), 2))
    auc_train = roc_auc_score(datased_prepared_encoded.train_target, results_train)
    print'{}{}%'.format('ROC AUC score: ', round((auc_train*100), 2))
    
    logr.fit( datased_prepared_encoded.test, datased_prepared_encoded.target_test )
    results_test = logr.predict(datased_prepared_encoded.test)
    #results_test = (results_test > 0.5).astype(int) 
    
    print ("\nModel Evaluation Statistics Accuracy - Area under the curve AUC.\nTesting : ")
    acc_test = accuracy_score(datased_prepared_encoded.target_test, results_test)
    print'{}{}%'.format('Accuraccy score: ', round((acc_test*100), 2))
    auc_test = roc_auc_score(datased_prepared_encoded.target_test, results_test)
    print'{}{}%'.format('ROC AUC score: ', round((auc_test*100), 2))
    
    return Bunch(train_results = results_train.copy(),
                 ACC_train = acc_train.copy(),
                 AUC_train = auc_train.copy(),
                 test_results = results_test.copy(), 
                 ACC_test = acc_test.copy(),
                 AUC_test = auc_test.copy()
                 )

#=============================================================================#
#  2. DECISION TREE CLASSIFIER using sklearn
#-----------------------------------------------------------------------------#
def decision_tree_classifier(data,target):
    predictors = ['age','workclass','education','educational-num',
                  'marital-status', 'occupation','relationship','race','gender',
                  'capital-gain','capital-loss','hours-per-week', 'native-country']
    
    tree_count = 10 
    bag_proportion = 0.6 
    predictions = []
    
    sss = StratifiedShuffleSplit(target, 1, test_size=0.25, random_state=1) 
    for train_index, test_index in sss:
        train_data = data.iloc[train_index] 
        test_data = data.iloc[test_index]  
        for i in range(tree_count):
            bag = train_data.sample(frac=bag_proportion, replace = True, random_state=i)
            X_train, X_test = bag[predictors], test_data[predictors]
            y_train, y_test = bag["income"], test_data["income"]
            clf = DecisionTreeClassifier(random_state=1, min_samples_leaf=75) 
            clf.fit(X_train, y_train) 
            predictions.append(clf.predict_proba(X_test)[:,1])
    
    combined = np.sum(predictions, axis=0)/10 
    rounded = np.round(combined) # we have to round pretiction to the ceeling
    
    acc_test = accuracy_score(rounded, y_test)
    auc_test = roc_auc_score(rounded, y_test)
    
    print'{}{}%'.format('Accuraccy score: ', round(accuracy_score(rounded, y_test) * 100 , 2)) 
    print'{}{}%'.format('ROC AUC score: ', round(roc_auc_score(rounded, y_test)* 100 , 2)) 
    
    return Bunch(predic = predictions,
                 test_results = rounded, 
                 ACC_test = acc_test.copy(),
                 AUC_test = auc_test.copy()
                 )

#=============================================================================#
#  2. DECISION TREE REGRESSION using sklearn
#-----------------------------------------------------------------------------#
def decision_tree_regre(data,target):
    predictors = ['age','workclass','education','educational-num',
                  'marital-status', 'occupation','relationship','race','gender',
                  'capital-gain','capital-loss','hours-per-week', 'native-country']
    
    tree_count = 10 
    bag_proportion = 0.6 
    predictions = []
    
    sss = StratifiedShuffleSplit(target, 1, test_size=0.25, random_state=1) 
    for train_index, test_index in sss:
        train_data = data.iloc[train_index] 
        test_data = data.iloc[test_index]  
        for i in range(tree_count):
            bag = train_data.sample(frac=bag_proportion, replace = True, random_state=i)
            X_train, X_test = bag[predictors], test_data[predictors]
            y_train, y_test = bag["income"], test_data["income"]
            clf = DecisionTreeRegressor()
            clf.fit(X_train, y_train) 
            predictions.append(clf.predict(X_test))
    
    combined = np.sum(predictions, axis=0)/10 
    rounded = np.round(combined) # we have to round pretiction to the ceeling
    
    acc_test = accuracy_score(rounded, y_test)
    auc_test = roc_auc_score(rounded, y_test)
    
    print'{}{}%'.format('Accuraccy score: ', round(accuracy_score(rounded, y_test) * 100 , 2)) 
    print'{}{}%'.format('ROC AUC score: ', round(roc_auc_score(rounded, y_test)* 100 , 2)) 
    
    return Bunch(predic = predictions,
                 test_results = rounded, 
                 ACC_test = acc_test.copy(),
                 AUC_test = auc_test.copy()
                 )

###############################################################################
################################ MAIN #########################################
###############################################################################
#-----------------------------------------------------------------------------#
# 1. DATA PREPARATION AND VISUALIZATION 
#-----------------------------------------------------------------------------#
# 1.1. reading data and creatind dictonary to storte rawdata, target, columns etc.
dataset_raw = load_csv(filename) #ok tested

# 1.2. Checking RawData, iterating columns, printing unique values and counting them 
checking_data(dataset_raw.rawdata) #ok tested

# 1.3. Data Cleaning, deleting unnecesary information rows with ? symbols 
dataset_cleaned = clean_data(dataset_raw.rawdata) #ok tested

# 1.4. Raw data preparation 
dataset_prepared = prepare_data(dataset_cleaned.newdata) #ok tested

# 1.5. Converting Categorical variables to numeric
dataset_encoded = number_encode_features(dataset_cleaned.newdata) #ok tested

# 1.6. Data Vizualization to understand variables and numbers with pictures
visualize_data(dataset_encoded.num_data) # ok tested

# 1.7. Prepare data to Machine Learning Analysis
datased_prepared_encoded = prepare_data(dataset_encoded.num_data)

#-----------------------------------------------------------------------------#
# 2. MACHINE LEARNING 
#-----------------------------------------------------------------------------#
# 2.1 Correlation (plotting and generating correlation dataframe)
dataset_correlation = correlation(dataset_encoded.num_data)
print ("------Correlation Table-------------")
print (dataset_correlation.data_df_corr)

# 2.2. lOGISTIC REGRESSION statsmodels library
result_log_reg_sm = log_regre_statsmodels()

# 2.3 lOGISTIC REGRESSION ssklearn library
result_log_reg_sklearn = log_regre_sklearn()

# 2.4 DECISION TREE CLASSIFIER sklearn DecisionTreeClassifier library
result_decision_tree_clasif = decision_tree_classifier(datased_prepared_encoded.rawdata, datased_prepared_encoded.data_target)

# 2.5 DECISION TREE REGRESSION sklearn DecisionTreeRegressor library
result_decision_tree_regre = decision_tree_regre(datased_prepared_encoded.rawdata, datased_prepared_encoded.data_target)