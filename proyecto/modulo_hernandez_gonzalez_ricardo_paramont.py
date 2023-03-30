import pandas as pd
import numpy as np

from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk import FreqDist
from textblob import TextBlob
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import plot_confusion_matrix, roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.base import clone
from sklearn.preprocessing import label_binarize

import re
import unicodedata
import nltk
import unicodedata


import matplotlib.pyplot as plt


#####################################################################################
#################################### Funciones ######################################
#####################################################################################

def completitud(df):
    """Checks percentage of non missing values.

    Parameters
    ----------
    df : pandas.DataFrame
        Data

    Returns
    -------
    pandas.DataFrame
        dataframe with the columns of:
            columna: column
            total: total number of missings
            completitud: percentage of non missing values
    """
    comp=pd.DataFrame(df.isnull().sum())
    comp.reset_index(inplace=True)
    comp=comp.rename(columns={"index":"columna",0:"total"})
    comp["completitud"]=round((1-comp["total"]/df.shape[0])*100, 3)
    comp=comp.sort_values(by="completitud",ascending=True)
    comp.reset_index(drop=True,inplace=True)
    return comp

def calc_vif(X):
    """Calculates the variance inflation factor for each column of the dataframe.

    Parameters
    ----------
    X : pandas.DataFrame
        Dataframe to be explored

    Returns
    -------
    pandas.DataFrame
        dataframe showing the vif of every column.
    """
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)


def bestmodel_search(X_train, y_train, model, params={}, cv=True,
                           k_fold=5, score='accuracy', mode='grid', 
                           n_iter=10, random_state=0):
    """searches for model with the best score provided within a grid of the models hyper
    parameters. The search can be grid search or random search with or without the use of
    cross validation.

    Parameters
    ----------
    X_train : pandas.DataFrame or matrix
        Training dataframe.
    y_train : pandas.Series
        Training true objective values.
    model : model to be trained
        type of regression or classification model.
    params : dict, optional
        hyperparameter options, by default {}
    cv : bool, optional
        checks if cross validation will be used, by default True
    k_fold : int, optional
        fols for cross validation, by default 5
    score : str, optional
        scoring value to evaluate models, by default 'accuracy'
    mode : str, optional
        by default 'grid'
        type of search. 
        'random': random search, 
        'grid': grid search
    n_iter : int, optional
        number of iterations to be run with random search, by default 10
    random_state : int, optional
        random state for random serach, by default 0

    Returns
    -------
    model
        model with the best scoring value

    Raises
    ------
    Error
        It'll raise an error if mode has other value than 'grid' or 'random'.
    """
    kwargs = dict(
        cv = StratifiedKFold(k_fold) if cv else [(slice(None), slice(None))], #given list disables cv (gives just one fold)
        verbose=True,
        scoring=score,
        estimator=model,
        n_jobs=-1,
    )

    if mode == 'grid':
        kwargs['param_grid']=params
        search = GridSearchCV(**kwargs)
    elif mode == 'random':
        kwargs['param_distributions']=params
        kwargs['random_state']=random_state
        kwargs['n_iter']=n_iter
        search = RandomizedSearchCV(**kwargs)
    else:
        raise Error("Enter valod mode: ['grid' , 'random']")

    search.fit(X_train,y_train)
    print(f"Best Score ({score}) :  {search.best_score_}")
    print(f"Best Params :  {search.best_params_}")
    return search.best_estimator_



def roc_plot(fpr, tpr, roc_auc, class_name):
    """plots roc curve using matplotlib

    Parameters
    ----------
    fpr : array like
        false positive rate, obtained from sklearn.metrics.roc_auc 
    tpr : array like
        tru positive rate, obtained from sklearn.metrics.roc_auc 
    roc_auc : number
        value obtained from sklearn.metrics.auc
    class_name : string
        name of the class compared in one vs rest
    """
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'One vs All, range: {class_name} per year')
    plt.legend(loc="lower right")
    plt.show()
    
def roc_plot_multiclass(X, y, model, class_dict):
    """makes use of roc_plot function to plot roc curve for every class in the objective
    in a one vs rest manner.

    Parameters
    ----------
    X : pd.DataFrame, matrix like
        Input values for the model.
    y : pd.Series, array-like
        True values of the objective.
    model : model
        model to be evaluated
    class_dict : dictionary
        dictionary containing the real name of the classes in the objective.
    """
    y_binarized = label_binarize(y, list(class_dict.keys()))
    y_score = model.predict_proba(X)
    for i in class_dict:
        fpr, tpr, roc_auc = roc_curve(y_binarized[:,i], y_score[:,i])
        roc_auc = auc(fpr, tpr)
        roc_plot(fpr, tpr, roc_auc, class_dict[i])
        
    
def metrics_master_multiclass(X_train, y_train, X_test, y_test, model, 
                              class_dict={
                                  0:'less than $85k',
                                  1:'between 85k and 125k',
                                  2:'more than 125k'
                              }
                             ):
    """displays metrics of the given model for its train and test datasets.

    Parameters
    ----------
    X_train : pd.DataFrame, matrix like
        training dataset
    y_train : pd.Series, array-like
        true values of the objective for training
    X_test : pd.DataFrame, matrix like
        testing dataset
    y_test : pd.Series, array-like
        true values of the objective for testing
    model : model
        model to be evaluated
    class_dict : dict, optional
        dictionary with classes of the objective, by default { 0:'less than k', 1:'between 85k and 125k', 2:'more than 125k' }
    """
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    print("\033[1m Métricas Train \033[0m")
    print(classification_report(y_train, y_pred_train, digits=3))
    print('\n')
    print("\033[1m Métricas Test \033[0m")
    print(classification_report(y_test, y_pred_test, digits=3))
    print('\n')
    
    print("\033[1m Matríz de confusión de Train \033[0m")
    plot_confusion_matrix(model, X_train, y_train)
    plt.show()
    print("\033[1m Matríz de confusión de Test \033[0m")
    plot_confusion_matrix(model, X_test, y_test)
    plt.show()
    
    print('\033[1m ROC train \033[0m')
    roc_plot_multiclass(X_train, y_train, model, class_dict)
    print('\033[1m ROC test \033[0m')
    roc_plot_multiclass(X_test, y_test, model, class_dict)

def stability_class(y_pred, y_prob, class_name):
    """shows prediction probability distribution for a certain class in a given model

    Parameters
    ----------
    y_pred : pd.Series, array-like
        predicted values of the objective
    y_prob : pd.Series, array-like
        predicted probability values of the objective
    class_name : string
        name of the class being evaluated.

    Returns
    -------
    pd.DataFrame
        DataFrame showing prediction probability distribution for the given class
    """
    X_aux = pd.DataFrame()
    X_aux['pred'] = y_pred
    X_aux['prob'] = y_prob
    
    stability = pd.DataFrame()
    i=0
    while round(i,1) < 1:
        range_ = f'({round(i,1)},{round(i+0.1,1)}]'
        zero_count = X_aux['pred'][X_aux['pred']==0][X_aux['prob'].between(i,i+0.1)].count()
        one_count = X_aux['pred'][X_aux['pred']==1][X_aux['prob'].between(i,i+0.1)].count()
        aux = pd.DataFrame([[range_,zero_count,one_count]],columns=['Prob','Other classes',f'{class_name}'])
        stability=stability.append(aux)
        i += 0.1
    return stability


def stability_multiclass(X, model, class_dict={0:'less than $85k',
                                               1:'between 85k and 125k',
                                               2:'more than 125k'}):
    """makes use of the stability_class function to show a dataframe with the prediction 
    probability distribution of each class of the objective, for a given mdel.


    Parameters
    ----------
    X : pandas.DataFrame
        input dataset for the model.
    model : model
        model to be evaluated
    class_dict : dict, optional
        dict with class names, by default {0:'less than k', 1:'between 85k and 125k', 2:'more than 125k'}
    """
    y_pred = label_binarize(model.predict(X), list(class_dict.keys()))
    y_prob = model.predict_proba(X)
    for i in class_dict:
        print(class_dict[i])
        display(stability_class(y_pred[:,i], y_prob[:,i],class_dict[i]))
        print('\n')
    


#####################################################################################
################################### Transformers ####################################
#####################################################################################

class ImportDF(TransformerMixin):
    """Imports dataset from csv file and transforms it to pandas.DataFrame
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, file_name):
        X_aux = pd.read_csv(file_name,na_values=[-1,'-1','Unknown / Non-Applicable'])
        X_aux = X_aux.drop(X_aux.columns[:2], axis=1)
        return X_aux

class LabelColumns(TransformerMixin):
    """Labels columns, specific for this problem.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_aux = X.copy()
         #continuas
        c_feats = []
         #discretas
        v_feats = ['job_title','rating','location','headquarters','size','founded','type_of_ownership','industry','sector','revenue','competitors','easy_apply','salary_estimate_source','salary_estimate']
         #fehcas
        d_feats = []
         #texto
        t_feats = ['job_description','company_name']
        X_aux.columns = X_aux.columns.str.replace(' ','_').map(str.lower)
        def label_columns(df,feats,prefix):
            feats_new=[prefix+x for x in feats]
            df=df.rename(columns=dict(zip(feats,feats_new)))
            return df
        X_aux = label_columns(X_aux,c_feats,"c_")
        X_aux = label_columns(X_aux,v_feats,"v_")
        X_aux = label_columns(X_aux,t_feats,"t_")
        X_aux = label_columns(X_aux,d_feats,"d_")
        return X_aux

class DeleteColumns(TransformerMixin):
    """Deletes indicated columns

    Parameters
    ----------
    columns : array-like of strings
        columns to be deleted
    """
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_aux = X.copy()
        X_aux = X_aux.drop(columns=self.columns)
        return X_aux

class CleanText(TransformerMixin):
    """Cleans text, transforming upper case  to lower case and deleting special characters.
    Cleaning is specific for this problem
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_aux = X.copy()
        def clean_text(text, pattern="[^a-zA-Z0-9 ]",replace=" "):
            cleaned_text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore')
            cleaned_text = re.sub(pattern, replace, cleaned_text.decode("utf-8"), flags=re.UNICODE)
            cleaned_text = u' '.join(cleaned_text.strip().lower().split())
            return cleaned_text
        X_aux['v_job_title'] = X_aux["v_job_title"].map(lambda x:clean_text(x.lower(), pattern="[^a-zA-Z ]",replace=""))
        X_aux['v_location'] = X_aux["v_location"].map(lambda x:clean_text(x.lower(), pattern="[^a-zA-Z, ]",replace=""))
        X_aux["v_headquarters"] = X_aux["v_headquarters"].map(lambda x:clean_text(x.lower(), pattern="[^a-zA-Z, ]",replace=""),na_action='ignore')
        X_aux['v_industry'] = X_aux['v_industry'].map(lambda x:clean_text(x.lower(), pattern="[^a-zA-Z ]",replace=""),na_action='ignore')
        X_aux['v_sector'] = X_aux['v_sector'].map(lambda x:clean_text(x.lower(), pattern="[^a-zA-Z ]",replace=""),na_action='ignore')
        X_aux['v_type_of_ownership'] = X_aux['v_type_of_ownership'].map(lambda x:clean_text(x.lower(), pattern="[^a-zA-Z-/ ]",replace=""),na_action='ignore')
        X_aux['v_size'] = X_aux['v_size'].map(lambda x:clean_text(x.lower(), pattern="[^a-zA-Z0-9 ]",replace=""),na_action='ignore')
        X_aux['t_company_name'] = X_aux['t_company_name'].map(lambda x:clean_text(x.lower(), pattern="[^a-zA-Z- ]",replace=""))
        X_aux['t_job_description'] = X_aux['t_job_description'].map(lambda x:clean_text(x.lower(), pattern="[^a-zA-Z ]",replace=""))
        return X_aux

class GetSalaryMean(TransformerMixin):
    """Transforms 'v_salary_estimate' column into a numerical value, the average of the 
    minimum and maximum estimates
    """
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_aux = X.copy()
        
        def transform_salary(text):
            nums = re.findall('(\d{1,3})', text)
            min_sal = int(nums[0])
            max_sal = int(nums[1])
            #Si el salario está dado por hora, se pasa a año considerando:
            #   trabajo tiempo completo 8h/día
            #   calendario laboral estadounidense de 2019 261dias/anno
            if 'Per Hour' in text:
                return (min_sal+max_sal)/2*8*261
            else:
                return (min_sal+max_sal)/2*1000
        
        X_aux['v_salary_estimate'] = X_aux['v_salary_estimate'].map(transform_salary)
        return X_aux

class CategorizeSalary(TransformerMixin):
    """Categorizes 'v_salary_estimate' in three categories.
    """
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_aux = X.copy()
        
        def categorize(num):
            if num < 85_000:
                return 0
            elif num < 125_000:
                return 1
            else:
                return 2
        X_aux['v_salary_estimate'] = X_aux['v_salary_estimate'].map(categorize)
        return X_aux

    def inverse_transform(self, y):
        y_aux = y.copy()

        def inverse_categorize(num):
            if num == 0:
                return "85,000 or less "
            elif num == 1:
                return "between 85,000, and 125,000"
            else:
                return "more than 1250,000"
        y_aux = y_aux.map(inverse_categorize)
        return y_aux

class HighRank(TransformerMixin):
    """Returns dummy column that indicates if the 'v_job_title' contains a word indicating
    if the job is of high rank
    """
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_aux = X.copy()
        search = "senior|manager|chief|supervisor|director|executive|boss|officer|specialist|lead|principal"
        X_aux['v_is_high_rank'] = X_aux['v_job_title'].str.contains(search).astype(int)
        return X_aux

class NormalizeJobTitle(TransformerMixin):
    """Normalizes the column 'v_jov_title', to reduce the number of unique values
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_aux = X.copy()
        
        #Normalización con el uso de regex
        X_aux["v_job_title"] = X_aux["v_job_title"].str.replace(r'(^.*data.*scien.*$)', 'data scientist')
        X_aux["v_job_title"] = X_aux["v_job_title"].str.replace(r'(^.*data.*engin.*$)', 'data engineer')
        X_aux["v_job_title"] = X_aux["v_job_title"].str.replace(r'(^.*data.*anal.*$)', 'data analyst')
        X_aux["v_job_title"] = X_aux["v_job_title"].str.replace(r'(^.*machine.*learning.*$)', 'machine learning professional')
        X_aux["v_job_title"] = X_aux["v_job_title"].str.replace(r'(^.*business.* anal.*$)', 'business intelligence analyst')
        X_aux["v_job_title"] = X_aux["v_job_title"].str.replace(r'(^((?!data).)*scientist.*$)', 'specific discipline scientist')
        X_aux["v_job_title"] = X_aux["v_job_title"].str.replace(r'(^((?!(data)|(business)).)*anal.*$)', 'analyst of other nature')
        X_aux["v_job_title"] = X_aux["v_job_title"].str.replace(r'(^.*data.*architect.*$)', 'data architect')
        X_aux["v_job_title"] = X_aux["v_job_title"].str.replace(r'(^.*cybersecurity.*$)', 'cybersecurity applied to data science')
        #Separando a los trabajos que no pudieron ser agrupados en las categorias de arriba
        # y que tinen una sola ocurrencia
        counts = pd.DataFrame(X_aux["v_job_title"].value_counts())
        one_occurrence = counts[counts['v_job_title']==1].index
        dictio = dict(zip(one_occurrence,["highly specific"]*len(one_occurrence)))
        X_aux["v_job_title"] = X_aux["v_job_title"].replace(dictio)
        #Separando categorias menored en otros
        dictio = dict(zip(list(X_aux["v_job_title"].value_counts().index.tolist()[-9:]),["others"]*(9)))
        X_aux["v_job_title"] = X_aux["v_job_title"].replace(dictio)
        
        return X_aux

class NormalizeLocation(TransformerMixin):
    """divides location column in a city and state column

    Parameters
    ----------
    location : string
        column to be transformed
    """
    def __init__(self, location, prefix=False):
        #location is the columns that contains location
        self.location = location
        self.prefix = prefix
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_aux = X.copy()
        if self.prefix:
            prefix = self.location+'_'
        else:
            prefix = 'v_'
        
        X_aux[[prefix+'city',prefix+'state']] = X_aux[self.location].str.split(', ',1, expand=True)
        X_aux = X_aux.drop(columns=self.location)
        return X_aux

class NormalizeToOthers(TransformerMixin):
    """Normalizes column, grouping categories with less than given thresshold into
    the category 'others'.

    Parameters
    ----------
    column_threshold : tuple: (string, int)
        tupple with column name and threshold number
    """
    def __init__(self, column_threshold):
        self.column_threshold = column_threshold
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_aux = X.copy()
        
        for column, threshold in self.column_threshold:
            #Agrupamos los que tienen valores iguales o menores al threshold
            # dentro de la categoria others.
            counts = pd.DataFrame(X_aux[column].value_counts())
            low_occurrence = counts[counts[column]<=threshold].index
            dictio = dict(zip(low_occurrence,["others"]*len(low_occurrence)))
            X_aux[column] = X_aux[column].replace(dictio)
            
        return X_aux

from sklearn.impute import SimpleImputer

class mySimpleImputer(SimpleImputer):
    """imputes column with given strategy

    Parameters
    ----------
    columns : list of string
        list with columns to be imputed
    strategy: string
        allowed strategy to impute column
    missing_values: miss value
        missing values in the columns
    """
    def __init__(self, columns, strategy, missing_values=np.nan):
        self.columns = columns
        super().__init__(missing_values=missing_values, strategy=strategy)
    
    def fit(self, X, y=None):
        return super().fit(X[self.columns])
    
    def transform(self, X):
        X_aux = X.copy()
        X_aux = X_aux.fillna(value=np.nan) # cambiando valores como None a np.nan
        X_aux[self.columns] = super().transform(X_aux[self.columns])
        return X_aux

class CompetitorsTransformer(TransformerMixin):
    """Transforms given column into dummy with 1 if column contains value other than 0,
    returns extra column with number of competitors

    Parameters
    ----------
    column : string
        column to be transformed
    """
    def __init__(self, column):
        self.column = column
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_aux = X.copy()
        X_competitors = X_aux[self.column]
        X_aux[self.column+'_count'] = X_competitors[~X_competitors.isnull()].map(lambda cell:len(cell.split(', ')))
        X_aux[self.column+'_count'] = X_aux[self.column+'_count'].fillna(0)
        X_aux[self.column] = (~X_competitors.isnull()).astype(int)
        return X_aux

class BigCity(TransformerMixin):
    """returns dummy column 'v_big_city' if the location column contains one of the biggest
    cities in the US

    Parameters
    ----------
    column : string
        city column
    """
    def __init__(self, column):
        self.column = column
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_aux = X.copy()
        top_20_city = list(map(lambda city:city.lower(),
                ['New York','Los Angeles','Chicago','Houston','Phoenix',
                 'Philadelphia','San Antonio','San Diego','Dallas','San Jose',
                 'Austin','Jacksonville','Fort Worth','Columbus','Charlotte',
                 'San Francisco','Indianapolis','Seattle','Denver','Washington']))
        X_aux['v_big_city'] = X_aux[self.column].isin(top_20_city).astype(int)
        return X_aux

class BigTech(TransformerMixin):
    """returns dummy column 'v_is_big_tech' if the 'v_company_name' contains one of the 
    big tech companies
    """
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_aux = X.copy()
        search = """
        facebook|amazon| aws |apple|netflix|google|microsoft|ibm|spotify|tesla|uber|
        twitter|alphabet|visa|nvidia|intel|adobe|cisco|at&t|oracle|airbnb|samsung|
        foxconn|huawei|dell|sony| hp | lg |lenovo
        """
        X_aux['v_is_big_tech'] = X_aux['t_company_name'].str.contains(search).astype(int)
        return X_aux

class myOneHotEncoder(TransformerMixin):
    """uses sklearn OneHotEncoder class to encode given columns

    Parameters
    ----------
    columns : list of strings
        columns to be encoded
    """
    def __init__(self, columns, sparse=False, handle_unknown='ignore'):
        self.columns = columns
        self.enc = OneHotEncoder(sparse=sparse, handle_unknown=handle_unknown)
         
    def fit(self, X, y=None):
        return self.enc.fit(X[self.columns])
    
    def transform(self, X):
        X_aux = X.copy()
        enc_df = pd.DataFrame(
            self.enc.transform(X_aux[self.columns]),
            columns=self.enc.get_feature_names()
        )
        X_aux = X_aux.reset_index(drop=True).join(enc_df)
        X_aux = X_aux.drop(columns = self.columns)
        return X_aux

class myOrdinalEncoder(TransformerMixin):
    """uses sklearn OrdinalEncoder class to encode given columns

    Parameters
    ----------
    columns : list of strings
        columns to be encoded
    """
    def __init__(self, columns, categories):
        self.columns = columns
        self.enc = OrdinalEncoder(categories=categories)
         
    def fit(self, X, y=None):
        X[self.columns] = X[self.columns].fillna('unknown')
        return self.enc.fit(X[self.columns])
    
    def transform(self, X):
        X_aux = X.copy()
        X_aux[self.columns] = X_aux[self.columns].fillna('unknown')
        enc_df = pd.DataFrame(
            self.enc.transform(X_aux[self.columns]),
            columns=self.columns
        )
        X_aux = X_aux.drop(columns=self.columns)
        X_aux = X_aux.reset_index(drop=True).join(enc_df)
        return X_aux

class TextLength(TransformerMixin):
    """Returns length of the text value depending on the number or characters or words
    (words are counted as strings separated by blanks)

    Parameters
    ----------
    columns : list of strings
        columns to meassure
    prefix: boolean
        checks if prefix should be added to the columns of length
    """
    def __init__(self, columns, prefix=True):
        self.columns = columns
        self.prefix = prefix
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_aux = X.copy()
        for column in self.columns:
            pref = column if self.prefix else 'v'
            X_aux[pref+'_char_length'] = X_aux[column].map(len)
            X_aux[pref+'_word_length'] = X_aux[column].map(lambda text:len(text.split()))
        return X_aux

class DeleteStopwords(TransformerMixin):
    """Deletes stopwords of given column

    Parameters
    ----------
    column : string
        text column to be cleaned, default='t_texto'
    """
    def __init__(self, column='t_texto', lang='english'):
        self.column = column
        self.lang = lang
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_aux = X.copy()
        self.stop_words = nltk.corpus.stopwords.words(self.lang)
        X_aux[self.column] = X_aux[self.column].map(
            lambda text:" ".join([x for x in text.split(" ") if x not in self.stop_words]))
        return X_aux

class MyCountVectorizer(TransformerMixin):
    """uses sklearn CountVectorizer class to vectorize text column given

    Parameters
    ----------
    column : string
        column to be vectorized
    min_df : number
        minimum number of occurence of the words to be vectorized
    """
    def __init__(self,column,min_df):
        self.column = column
        #min_df controla minimo de ocurrencia porcentual de la palabra
        self.model = CountVectorizer(min_df=min_df)
        
    def fit(self, X, y=None):
        return self.model.fit(X[self.column])
    
    def transform(self, X):
        X_aux = X.copy()
        array_aux = self.model.transform(X_aux[self.column])
        column_names = ['vect_'+s for s in self.model.get_feature_names()]
        df_aux = pd.DataFrame(array_aux.toarray(),columns=column_names)
        #insertando vectorización en array
        X_aux = X_aux.reset_index(drop=True)
        X_aux = pd.concat([X_aux,df_aux],axis=1)
        #Se elimina texto
        X_aux=X_aux.drop(columns=self.column)
        return X_aux