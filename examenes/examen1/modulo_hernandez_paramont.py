#Utilidades
import sys
import warnings
import pickle
from IPython.display import HTML, display
from math import sin, cos, sqrt, atan2, radians
#bibliotecas para manejo de datos
import pandas as pd
import numpy as np
from scipy import stats

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

#bibliotecas para graficar
import plotly
import plotly.graph_objects as go
import plotly.express as px
import cufflinks as cf
import stylecloud
from PIL import Image
from plotly.offline import plot,iplot
pd.options.plotting.backend = "plotly"
cf.go_offline()
pd.set_option("display.max_columns",200)
import matplotlib.pyplot as plt


######################################################################################
#################################### Functions #######################################
######################################################################################

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

def z_score_outliers(df, cols):
    X_aux = df.copy()
    total_outliers = []
    indices = []
    for col in cols:
        z=np.abs(stats.zscore(X_aux[col],nan_policy='omit'))
        total_outliers.append(len(X_aux[[col]][(z>=3)]))
        indices.append(list((X_aux[[col]][(z>=3)].index)))
    
    results=pd.DataFrame()
    results["features"]=cols
    results["n_outliers"]=total_outliers
    results["n_outliers_%"]=round((results["n_outliers"]/len(X_aux)),2)
    results["indices"]=indices
    return results

def bestmodel_searchCV(X_train, y_train, model, params={}, cv=True,
                           k_fold=5, score='roc_auc', mode='grid', 
                           n_iter=10, random_state=0):
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


def df_cm(cm):
    df = pd.DataFrame({'Observacion_low':cm[0][:],
                       'Observacion_high':cm[1][:],},
                      index=['Prediccion_low','Prediccion_high'])
    return df

#metricas a partir de matriz de confusión
def metrics(cm):
    vp=cm[0][0]
    fp=cm[0][1]
    fn=cm[1][0]
    vn=cm[1][1]
    exactitud = ( vp+vn )/( vp+fp+fn+vn )
    print(f"   Exactitud : {exactitud:.3f}")
    
    precision=vp/(vp+fp)
    print(f"   Precision : {precision:.3f}")

    recall=(vp/(vp+fn))
    print(f"   Recall : {recall:.3f}")

    f1_score=((2*precision*recall)/(recall+precision))
    print(f"   f1_score : {f1_score:.3f}")
    
    TPR=recall
    print(f"   TPR : {TPR:.3f}")

    FPR=(fp/(fp+vn))
    print(f"   FPR : { FPR:.3f}")    
    
    return precision,recall,f1_score,TPR,FPR

def roc_fig(y_test, y_score):
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr,tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC={roc_auc})')
    plt.legend(loc="lower right")
    
    return plt.show()
    
    
def metrics_master_class(X_train, y_train, X_test, y_test, model):
    y_pred_train=model.predict(X_train)
    y_pred_test=model.predict(X_test)
    
    y_score_train=model.predict_proba(X_train)[:,1]
    y_score_test=model.predict_proba(X_test)[:,1]
    
    cm_train = confusion_matrix(y_train,y_pred_train)
    cm_test = confusion_matrix(y_test,y_pred_test)
    
    print("\033[1m Métricas Train \033[0m")
    metrics(cm_train)
    print('\n')
    print("\033[1m Métricas Test \033[0m")
    metrics(cm_test)
    print('\n')
    print("\033[1m Matríz de confusión de Train \033[0m")
    display(df_cm(cm_train))
    print('\n')
    print("\033[1m Matríz de confusión de Test \033[0m")
    display(df_cm(cm_test))
    print('\033[1m ROC train \033[0m')
    display(roc_fig(y_train, y_score_train))
    print('\033[1m ROC test \033[0m')
    display(roc_fig(y_test, y_score_test))

def estabilidad_class(X, model):
    X_aux = X.copy()
    X_aux['pred'] = model.predict(X)
    X_aux['proba'] = model.predict_proba(X)[:,1]
    
    estabilidad = pd.DataFrame()
    i=0
    while round(i,1) < 1:
        range_ = f'({round(i,1)},{round(i+0.1,1)}]'
        zero_count = X_aux['pred'][X_aux['pred']==0][X_aux['proba'].between(i,i+0.1)].count()
        one_count = X_aux['pred'][X_aux['pred']==1][X_aux['proba'].between(i,i+0.1)].count()
        aux = pd.DataFrame([[range_,zero_count,one_count]],columns=['Proba','low','high'])
        estabilidad=estabilidad.append(aux)
        i += 0.1
    return estabilidad

def metrics_regr(y_real, y_pred, n_rows, n_cols):
    mae=mean_absolute_error(y_real,y_pred)
    mse=mean_squared_error(y_real,y_pred)
    rmse=np.sqrt(mse)
    r2=r2_score(y_real,y_pred)
    r2_adj=1-(((n_rows-1)/(n_rows-n_cols-1)))*(1-r2)
    mape=mean_absolute_percentage_error(y_real,y_pred)
    
    print(f"MAE : {mae:.3f}")
    print(f"MSE : {mse:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"MAPE : {mape:.3f}")
    print(f"R2 : {r2:.3f}")
    print(f"R2 Adj: {r2_adj:.3f}")

def metrics_master_regr(X_train, y_train, X_test, y_test, model):
    y_pred_train=model.predict(X_train)
    y_pred_test=model.predict(X_test)
    
    n_rows_train = len(X_train)
    n_rows_test = len(X_test)
    n_cols = len(X_train.columns)
    
    print("\033[1m Métricas Train \033[0m")
    metrics_regr(y_train, y_pred_train, n_rows_train, n_cols)
    print('\n')
    print("\033[1m Métricas Test \033[0m")
    metrics_regr(y_test, y_pred_test, n_rows_test, n_cols)

def estabilidad_regr_aux(X, y, model, lower_model, upper_model, save, train=True):
    y_pred = model.predict(X)
    
    df_aux = pd.DataFrame()
    df_aux["true"]=y
    df_aux["pred"]=y_pred
    df_aux["lower_pred"]=lower_model.predict(X)
    df_aux["upper_pred"]=upper_model.predict(X)
    df_aux.sample(200).reset_index(drop=True).iplot()
    if save:
        mode = 'train' if train else 'test'
        df_aux.sample(200).to_csv(f"bandas_{model}_{mode}.csv")

def estabilidad_regr(X_train, y_train, X_test, y_test, model, save=True):
    lower_alpha=0.1
    upper_alpha=0.9
    lower_model = GradientBoostingRegressor(loss = "quantile",                    
                                            alpha = lower_alpha)  
    upper_model = GradientBoostingRegressor(loss = "quantile", 
                                            alpha = upper_alpha)
    lower_model.fit(X_train, y_train) 
    upper_model.fit(X_train, y_train)
    print("\033[1m Train \033[0m")
    estabilidad_regr_aux(X_train, y_train, model, lower_model, upper_model, save)
    print('\n')
    print("\033[1m Test \033[0m")
    estabilidad_regr_aux(X_test, y_test, model, lower_model, upper_model, save, train=False)
    


######################################################################################
############################## Transformer Classess ##################################
######################################################################################

class DeleteColumns(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_aux = X.copy()
        return X_aux.drop(columns=self.columns)

class LabelColumns(TransformerMixin):
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_aux = X.copy()
         #continuas
        c_feats = self.feat_dict['c_feats']
         #discretas
        v_feats = self.feat_dict['v_feats']         
        #fehcas
        d_feats = self.feat_dict['d_feats']
         #texto
        t_feats = self.feat_dict['t_feats']
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

class ToDatetime(TransformerMixin):
    def __init__(self, columns, format="%Y/%m/%d %H:%M:%S"):
        self.columns = columns
        self.format = format
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_aux = X.copy()
        for column in self.columns:
            X_aux[column] = pd.to_datetime(X_aux[column], infer_datetime_format=True)
        return X_aux

class DateInfoExtraction(TransformerMixin):
    def __init__(self,columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, prefix=False):
        X_aux = X.copy()
        for column in self.columns:
            pref = column if prefix else 'v'
            X_aux[pref+'_year'] = X_aux[column].dt.year
            X_aux[pref+'_quarter'] = X_aux[column].dt.quarter
            X_aux[pref+'_month'] = X_aux[column].dt.month
            X_aux[pref+'_semester'] = (X_aux[pref+'_month']>6).astype(int)+1
            X_aux[pref+'_day_of_month'] = X_aux[column].dt.day
            X_aux[pref+'_day_of_week'] = X_aux[column].dt.dayofweek
            X_aux[pref+'_weekend'] = (X_aux[pref+'_day_of_week'] >4).astype(int)
            X_aux = X_aux.drop(columns=column)
        return X_aux


from sklearn.base import BaseEstimator, TransformerMixin
class DistLonLat(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_aux = X.copy()
        
        def eucli_dist(list):
            # approximate radius of earth in km
            R = 6373.0
            lat1 = radians(list[0])
            lon1 = radians(list[1])
            lat2 = radians(list[2])
            lon2 = radians(list[3])
            dlon = lon2 - lon1
            dlat = lat2 - lat1

            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            distance = R * c
            return distance
        
        def manhattan_dist(list):
            # approximate radius of earth in km
            R = 6373.0
            lat1 = radians(list[0])
            lon1 = radians(list[1])
            lat2 = radians(list[2])
            lon2 = radians(list[3])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            #latitude
            a_lat = sin(dlat / 2)**2 
            c_lat = 2 * atan2(sqrt(a_lat), sqrt(1 - a_lat))
            distance_lat = R * c_lat
            #longitude
            a_lon = sin(dlon / 2)**2 
            c_lon = 2 * atan2(sqrt(a_lon), sqrt(1 - a_lon))
            distance_lon = R * c_lon
            
            return abs(distance_lat)+ abs(distance_lon)
            
        X_aux['c_dist_ecuclidian'] = X_aux[self.columns].apply(eucli_dist, axis=1)
        X_aux['c_dist_manhattan'] = X_aux[self.columns].apply(manhattan_dist, axis=1)
        return X_aux


class MyLabelEncoder(LabelEncoder):
    def __init__(self,column):
        self.column = column
        
    def fit(self, X, y=None):
        X_aux = X.copy()
        return super().fit(X_aux[self.column])
        
    
    def transform(self, X):
        X_aux = X.copy()
        X_aux[self.column] = super().transform(X_aux[self.column])
        return X_aux

class myOrdinalEncoder(TransformerMixin):
    def __init__(self, columns, categories):
        self.columns = columns
        self.enc = OrdinalEncoder(categories=categories)
         
    def fit(self, X, y=None):
        #X[self.columns] = X[self.columns].fillna('unknown')
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

######################################################################################
################################ Plot Functtions #####################################
######################################################################################


def my_histogram(df,col,bins,title="",x_title="",y_title="conteo"):
    """generates plotly histogram

    Parameters
    ----------
    df : pandas.DataFrame
        data frame to extract data from
    col : [string
        column from data frame to plot
    bins : int
        number of bins for histogram
    title : str, optional
        title of the plot, by default ""
    x_title : str, optional
        x axis title, by default ""
    y_title : str, optional
        y axis title, by default "conteo"

    Returns
    -------
    plotly figure
    """
    layout = go.Layout(font_family="Courier New, monospace",
        font_color="black",title_text=title,title_font_size=20,
        xaxis= {"title": {"text": x_title,"font": {"family": 'Courier New, monospace',"size":12,"color": '#002e4d'}}},
        yaxis= {"title": {"text": y_title,"font": {"family": 'Courier New, monospace',"size": 12, "color": '#002e4d'}}},               
        title_font_family="Arial",title_font_color="#002020",
        template="plotly_white", plot_bgcolor="rgb(168,168,168)")
    fig=df[[col]].iplot(kind='histogram',x=col,bins=bins,title=title,asFigure=True,layout=layout,sortbars=True,linecolor='#2b2b2b')
    fig.update_traces(marker_color='#045C8C',opacity=0.7)
    return fig

def my_bar_count(df,x,title="",x_title="",y_title=""):
    """ counts categories in the variable and generates plotly bar plot

    Parameters
    ----------
    df : pandas.DataFrame
        data frame to extract data from
    col : [string
        column from data frame to plot
    title : str, optional
        title of the plot, by default ""
    x_title : str, optional
        x axis title, by default ""
    y_title : str, optional
        y axis title, by default ""

    Returns
    -------
    plotly figure
    """
    layout = go.Layout(font_family="Courier New, monospace",
        font_color="black",title_text=title,title_font_size=20,
        xaxis= {"title": {"text": x_title,"font": {"family": 'Courier New, monospace',"size": 12, "color": '#002e4d'}}},
        yaxis= {"title": {"text": y_title,"font": {"family": 'Courier New, monospace',"size": 12, "color": '#002e4d'}}},
        title_font_family="Arial",title_font_color="#003030",
        template="plotly_white", plot_bgcolor="rgb(168,168,168)")
    aux=pd.DataFrame(df[x].value_counts()).reset_index().rename(columns={"index":"conteo"})
    fig=aux.iplot(kind='bar',x="conteo",y=x,title=title,asFigure=True,barmode="overlay",sortbars=True,color='#2b2b2b',layout=layout,width=5)
    fig.update_layout(width=800)
    fig.update_traces(marker_color='#045C8C',opacity=0.7)
    return fig

def my_bar(df,x,y,title="",x_title="",y_title=""):
    """ generates plotly bar plot

    Parameters
    ----------
    df : pandas.DataFrame
        data frame to extract data from
    x : string
        column that defines independent values (x axis) of the plot
    y : string
        column that defines dependent values (y axis) of the plot
    title : str, optional
        title of the plot, by default ""
    x_title : str, optional
        x axis title, by default ""
    y_title : str, optional
        y axis title, by default ""

    Returns
    -------
    plotly figure
    """
    layout = go.Layout(font_family="Courier New, monospace",
        font_color="black",title_text=title,title_font_size=20,
        xaxis= {"title": {"text": x_title,"font": {"family": 'Courier New, monospace',"size": 12, "color": '#002e4d'}}},
        yaxis= {"title": {"text": y_title,"font": {"family": 'Courier New, monospace',"size": 12, "color": '#002e4d'}}},
        title_font_family="Arial",title_font_color="#002020",
        template="plotly_white", plot_bgcolor="rgb(168,168,168)")
    fig=df.iplot(kind='bar',x=x,y=y,title=title,asFigure=True,barmode="overlay",sortbars=True,color='#2b2b2b',layout=layout,width=5)
    fig.update_layout(width=800)
    fig.update_traces(marker_color='#045C8C',opacity=0.7)
    return fig

def my_pie_count(df,col,title=""):
    """ counts categories in the variable and generates plotly pie plot

    Parameters
    ----------
    df : pandas.DataFrame
        data frame to extract data from
    col : string
        column from data frame to plot
    title : str, optional
        title of the plot, by default ""

    Returns
    -------
    plotly figure
    """
    layout = go.Layout(template="plotly_white")
    colors=['#4676d0','#95b0e4','#19293c','#6faa9f','#ccceb1','#344647','#02160f','#779a7c','#070919','#2b2b2b','#121212']
    aux=pd.DataFrame(df[col].value_counts()).reset_index().rename(columns={"index":"conteo"})
    fig=aux.iplot(kind='pie',labels='conteo',values=col,title=title,asFigure=True,theme="white")
    fig.update_traces(textfont_size=10,opacity=0.65,
                  marker=dict(colors=colors))
    fig.update_layout(font_family="Courier New, monospace",
    font_color="black",title_text=title,title_font_size=20,title_font_family="Arial",title_font_color="#002020",template="plotly_white")
    return fig

def my_pie(df,labels,values,title=""):
    """ generates plotly pie plot

    Parameters
    ----------
    df : pandas.DataFrame
        data frame to extract data from
    labels : string
        column that defines independent values (categories) of the plot
    values  : strings
        column that defines dependent values (quantity of categories) of the plot
    title : str, optional
        title of the plot, by default ""

    Returns
    -------
    plotly figure
    """
    layout = go.Layout(template="plotly_white")
    colors=['#4676d0','#95b0e4','#19293c','#6faa9f','#ccceb1','#344647','#02160f','#779a7c','#070919','#2b2b2b','#121212']*2
    fig=df.iplot(kind='pie',labels=labels,values=values,title=title,asFigure=True,theme="white")
    fig.update_traces(textfont_size=10,opacity=0.65,
                  marker=dict(colors=colors))
    fig.update_layout(font_family="Courier New, monospace",
    font_color="black",title_text=title,title_font_size=20,title_font_family="Arial",title_font_color="#002020",template="plotly_white")
    return fig

def my_box(df,columns,values,title="",x_title="",y_title=""):
    """ generates plotly box plot

    Parameters
    ----------
    df : pandas.DataFrame
        data frame to extract data from
    columns : string
        column that defines independent values (categories) of the plot
    values  : strings
        column that defines dependent values (values' distribution) of the plot
    title : str, optional
        title of the plot, by default ""
    x_title : str, optional
        x axis title, by default ""
    y_title : str, optional
        y axis title, by default ""

    Returns
    -------
    plotly figure
    """
    colors=['#4676d0','#19293c','#6faa9f','#ccceb1','#344647','#02160f','#779a7c','#070919','#2b2b2b','#121212']
    layout = go.Layout(font_family="Courier New, monospace",
        font_color="black",title_text=title,title_font_size=20,
        xaxis= {"title": {"text": x_title,"font": {"family": 'Courier New, monospace',"size": 12,"color": '#002e4d'}}},
        yaxis= {"title": {"text": y_title,"font": {"family": 'Courier New, monospace',"size": 12, "color": '#002e4d'}}},
        title_font_family="Arial",title_font_color="#002020",
        template="plotly_white", plot_bgcolor="rgb(208,208,208)")
    fig=df.pivot(columns=columns,values=values).iplot(kind='box',title=title,asFigure=True,theme="white",layout=layout,color=colors)
    return fig
