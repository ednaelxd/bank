import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
# from sklearn.model_selection import StratifiedKFold

from imblearn.over_sampling import SMOTE#, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import matplotlib.pyplot as plt

import seaborn as sns
import locale

pd.set_option('max_colwidth', 400)
pd.set_option('display.max_rows', 500)

# decision tree evaluated on imbalanced dataset with SMOTE oversampling
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
# define pipeline
steps = [('over', SMOTE()), ('model', DecisionTreeClassifier())]
pipeline = Pipeline(steps=steps)
# evaluate pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='f1_micro', cv=cv, n_jobs=-1)
print('Mean ROC AUC: %.3f' % mean(scores))

def train_and_search(model, param_grid, X_train, y_train, metric="f1_micro", seed=42, search_type="grid", n_jobs=-1, verbose=10, 
                     sampler="SMOTE", n_repeats=3):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=n_repeats, random_state=seed)
    # kf = KFold(n_splits=10, random_state=42, shuffle=True) 
    # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    # using a specif pipeline for unbalanced datasets and undersampling of the most frequent classes
    imba_pipeline = Pipeline(steps=[('sampler', SMOTE(random_state=seed)),
                                    ('model', model),
                                    ('cross-validate', cross_val_score),
                                    (v)])

    new_param_grid = {'model__' + key: param_grid[key] for key in param_grid}

    params = {"estimator": imba_pipeline,
              "param_grid": new_param_grid,
              "scoring": metric,
              "n_jobs": n_jobs,
              "verbose": verbose,
              "return_train_score": True,
              "cv": cv
            }

    if search_type == "grid":
        search = GridSearchCV(**params)
    else:
        search = RandomizedSearchCV(**params)

    search.fit(X_train, y_train)
    print(f'Best parameters: {search.best_params_}')
    results = pd.DataFrame(search.cv_results_)
    return search, results

def plot_results(gscv, index='dar__ordar', columns='dar__ordriv'):
    """Select two hyperparameters from which we plot the fluctuations"""
    index = 'param_' + 'model__' + index
    columns = 'param_' + 'model__' + columns

    # prepare the results into a pandas.DataFrame
    df = pd.DataFrame(gscv.cv_results_)

    # Remove the other by selecting their best values (from gscv.best_params_)
    other = [c for c in df.columns if c[:6] == 'param_']
    other.remove(index)
    other.remove(columns)
    for col in other:
        df = df[df[col] == gscv.best_params_[col[6:]]]

    # Create pivot tables for easy plotting
    table_mean = df.pivot_table(index=index, columns=columns,
                                values=['mean_test_score'])
    table_std = df.pivot_table(index=index, columns=columns,
                               values=['std_test_score'])

    # plot the pivot tables
    plt.figure()
    ax = plt.gca()
    for col_mean, col_std in zip(table_mean.columns, table_std.columns):
        table_mean[col_mean].plot(ax=ax, yerr=table_std[col_std], marker='o',
                                  label=col_mean)
    plt.title('Grid-search results (higher is better)')
    plt.ylabel('log-likelihood compared to an AR(0)')
    plt.legend(title=table_mean.columns.names)
    plt.show()
    

def handle_param_names(best_params):
    return  {key[len('model__'):]: best_params[key] for key in best_params.keys()}

def load_data():
    df = pd.read_csv("../dataset/bank-preprocessed.csv")
    return df

def show_value_counts(serie, value_counted=False, dask=False, column_desc=None, grain='Rows', 
                      size=None, total=None, title=None, height=10, width=5, index=None, save_to=False, ax=None):
    fig = plt.figure()
    fig.set_size_inches(width, height)

    if ax is None:
        ax = plt.subplot(1,1,1)

    if not value_counted:
        serie = serie.value_counts()
    
    if dask:
        serie = serie.compute()
        
    serie = serie.sort_values(ascending=True)

    if not total:
        total = serie.sum()
    
    corte = ''
    
    if (index):
        serie = serie.rename(index)
    
    if serie.index.dtype != 'object':
        if serie.index.dtype == 'float64':
            serie.index = serie.index.map(int)
        serie.index = serie.index.map(str)
    serie.index = serie.index.map(str)
    
    if size and len(serie) > size:
        serie = serie.sort_values(ascending=False)
        serie = serie[:size]
        serie = serie.sort_values(ascending=True)
        corte = ' ({} mais frequentes)'.format(size)
    
    if not title:
        if column_desc:
            column = column_desc
        else:
            column = serie.name
        title = "{} by {}{}".format(grain, column, corte)
   
    ax.barh(serie.index, serie, align='center', color='c', ecolor='black')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    percentage = serie/total*100
    number_distance = serie.max()*0.005
    
    for i, v in enumerate(serie):
        pct = locale.format_string('%.2f', percentage[i], True)
        ax.text(v+number_distance , i-0.2, '{0:,} ({1}%)'.format(v, pct), color='k')
    ax.set(title=title,
           xlabel='',
           ylabel='')
    sns.despine(left=True, bottom=True)

    if save_to:
        plt.savefig(save_to)
    # plt.show()

def plot_grid(functions_and_parameters, n_cols=3, values_mapping=None, width_scale=5.5, height_scale=4):
    Tot = len(functions_and_parameters)
    # Compute Rows required
    n_rows = Tot // n_cols
    n_rows += Tot % n_cols
    fig = plt.figure(1)
    fig.set_figwidth(n_cols*width_scale)
    fig.set_figheight(n_rows*height_scale)
    for position, functions_and_parameter in enumerate(functions_and_parameters):
        ax = fig.add_subplot(n_rows, n_cols, position+1)
        kwargs = {'ax': ax}
        function = functions_and_parameter[0]
        parameters = functions_and_parameter[1]
        function(**parameters, **kwargs)
    plt.show()


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_cf_matrix(y_test, y_pred):
    cm=confusion_matrix(y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    f, ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="black", ax=ax, cmap="YlGnBu",
                fmt='.2f')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show(block=False)