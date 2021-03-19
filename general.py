'''
version=0.1
'''
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error

def validate(X, y, model1=None, model1_scores=None, model2=None, statement=None, n=10, cv=5, scoring=mean_absolute_error):
    """
    compare model2 against model1 or its scores 
    and return t-test
    """
    if model1:
        scores_first_model = np.array([])
    elif model1_scores is not None:        
        scores_first_model = model1_scores
    else:
        print('No model1 is specified')
        return
        
    scores_second_model = np.array([])
    for i in tqdm(range(n)):
#         fold = KFold(n_splits=cv, shuffle=True, random_state=i)
        
        if model1:
#             first_model_this_split = cross_val_score(estimator=model1,
#                                                      X=X, y=y, n_jobs=-1,
#                                                      cv=fold, scoring=scoring)
            
            first_model_this_split = trimmed_cross_val(estimator=model1, X=X, y=y,\
                                                       statement=statement, scoring=scoring, random_state=i)
            scores_first_model = np.append(scores_first_model,
                                         first_model_this_split)
        
        
#         second_model_this_split = cross_val_score(estimator=model2,
#                                                 X=X, y=y, n_jobs=-1,
#                                                 cv=fold, scoring=scoring)
        second_model_this_split = trimmed_cross_val(estimator=model2, X=X, y=y,\
                                                       statement=statement, scoring=scoring, random_state=i)
        scores_second_model = np.append(scores_second_model,
                                     second_model_this_split)
    return ttest_rel(scores_first_model, scores_second_model),\
            scores_first_model, scores_second_model

def write_submission_file(prediction, filename,
                          path_to_sample='../../data/medium/raw_data/sample_submission.csv'):
    submission = pd.read_csv(path_to_sample, index_col='id')
    
    submission['log_recommends'] = prediction
    submission.to_csv(filename)

def trim(data, std_range=6):
    n = std_range
    mean = data.mean()
    std = data.std()
#     data_trimmed = data[np.abs(data - mean) <= std * n]
#     data_outliers = data[np.abs(data - mean) > std * n]
    return (np.abs(data - mean) <= std * n)

def trimmed_cross_val(estimator,X,y,scoring=mean_absolute_error,n_splits=5,statement=None,random_state=17):
    """
    Evaluate a score by cross-validation. 
    The fit method will be performed on the entire train subset at each iteration,
    the predict method and scoring will be performed only for objects from test subset where statement is True
    
    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
    X : pandas.DataFrame
        The data to fit.
    y : pandas.Series
        The target variable to try to predict.
    scoring : callable 
        The scoring function of signature scoring(y_true,y_pred).
    statement : boolean numpy.array of shape equal to y.shape
        The mask showing the objects we want to evaluate estimator on.
    n_splits : int
        Number of folds for cross-validation
    random_state : int
        Random_state for KFold and StratifiedKFold    
    
    Returns
    -----------
    scores : array of float, shape=(n_splits,)
    
    """
    if statement is None:
        cv = KFold(n_splits=n_splits,shuffle=True,random_state=random_state)
        cv_iter = list(cv.split(X, y))
    else:
        cv = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=random_state)
        cv_iter = list(cv.split(X, statement))
    scores = []
    
    for train, test in cv_iter:
        estimator.fit(X[train],y[train])
        if statement is not None:
            y_statement = y[test][statement[test]]
            pred_statement = estimator.predict(X[test][statement[test]])
        else:
            y_statement = y[test]
            pred_statement = estimator.predict(X[test])
        scores.append(scoring(y_statement,pred_statement))
    return np.array(scores)