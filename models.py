from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from xgboost import XGBRegressor
from xgboost import XGBClassifier

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score



def LinRegSklearn(X, y, model=None, **kwargs):
    '''
    Works with regression task
    get X - features, y - label, model - model to use. LinReg if None
    
    returns predictions, r2, mse
    '''
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    
    if model is None:
        model = ElasticNet(**kwargs,
                      tol = 1e-10)
    if model == 'Tree':
        model = DecisionTreeRegressor()
    model.fit(X_train,y_train)
    
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared = False)
    
    return (y_pred, y_test, r2, rmse)


def LogRegSklearn(X, y, model=None, **kwargs):
    '''
    Works with classification task
    get X - features, y - label, model - model to use.
    
    returns predictions, r2, mse
    '''
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify=y)
    

    model = LogisticRegression(**kwargs,
                      tol = 1e-10)

    model.fit(X_train,y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    return (y_pred, y_test, metrics)
    

def find_best_utils(X_train, y_train, X_val, y_val, param, **kwargs):
    
    if param == 'alpha':
        param_ls = [0.1, 0.5, 1, 3, 3.1, 3.15, ]
    elif param == 'l1':
        param_ls = [0.01, 0.1,  1]
    elif param == 'normalize':
        param_ls = [False]
    elif param == 'max_iter':
        param_ls = [1000, 2000, 3000]
    
    param_dic = {}

    for p in param_ls:

        model = ElasticNet(**kwargs)
        model.fit(X_train,y_train)

        y_pred = model.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared = False)

        param_dic[p] = rmse
        
    best_param = sorted(param_dic.items(), key=lambda item: item[1])[0][0]
    return best_param


def find_best_params(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.3, random_state = 42)
    
    best_alpha = find_best_utils(X_train, y_train, X_val, y_val, 'alpha')
    
    best_l1 = find_best_utils(X_train, y_train, X_val, y_val, 'l1', alpha=best_alpha)
    
    best_norm = find_best_utils(X_train, y_train, X_val, y_val, 'normalize', alpha=best_alpha, 
                               l1_ratio = best_l1)
    
    best_max_iter = find_best_utils(X_train, y_train, X_val, y_val, 'max_iter', alpha=best_alpha, 
                               l1_ratio = best_l1, normalize = best_norm)
    
    model = ElasticNet(alpha=best_alpha, l1_ratio = best_l1, normalize = best_norm, max_iter = best_max_iter)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_val)

    rmse = mean_squared_error(y_val, y_pred, squared = False)

    
    return (best_alpha, best_l1, best_norm, best_max_iter)

def LinRegStatmodels(X,y):
    X = sm.add_constant(X)
    for iter in range(len(X.columns)+1):
        mod = sm.OLS(y, X)
        res = mod.fit()
    
        columns_significant = res.pvalues.index
        columns_not_significant = res.pvalues[(res.pvalues > 0.05)].index
        n_columns_not_significant = len(columns_not_significant)

        
        if n_columns_not_significant > 0:
            col_to_drop = columns_not_significant[-1]
            columns_significant = columns_significant.drop(col_to_drop)
            X = X[columns_significant]
        else:
            return res  

    return None   
    
    
def XGBReg(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    max_depth = [2, 3]
    colsample = [0.6, 0.7]
    n_estimators = [100, 200, 400, 600]
    eta = [0.1, 0.3, 0.5]
    score = {}

    for n in n_estimators:
        for md in max_depth:
            for e in eta:
                for cs in colsample:
                    model = XGBRegressor(n_estimators=n, max_depth=md, eta=e, colsample_bytree=cs)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score[(n,md,e,cs)] = mean_squared_error(y_pred,y_test, squared=False)

    
    opt_params, rmse = sorted(score.items(), key=lambda item: item[1])[0]
    model = XGBRegressor(n_estimators = opt_params[0], max_depth=opt_params[1], eta=opt_params[2], colsample_bytree=opt_params[3])
    model.fit(X_train, y_train)

    return opt_params, rmse, model

def XGBCls(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y)

    max_depth = [2, 3]
    colsample = [0.6, 0.7]
    n_estimators = [100, 200,400,  600]
    score = {}

    for n in n_estimators:
        for md in max_depth:
            for cs in colsample:
                model = XGBClassifier(n_estimators=n, max_depth=md, colsample_bytree=cs)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score[(n,md,cs)] = roc_auc_score(y_pred,y_test)

    
    opt_params, roc_auc = sorted(score.items(), key=lambda item: item[1])[0]
    model = XGBClassifier(n_estimators = opt_params[0], max_depth=opt_params[1], colsample_bytree=opt_params[2])
    model.fit(X_train, y_train)

    return opt_params, roc_auc, model

def RFReg(X, y, task = 'regression'):
    if task == 'regression':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y)


    max_depth = [None, 2, 3]
    min_samples_split = [2, 3, 4, 5]
    n_estimators = [100, 200, 300]
    min_samples_leaf = [1, 2, 3]
    score = {}

    for n in n_estimators:
        for md in max_depth:
            for msl in min_samples_leaf:
                for mss in min_samples_split:
                    if task == 'regression':
                        model = RandomForestRegressor(n_estimators=n, max_depth=md, min_samples_leaf=msl, min_samples_split=mss, n_jobs=-1)
                    else:
                        model = RandomForestClassifier(n_estimators=n, max_depth=md, min_samples_leaf=msl, min_samples_split=mss, n_jobs=-1)

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    if task == 'regression':
                        score[(n,md,msl,mss)] = mean_squared_error(y_pred,y_test, squared=False)
                    else:
                        score[(n,md,msl,mss)] = roc_auc_score(y_test, y_pred)

    if task == 'regression':
        rev = False
    else:
        rev = True

    opt_params, rmse = sorted(score.items(), key=lambda item: item[1], reverse = rev)[0]

    if task == 'regression':
        model = RandomForestRegressor(n_estimators=opt_params[0], max_depth=opt_params[1], min_samples_leaf=opt_params[2], min_samples_split=opt_params[3], n_jobs=-1)
    else:
        model = RandomForestClassifier(n_estimators=opt_params[0], max_depth=opt_params[1], min_samples_leaf=opt_params[2], min_samples_split=opt_params[3], n_jobs=-1)

    model.fit(X_train, y_train)

    return opt_params, rmse, model