#%%
import pandas as pd
import numpy as np
from PlotsUtils import plot_model


#%%
def OLSRegression(X_train, y_train, X_test, y_test, plot_test=True):
    """
    Función para entrenar y validar un modelo de regresión ordinaria
    """

    # 1. Librerias
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error

    # 2. Train Model-------------------------------------------
    ols_model = LinearRegression()
    ols_model.fit(X_train, y_train)
    
    y_pred_train = ols_model.predict(X_train)
    
    print("\nValidation model with train data")
    print("     Rsquare: {:.2f}".format(ols_model.score(X_train, y_train)))
    print("     Mean Square Error: {:.2f}".format(mean_squared_error(y_train, y_pred_train)))
    print("     Root Mean Square Error: {:.2f}".format(np.sqrt(mean_squared_error(y_train, y_pred_train))))

    y_pred_test = ols_model.predict(X_test)
    print("\nValidation model with test data")
    print("     Rsquare: {:.2f}".format(ols_model.score(X_test, y_test)))
    print("     Mean Square Error: {:.2f}".format(mean_squared_error(y_test, y_pred_test)))
    print("     Root Mean Square Error: {:.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred_test))))
        
    ols_coef = pd.DataFrame({'Variable': list(X_train.columns), 'Coeficiente': ols_model.coef_})
    ols_coef['Signo'] = np.where(ols_coef.Coeficiente > 0, 'Positivo', 'Negatigo')
    ols_coef['Coeficiente'] = np.absolute(ols_coef.Coeficiente)

     # 3. Analizar predicciones y residuales---------------------------------
    
    print("-"*80)
    print("\nAnálisis de las predicciones y los residuales")

    if(plot_test):
        plot_model(y_test, y_pred_test)
    else:
        plot_model(y_train, y_pred_train)
    
    
    return(ols_model, ols_coef)


#%%
def RidgeRegression(X_train, y_train, X_test, y_test, alphas, plot_test=True):
    """
    Función para entrenar y validar modelo Ridge
    """
    
    # 1. Libraries
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.metrics import mean_squared_error, r2_score
    #from sklearn.model_selection import KFold, RepeatedKFold

    # 2. Cross Validation------------------------------------------------
    # Configurar la validación cruzada By default, it performs Leave-One-Out Cross-Validation
    ridgecv = RidgeCV(alphas=alphas)
    ridgecv.fit(X_train, y_train)

    # 3. Train Ridge Model------------------------------------------
    
    ridge = Ridge(alpha=ridgecv.alpha_)
    #ridge.set_params()
    
    print("-"*80)
    print("Model Parameter")

    print("     Alpha: {:.2f}".format(ridgecv.alpha_))
    
    ridge.fit(X_train, y_train)
    y_pred_train = ridge.predict(X_train)
    
    print("\nValidation model with train data")
    print("     Rsquare: {:.2f}".format(ridge.score(X_train, y_train)))
    print("     Mean Square Error: {:.2f}".format(mean_squared_error(y_train, y_pred_train)))
    print("     Root Mean Square Error: {:.2f}".format(np.sqrt(mean_squared_error(y_train, y_pred_train))))

    y_pred_test = ridge.predict(X_test)
    print("\nValidation model with test data")
    print("     Rsquare: {:.2f}".format(ridge.score(X_test, y_test)))
    print("     Mean Square Error: {:.2f}".format(mean_squared_error(y_test, y_pred_test)))
    print("     Root Mean Square Error: {:.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred_test))))
        
    ridge_coef = pd.DataFrame({'Variable': list(X_train.columns), 'Coeficiente': ridge.coef_})
    ridge_coef['Signo'] = np.where(ridge_coef.Coeficiente > 0, 'Positivo', 'Negatigo')
    ridge_coef['Coeficiente'] = np.absolute(ridge_coef.Coeficiente)
    
    #ridge_coef.sort_values(by='Coeficiente', ascending=False)

    # 4. Analizar predicciones y residuales---------------------------------
    
    print("-"*80)
    print("\nAnálisis de las predicciones y los residuales")

    if(plot_test):
        plot_model(y_test, y_pred_test)
    else:
        plot_model(y_train, y_pred_train)
    
    
    return(ridge, ridge_coef)

#%%
def LassoRegression(X_train, y_train, X_test, y_test, alphas, plot_test=True):
    """
    Función para entrenar modelo Lasso
    """
    
    # 1. Libraries
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.metrics import mean_squared_error, r2_score
    #from sklearn.model_selection import KFold, RepeatedKFold

    # 2. Cross Validation------------------------------------------------
    # Configurar la validación cruzada By default, it performs Leave-One-Out Cross-Validation
    lassocv = LassoCV(alphas=alphas)
    lassocv.fit(X_train, y_train)

    # 3. Train Ridge Model------------------------------------------
    
    lasso = Lasso(alpha=lassocv.alpha_)
    #ridge.set_params()
    
    print("-"*80)
    print("Model Parameter")

    print("     Alpha: {:.2f}".format(lassocv.alpha_))
    
    lasso.fit(X_train, y_train)
    y_pred_train = lasso.predict(X_train)
    
    print("\nValidation model with train data")
    print("     Rsquare: {:.2f}".format(lasso.score(X_train, y_train)))
    print("     Mean Square Error: {:.2f}".format(mean_squared_error(y_train, y_pred_train)))
    print("     Root Mean Square Error: {:.2f}".format(np.sqrt(mean_squared_error(y_train, y_pred_train))))

    y_pred_test = lasso.predict(X_test)
    print("\nValidation model with test data")
    print("     Rsquare: {:.2f}".format(lasso.score(X_test, y_test)))
    print("     Mean Square Error: {:.2f}".format(mean_squared_error(y_test, y_pred_test)))
    print("     Root Mean Square Error: {:.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred_test))))
        
    lasso_coef = pd.DataFrame({'Variable': list(X_train.columns), 'Coeficiente': lasso.coef_})
    lasso_coef['Signo'] = np.where(lasso_coef.Coeficiente > 0, 'Positivo', 'Negatigo')
    lasso_coef['Coeficiente'] = np.absolute(lasso_coef.Coeficiente)
    
    #ridge_coef.sort_values(by='Coeficiente', ascending=False)

    # 4. Analizar predicciones y residuales---------------------------------
    
    print("-"*80)
    print("\nAnálisis de las predicciones y los residuales")

    if(plot_test):
        plot_model(y_test, y_pred_test)
    else:
        plot_model(y_train, y_pred_train)
 
    return(lasso, lasso_coef)

#%%
def CARTRegressor(X_train, y_train, X_test, y_test, plot_test=True):
    """
    Function for train decision tree model
    """
    # 1. Libraries
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score

    # 2. Hypterparameter tunning
    cart_model = DecisionTreeRegressor()
    cart_dict_parameter = {
        "max_depth":[2,4,6,8,10],
        "min_samples_leaf":[1,2,4,6,8,10]
        }
    
    # Search the best hyperparameter
    cart_tuning = GridSearchCV(cart_model, cart_dict_parameter, cv=5)
    cart_tuning.fit(X_train, y_train)

    # 3. Train tree model with the best hyperparamter
    cart_model.set_params(**cart_tuning.best_params_)
    cart_model.fit(X_train, y_train)
    y_pred_train = cart_model.predict(X_train)
    
    print("-"*80)
    print("Model Parameter: \n")
    print("{}".format(cart_model.get_params))
    
    print("\nValidation model with train data")
    print("     Rsquare: {:.2f}".format(cart_model.score(X_train, y_train)))
    print("     Mean Square Error: {:.2f}".format(mean_squared_error(y_train, y_pred_train)))
    print("     Root Mean Square Error: {:.2f}".format(np.sqrt(mean_squared_error(y_train, y_pred_train))))

    y_pred_test = cart_model.predict(X_test)
    print("\nValidation model with test data")
    print("     Rsquare: {:.2f}".format(cart_model.score(X_test, y_test)))
    print("     Mean Square Error: {:.2f}".format(mean_squared_error(y_test, y_pred_test)))
    print("     Root Mean Square Error: {:.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred_test))))

    # 4. Analizar predicciones y residuales---------------------------------
    
    print("-"*80)
    print("\nAnálisis de las predicciones y los residuales")

    if(plot_test):
        plot_model(y_test, y_pred_test)
    else:
        plot_model(y_train, y_pred_train)
        
    return(cart_model)

#%%
def RandForestRegressor(X_train, y_train, X_test, y_test, parameter=None, hyper_tunning=False, plot_test=True):
    """
    Function for train Random Forest Regressión
    """
    # 1. Libraries
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score

    # 2. Hyperparameter tunning
        
    rf_model = RandomForestRegressor()
    if(hyper_tunning):
        rf_dist = {"max_depth": [3,6,9,11],
               "max_features": [3,6,9,11],
               "n_estimators": [100,300,500,1000]}
        rf_tunning = GridSearchCV(rf_model, rf_dist, cv=5)
        rf_tunning.fit(X_train, y_train)
        rf_model.set_params(**rf_tunning.best_params_)
    else:
        rf_model.set_params(**parameter)

    # 3. Train Random Forest with the best hyperparameter
    
    rf_model.fit(X_train, y_train)
    y_pred_train = rf_model.predict(X_train)

    print("-"*80)
    print("Model Parameter: \n")
    print("{}".format(rf_model.get_params))
    
    print("\nValidation model with train data")
    print("     Rsquare: {:.2f}".format(rf_model.score(X_train, y_train)))
    print("     Mean Square Error: {:.2f}".format(mean_squared_error(y_train, y_pred_train)))
    print("     Root Mean Square Error: {:.2f}".format(np.sqrt(mean_squared_error(y_train, y_pred_train))))

    y_pred_test = rf_model.predict(X_test)
    print("\nValidation model with test data")
    print("     Rsquare: {:.2f}".format(rf_model.score(X_test, y_test)))
    print("     Mean Square Error: {:.2f}".format(mean_squared_error(y_test, y_pred_test)))
    print("     Root Mean Square Error: {:.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred_test))))

    # 4. Analizar predicciones y residuales---------------------------------
    
    print("-"*80)
    print("\nAnálisis de las predicciones y los residuales")

    if(plot_test):
        plot_model(y_test, y_pred_test)
    else:
        plot_model(y_train, y_pred_train)
        
    return(rf_model)


#%%
def XgbRegressor(X_train, y_train, X_test, y_test, parameter=None, hyper_tunning=False, plot_test=True):

    # 1. Libraries
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score


    # 2. Hyperparameter tunning
    xgb_model = xgb.XGBRegressor(booster = 'gbtree', objective='reg:squarederror', n_jobs=1)
    
    if(hyper_tunning):
        xgb_tunning = GridSearchCV(xgb_model,
                                    {'max_depth': [2,3,5],
                                     'n_estimators': [10, 30, 50],
                                     'learning_rate': [0.05, 0.07, 0.09]}, 
                                     verbose=1, n_jobs=1, cv=5)
        xgb_tunning.fit(X_train, y_train)
        xgb_model.set_params(**xgb_tunning.best_params_)
    else:
        xgb_model.set_params(**parameter)

    # 3. Train Extreme Gradient Boosting with the best hyperparameter
    
    xgb_model.fit(X_train, y_train)
    y_pred_train = xgb_model.predict(X_train)

    print("-"*80)
    print("Model Parameter: \n")
    print("{}".format(xgb_model.get_params))
    
    print("\nValidation model with train data")
    print("     Rsquare: {:.2f}".format(xgb_model.score(X_train, y_train)))
    print("     Mean Square Error: {:.2f}".format(mean_squared_error(y_train, y_pred_train)))
    print("     Root Mean Square Error: {:.2f}".format(np.sqrt(mean_squared_error(y_train, y_pred_train))))

    y_pred_test = xgb_model.predict(X_test)
    print("\nValidation model with test data")
    print("     Rsquare: {:.2f}".format(xgb_model.score(X_test, y_test)))
    print("     Mean Square Error: {:.2f}".format(mean_squared_error(y_test, y_pred_test)))
    print("     Root Mean Square Error: {:.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred_test))))

    # 4. Analizar predicciones y residuales---------------------------------
    
    print("-"*80)
    print("\nAnálisis de las predicciones y los residuales")

    if(plot_test):
        plot_model(y_test, y_pred_test)
    else:
        plot_model(y_train, y_pred_train)

    return(xgb_model)

#%%
def NNETRegressor(X_train, y_train, X_test, y_test, parameter=None, hyper_tunning=False, plot_test=True):
    """
    Function for train neural network
    """
    # 1. Libraries
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error, r2_score


    # 2. Hyperparameter tunning
    nnet_model = MLPRegressor(max_iter=1000)

    if(hyper_tunning):
        nnet_tunning = GridSearchCV(nnet_model,
                                    param_grid={"activation": ["logistic", "tanh", "relu"],
                                                    "alpha": 10.0**-np.arange(1, 7)}, 
                                    cv=5)
        nnet_tunning.fit(X_train, y_train)
        nnet_model.set_params(**nnet_tunning.best_params_)
    else:
        nnet_model.set_params(**parameter)

    # 3. Train Extreme Gradient Boosting with the best hyperparameter

    nnet_model.fit(X_train, y_train)
    y_pred_train = nnet_model.predict(X_train)

    print("-"*80)
    print("Model Parameter: \n")
    print("{}".format(nnet_model.get_params))

    print("\nValidation model with train data")
    print("     Rsquare: {:.2f}".format(nnet_model.score(X_train, y_train)))
    print("     Mean Square Error: {:.2f}".format(mean_squared_error(y_train, y_pred_train)))
    print("     Root Mean Square Error: {:.2f}".format(np.sqrt(mean_squared_error(y_train, y_pred_train))))

    y_pred_test = nnet_model.predict(X_test)
    print("\nValidation model with test data")
    print("     Rsquare: {:.2f}".format(nnet_model.score(X_test, y_test)))
    print("     Mean Square Error: {:.2f}".format(mean_squared_error(y_test, y_pred_test)))
    print("     Root Mean Square Error: {:.2f}".format(np.sqrt(mean_squared_error(y_test, y_pred_test))))

    # 4. Analizar predicciones y residuales---------------------------------

    print("-"*80)
    print("\nAnálisis de las predicciones y los residuales")

    if(plot_test):
        plot_model(y_test, y_pred_test)
    else:
        plot_model(y_train, y_pred_train)

    return(nnet_model)
