################################################################################
#                        skforecast.model_selection                            #
#                                                                              #
# This work by Joaquín Amat Rodrigo is licensed under a Creative Commons       #
# Attribution 4.0 International License.                                       #
################################################################################
# coding=utf-8


import typing
from typing import Union, Dict, List, Tuple
import numpy as np
import pandas as pd
import logging
import tqdm
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.ar_model import AutoReg

from .model_selection import time_series_spliter

logging.basicConfig(
    format = '%(asctime)-5s %(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.INFO,
)


def backtesting_sarimax_statsmodels(
    y: Union[np.ndarray, pd.Series],
    initial_train_size: int,
    steps: int,
    metric: str,
    order: tuple=(1, 0, 0), 
    seasonal_order: tuple=(0, 0, 0, 0),
    trend: str=None,
    exog: Union[np.ndarray, pd.Series, pd.DataFrame]=None,
    verbose: bool=False
) -> Tuple[np.array, np.array]:
    '''
    
    Backtesting (validation) of `SARIMAX` model from statsmodels v0.12. The model
    is trained only once using the `initial_train_size` first observations. In each
    iteration, a number of `steps` predictions are evaluated. This evaluation is
    much faster than cross-validation since the model is trained only once.
    
    https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_forecasting.html
    
    Parameters
    ----------
        
    y : 1D np.ndarray, pd.Series
        Training time series values. 
        
    order: tuple 
        The (p,d,q) order of the model for the number of AR parameters, differences,
        and MA parameters. d must be an integer indicating the integration order
        of the process, while p and q may either be an integers indicating the AR
        and MA orders (so that all lags up to those orders are included) or else
        iterables giving specific AR and / or MA lags to include. Default is an
        AR(1) model: (1,0,0).
        
    seasonal_order: tuple
        The (P,D,Q,s) order of the seasonal component of the model for the AR parameters,
        differences, MA parameters, and periodicity. D must be an integer
        indicating the integration order of the process, while P and Q may either
        be an integers indicating the AR and MA orders (so that all lags up to
        those orders are included) or else iterables giving specific AR and / or
        MA lags to include. s is an integer giving the periodicity (number of
        periods in season), often it is 4 for quarterly data or 12 for monthly data.
        Default is no seasonal effect.
        
    trend: str {‘n’,’c’,’t’,’ct’}
        Parameter controlling the deterministic trend polynomial A(t). Can be
        specified as a string where ‘c’ indicates a constant (i.e. a degree zero
        component of the trend polynomial), ‘t’ indicates a linear trend with time,
        and ‘ct’ is both. Can also be specified as an iterable defining the non-zero
        polynomial exponents to include, in increasing order. For example, [1,1,0,1]
        denotes a+bt+ct3. Default is to not include a trend component.
    
    initial_train_size: int 
        Number of samples in the initial train split.
        
    steps : int
        Number of steps to predict.
        
    metric : {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}
        Metric used to quantify the goodness of fit of the model.
        
    exog : np.ndarray, pd.Series, pd.DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
            
    verbose : bool, default `False`
        Print number of folds used for backtesting.
        
    Returns 
    -------
    backtest_predictions: 1D np.ndarray
        Value of predictions.
        
    metric_value: np.ndarray shape (1,)
        Value of the metric.
    '''
    

    if metric not in ['mean_squared_error', 'mean_absolute_error',
                      'mean_absolute_percentage_error']:
        raise Exception(
            f"Allowed metrics are: 'mean_squared_error', 'mean_absolute_error' and "
            f"'mean_absolute_percentage_error'. Got {metric}."
        )
    
    backtest_predictions = []
    
    metrics = {
        'mean_squared_error': mean_squared_error,
        'mean_absolute_error': mean_absolute_error,
        'mean_absolute_percentage_error': mean_absolute_percentage_error
    }
    
    metric = metrics[metric]
    
    if isinstance(y, pd.Series):
        y = y.to_numpy(copy=True)
        
    if isinstance(exog, (pd.Series, pd.DataFrame)):
        exog = exog.to_numpy(copy=True)
        
    if exog is None:
        model = SARIMAX(
                    endog = y[:initial_train_size],
                    order = order,
                    seasonal_order = seasonal_order,
                    trend = trend
                ).fit(disp=0)
        
    else:
        model = SARIMAX(
                    endog = y[:initial_train_size],
                    exog  = exog[:initial_train_size],
                    order = order,
                    seasonal_order = seasonal_order,
                    trend = trend
                ).fit(disp=0)
    
    
    folds     = (len(y) - initial_train_size) // steps + 1
    remainder = (len(y) - initial_train_size) % steps
    
    if verbose:
        print(f"Number of observations used for training: {initial_train_size}")
        print(f"Number of observations used for testing: {len(y) - initial_train_size}")
        print(f"    Number of folds: {folds - 1 * (remainder == 0)}")
        print(f"    Number of steps per fold: {steps}")
        if remainder != 0:
            print(f"    Last fold only includes {remainder} observations")
      
    for i in range(folds):
        last_window_end   = initial_train_size + i * steps
        last_window_start = (initial_train_size + i * steps) - steps 
        last_window       = y[last_window_start:last_window_end]
        
        if i == 0:
            if exog is None:
                pred = model.forecast(steps=steps)
                            
            else:
                pred = model.forecast(
                            steps       = steps,
                            exog        = exog[last_window_end:last_window_end + steps]
                        )
                
        elif i < folds - 1:
            # Update internal values stored by SARIMAX
            model = model.extend(last_window)
        
            if exog is None:
                pred = model.forecast(steps=steps)
                            
            else:
                pred = model.forecast(
                            steps       = steps,
                            exog        = exog[last_window_end:last_window_end + steps]
                        )
                
        elif remainder != 0:
            steps = remainder
            # Update internal values stored by SARIMAX
            model = model.extend(last_window)
            
            if exog is None:
                pred = model.forecast(steps=steps)
                
            else:
                pred = model.forecast(
                            steps       = steps,
                            exog        = exog[last_window_end:last_window_end + steps]
                        )
        else:
            continue
        
        backtest_predictions.append(pred)
    
    backtest_predictions = np.concatenate(backtest_predictions)
    metric_value = metric(
                        y_true = y[initial_train_size: initial_train_size + len(backtest_predictions)],
                        y_pred = backtest_predictions
                   )

    return np.array([metric_value]), backtest_predictions


def cv_sarimax_statsmodels(
    y: Union[np.ndarray, pd.Series],
    initial_train_size: int,
    steps: int,
    metric: str,
    order: tuple=(1, 0, 0), 
    seasonal_order: tuple=(0, 0, 0, 0),
    trend: str=None,
    exog: Union[np.ndarray, pd.Series, pd.DataFrame]=None,
    allow_incomplete_fold: bool=True,
    verbose: bool=False
) -> Tuple[np.array, np.array]:
    '''
        
    Cross-validation of `SARIMAX` model from statsmodels v0.12. The order of data
    is maintained and the training set increases in each iteration.
    
    Parameters
    ----------
        
    y : 1D np.ndarray, pd.Series
        Training time series values. 
        
    order: tuple 
        The (p,d,q) order of the model for the number of AR parameters, differences,
        and MA parameters. d must be an integer indicating the integration order
        of the process, while p and q may either be an integers indicating the AR
        and MA orders (so that all lags up to those orders are included) or else
        iterables giving specific AR and / or MA lags to include. Default is an
        AR(1) model: (1,0,0).
        
    seasonal_order: tuple
        The (P,D,Q,s) order of the seasonal component of the model for the AR parameters,
        differences, MA parameters, and periodicity. D must be an integer
        indicating the integration order of the process, while P and Q may either
        be an integers indicating the AR and MA orders (so that all lags up to
        those orders are included) or else iterables giving specific AR and / or
        MA lags to include. s is an integer giving the periodicity (number of
        periods in season), often it is 4 for quarterly data or 12 for monthly data.
        Default is no seasonal effect.
        
    trend: str {‘n’,’c’,’t’,’ct’}
        Parameter controlling the deterministic trend polynomial A(t). Can be
        specified as a string where ‘c’ indicates a constant (i.e. a degree zero
        component of the trend polynomial), ‘t’ indicates a linear trend with time,
        and ‘ct’ is both. Can also be specified as an iterable defining the non-zero
        polynomial exponents to include, in increasing order. For example, [1,1,0,1]
        denotes a+bt+ct3. Default is to not include a trend component.
    
    initial_train_size: int 
        Number of samples in the initial train split.
        
    steps : int
        Number of steps to predict.
        
    metric : {'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'}
        Metric used to quantify the goodness of fit of the model.
        
    exog : np.ndarray, pd.Series, pd.DataFrame, default `None`
        Exogenous variable/s included as predictor/s. Must have the same
        number of observations as `y` and should be aligned so that y[i] is
        regressed on exog[i].
            
    verbose : bool, default `False`
        Print number of folds used for cross-validation.
        
    Returns 
    -------
    cv_predictions: 1D np.ndarray
        Predictions.
        
    cv_metrics: 1D np.ndarray
        Value of the metric for each partition.
    '''
    

    if metric not in ['mean_squared_error', 'mean_absolute_error',
                      'mean_absolute_percentage_error']:
        raise Exception(
            f"Allowed metrics are: 'mean_squared_error', 'mean_absolute_error' and "
            f"'mean_absolute_percentage_error'. Got {metric}."
        )
        
    metrics = {
        'mean_squared_error': mean_squared_error,
        'mean_absolute_error': mean_absolute_error,
        'mean_absolute_percentage_error': mean_absolute_percentage_error
    }
    
    metric = metrics[metric]
    
    if isinstance(y, pd.Series):
        y = y.to_numpy(copy=True)
        
    if isinstance(exog, (pd.Series, pd.DataFrame)):
        exog = exog.to_numpy(copy=True)
        
    cv_predictions = []
    cv_metrics = []
    
    splits = time_series_spliter(
                y                     = y,
                initial_train_size    = initial_train_size,
                steps                 = steps,
                allow_incomplete_fold = allow_incomplete_fold,
                verbose               = verbose
             )
    
    for train_index, test_index in splits:
        
        if exog is None:
            model = SARIMAX(
                    endog = y[train_index],
                    order = order,
                    seasonal_order = seasonal_order,
                    trend = trend
                ).fit(disp=0)
            
            pred = model.forecast(steps=len(test_index))
            
        else:         
            model = SARIMAX(
                    endog = y[train_index],
                    exog  = exog[train_index],
                    order = order,
                    seasonal_order = seasonal_order,
                    trend = trend
                ).fit(disp=0)
            
            pred = model.forecast(steps=len(test_index), exog=exog[test_index])
    
               
        metric_value = metric(
                            y_true = y[test_index],
                            y_pred = pred
                       )
        
        cv_metrics.append(metric_value)
        cv_predictions.append(pred)
                          
    return np.array(cv_metrics), np.concatenate(cv_predictions)