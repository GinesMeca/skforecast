# Unit test create_train_X_y ForecasterAutoregMultiSeries
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries
from skforecast.exceptions import MissingValuesWarning
from skforecast.exceptions import IgnoredArgumentWarning
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def test_create_train_X_y_TypeError_when_exog_is_categorical_of_no_int():
    """
    Test TypeError is raised when exog is categorical with no int values.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(3)),  
                           '2': pd.Series(np.arange(3))})
    exog = pd.Series(['A', 'B', 'C'], name='exog', dtype='category')
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=2)

    err_msg = re.escape(
        ("Categorical columns in exog must contain only integer values. "
         "See skforecast docs for more info about how to include "
         "categorical features https://skforecast.org/"
         "latest/user_guides/categorical-features.html")
    )
    with pytest.raises(TypeError, match = err_msg):
        forecaster.create_train_X_y(series=series, exog=exog)


def test_create_train_X_y_ValueError_when_Forecaster_fitted_and_different_columns_names():
    """
    Test ValueError is raised when the forecaster is fitted and the columns names
    of the series are different from the columns names used to fit the forecaster.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fitted = True
    forecaster.series_col_names = ['l1', 'l2']

    new_series = pd.DataFrame({
        'l1': pd.Series(np.arange(10)),
        'l4': pd.Series(np.arange(10))
    })

    err_msg = re.escape(
        (f"Once the Forecaster has been trained, `series` must have the "
         f"same columns as the series used during training:\n" 
         f" Got      : ['l1', 'l4']\n"
         f" Expected : ['l1', 'l2']")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(series=new_series)


def test_create_train_X_y_ValueError_when_Forecaster_fitted_without_exog_and_exog_is_not_None():
    """
    Test ValueError is raised when the forecaster was fitted without exog and
    exog is not None.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fitted = True
    forecaster.series_col_names = ['l1', 'l2']
    forecaster.exog_col_names = None

    series = pd.DataFrame({
        'l1': pd.Series(np.arange(10)),
        'l2': pd.Series(np.arange(10))
    })
    exog = pd.Series(np.arange(10), name='exog')

    err_msg = re.escape(
        ("Once the Forecaster has been trained, `exog` must be `None` "
         "because no exogenous variables were added during training.")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(series=series, exog=exog)


def test_create_train_X_y_ValueError_when_Forecaster_fitted_and_different_exog_columns_names():
    """
    Test ValueError is raised when the forecaster is fitted and the columns names
    of exog are different from the columns names used to fit the forecaster.
    """
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)
    forecaster.fitted = True
    forecaster.series_col_names = ['l1', 'l2']
    forecaster.exog_col_names = ['exog']

    series = pd.DataFrame({
        'l1': pd.Series(np.arange(10)),
        'l2': pd.Series(np.arange(10))
    })
    new_exog = pd.Series(np.arange(10), name='exog2')

    err_msg = re.escape(
        (f"Once the Forecaster has been trained, `exog` must have the "
         f"same columns as the series used during training:\n" 
         f" Got      : ['exog2']\n"
         f" Expected : ['exog']")
    )
    with pytest.raises(ValueError, match = err_msg):
        forecaster.create_train_X_y(series=series, exog=new_exog)


def test_create_train_X_y_output_when_series_and_exog_is_None():
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is None.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(7, dtype=float)), 
                           '2': pd.Series(np.arange(7, dtype=float))})
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3)

    results = forecaster.create_train_X_y(series=series)
    expected = (
        pd.DataFrame(
            data = np.array([[-0.5, -1. , -1.5, 1., 0.],
                             [ 0. , -0.5, -1. , 1., 0.],
                             [ 0.5,  0. , -0.5, 1., 0.],
                             [ 1. ,  0.5,  0. , 1., 0.],
                             [-0.5, -1. , -1.5, 0., 1.],
                             [ 0. , -0.5, -1. , 0., 1.],
                             [ 0.5,  0. , -0.5, 0., 1.],
                             [ 1. ,  0.5,  0. , 0., 1.]]),
            index   = pd.Index([3, 4, 5, 6, 3, 4, 5, 6]),
            columns = ['lag_1', 'lag_2', 'lag_3', '1', '2']
        ),
        pd.Series(
            data  = np.array([0., 0.5, 1., 1.5, 0., 0.5, 1., 1.5]),
            index = pd.Index([3, 4, 5, 6, 3, 4, 5, 6]),
            name  = 'y',
            dtype = float
        ),
        {'1': pd.RangeIndex(start=0, stop=7, step=1),
         '2': pd.RangeIndex(start=0, stop=7, step=1)},
        ['1', '2'],
        None,
        None
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert isinstance(results[4], type(None))
    assert isinstance(results[5], type(None))


def test_create_train_X_y_output_when_series_and_exog_no_pandas_index():
    """
    Test the output of create_train_X_y when series and exog have no pandas index 
    that doesn't start at 0.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    series.index = np.arange(6, 16)
    exog = pd.Series(np.arange(100, 110), index=np.arange(6, 16), 
                     name='exog', dtype=float)
    
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5,
                                              transformer_series=None)
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 1., 0., 105.],
                             [5., 4., 3., 2., 1., 1., 0., 106.],
                             [6., 5., 4., 3., 2., 1., 0., 107.],
                             [7., 6., 5., 4., 3., 1., 0., 108.],
                             [8., 7., 6., 5., 4., 1., 0., 109.],
                             [4., 3., 2., 1., 0., 0., 1., 105.],
                             [5., 4., 3., 2., 1., 0., 1., 106.],
                             [6., 5., 4., 3., 2., 0., 1., 107.],
                             [7., 6., 5., 4., 3., 0., 1., 108.],
                             [8., 7., 6., 5., 4., 0., 1., 109.]]),
            index   = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'l1', 'l2', 'exog']
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.Index([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            name  = 'y',
            dtype = float
        ),
        {'l1': pd.RangeIndex(start=0, stop=10, step=1),
         'l2': pd.RangeIndex(start=0, stop=10, step=1)},
        ['l1', 'l2'],
        ['exog'],
        {'exog': exog.dtypes}
    )

    pd.testing.assert_frame_equal(results[0], expected[0])
    pd.testing.assert_series_equal(results[1], expected[1])
    for k in results[2].keys():
        pd.testing.assert_index_equal(results[2][k], expected[2][k])
    assert results[3] == expected[3]
    assert results[4] == expected[4]
    for k in results[5].keys():
        assert results[5] == expected[5]


# TODO: CONTINUE FROM HERE
# =============================================================================


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt : f'dtype: {dt}')
def test_create_train_X_y_output_when_series_10_and_exog_is_series_of_float_int(dtype):
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is a pandas series of floats or ints.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(10, dtype=float)), 
                           '2': pd.Series(np.arange(10, dtype=float))})
    exog = pd.Series(np.arange(100, 110), name='exog', dtype=dtype)

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5,
                                              transformer_series=None)
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105., 1., 0.],
                             [5., 4., 3., 2., 1., 106., 1., 0.],
                             [6., 5., 4., 3., 2., 107., 1., 0.],
                             [7., 6., 5., 4., 3., 108., 1., 0.],
                             [8., 7., 6., 5., 4., 109., 1., 0.],
                             [4., 3., 2., 1., 0., 105., 0., 1.],
                             [5., 4., 3., 2., 1., 106., 0., 1.],
                             [6., 5., 4., 3., 2., 107., 0., 1.],
                             [7., 6., 5., 4., 3., 108., 0., 1.],
                             [8., 7., 6., 5., 4., 109., 0., 1.]]),
            index   = pd.RangeIndex(start=0, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'exog', '1', '2']
        ).astype({'exog': dtype}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=0, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        pd.RangeIndex(start=0, stop=len(series), step=1),
        pd.Index(np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]))
    )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            np.testing.assert_array_equal(results[i], expected[i])


@pytest.mark.parametrize("dtype", 
                         [float, int], 
                         ids = lambda dt : f'dtype: {dt}')
def test_create_train_X_y_output_when_series_10_and_exog_is_dataframe_of_float_int(dtype):
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns of floats or ints.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(10, dtype=float)), 
                           '2': pd.Series(np.arange(10, dtype=float))})
    exog = pd.DataFrame({'exog_1': np.arange(100, 110, dtype=dtype),
                         'exog_2': np.arange(1000, 1010, dtype=dtype)})

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5,
                                              transformer_series=None)
    results = forecaster.create_train_X_y(series=series, exog=exog)    

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105., 1005., 1., 0.],
                             [5., 4., 3., 2., 1., 106., 1006., 1., 0.],
                             [6., 5., 4., 3., 2., 107., 1007., 1., 0.],
                             [7., 6., 5., 4., 3., 108., 1008., 1., 0.],
                             [8., 7., 6., 5., 4., 109., 1009., 1., 0.],
                             [4., 3., 2., 1., 0., 105., 1005., 0., 1.],
                             [5., 4., 3., 2., 1., 106., 1006., 0., 1.],
                             [6., 5., 4., 3., 2., 107., 1007., 0., 1.],
                             [7., 6., 5., 4., 3., 108., 1008., 0., 1.],
                             [8., 7., 6., 5., 4., 109., 1009., 0., 1.]]),
            index   = pd.RangeIndex(start=0, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'exog_1', 'exog_2', '1', '2']
        ).astype({'exog_1': dtype, 'exog_2': dtype}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=0, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        pd.RangeIndex(start=0, stop=len(series), step=1),
        pd.Index(np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]))
    )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            np.testing.assert_array_equal(results[i], expected[i])


@pytest.mark.parametrize("exog_values, dtype", 
                         [([True]    , bool), 
                          (['string'], str)], 
                         ids = lambda dt : f'values, dtype: {dt}')
def test_create_train_X_y_output_when_series_10_and_exog_is_series_of_bool_str(exog_values, dtype):
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is a pandas series of bool or str.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    exog = pd.Series(exog_values*10, name='exog', dtype=dtype)

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5,
                                              transformer_series=None)
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.],
                             [4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=0, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog = exog_values*5 + exog_values*5, 
                 l1   = [1.]*5 + [0.]*5, 
                 l2   = [0.]*5 + [1.]*5).astype({'exog': dtype}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=0, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        pd.RangeIndex(start=0, stop=len(series), step=1),
        pd.Index(np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]))
    )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            np.testing.assert_array_equal(results[i], expected[i])


@pytest.mark.parametrize("v_exog_1   , v_exog_2  , dtype", 
                         [([True]    , [False]   , bool), 
                          (['string'], ['string'], str)], 
                         ids = lambda dt : f'values, dtype: {dt}')
def test_create_train_X_y_output_when_series_10_and_exog_is_dataframe_of_bool_str(v_exog_1, v_exog_2, dtype):
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns of bool or str.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    exog = pd.DataFrame({'exog_1': v_exog_1*10,
                         'exog_2': v_exog_2*10})

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5,
                                              transformer_series=None)
    results = forecaster.create_train_X_y(series=series, exog=exog)    

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.],
                             [4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=0, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog_1 = v_exog_1*5 + v_exog_1*5, 
                 exog_2 = v_exog_2*5 + v_exog_2*5, 
                 l1     = [1.]*5 + [0.]*5, 
                 l2     = [0.]*5 + [1.]*5).astype({'exog_1': dtype, 
                                                   'exog_2': dtype}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=0, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        pd.RangeIndex(start=0, stop=len(series), step=1),
        pd.Index(np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]))
    )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            np.testing.assert_array_equal(results[i], expected[i])


def test_create_train_X_y_output_when_series_10_and_exog_is_series_of_category():
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is a pandas series of category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    exog = pd.Series(range(10), name='exog', dtype='category')

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5,
                                              transformer_series=None)
    results = forecaster.create_train_X_y(series=series, exog=exog)   

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.],
                             [4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=0, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog = pd.Categorical([5, 6, 7, 8, 9]*2, categories=range(10)), 
                 l1   = [1.]*5 + [0.]*5, 
                 l2   = [0.]*5 + [1.]*5),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=0, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        pd.RangeIndex(start=0, stop=len(series), step=1),
        pd.Index(np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]))
    )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            np.testing.assert_array_equal(results[i], expected[i])


def test_create_train_X_y_output_when_series_10_and_exog_is_dataframe_of_category():
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns of category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    exog = pd.DataFrame({'exog_1': pd.Categorical(range(10)),
                         'exog_2': pd.Categorical(range(100, 110))})

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5,
                                              transformer_series=None)
    results = forecaster.create_train_X_y(series=series, exog=exog)   

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.],
                             [4., 3., 2., 1., 0.],
                             [5., 4., 3., 2., 1.],
                             [6., 5., 4., 3., 2.],
                             [7., 6., 5., 4., 3.],
                             [8., 7., 6., 5., 4.]]),
            index   = pd.RangeIndex(start=0, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5']
        ).assign(exog_1 = pd.Categorical([5, 6, 7, 8, 9]*2, categories=range(10)),
                 exog_2 = pd.Categorical([105, 106, 107, 108, 109]*2, categories=range(100, 110)), 
                 l1     = [1.]*5 + [0.]*5, 
                 l2     = [0.]*5 + [1.]*5),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=0, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        pd.RangeIndex(start=0, stop=len(series), step=1),
        pd.Index(np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]))
    )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            np.testing.assert_array_equal(results[i], expected[i])


def test_create_train_X_y_output_when_series_10_and_exog_is_dataframe_of_float_int_category():
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns of float, int, category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    exog = pd.DataFrame({'exog_1': pd.Series(np.arange(100, 110), dtype=float),
                         'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
                         'exog_3': pd.Categorical(range(100, 110))})

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5,
                                              transformer_series=None)
    results = forecaster.create_train_X_y(series=series, exog=exog)   

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105., 1005.],
                             [5., 4., 3., 2., 1., 106., 1006.],
                             [6., 5., 4., 3., 2., 107., 1007.],
                             [7., 6., 5., 4., 3., 108., 1008.],
                             [8., 7., 6., 5., 4., 109., 1009.],
                             [4., 3., 2., 1., 0., 105., 1005.],
                             [5., 4., 3., 2., 1., 106., 1006.],
                             [6., 5., 4., 3., 2., 107., 1007.],
                             [7., 6., 5., 4., 3., 108., 1008.],
                             [8., 7., 6., 5., 4., 109., 1009.]],
                             dtype=float),
            index   = pd.RangeIndex(start=0, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'exog_1', 'exog_2']
        ).assign(exog_3 = pd.Categorical([105, 106, 107, 108, 109]*2, categories=range(100, 110)), 
                 l1     = [1.]*5 + [0.]*5, 
                 l2     = [0.]*5 + [1.]*5).astype({'exog_1': float, 
                                                   'exog_2': int}),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]),
            index = pd.RangeIndex(start=0, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        pd.RangeIndex(start=0, stop=len(series), step=1),
        pd.Index(np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]))
    )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            np.testing.assert_array_equal(results[i], expected[i])


def test_create_train_X_y_output_when_series_and_exog_is_dataframe_datetime_index():
    """
    Test the output of create_train_X_y when series has 2 columns and 
    exog is a pandas dataframe with two columns and datetime index.
    """
    series = pd.DataFrame({'1': np.arange(7, dtype=float), 
                           '2': np.arange(7, dtype=float)},
                           index = pd.date_range("1990-01-01", periods=7, freq='D'))
    exog = pd.DataFrame({'exog_1' : np.arange(100, 107, dtype=float),
                         'exog_2' : np.arange(1000, 1007, dtype=float)},
                        index = pd.date_range("1990-01-01", periods=7, freq='D'))
                         
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=3,
                                              transformer_series=None)
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([[2.0, 1.0, 0.0, 103., 1003., 1., 0.],
                             [3.0, 2.0, 1.0, 104., 1004., 1., 0.],
                             [4.0, 3.0, 2.0, 105., 1005., 1., 0.],
                             [5.0, 4.0, 3.0, 106., 1006., 1., 0.],
                             [2.0, 1.0, 0.0, 103., 1003., 0., 1.],
                             [3.0, 2.0, 1.0, 104., 1004., 0., 1.],
                             [4.0, 3.0, 2.0, 105., 1005., 0., 1.],
                             [5.0, 4.0, 3.0, 106., 1006., 0., 1.]]),
            index   = pd.RangeIndex(start=0, stop=8, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'exog_1', 'exog_2', '1', '2']
        ),
        pd.Series(
            data  = np.array([3., 4., 5., 6., 3., 4., 5., 6.]),
            index = pd.RangeIndex(start=0, stop=8, step=1),
            name  = 'y',
            dtype = float
        ),
        pd.date_range("1990-01-01", periods=7, freq='D'),
        pd.Index(pd.DatetimeIndex(['1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07', 
                                   '1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07']))
    )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            np.testing.assert_array_equal(results[i], expected[i])


def test_create_train_X_y_output_when_series_10_and_transformer_series_is_StandardScaler():
    """
    Test the output of create_train_X_y when exog is None and transformer_series
    is StandardScaler.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series(np.arange(10, dtype=float))})
    forecaster = ForecasterAutoregMultiSeries(
                    regressor          = LinearRegression(),
                    lags               = 5,
                    transformer_series = StandardScaler()
                )
    results = forecaster.create_train_X_y(series=series)
    expected = (
        pd.DataFrame(
            data = np.array([
                       [-0.17407766, -0.52223297, -0.87038828, -1.21854359, -1.5666989 , 1.,  0.],
                       [ 0.17407766, -0.17407766, -0.52223297, -0.87038828, -1.21854359, 1.,  0.],
                       [ 0.52223297,  0.17407766, -0.17407766, -0.52223297, -0.87038828, 1.,  0.],
                       [ 0.87038828,  0.52223297,  0.17407766, -0.17407766, -0.52223297, 1.,  0.],
                       [ 1.21854359,  0.87038828,  0.52223297,  0.17407766, -0.17407766, 1.,  0.],
                       [-0.17407766, -0.52223297, -0.87038828, -1.21854359, -1.5666989 , 0.,  1.],
                       [ 0.17407766, -0.17407766, -0.52223297, -0.87038828, -1.21854359, 0.,  1.],
                       [ 0.52223297,  0.17407766, -0.17407766, -0.52223297, -0.87038828, 0.,  1.],
                       [ 0.87038828,  0.52223297,  0.17407766, -0.17407766, -0.52223297, 0.,  1.],
                       [ 1.21854359,  0.87038828,  0.52223297,  0.17407766, -0.17407766, 0.,  1.]]),
            index   = pd.RangeIndex(start=0, stop=10, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'l1', 'l2']
        ),
        pd.Series(
            data  = np.array([0.17407766, 0.52223297, 0.87038828, 1.21854359, 1.5666989 ,
                              0.17407766, 0.52223297, 0.87038828, 1.21854359, 1.5666989 ]),
            index = pd.RangeIndex(start=0, stop=10, step=1),
            name  = 'y',
            dtype = float
        ),
        pd.RangeIndex(start=0, stop=len(series), step=1),
        pd.Index(np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9]))
    )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            np.testing.assert_array_equal(results[i], expected[i])


def test_create_train_X_y_output_when_exog_is_None_and_transformer_exog_is_not_None():
    """
    Test the output of create_train_X_y when exog is None and transformer_exog
    is not None.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(7, dtype=float)), 
                           '2': pd.Series(np.arange(7, dtype=float))})
    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     transformer_series = None,
                     transformer_exog   = StandardScaler()
                 )

    results = forecaster.create_train_X_y(series=series)
    expected = (
        pd.DataFrame(
            data = np.array([[2.0, 1.0, 0.0, 1., 0.],
                             [3.0, 2.0, 1.0, 1., 0.],
                             [4.0, 3.0, 2.0, 1., 0.],
                             [5.0, 4.0, 3.0, 1., 0.],
                             [2.0, 1.0, 0.0, 0., 1.],
                             [3.0, 2.0, 1.0, 0., 1.],
                             [4.0, 3.0, 2.0, 0., 1.],
                             [5.0, 4.0, 3.0, 0., 1.]]),
            index   = pd.RangeIndex(start=0, stop=8, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', '1', '2']
        ),
        pd.Series(
            data  = np.array([3., 4., 5., 6., 3., 4., 5., 6.]),
            index = pd.RangeIndex(start=0, stop=8, step=1),
            name  = 'y',
            dtype = float
        ),
        pd.RangeIndex(start=0, stop=len(series), step=1),
        pd.Index(np.array([3., 4., 5., 6., 3., 4., 5., 6.]))
    )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            np.testing.assert_array_equal(results[i], expected[i])


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'1': StandardScaler(), '2': StandardScaler()}], 
                         ids = lambda tr : f'transformer_series type: {type(tr)}')
def test_create_train_X_y_output_when_transformer_series_and_transformer_exog(transformer_series):
    """
    Test the output of create_train_X_y when using transformer_series and 
    transformer_exog.
    """
    series = pd.DataFrame({'1': np.arange(10, dtype=float), 
                           '2': np.arange(10, dtype=float)},
                           index = pd.date_range("1990-01-01", periods=10, freq='D'))
    exog = pd.DataFrame({
               'exog_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 24.4, 87.2, 47.4, 23.8],
               'exog_2': ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b']},
                index = pd.date_range("1990-01-01", periods=10, freq='D'))

    transformer_exog = ColumnTransformer(
                            [('scale', StandardScaler(), ['exog_1']),
                             ('onehot', OneHotEncoder(), ['exog_2'])],
                            remainder = 'passthrough',
                            verbose_feature_names_out = False
                        )

    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog
                 )
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([
                       [-0.87038828, -1.21854359, -1.5666989 ,  0.67431975, 1., 0., 1., 0.],
                       [-0.52223297, -0.87038828, -1.21854359,  0.37482376, 1., 0., 1., 0.],
                       [-0.17407766, -0.52223297, -0.87038828, -0.04719331, 0., 1., 1., 0.],
                       [ 0.17407766, -0.17407766, -0.52223297, -0.81862236, 0., 1., 1., 0.],
                       [ 0.52223297,  0.17407766, -0.17407766,  2.03112731, 0., 1., 1., 0.],
                       [ 0.87038828,  0.52223297,  0.17407766,  0.22507577, 0., 1., 1., 0.],
                       [ 1.21854359,  0.87038828,  0.52223297, -0.84584926, 0., 1., 1., 0.],
                       [-0.87038828, -1.21854359, -1.5666989 ,  0.67431975, 1., 0., 0., 1.],
                       [-0.52223297, -0.87038828, -1.21854359,  0.37482376, 1., 0., 0., 1.],
                       [-0.17407766, -0.52223297, -0.87038828, -0.04719331, 0., 1., 0., 1.],
                       [ 0.17407766, -0.17407766, -0.52223297, -0.81862236, 0., 1., 0., 1.],
                       [ 0.52223297,  0.17407766, -0.17407766,  2.03112731, 0., 1., 0., 1.],
                       [ 0.87038828,  0.52223297,  0.17407766,  0.22507577, 0., 1., 0., 1.],
                       [ 1.21854359,  0.87038828,  0.52223297, -0.84584926, 0., 1., 0., 1.]]),
            index   = pd.RangeIndex(start=0, stop=14, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'exog_1',
                       'exog_2_a', 'exog_2_b', '1', '2']
        ),
        pd.Series(
            data  = np.array([-0.52223297, -0.17407766,  0.17407766,  0.52223297,  0.87038828,
                               1.21854359,  1.5666989 , -0.52223297, -0.17407766,  0.17407766,
                               0.52223297,  0.87038828,  1.21854359,  1.5666989 ]),
            index = pd.RangeIndex(start=0, stop=14, step=1),
            name  = 'y',
            dtype = float
        ),
        pd.date_range("1990-01-01", periods=10, freq='D'),
        pd.Index(pd.DatetimeIndex(['1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07', 
                                   '1990-01-08', '1990-01-09', '1990-01-10', '1990-01-04',
                                   '1990-01-05', '1990-01-06', '1990-01-07', '1990-01-08',
                                   '1990-01-09', '1990-01-10']))
    )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            np.testing.assert_array_equal(results[i], expected[i])


def test_create_train_X_y_output_when_series_different_length_and_exog_is_dataframe_of_float_int_category():
    """
    Test the output of create_train_X_y when series has 2 columns with different 
    lengths and exog is a pandas dataframe with two columns of float, int, category.
    """
    series = pd.DataFrame({'l1': pd.Series(np.arange(10, dtype=float)), 
                           'l2': pd.Series([np.nan, np.nan, 2., 3., 4., 5., 6., 7., 8., 9.])})
    series.index = pd.date_range("1990-01-01", periods=10, freq='D')
    exog = pd.DataFrame({'exog_1': pd.Series(np.arange(100, 110), dtype=float),
                         'exog_2': pd.Series(np.arange(1000, 1010), dtype=int),
                         'exog_3': pd.Categorical(range(100, 110))})
    exog.index = pd.date_range("1990-01-01", periods=10, freq='D')

    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=5,
                                              transformer_series=None)
    results = forecaster.create_train_X_y(series=series, exog=exog)   

    expected = (
        pd.DataFrame(
            data = np.array([[4., 3., 2., 1., 0., 105., 1005.],
                             [5., 4., 3., 2., 1., 106., 1006.],
                             [6., 5., 4., 3., 2., 107., 1007.],
                             [7., 6., 5., 4., 3., 108., 1008.],
                             [8., 7., 6., 5., 4., 109., 1009.],
                             [6., 5., 4., 3., 2., 107., 1007.],
                             [7., 6., 5., 4., 3., 108., 1008.],
                             [8., 7., 6., 5., 4., 109., 1009.]],
                             dtype=float),
            index   = pd.RangeIndex(start=0, stop=8, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                       'exog_1', 'exog_2']
        ).assign(exog_3 = pd.Categorical([105, 106, 107, 108, 109, 
                                          107, 108, 109], categories=range(100, 110)), 
                 l1     = [1.]*5 + [0.]*3, 
                 l2     = [0.]*5 + [1.]*3
        ).astype({'exog_1': float, 
                  'exog_2': int}
        ),
        pd.Series(
            data  = np.array([5, 6, 7, 8, 9, 7, 8, 9]),
            index = pd.RangeIndex(start=0, stop=8, step=1),
            name  = 'y',
            dtype = float
        ),
        pd.date_range("1990-01-01", periods=10, freq='D'),
        pd.DatetimeIndex(['1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10',
                          '1990-01-08', '1990-01-09', '1990-01-10'],
                         dtype='datetime64[ns]', freq=None
        )
    )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            np.testing.assert_array_equal(results[i], expected[i])


@pytest.mark.parametrize("transformer_series", 
                         [StandardScaler(),
                          {'l1': StandardScaler(), 'l2': StandardScaler(), 'l3': StandardScaler()}], 
                         ids = lambda tr : f'transformer_series type: {type(tr)}')
def test_create_train_X_y_output_when_transformer_series_and_transformer_exog_with_different_series_lengths(transformer_series):
    """
    Test the output of create_train_X_y when using transformer_series and 
    transformer_exog with series with different lengths.
    """
    series = pd.DataFrame({'l1': np.arange(10, dtype=float), 
                           'l2': pd.Series([np.nan, np.nan, 
                                            2., 3., 4., 5., 6., 7., 8., 9.]), 
                           'l3': pd.Series([np.nan, np.nan, np.nan, np.nan, 
                                            4., 5., 6., 7., 8., 9.])})
    series.index = pd.date_range("1990-01-01", periods=10, freq='D')
    exog = pd.DataFrame({
               'exog_1': [7.5, 24.4, 60.3, 57.3, 50.7, 41.4, 24.4, 87.2, 47.4, 23.8],
               'exog_2': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b']},
                index = pd.date_range("1990-01-01", periods=10, freq='D'))

    transformer_exog = ColumnTransformer(
                           [('scale', StandardScaler(), ['exog_1']),
                            ('onehot', OneHotEncoder(), ['exog_2'])],
                           remainder = 'passthrough',
                           verbose_feature_names_out = False
                       )

    forecaster = ForecasterAutoregMultiSeries(
                     regressor          = LinearRegression(),
                     lags               = 3,
                     transformer_series = transformer_series,
                     transformer_exog   = transformer_exog
                 )
    results = forecaster.create_train_X_y(series=series, exog=exog)

    expected = (
        pd.DataFrame(
            data = np.array([
                       [-0.8703882797784892,  -1.2185435916898848,  -1.5666989036012806,   0.6743197452466179,  0.0, 1.0, 1.0, 0.0, 0.0],
                       [-0.5222329678670935,  -0.8703882797784892,  -1.2185435916898848,   0.3748237614897084,  1.0, 0.0, 1.0, 0.0, 0.0],
                       [-0.17407765595569785, -0.5222329678670935,  -0.8703882797784892,  -0.04719330653139179, 0.0, 1.0, 1.0, 0.0, 0.0],
                       [ 0.17407765595569785, -0.17407765595569785, -0.5222329678670935,  -0.8186223556022197,  1.0, 0.0, 1.0, 0.0, 0.0],
                       [ 0.5222329678670935,   0.17407765595569785, -0.17407765595569785,  2.0311273080241334,  0.0, 1.0, 1.0, 0.0, 0.0],
                       [ 0.8703882797784892,   0.5222329678670935,   0.17407765595569785,  0.2250757696112534,  1.0, 0.0, 1.0, 0.0, 0.0],
                       [ 1.2185435916898848,   0.8703882797784892,   0.5222329678670935,  -0.8458492632164842,  0.0, 1.0, 1.0, 0.0, 0.0],
                       [-0.6546536707079772,  -1.091089451179962,   -1.5275252316519468,  -0.04719330653139179, 0.0, 1.0, 0.0, 1.0, 0.0],
                       [-0.2182178902359924,  -0.6546536707079772,  -1.091089451179962,   -0.8186223556022197,  1.0, 0.0, 0.0, 1.0, 0.0],
                       [ 0.2182178902359924,  -0.2182178902359924,  -0.6546536707079772,   2.0311273080241334,  0.0, 1.0, 0.0, 1.0, 0.0],
                       [ 0.6546536707079772,   0.2182178902359924,  -0.2182178902359924,   0.2250757696112534,  1.0, 0.0, 0.0, 1.0, 0.0],
                       [ 1.091089451179962,    0.6546536707079772,   0.2182178902359924,  -0.8458492632164842,  0.0, 1.0, 0.0, 1.0, 0.0],
                       [-0.29277002188455997, -0.8783100656536799,  -1.4638501094227998,   2.0311273080241334,  0.0, 1.0, 0.0, 0.0, 1.0],
                       [ 0.29277002188455997, -0.29277002188455997, -0.8783100656536799,   0.2250757696112534,  1.0, 0.0, 0.0, 0.0, 1.0],
                       [ 0.8783100656536799,   0.29277002188455997, -0.29277002188455997, -0.8458492632164842,  0.0, 1.0, 0.0, 0.0, 1.0]]),
            index   = pd.RangeIndex(start=0, stop=15, step=1),
            columns = ['lag_1', 'lag_2', 'lag_3', 'exog_1',
                       'exog_2_a', 'exog_2_b', 'l1', 'l2', 'l3']
        ),
        pd.Series(
            data  = np.array([-0.5222329678670935, -0.17407765595569785, 0.17407765595569785, 0.5222329678670935, 0.8703882797784892, 1.2185435916898848, 1.5666989036012806, 
                              -0.2182178902359924, 0.2182178902359924, 0.6546536707079772, 1.091089451179962, 1.5275252316519468, 
                              0.29277002188455997, 0.8783100656536799, 1.4638501094227998]),
            index = pd.RangeIndex(start=0, stop=15, step=1),
            name  = 'y',
            dtype = float
        ),
        pd.date_range("1990-01-01", periods=10, freq='D'),
        pd.DatetimeIndex(['1990-01-04', '1990-01-05', '1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10',
                          '1990-01-06', '1990-01-07', '1990-01-08', '1990-01-09', '1990-01-10',
                          '1990-01-08', '1990-01-09', '1990-01-10'],
                         dtype='datetime64[ns]', freq=None
        )
    )

    for i in range(len(expected)):
        if isinstance(expected[i], pd.DataFrame):
            pd.testing.assert_frame_equal(results[i], expected[i])
        elif isinstance(expected[i], pd.Series):
            pd.testing.assert_series_equal(results[i], expected[i])
        else:
            np.testing.assert_array_equal(results[i], expected[i])












# TODO: Complete
def test_create_train_X_y_MissingValuesWarning_when_exog_has_missing_values():
    """
    Test create_train_X_y is issues a MissingValuesWarning when X_train has 
    missing values and drop_nan.
    """
    series = pd.DataFrame({'1': pd.Series(np.arange(4)),  
                           '2': pd.Series(np.arange(4))})
    exog = pd.Series([1, 2, 3, np.nan], name='exog')
    forecaster = ForecasterAutoregMultiSeries(LinearRegression(), lags=2)

    warn_msg = re.escape(
        ("NaNs detected in `X_train`. Some regressor do not allow "
         "NaN values during training. If you want to drop them, "
         "set `drop_nan = True`.")
    )
    with pytest.warns(MissingValuesWarning, match = warn_msg):
        forecaster.create_train_X_y(series=series, exog=exog, drop_nan=False)