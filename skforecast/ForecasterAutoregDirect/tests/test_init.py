# Unit test __init__ ForecasterAutoregDirect
# ==============================================================================
import re
import pytest
import numpy as np
import pandas as pd
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from sklearn.linear_model import LinearRegression


@pytest.mark.parametrize("n_jobs", 
                         [1.0, 'not_int_auto'], 
                         ids = lambda value : f'n_jobs: {value}')
def test_init_TypeError_when_n_jobs_not_int_or_auto(n_jobs):
    """
    Test TypeError is raised in when n_jobs is not an integer or 'auto'.
    """
    err_msg = re.escape(f"`n_jobs` must be an integer or `'auto'`. Got {type(n_jobs)}.")
    with pytest.raises(TypeError, match = err_msg):
        ForecasterAutoregDirect(LinearRegression(), lags=2, steps=2, n_jobs=n_jobs)