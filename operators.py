import pandas as pd
import numpy as np
import statsmodels.api as sm

# class Operator:
#     def __init__():

# operators_dict = dict(
#     ts_rank = ts_rank,    
# )

# arithmetic
def log(x: pd.DataFrame):
    return np.log(x[x!=0])

def sqrt(x: pd.DataFrame):
    return np.sqrt(x)

def sign(x: pd.DataFrame):
    return np.sign(x)

# time_series
def ts_rank(x: pd.DataFrame, d:int):
    return x.rolling(d).rank(pct=True)

def ts_corr(x: pd.DataFrame, y:pd.DataFrame, d:int):
    return x.rolling(d).corr(y)

# cross_section
def cs_rank(x: pd.DataFrame):
    return x.rank(axis=1, pct=True)

def cs_normalize(x: pd.DataFrame):
    row_means = x.mean(axis=1)
    row_stds = x.std(axis=1)
    normalized_df = (x.sub(row_means, axis=0)).div(row_stds, axis=0)
    return normalized_df

def abs(x: pd.DataFrame) -> pd.DataFrame:
    return np.abs(x)#x.abs()

def delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.shift(d)

def correlation(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d).corr(y)#.replace([-np.inf, np.inf], 0).fillna(value=0)

def covariance(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d).cov(y)#.replace([-np.inf, np.inf], 0).fillna(value=0)

def cs_scale(x: pd.DataFrame, a:int=1) -> pd.DataFrame:
    return x.mul(a).div(x.abs().sum(axis = 1), axis='index')

def ts_delta(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.diff(d)

def signedpower(x: pd.DataFrame, a:int) -> pd.DataFrame:
    return x**a

def ts_decay(x: pd.DataFrame, d:int) -> pd.DataFrame:
    # 過去 d 天的加權移動平均線，權重線性衰減 d, d ‒ 1, ..., 1（重新調整為總和為 1）
    result = x.values.copy()
    with np.errstate(all="ignore"):
        for i in range(1, d):
            result[i:] += (i+1) * x.values[:-i]
    result[:d] = np.nan
    return pd.DataFrame(result / np.arange(1, d+1).sum(),index = x.index,columns = x.columns)


def ts_min(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).min()

def ts_max(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).max()

def ts_argmin(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).apply(np.nanargmin, raw=True)+1

def ts_argmax(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).apply(np.nanargmax, raw=True)+1

def min(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return np.minimum(x,y)

def max(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return np.maximum(x,y)

def ts_sum(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).sum()

def ts_product(x: pd.DataFrame, d:int) -> pd.DataFrame:
    #return x.rolling(d, min_periods=d//2).apply(np.prod, raw=True)
    result = x.values.copy()
    with np.errstate(all="ignore"):
        for i in range(1, d):
            result[i:] *= x.values[:-i]
    return pd.DataFrame(result,index = x.index,columns = x.columns)

def ts_stddev(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).std()

def where(condition: pd.DataFrame, choiceA: pd.DataFrame, choiceB: pd.DataFrame) -> pd.DataFrame:
    condition_copy = pd.DataFrame(np.nan, index = condition.index, columns=condition.columns)
    condition_copy[condition] = choiceA
    condition_copy[~condition] = choiceB
    return condition_copy

def ts_mean(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).mean()

### TO DO~~~
def ts_regression(y: pd.DataFrame, x: pd.DataFrame, d: int, rettype: int = 0) -> pd.DataFrame:
    # Initialize a DataFrame to store the regression results
    results = pd.DataFrame(index=y.index, columns=['rettype_' + str(rettype)])
    
    # Perform the rolling regression
    for i in range(d-1, len(y)):
        # Slice the window for dependent and independent variables
        y_window = y.iloc[i-d+1:i+1].dropna()
        x_window = x.iloc[i-d+1:i+1].loc[y_window.index]  # Align x with y after dropping NaNs
        
        # Check if after dropping NaNs, there are enough data points to perform regression
        if len(y_window) < d or len(x_window) < d:
            continue
        
        # Add a constant term for the intercept
        x_with_const = sm.add_constant(x_window, has_constant='add')
        
        # Fit the regression model, skipping any NaNs or Infs
        model = sm.OLS(y_window, x_with_const, missing='drop').fit()
        
        # Based on rettype, store the appropriate value
        if rettype == 0:
            results.iloc[i] = model.resid.iloc[-1]
        elif rettype == 1:
            results.iloc[i] = model.params.get('const', np.nan)
        elif rettype == 2:
            results.iloc[i] = model.params.get(x_window.columns[0], np.nan)
        elif rettype == 3:
            results.iloc[i] = model.predict(x_with_const).iloc[-1]
        elif rettype == 4:
            results.iloc[i] = model.ssr
        elif rettype == 5:
            results.iloc[i] = model.ess
        elif rettype == 6:
            results.iloc[i] = model.rsquared
        # Additional rettypes can be added here for other statistics
    
    # Return the DataFrame with the regression results
    return results

