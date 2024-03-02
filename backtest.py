import sys
sys.path.append('/Users/tedting/Documents')
from Alpha.operators import *
import pandas as pd
import numpy as np

def weight(alpha:pd.DataFrame, strategy = 'LS'):
    demean = alpha.sub(alpha.mean(axis = 1), axis = 0)
    weight = demean.div(demean.abs().sum(axis = 1), axis = 0) # 有浮點數問題待解決

    # 多、空或多空
    if strategy == 'LS':
        weight = weight
    elif strategy == 'LO':
        weight = weight[weight > 0]*2
    elif strategy == 'SO':
        weight = weight[weight < 0]*2
    else:
        raise ValueError("Please use 'LS', 'LO', or 'SO' in strategy")
    weight = weight.fillna(0)
    return weight # pd.DataFrame


def get_weight(alpha:pd.DataFrame, strategy = 'LS'):
    demean = alpha.sub(alpha.mean(axis = 1), axis = 0)
    weight = demean.div(demean.abs().sum(axis = 1), axis = 0) # 有浮點數問題待解決

    # 多、空或多空
    if strategy == 'LS':
        weight = weight
    elif strategy == 'LO':
        weight = weight[weight > 0]*2
    elif strategy == 'SO':
        weight = weight[weight < 0]*2
    else:
        raise ValueError("Please use 'LS', 'LO', or 'SO' in strategy")
    weight = weight.fillna(0)
    return weight # pd.DataFrame

# def LS_rank_neutrolize(df:np.array)-> np.array:
#     for index, row in df.iterrows():
#         ranked_values = pd.Series(row.values).rank()
#         pct_values = [(v - min(ranked_values)) / (max(ranked_values) - min(ranked_values)) for v in ranked_values]
#         pct_values = pd.Series(pct_values)
#         neutrolized_value = pct_values -0.5
#         x = pd.DataFrame(neutrolized_value)
#     return(x) # can't get the correct answer

def fee(alpha:pd.DataFrame, 
        strategy = 'LS', 
        buy_fee:float = 0.001425*0.3, 
        sell_fee:float = 0.001425*0.3+0.003, 
        start_time='2013-01-01',
        end_time='2023-08-11'):
    weights = weight(alpha, strategy = strategy).loc[start_time:end_time]
    delta_weight = weights.shift(1) - weights
    buy_fees = delta_weight[delta_weight > 0]*(buy_fee)
    buy_fees = buy_fees.fillna(0)
    sell_fees = delta_weight.abs()[delta_weight < 0]*(sell_fee)
    sell_fees = sell_fees.fillna(0)
    fee = buy_fees + sell_fees
    daily_fee = fee.sum(axis = 1)
    return daily_fee # pd.Series

def bt(alpha:pd.DataFrame,
       expreturn:pd.DataFrame, 
       strategy='LS', 
       buy_fee:float=0.001425*0.3, sell_fee:float=0.001425*0.3+0.003,
       start_time='2013-01-01',
       end_time='2023-08-11'):
    daily_weight = weight(alpha, strategy = strategy).loc[start_time:end_time]
    expreturn = expreturn.loc[start_time:end_time]
    daily_fee = fee(alpha, buy_fee=buy_fee, sell_fee=sell_fee, start_time=start_time, end_time=end_time)
    daily_profit = (daily_weight * expreturn).sum(axis=1)
    daily_returns = daily_profit - daily_fee
    
    summary_df = pd.DataFrame({
    'Sharpe Ratio':[sharpe(daily_returns)],
    'Annualized Ret': [annual_returns(daily_returns)],
    'Max Drawdown':[MDD(daily_returns)],
    'Turnover':[turnover(alpha)], 
    'Std':[daily_returns.std()],
    'IC':[IC(alpha, expreturn, start_time, end_time)],
    'IR':[IR(alpha, expreturn, start_time, end_time)]# 這邊的input不同要注意
    }, index = ['Performance'])
    return daily_returns, summary_df # pd.DataFrame

# Index
def sharpe(daily_returns:pd.Series):
    Sharpe_ratio = round(daily_returns.mean()/daily_returns.std()*252**0.5,4)
    return Sharpe_ratio 

def annual_returns(daily_returns:pd.Series):
    LS_compound = (daily_returns+1).cumprod()
    days = len(LS_compound)
    total_return = LS_compound.iloc[-1]/LS_compound.iloc[0]-1
    annual_returns =  (total_return + 1)**(252/days) - 1
    return annual_returns

def turnover(alpha:pd.DataFrame):
    delta_weight = weight(alpha).shift(1) - weight(alpha)
    daily_tradingvalue = delta_weight.abs().sum(axis = 1)
    turnover = daily_tradingvalue.sum()/len(daily_tradingvalue)
    return turnover

def MDD(daily_returns:pd.Series):
    LS_compound = (daily_returns+1).cumprod()
    drawdowns = []
    peak = LS_compound[0]
    for price in LS_compound:
        if price > peak:
            peak = price
        drawdown = (price - peak) / peak
        drawdowns.append(drawdown)
    max_drawdown = np.min(drawdowns)
    return abs(max_drawdown)

def IC(alpha:pd.DataFrame, expreturn:pd.DataFrame, start_time='2014-01-01', end_time='2023-08-11'):
    f = alpha.loc[start_time:end_time]  # t期因子值
    r = expreturn.loc[start_time:end_time]  # t+1期股票实际收益率
    ic = f.corrwith(r, axis=1)
    return(ic.mean())

def IR(alpha:pd.DataFrame, expreturn:pd.DataFrame, start_time='2014-01-01', end_time='2023-08-11'):
    f = alpha.loc[start_time:end_time]  # t期因子值
    r = expreturn.loc[start_time:end_time]  # t+1期股票实际收益率
    ic = f.corrwith(r, axis=1)
    ir = ic.mean()/ic.std()
    return(ir)
    
   
