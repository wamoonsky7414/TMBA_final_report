import pandas as pd
import numpy as np
import os
import random
from typing import Iterable
import warnings
import sys
sys.path.append('/Users/tedting/Documents')
from Alpha.operators import *
from Alpha.backtest import *
warnings.simplefilter(action='ignore', category=FutureWarning)

### WR means Wide Range !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
target_folder_path = '/Users/tedting/Documents/Alpha/data'
t_file = rf'{target_folder_path}/data_DM/WR_trash_filter_list_q_20240115_101_250.pkl'

### 共通資料
adj_open = pd.read_pickle(rf'{target_folder_path}/adj_open.pkl').loc['2013':]
exp_returns = adj_open.pct_change().shift(-2)

Open = pd.read_pickle(rf'{target_folder_path}/open.pkl')
Close = pd.read_pickle(rf'{target_folder_path}/收盤價(元).pkl')
Low = pd.read_pickle(rf'{target_folder_path}/最低價(元).pkl')
High = pd.read_pickle(rf'{target_folder_path}/最高價(元).pkl')
Volume = pd.read_pickle(rf'{target_folder_path}/成交量(千股).pkl')
Limit = pd.read_pickle(rf'{target_folder_path}/limit.pkl')

ZTXA = pd.read_pickle(rf'{target_folder_path}/ZTXA_close.pkl')
Benchmark = ZTXA['ZTXA 台指近月期貨指數'].pct_change().shift(-2)

Volume_y_avg = Volume.rolling(252).mean()
Close_y_avg = Close.rolling(252).mean()
heavy_Filter = Close_y_avg < 1000 
Volume_Filter = Volume_y_avg > 200 #問一下這個數字對嗎
trade_volume_Filter = Close_y_avg * Volume_y_avg > 60000  #問一下這個數字對嗎
small_aum_Filter = Volume_Filter & trade_volume_Filter
small_aum_Filter_L = Volume_Filter & trade_volume_Filter & heavy_Filter & Limit


### dataset資料
dataused_start = 500
data_folder_path = '/Volumes/Database/Financial_report_90'  # 替換為您的資料夾路徑
file_list = os.listdir(data_folder_path)
file_list_selected = file_list[dataused_start:]
dataframes = []
for file in file_list_selected:
    file_path = os.path.join(data_folder_path, file)
    df = pd.read_pickle(file_path)
    dataframes.append(df)

data_list =[rf'dataframes[{num}]' for num in range(len(dataframes))]
arithmetic_list = ['+', '-', '*', '/']
ts_operators_list = ['ts_delta(', 'ts_max(', 'ts_min(', 'ts_rank(', 'ts_stddev(', 'ts_sum(', 'ts_decay(', 'ts_mean(', 'ts_product(']
period_list = [f',{num})' for num in range(1, 505)]
cs_operators_list = ['abs(', 'log(', 'cs_rank(']
D = data_list
A = arithmetic_list
T = ts_operators_list
P = period_list
C = cs_operators_list

### backtest_function
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

def bt_algo(expression,
       expreturn:pd.DataFrame, 
       strategy='LS',
       filter='None',
       buy_fee:float=0.001425*0.3, sell_fee:float=0.001425*0.3+0.003,
       start_time='2013-01-01',
       end_time='2021-12-31'):
    if filter == 'None':
        alpha = eval(f'{expression}')
    elif filter == 'small_aum_Filter':
        alpha = eval(f'{expression}')[small_aum_Filter]
    elif filter == 'small_aum_Filter_L':
        alpha = eval(f'{expression}')[small_aum_Filter_L]
    else:
        raise ValueError("Please use 'None', 'small_aum_Filter', or 'small_aum_Filter_L' in filter")
    
    daily_weight = weight(alpha, strategy = strategy).loc[start_time:end_time]
    expreturn = expreturn.loc[start_time:end_time]
    daily_fee = fee(alpha, buy_fee=buy_fee, sell_fee=sell_fee, start_time=start_time, end_time=end_time)
    daily_profit = (daily_weight * expreturn).sum(axis=1)
    daily_returns = daily_profit - daily_fee
    
    summary_df = pd.DataFrame({
    'expression':[expression],
    'FitnessWQ':[round(sharpe(daily_returns)*sqrt(abs(annual_returns(daily_returns)/turnover(alpha))),4)],
    'Sharpe':[round(sharpe(daily_returns), 4)],
    'CAGR': [round(annual_returns(daily_returns), 4)],
    'MDD':[round(MDD(daily_returns),4)],
    'Turnover':[round(turnover(alpha), 4)]
    }, index = ['Performance'])
    return daily_returns, summary_df # pd.DataFrame

def bt_fornow(expression,
       expreturn:pd.DataFrame, 
       strategy='LS',
       filter='None',
       buy_fee:float=0.001425*0.3, sell_fee:float=0.001425*0.3+0.003,
       start_time='2013-01-01',
       end_time='2023-08-11'):
    if filter == 'None':
        alpha = expression
    elif filter == 'small_aum_Filter':
        alpha = expression[small_aum_Filter]
    elif filter == 'small_aum_Filter_L':
        alpha = expression[small_aum_Filter_L]
    else:
        raise ValueError("Please use 'None', 'small_aum_Filter', or 'small_aum_Filter_L' in filter")
    
    daily_weight = weight(alpha, strategy = strategy).loc[start_time:end_time]
    expreturn = expreturn.loc[start_time:end_time]
    daily_fee = fee(alpha, buy_fee=buy_fee, sell_fee=sell_fee, start_time=start_time, end_time=end_time)
    daily_profit = (daily_weight * expreturn).sum(axis=1)
    daily_returns = daily_profit - daily_fee
    
    summary_df = pd.DataFrame({
    'expression':[expression],
    'FitnessWQ':[round(sharpe(daily_returns)*sqrt(abs(annual_returns(daily_returns)/turnover(alpha))),4)],
    'Sharpe':[round(sharpe(daily_returns), 4)],
    'CAGR': [round(annual_returns(daily_returns), 4)],
    'MDD':[round(MDD(daily_returns),4)],
    'Turnover':[round(turnover(alpha), 4)]
    }, index = ['Performance'])
    return daily_returns, summary_df # pd.DataFrame

### algorithom_fuction
def trash_filter():
    try:
        trash_filter_list = pd.read_pickle(t_file)
    except FileNotFoundError:
        trash_filter_list = []
    return trash_filter_list

def first_layer_generate(trash_filter_list:pd.Series, sample_num = 10000):
    expression_list = []
    for _ in range(sample_num):
        TorC = random.random()
        if TorC < 0.5:
            expression = rf'{random.choice(T)} {random.choice(D)} {random.choice(P)}'
        else:
            expression = rf'{random.choice(C)} {random.choice(D)} )'
        expression_list.append(expression)
    expression_list = [item for item in expression_list if item not in trash_filter_list]
    return expression_list

def second_layer_generate(expression_list:pd.DataFrame, trash_filter_list:pd.Series):
    new_expression_list = expression_list.copy()
    for expression in expression_list:
        _, _, e2 = expression.split(' ', 2)
        if e2 == ')':
            new_expression = rf'{random.choice(T)} {expression} {random.choice(P)}'
        else:
            new_expression = rf'{random.choice(C)} {expression} )'
        new_expression_list.append(new_expression)
    new_expression_list = [item for item in new_expression_list if item not in trash_filter_list]
    return new_expression_list



