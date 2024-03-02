import pandas as pd
import numpy as np
import os
import random
from typing import Iterable

import sys
sys.path.append('/Users/tedting/Documents')
from Alpha.operators import *
from Alpha.backtest import *

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

target_folder_path = rf'/Users/tedting/Documents/Alpha/data'

### 共通資料
adj_open = pd.read_pickle(rf"{target_folder_path}/adj_open.pkl").loc['2013':]
exp_returns = adj_open.pct_change().shift(-2)

Close = pd.read_pickle(rf'{target_folder_path}/收盤價(元).pkl')
Volume = pd.read_pickle(rf'{target_folder_path}/成交量(千股).pkl')
Limit = pd.read_pickle(rf'{target_folder_path}/limit.pkl')

ZTXA = pd.read_pickle(rf'{target_folder_path}/ZTXA_close.pkl')
Benchmark = ZTXA['ZTXA 台指近月期貨指數'].pct_change().shift(-2)

Volume_y_avg = Volume.rolling(252).mean()
Close_y_avg = Close.rolling(252).mean()
heavy_Filter = Close_y_avg < 1000 
Volume_Filter = Volume_y_avg > 200
trade_volume_Filter = Close_y_avg * Volume_y_avg > 60000
small_aum_Filter = Volume_Filter & trade_volume_Filter
small_aum_Filter_L = Volume_Filter & trade_volume_Filter & heavy_Filter & Limit

### dataset資料
一以下 = pd.read_pickle(rf'{target_folder_path}/data_DM/1 張以下(人數).pkl')
一到五 = pd.read_pickle(rf'{target_folder_path}/data_DM/1 -5  張(人數).pkl')
五到十 = pd.read_pickle(rf'{target_folder_path}/data_DM/5 -10 張(人數).pkl')
十到十五 = pd.read_pickle(rf'{target_folder_path}/data_DM/10-15 張(人數).pkl')
十五到二十 = pd.read_pickle(rf'{target_folder_path}/data_DM/15-20 張(人數).pkl')
二十到三十 = pd.read_pickle(rf'{target_folder_path}/data_DM/20-30 張(人數).pkl')
三十到四十 = pd.read_pickle(rf'{target_folder_path}/data_DM/30-40 張(人數).pkl')
四十到五十 = pd.read_pickle(rf'{target_folder_path}/data_DM/40-50 張(人數).pkl')
五十到一百 = pd.read_pickle(rf'{target_folder_path}/data_DM/50-100 張(人數).pkl')
一百到二百 = pd.read_pickle(rf'{target_folder_path}/data_DM/100-200 張(人數).pkl')
二百到四百 = pd.read_pickle(rf'{target_folder_path}/data_DM/200-400 張(人數).pkl')
四百到六百 = pd.read_pickle(rf'{target_folder_path}/data_DM/400-600 張(人數).pkl')
六百到八百 = pd.read_pickle(rf'{target_folder_path}/data_DM/600-800 張(人數).pkl')
八百到一千 = pd.read_pickle(rf'{target_folder_path}/data_DM/800-1000張(人數).pkl')
一千以上 = pd.read_pickle(rf'{target_folder_path}/data_DM/1000張以上  (人數).pkl')
df_0 = pd.DataFrame(0, index=一千以上.index, columns=一千以上.columns, dtype=float)
df_1 = pd.DataFrame(1, index=一千以上.index, columns=一千以上.columns, dtype=float)

data_list = ['Close', 'Volume',
    '一以下', '一到五', '五到十', '十到十五', '十五到二十', '二十到三十', '三十到四十', '四十到五十',
    '五十到一百', '一百到二百', '二百到四百', '四百到六百', '六百到八百', '八百到一千', '一千以上',
    'df_0', 'df_1']

arithmetic_list = ['+', '-', '*', '/']
ts_operators_list = ['ts_delta(', 'ts_max(', 'ts_min(', 'ts_rank(', 'ts_stddev(', 'ts_sum(', 'ts_decay(', 'ts_mean(', 'ts_product(']
random_numbers = [random.randint(1, 504) for _ in range(504)]
period_list = [rf',{num})' for num in random_numbers] # 1是for decay 不變
cs_operators_list = ['abs(', 'log(', 'cs_rank(']
Os_List = ['ts_delta(', 'ts_max(', 'ts_min(', 'ts_rank(', 'ts_stddev(', 'ts_sum(', 'ts_decay(', 'ts_mean(', 'ts_product(', 'abs(', 'log(', 'cs_rank(']
D = data_list
A = arithmetic_list
O = ts_operators_list
P = period_list
C = cs_operators_list
O = Os_List

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
        start_time='2014-01-01',
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
       start_time='2014-01-01',
       end_time='2023-08-11'):
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
    'Sharpe':[round(sharpe(daily_returns), 4)],
    'CAGR': [round(annual_returns(daily_returns), 4)],
    'MDD':[round(MDD(daily_returns),4)],
    'Turnover':[round(turnover(alpha), 4)]
    }, index = ['Performance'])
    return daily_returns, summary_df # pd.DataFrame

### GA_function
def generate_expression():
    Ds = [random.choice(D) for _ in range(2)]
    Ps = [random.choice(P) for _ in range(4)]
    Os = [random.choice(O) for _ in range(4)]
    ### example: ts_rank( ts_rank( cs_rank( D ) A cs_rank( D ) ,5) ,5)
    expression_ny = f'{Os[0]} {Os[1]} {Os[2]} {Ds[0]} ) {random.choice(A)} {Os[3]} {Ds[1]} ) ) )' 
    O0, O1, O2, D0, P0, A0, O3, D1, P1, P2, P3 = expression_ny.split(' ', 10)
    if O0 not in cs_operators_list:
        P3 = Ps[0]
    if O1 not in cs_operators_list:
        P2 = Ps[1]
    if O2 not in cs_operators_list:
        P0 = Ps[2]
    if O3 not in cs_operators_list:
        P1 = Ps[3]
    expression = rf'{O0} {O1} {O2} {D0} {P0} {A0} {O3} {D1} {P1} {P2} {P3}'
    return expression


def crossover(expression_list, crossoverrate=0.5):
    working_list = expression_list.copy()
    if len(working_list) % 2 != 0:
        working_list.pop()

    while len(working_list) > 0:
        selected = random.sample(working_list, 2) # 選到的池子內相互交配(池子內沒有先後強弱之分)
        working_list.remove(selected[0])
        working_list.remove(selected[1])

        new_born1 = selected[0]
        new_born2 = selected[1]
        if random.random() < crossoverrate:
            O00, O01, O02, D00, P00, A00, O03, D01, P01, P02, P03 = new_born1.split(' ', 10)
            O10, O11, O12, D10, P10, A10, O13, D11, P11, P12, P13 = new_born2.split(' ', 10)
            new_born1 = f'{O10} {O01} {O02} {D00} {P00} {A00} {O03} {D01} {P01} {P02} {P13}'
            new_born2 = f'{O00} {O11} {O12} {D10} {P10} {A10} {O13} {D11} {P11} {P12} {P03}'
        if random.random() < crossoverrate:
            O00, O01, O02, D00, P00, A00, O03, D01, P01, P02, P03 = new_born1.split(' ', 10)
            O10, O11, O12, D10, P10, A10, O13, D11, P11, P12, P13 = new_born2.split(' ', 10)
            new_born1 = f'{O00} {O11} {O02} {D00} {P00} {A00} {O03} {D01} {P01} {P12} {P03}'
            new_born2 = f'{O10} {O01} {O12} {D10} {P10} {A10} {O13} {D11} {P11} {P02} {P13}'
        if random.random() < crossoverrate:
            O00, O01, O02, D00, P00, A00, O03, D01, P01, P02, P03 = new_born1.split(' ', 10)
            O10, O11, O12, D10, P10, A10, O13, D11, P11, P12, P13 = new_born2.split(' ', 10)
            new_born1 = f'{O00} {O01} {O12} {D00} {P10} {A00} {O03} {D01} {P01} {P02} {P03}'
            new_born2 = f'{O10} {O11} {O02} {D10} {P00} {A10} {O13} {D11} {P11} {P12} {P13}'
        if random.random() < crossoverrate:
            O00, O01, O02, D00, P00, A00, O03, D01, P01, P02, P03 = new_born1.split(' ', 10)
            O10, O11, O12, D10, P10, A10, O13, D11, P11, P12, P13 = new_born2.split(' ', 10)
            new_born1 = f'{O00} {O01} {O02} {D10} {P00} {A00} {O03} {D01} {P01} {P02} {P03}'
            new_born2 = f'{O10} {O11} {O12} {D00} {P10} {A10} {O13} {D11} {P11} {P12} {P13}'
        if random.random() < crossoverrate:
            O00, O01, O02, D00, P00, A00, O03, D01, P01, P02, P03 = new_born1.split(' ', 10)
            O10, O11, O12, D10, P10, A10, O13, D11, P11, P12, P13 = new_born2.split(' ', 10)
            new_born1 = f'{O00} {O01} {O02} {D00} {P00} {A10} {O03} {D01} {P01} {P02} {P03}'
            new_born2 = f'{O10} {O11} {O12} {D10} {P10} {A00} {O13} {D11} {P11} {P12} {P13}'
        if random.random() < crossoverrate:
            O00, O01, O02, D00, P00, A00, O03, D01, P01, P02, P03 = new_born1.split(' ', 10)
            O10, O11, O12, D10, P10, A10, O13, D11, P11, P12, P13 = new_born2.split(' ', 10)
            new_born1 = f'{O00} {O01} {O02} {D00} {P00} {A00} {O13} {D01} {P11} {P02} {P03}'
            new_born2 = f'{O10} {O11} {O12} {D10} {P10} {A10} {O03} {D11} {P01} {P12} {P13}'
        if random.random() < crossoverrate:
            O00, O01, O02, D00, P00, A00, O03, D01, P01, P02, P03 = new_born1.split(' ', 10)
            O10, O11, O12, D10, P10, A10, O13, D11, P11, P12, P13 = new_born2.split(' ', 10)
            new_born1 = f'{O00} {O01} {O02} {D00} {P00} {A00} {O03} {D11} {P01} {P02} {P03}'
            new_born2 = f'{O10} {O11} {O12} {D10} {P10} {A10} {O13} {D01} {P11} {P12} {P13}'
        expression_list.append(new_born1)
        expression_list.append(new_born2)
    return expression_list

def mutate(expression_list, mutationrate = 0.01, intensity = 0.5):
    Ds = [random.choice(D) for _ in range(2)]
    Ps = [random.choice(P) for _ in range(4)]
    Os = [random.choice(O) for _ in range(4)]
    for index, expression in enumerate(expression_list):
        print
        if random.random() < mutationrate:
            new_mutation = expression
            if random.random() < intensity:
                O0, O1, O2, D0, P0, A0, O3, D1, P1, P2, P3 = new_mutation.split(' ', 10)
                new_mutation = f'{Os[0]} {O1} {O2} {D0} {P0} {A0} {O3} {D1} {P1} {P2} {Ps[0]}'
            if random.random() < intensity:
                O0, O1, O2, D0, P0, A0, O3, D1, P1, P2, P3 = new_mutation.split(' ', 10)
                new_mutation = f'{O0} {Os[1]} {O2} {D0} {P0} {A0} {O3} {D1} {P1} {Ps[1]} {P3}'
            if random.random() < intensity:
                O0, O1, O2, D0, P0, A0, O3, D1, P1, P2, P3 = new_mutation.split(' ', 10)
                new_mutation = f'{O0} {O1} {Os[2]} {D0} {Ps[2]} {A0} {O3} {D1} {P1} {P2} {P3}'
            if random.random() < intensity:
                O0, O1, O2, D0, P0, A0, O3, D1, P1, P2, P3 = new_mutation.split(' ', 10)
                new_mutation = f'{O0} {O1} {O2} {Ds[0]} {P0} {A0} {O3} {D1} {P1} {P2} {P3}'
            if random.random() < intensity:
                O0, O1, O2, D0, P0, A0, O3, D1, P1, P2, P3 = new_mutation.split(' ', 10)
                new_mutation = f'{O0} {O1} {O2} {D0} {P0} {random.choice(A)} {O3} {D1} {P1} {P2} {P3}'
            if random.random() < intensity:
                O0, O1, O2, D0, P0, A0, O3, D1, P1, P2, P3 = new_mutation.split(' ', 10)
                new_mutation = f'{O0} {O1} {O2} {D0} {P0} {A0} {Os[3]} {D1} {Ps[3]} {P2} {P3}'
            if random.random() < intensity:
                O0, O1, O2, D0, P0, A0, O3, D1, P1, P2, P3 = new_mutation.split(' ', 10)
                new_mutation = f'{O0} {O1} {O2} {D0} {P0} {A0} {O3} {Ds[1]} {P1} {P2} {P3}'
            expression_list[index] = new_mutation
        else:
            expression_list[index] = expression
    return expression_list
