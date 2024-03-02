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
D = data_list
A = arithmetic_list
O = ts_operators_list
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
    Ds = [random.choice(D) for _ in range(4)]
    As = [random.choice(A) for _ in range(3)]
    Ts = [random.choice(O) for _ in range(2)]
    Ps = [random.choice(P) for _ in range(2)]
    Cs = [random.choice(C) for _ in range(4)]
    expression = f'{Ts[0]} {Cs[0]} {Ds[0]} ) {As[0]} {Cs[1]} {Ds[1]} ) {Ps[0]} {As[1]} {Ts[1]} {Cs[2]} {Ds[2]} ) {As[2]} {Cs[3]} {Ds[3]} ) {Ps[1]}' # 記得O與Ｐ不同位置不對應
    return expression

def gene_split(expression):
    T0, rest = expression.split(' ', 1)
    C0, rest = rest.split(' ', 1)
    D0, rest = rest.split(' ) ', 1)
    A0, rest = rest.split(' ', 1)
    C1, rest = rest.split(' ', 1)
    D1, rest = rest.split(' ) ', 1)
    P0, rest = rest.split(' ', 1)
    A1, rest = rest.split(' ', 1)
    T1, rest = rest.split(' ', 1)
    C2, rest = rest.split(' ', 1)
    D2, rest = rest.split(' ) ', 1)
    A2, rest = rest.split(' ', 1)
    C3, rest = rest.split(' ', 1)
    D3, P1 = rest.split(' ) ', 1)
    return T0, C0, D0, A0, C1, D1, P0, A1, T1, C2, D2, A2, C3, D3, P1

def perform_crossover(genes0_list, genes1_list, crossoverrate):
    for i in range(len(genes0_list)):
        if random.random() < crossoverrate:
            genes0_list[i], genes1_list[i] = genes0_list[i], genes1_list[i]
    return genes0_list, genes1_list

def crossover(expression_list, crossoverrate=0.5):
    working_list = expression_list.copy()
    if len(working_list) % 2 != 0:
        working_list.pop()
    
    while len(working_list) > 0:
        # 从表达式列表中随机选择两个不同的表达式
        selected = random.sample(working_list, 2) # 選到的池子內相互交配(池子內沒有先後強弱之分)
        working_list.remove(selected[0])
        working_list.remove(selected[1])
        genes0 = gene_split(selected[0]) # T00, C00, D00, A00, C01,  D01, P00, A01, T01, C02, D02, A02, C03, D03, P01 = gene_split(selected[0])
        genes1 = gene_split(selected[1]) # T10, C10, D10, A10, C11,  D11, P10, A11, T11, C12, D12, A12, C13, D13, P11 = gene_split(selected[1])
        # 執行交叉交易
        n_g1, n_g2 = perform_crossover(list(genes0), list(genes1), crossoverrate)
        new_born1 = f'{n_g1[0]} {n_g1[1]} {n_g1[2]} ) {n_g1[3]} {n_g1[4]} {n_g1[5]} ) {n_g1[6]} {n_g1[7]} {n_g1[8]} {n_g1[9]} {n_g1[10]} ) {n_g1[11]} {n_g1[12]} {n_g1[13]} ) {n_g1[14]}' 
        new_born2 = f'{n_g2[0]} {n_g2[1]} {n_g2[2]} ) {n_g2[3]} {n_g2[4]} {n_g2[5]} ) {n_g2[6]} {n_g2[7]} {n_g2[8]} {n_g2[9]} {n_g2[10]} ) {n_g2[11]} {n_g2[12]} {n_g2[13]} ) {n_g2[14]}' 
        expression_list.append(new_born1)
        expression_list.append(new_born2)
    return expression_list

def perform_mutate(genes, choices, index, intensity):
    if random.random() < intensity:
        genes[index] = random.choice(choices)
    return genes

def mutate(expression_list, mutationrate = 0.01, intensity = 0.5):
    Ds = [random.choice(D) for _ in range(4)]
    As = [random.choice(A) for _ in range(3)]
    Ts = [random.choice(O) for _ in range(2)]
    Ps = [random.choice(P) for _ in range(2)]
    Cs = [random.choice(C) for _ in range(4)]
    for index, expression in enumerate(expression_list):
        if random.random() < mutationrate:
            # T0, C0, D0, A0, C1, D1, P0, A1, T1, C2, D2, A2, C3, D3, P1 = gene_split(expression)
            # new_mutation = f'{T0} {C0} {D0} ) {A0} {C1} {D1} ) {P0} {A1} {T1} {C2} {D2} ) {A2} {C3} {D3} ) {P1}' 
            genes = list(gene_split(expression))

            # 对基因序列中的每个位置进行随机突变
            genes = perform_mutate(genes, Ts, 0, intensity)
            genes = perform_mutate(genes, Cs, 1, intensity)
            genes = perform_mutate(genes, Ds, 2, intensity)
            genes = perform_mutate(genes, As, 3, intensity)
            genes = perform_mutate(genes, Cs, 4, intensity)
            genes = perform_mutate(genes, Ds, 5, intensity)
            genes = perform_mutate(genes, Ps, 6, intensity)
            genes = perform_mutate(genes, As, 7, intensity)
            genes = perform_mutate(genes, Ts, 8, intensity)
            genes = perform_mutate(genes, Cs, 9, intensity)
            genes = perform_mutate(genes, Ds, 10, intensity)
            genes = perform_mutate(genes, As, 11, intensity)
            genes = perform_mutate(genes, Cs, 12, intensity)
            genes = perform_mutate(genes, Ds, 13, intensity)
            genes = perform_mutate(genes, Ps, 14, intensity)
            # ... 根据需要继续添加更多的突变操作

            # 重新组合基因序列
            new_mutation = f'{genes[0]} {genes[1]} {genes[2]} ) {genes[3]} {genes[4]} {genes[5]} ) {genes[6]} {genes[7]} {genes[8]} {genes[9]} {genes[10]} ) {genes[11]} {genes[12]} {genes[13]} ) {genes[14]}' 
            expression_list[index] = new_mutation
        else:
            expression_list[index] = expression

    return expression_list
