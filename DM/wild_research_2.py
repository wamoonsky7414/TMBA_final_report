import pandas as pd
import numpy as np
import sys
from WR_function_2 import *
from typing import Iterable
import datetime
import pickle
import random
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

save_file = 'WR_result_20240115_500.csv'
t_file = rf'{target_folder_path}/data_DM/WR_trash_filter_list_q_20240115_101_500.pkl'

### Backtest field
strategy = 'LO'
filter = 'small_aum_Filter'
buy_fee=0.001425*0.3
sell_fee=0.001425*0.3+0.003
start_time='2013-01-01'
end_time='2021-12-31'

### Parameter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
sample = 7000
FF_Fitness = 0.1
SF_Fitness = 0.2
fitness = 'Sharpe'

trash_filter_list = trash_filter()

expression_list = first_layer_generate(trash_filter_list, sample)
print('First_Layer_simulating')
final_summary = pd.DataFrame()
for num, expression in enumerate(expression_list):
    if num == len(expression_list)//4:
        print(rf'已完成 {num}')
    if num == len(expression_list)//2:
        print(rf'已完成 {num}')
    if num == len(expression_list)-100:
        print(rf'計算最後 100 個因子')

    stocks = 20
    alpha = eval(expression)[small_aum_Filter]
    top = alpha.apply(lambda row: row.nlargest(stocks), axis=1) #.fillna(0)
    top = top.applymap(lambda x: not pd.isna(x))
    top_equalweight = top.applymap(lambda x: 1/stocks if x else 0)
    # top_equalweight = ((AUM//stocks) // (Close[top]*1000))*Close[top]*1000 /AUM
            
    weight = top_equalweight.loc[start_time:end_time]
    expreturn = exp_returns.loc[start_time:end_time]
    delta_weight = weight.shift(1) - weight
    buy_fees = delta_weight[delta_weight > 0]*(buy_fee)
    buy_fees = buy_fees.fillna(0)
    sell_fees = delta_weight.abs()[delta_weight < 0]*(sell_fee)
    sell_fees = sell_fees.fillna(0)
    fee = buy_fees + sell_fees
    daily_fee = fee.sum(axis = 1)
    daily_profit = (weight * expreturn).sum(axis=1)
    daily_returns = daily_profit - daily_fee

    summary_df = pd.DataFrame({
    'expression':[expression],
    'Sharpe':[sharpe(daily_returns)],
    'Annualized Ret': [annual_returns(daily_returns)],
    'Max Drawdown':[MDD(daily_returns)],
    'Turnover':[turnover(alpha)], 
    'Std':[daily_returns.std()],
    }, index = ['Performance'])

    if (summary_df[fitness].item() < FF_Fitness) | (pd.isna(summary_df[fitness].item())):
        trash_filter_list.append(summary_df['expression'].item())
    else:
        final_summary = pd.concat([final_summary, summary_df], axis=0, ignore_index=True)
print('First simulate finish')

trash_filter_list = list(dict.fromkeys(trash_filter_list)) # 運用dict不能有重複值機制獲得不重複list
pickle_file = t_file
with open(pickle_file, 'wb') as file:
    pickle.dump(trash_filter_list, file)
sorted_df = final_summary.sort_values(by=[fitness], ascending=False)
print(sorted_df.head(10))
sorted_df.to_csv(rf'{target_folder_path}/data_DM/{save_file}', index=False, encoding='utf-8-sig')
expression_list = sorted_df['expression'].tolist()


expression_list = second_layer_generate(expression_list, trash_filter_list)
expression_list = set(expression_list)

print('Second_Layer_simulating')
final_summary = pd.DataFrame()
for num, expression in enumerate(expression_list):
    if num == len(expression_list)//4:
        print(rf'已完成 {num}')
    if num == len(expression_list)//2:
        print(rf'已完成 {num}')
    if num == len(expression_list)-100:
        print(rf'計算最後 100 個因子')

    stocks = 20
    alpha = eval(expression)[small_aum_Filter]
    top = alpha.apply(lambda row: row.nlargest(stocks), axis=1) #.fillna(0)
    top = top.applymap(lambda x: not pd.isna(x))
    top_equalweight = top.applymap(lambda x: 1/stocks if x else 0)
    # top_equalweight = ((AUM//stocks) // (Close[top]*1000))*Close[top]*1000 /AUM
            
    weight = top_equalweight.loc[start_time:end_time]
    expreturn = exp_returns.loc[start_time:end_time]
    delta_weight = weight.shift(1) - weight
    buy_fees = delta_weight[delta_weight > 0]*(buy_fee)
    buy_fees = buy_fees.fillna(0)
    sell_fees = delta_weight.abs()[delta_weight < 0]*(sell_fee)
    sell_fees = sell_fees.fillna(0)
    fee = buy_fees + sell_fees
    daily_fee = fee.sum(axis = 1)
    daily_profit = (weight * expreturn).sum(axis=1)
    daily_returns = daily_profit - daily_fee

    summary_df = pd.DataFrame({
    'expression':[expression],
    'Sharpe':[sharpe(daily_returns)],
    'Annualized Ret': [annual_returns(daily_returns)],
    'Max Drawdown':[MDD(daily_returns)],
    'Turnover':[turnover(alpha)], 
    'Std':[daily_returns.std()],
    }, index = ['Performance'])

    if (summary_df[fitness].item() < SF_Fitness) | (pd.isna(summary_df[fitness].item())):
        trash_filter_list.append(summary_df['expression'].item())
    else:
        final_summary = pd.concat([final_summary, summary_df], axis=0, ignore_index=True)
print('Second simulate finish')

trash_filter_list = list(dict.fromkeys(trash_filter_list))
pickle_file = t_file
with open(pickle_file, 'wb') as file:
    pickle.dump(trash_filter_list, file)        
sorted_df = final_summary.sort_values(by=[fitness], ascending=False)
print(sorted_df.head(10))

sorted_df.to_csv(rf'{target_folder_path}/data_DM/{save_file}', index=False, encoding='utf-8-sig')
expression_list = sorted_df['expression'].tolist()

### 變成存pickle檔，存因子(index)和分別的daily_return