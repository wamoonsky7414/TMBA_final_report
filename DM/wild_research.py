import pandas as pd
import numpy as np
import sys
from WR_function import *
from typing import Iterable
import datetime
import pickle
import random
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

save_file = 'WR_result_20240111_1.csv'

### Backtest field
strategy = 'LO'
filter = 'small_aum_Filter'
buy_fee=0.001425*0.3
sell_fee=0.001425*0.3+0.003
start_time='2013-01-01'
end_time='2021-12-30'

### Parameter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
sample = 100
FF_Fitness = 0.7
SF_Fitness = 0.9
fitness = 'Sharpe'

trash_filter_list = trash_filter()

expression_list = first_layer_generate(trash_filter_list, sample)
print('First_Layer_simulating')
final_summary = pd.DataFrame()
for num, expression in enumerate(expression_list):
    if num == int(len(expression_list)/4):
        print(rf'已完成 {num}')
    if num == int(len(expression_list)/2):
        print(rf'已完成 {num}')
    if num == len(expression_list)-100:
        print(rf'計算最後 100 個因子')

    daily_returns, result_summary = bt_algo(expression, exp_returns,strategy=strategy,filter=filter,buy_fee=buy_fee, sell_fee=sell_fee,start_time=start_time,end_time=end_time)
    if (result_summary[fitness].item() < FF_Fitness) | (pd.isna(result_summary[fitness].item())):
        trash_filter_list.append(result_summary['expression'].item())
    else:
        final_summary = pd.concat([final_summary, result_summary], axis=0, ignore_index=True)
print('First simulate finish')

trash_filter_list = list(dict.fromkeys(trash_filter_list)) # 運用dict不能有重複值機制獲得不重複list
pickle_file = rf'{target_folder_path}/data_DM/WR_trash_filter_list.pkl'
with open(pickle_file, 'wb') as file:
    pickle.dump(trash_filter_list, file)
sorted_df = final_summary.sort_values(by=[fitness], ascending=False)
print(sorted_df.head(10))
sorted_df.to_csv(rf'{target_folder_path}/data_DM/{save_file}', index=False, encoding='utf-8-sig')
expression_list = sorted_df['expression'].tolist()


expression_list = second_layer_generate(expression_list, trash_filter_list)
expression_list = set(expression_list)
print(len(expression_list))
print('Second_Layer_simulating')
final_summary = pd.DataFrame()
for num, expression in enumerate(expression_list):
    if num == int(len(expression_list)/4):
        print(rf'已完成 {num}')
    if num == int(len(expression_list)/2):
        print(rf'已完成 {num}')
    if num == len(expression_list)-100:
        print(rf'計算最後 100 個因子')

    daily_returns, result_summary = bt_algo(expression, exp_returns,strategy=strategy,filter=filter,buy_fee=buy_fee, sell_fee=sell_fee,start_time=start_time,end_time=end_time)
    if (result_summary[fitness].item() < SF_Fitness) | (pd.isna(result_summary[fitness].item())):
        trash_filter_list.append(result_summary['expression'].item())
    else:
        final_summary = pd.concat([final_summary, result_summary], axis=0, ignore_index=True)
print('Second simulate finish')

trash_filter_list = list(dict.fromkeys(trash_filter_list))
pickle_file = rf'{target_folder_path}/data_DM/WR_trash_filter_list.pkl'
with open(pickle_file, 'wb') as file:
    pickle.dump(trash_filter_list, file)        
sorted_df = final_summary.sort_values(by=[fitness], ascending=False)
print(sorted_df.head(10))

sorted_df.to_csv(rf'{target_folder_path}/data_DM/{save_file}', index=False, encoding='utf-8-sig')
expression_list = sorted_df['expression'].tolist()

### 變成存pickle檔，存因子(index)和分別的daily_return