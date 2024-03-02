import pandas as pd
import numpy as np
import sys
# from Alpha.DM.GA_function_SS import *
from GA_function_SS import *
from typing import Iterable
import datetime
import pickle
import random
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

### Parameter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
target_folder_path = '/Users/tedting/Documents/Alpha/data'

sample = 10000
evoluation_period = 15 ## 演算幾輪
fitness = 'Sharpe' # 'Sharpe', 'CAGR', 'MDD'
offspring_pool = 100 # 建議 100
crossoverrate = 0.5
mutationrate = 0.1
intensity = 0.5

# Backtest field
strategy='LO'
filter = 'small_aum_Filter'
buy_fee=0.001425*0.3
sell_fee=0.001425*0.3+0.003
start_time='2013-01-01'
end_time='2021-12-30'


### sample generation
sample_expressions = [generate_expression() for _ in range(sample)] 
expression_list = sample_expressions

convergence_curve = pd.DataFrame()
generation = 0
while generation < evoluation_period :
    final_summary = pd.DataFrame()
    print(f'Generation: {generation}')
    if generation == 0:
        print(f"Calcuating '{len(expression_list)}' sample's fitness")
    else:
        print(f"Calcuating '{len(expression_list)}' offspring's fitness")
    ### Fitness
    for expression in expression_list:
        stocks = 10
        alpha = eval(expression)[small_aum_Filter]
        top = alpha.apply(lambda row: row.nlargest(stocks), axis=1).fillna(0)
        top_equalweight = top.applymap(lambda x: 1/stocks if x != 0 else 0)

        weight = top_equalweight.loc[start_time:end_time]
        expreturn = exp_returns .loc[start_time:end_time]
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
        'Std':[daily_returns.std()],
        }, index = ['Performance'])
        final_summary = pd.concat([final_summary, summary_df], axis=0, ignore_index=True)
    
    if generation == 0:
        print('Sample generate over')
    else:
        print('Fitness calculate over')

    # Selection also included convergence_curve
    if fitness in ['Sharpe', 'CAGR']:
        sorted_df = final_summary.sort_values(by=fitness, ascending=False)
        convergence_curve_number = pd.DataFrame({
            'top_convergence_curve': [sorted_df[fitness].head(1).max()],    
            'top10_avg_convergence_curve': [sorted_df[fitness].head(10).mean()]
        }, index = [fitness])
        convergence_curve = pd.concat([convergence_curve, convergence_curve_number], axis=0, ignore_index=True)

        selected_df = sorted_df.head(offspring_pool)
        show_df = sorted_df.head(10)
        print(show_df)
        expression_list = selected_df['expression'].tolist()

        # 儲存到檔案
        selected_df.to_csv(rf'{target_folder_path}/data_DM/GA3_20240113_result.csv', index=False, encoding='utf-8-sig')
        convergence_curve.to_csv(rf'{target_folder_path}/data_DM/Convergence_20240113_curve.csv', index=False, encoding='utf-8-sig')
    else:
        raise ValueError("Please use 'Sharpe' or 'CAGR' in selection")

    if generation == (evoluation_period -1):
        print('Generate_over!!!')
        break

    ### Crossover
    expression_list = crossover(expression_list, crossoverrate=crossoverrate)
    ### Mutation
    expression_list = mutate(expression_list, mutationrate=mutationrate, intensity=intensity)
    expression_list = list(set(expression_list))
    print('Crossover and mutatation process over')
    generation += 1


