import pandas as pd
import numpy as np
import random
from typing import Iterable
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

### SS means shareholders structure !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
target_folder_path = '/Users/tedting/Documents/Alpha/data'

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
Volume_Filter = Volume_y_avg > 200
trade_volume_Filter = Close_y_avg * Volume_y_avg > 60000
small_aum_Filter = Volume_Filter & trade_volume_Filter
small_aum_Filter_L = Volume_Filter & trade_volume_Filter & heavy_Filter & Limit

### dataset資料(1)
本益比 = pd.read_pickle(rf'{target_folder_path}/本益比-TSE.pkl')
現金股利率 = pd.read_pickle(rf'{target_folder_path}/現金股利率.pkl')

### dataset資料(2)
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

data_list = ['Close', 'Volume', 'Low', 'High', '本益比', '現金股利率',
    '一以下',
    '一到五',
    '五到十',
    '十到十五',
    '十五到二十',
    '二十到三十',
    '三十到四十',
    '四十到五十',
    '五十到一百',
    '一百到二百',
    '二百到四百', 
    '四百到六百', 
    '六百到八百',
    '八百到一千',
    '一千以上',
    'df_1',
    'df_0']

Arithmetic_list = ['+', '-', '*', '/']
Operators_list = ['ts_delta(', 'ts_max(', 'ts_min(', 'ts_rank(', 'ts_stddev(', 'ts_sum(', 'ts_decay(']
Period_list = [',1)', ',5)', ',21)', ',63)', ',126)', ',252)'] # 1是for decay 不變
D = data_list
A = Arithmetic_list
O = Operators_list
P = Period_list

### operators
def ts_min(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).min()
def ts_max(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).max()
def ts_sum(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).sum()
def ts_stddev(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).std()
def ts_rank(x: pd.DataFrame, d:int):
    return x.rolling(d).rank(pct=True)
def ts_delta(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.diff(d)
def ts_decay(x: pd.DataFrame, d:int) -> pd.DataFrame: # d=1 時 會等於不做decay
    # 過去 d 天的加權移動平均線，權重線性衰減 d, d ‒ 1, ..., 1（重新調整為總和為 1）
    result = x.values.copy()
    with np.errstate(all="ignore"):
        for i in range(1, d):
            result[i:] += (i+1) * x.values[:-i]
    result[:d] = np.nan
    return pd.DataFrame(result / np.arange(1, d+1).sum(),index = x.index,columns = x.columns)

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
       start_time='2013-01-01',
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
    'Sharpe':[round(sharpe(daily_returns), 4)],
    'CAGR': [round(annual_returns(daily_returns), 4)],
    # 'MDD':[round(MDD(daily_returns),4)],
    # 'Turnover':[round(turnover(alpha), 4)]
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
    elif filter == 'small_aum_Filter2_L':
        alpha = expression[small_aum_Filter_L]
    else:
        raise ValueError("Please use 'None', 'small_aum_Filter', or 'small_aum_Filter2_L' in filter")
    
    daily_weight = weight(alpha, strategy = strategy).loc[start_time:end_time]
    expreturn = expreturn.loc[start_time:end_time]
    daily_fee = fee(alpha, buy_fee=buy_fee, sell_fee=sell_fee, start_time=start_time, end_time=end_time)
    daily_profit = (daily_weight * expreturn).sum(axis=1)
    daily_returns = daily_profit - daily_fee
    
    summary_df = pd.DataFrame({
    'expression':[expression], 
    'Sharpe':[round(sharpe(daily_returns), 4)],
    'CAGR': [round(annual_returns(daily_returns), 4)],
    # 'MDD':[round(MDD(daily_returns),4)],
    # 'Turnover':[round(turnover(alpha), 4)]
    }, index = ['Performance'])
    return daily_returns, summary_df # pd.DataFrame

### BT_Index
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
    peak = LS_compound.iloc[0]
    for price in LS_compound:
        if price > peak:
            peak = price
        drawdown = (price - peak) / peak
        drawdowns.append(drawdown)
    max_drawdown = np.min(drawdowns)
    return abs(max_drawdown)

### GA_function
def generate_expression():
    Os = [random.choice(O) for _ in range(3)]
    Ps = [random.choice(P) for _ in range(3)]
    Ds = [random.choice(D) for _ in range(9)]
    As = [random.choice(A) for _ in range(8)]
    expression = f'{Os[0]} {Os[1]} {Os[2]} ( {Ds[8]} {As[0]} {Ds[0]} {As[1]} {Ds[1]} {As[2]} {Ds[2]} {As[3]} {Ds[3]} / {Ds[4]} {As[5]} {Ds[5]} {As[6]} {Ds[6]} {As[7]} {Ds[7]} ) {Ps[0]} {Ps[1]} {Ps[2]}'
    # 記得O與Ｐ不同位置不對應
    return expression
def GENE_split(expression):
    O0, rest = expression.split(' ', 1)
    Opart, rest = rest.split(' ( ', 1)
    O1, O2 = Opart.split(' ', 1)
    DApart, Ppart = rest.split(' ) ', 16) 
    D1, A1, D2, A2, D3, A3, D4, A4, D5, A5, D6, A6, D7, A7, D8, A8, D9= DApart.split(' ', 16) # 9D+8A=17 有16個空格
    P1, P2, P3 = Ppart.split(' ', 2)
    return O0, O1, O2, D1, A1, D2, A2, D3, A3, D4, A4, D5, A5, D6, A6, D7, A7, D8, A8, D9, P1, P2, P3
def crossover(expression_list, crossoverrate=0.5):
    if len(expression_list) % 2 != 0:
        expression_list.pop()
        
    for i in range(0, len(expression_list), 2):
        O01, O02, O03, D01, A01, D02, A02, D03, A03, D04, A04, D05, A05, D06, A06, D07, A07, D08, A08, D09, P01, P02, P03 = GENE_split(expression_list[i])
        O11, O12, O13, D11, A11, D12, A12, D13, A13, D14, A14, D15, A15, D16, A16, D17, A17, D18, A18, D19, P11, P12, P13 = GENE_split(expression_list[i+1])
        new_born1 = f'{O01} {O02} {O03} ( {D01} {A01} {D02} {A02} {D03} {A03} {D04} {A04} {D05} {A05} {D06} {A06} {D07} {A07} {D08} {A08} {D09} ) {P01} {P02} {P03}'      
        new_born2 = f'{O11} {O12} {O13} ( {D11} {A11} {D12} {A12} {D13} {A13} {D14} {A14} {D15} {A15} {D16} {A16} {D17} {A17} {D18} {A18} {D19} ) {P11} {P12} {P13}' 
        if random.random() < crossoverrate:
            O01, O02, O03, D01, A01, D02, A02, D03, A03, D04, A04, D05, A05, D06, A06, D07, A07, D08, A08, D09, P01, P02, P03 = GENE_split(new_born1)
            O11, O12, O13, D11, A11, D12, A12, D13, A13, D14, A14, D15, A15, D16, A16, D17, A17, D18, A18, D19, P11, P12, P13 = GENE_split(new_born2)
            new_born1 = f'{O11} {O01} {O02} ( {D01} {A01} {D02} {A02} {D03} {A03} {D04} {A04} {D05} {A05} {D06} {A06} {D07} {A07} {D08} {A08} {D09} ) {P01} {P02} {P13}'      
            new_born2 = f'{O01} {O02} {O12} ( {D11} {A11} {D12} {A12} {D13} {A13} {D14} {A14} {D15} {A15} {D16} {A16} {D17} {A17} {D18} {A18} {D19} ) {P11} {P12} {P03}'
        if random.random() < crossoverrate:
            O01, O02, O03, D01, A01, D02, A02, D03, A03, D04, A04, D05, A05, D06, A06, D07, A07, D08, A08, D09, P01, P02, P03 = GENE_split(new_born1)
            O11, O12, O13, D11, A11, D12, A12, D13, A13, D14, A14, D15, A15, D16, A16, D17, A17, D18, A18, D19, P11, P12, P13 = GENE_split(new_born2)
            new_born1 = f'{O01} {O11} {O02} ( {D01} {A01} {D02} {A02} {D03} {A03} {D04} {A04} {D05} {A05} {D06} {A06} {D07} {A07} {D08} {A08} {D09} ) {P01} {P12} {P03}'      
            new_born2 = f'{O11} {O01} {O12} ( {D11} {A11} {D12} {A12} {D13} {A13} {D14} {A14} {D15} {A15} {D16} {A16} {D17} {A17} {D18} {A18} {D19} ) {P11} {P02} {P13}'
        if random.random() < crossoverrate:
            O01, O02, O03, D01, A01, D02, A02, D03, A03, D04, A04, D05, A05, D06, A06, D07, A07, D08, A08, D09, P01, P02, P03 = GENE_split(new_born1)
            O11, O12, O13, D11, A11, D12, A12, D13, A13, D14, A14, D15, A15, D16, A16, D17, A17, D18, A18, D19, P11, P12, P13 = GENE_split(new_born2)
            new_born1 = f'{O01} {O02} {O13} ( {D01} {A01} {D02} {A02} {D03} {A03} {D04} {A04} {D05} {A05} {D06} {A06} {D07} {A07} {D08} {A08} {D09} ) {P11} {P02} {P03}'      
            new_born2 = f'{O11} {O12} {O03} ( {D11} {A11} {D12} {A12} {D13} {A13} {D14} {A14} {D15} {A15} {D16} {A16} {D17} {A17} {D18} {A18} {D19} ) {P01} {P12} {P13}'
        if random.random() < crossoverrate:
            O01, O02, O03, D01, A01, D02, A02, D03, A03, D04, A04, D05, A05, D06, A06, D07, A07, D08, A08, D09, P01, P02, P03 = GENE_split(new_born1)
            O11, O12, O13, D11, A11, D12, A12, D13, A13, D14, A14, D15, A15, D16, A16, D17, A17, D18, A18, D19, P11, P12, P13 = GENE_split(new_born2)
            new_born1 = f'{O01} {O02} {O03} ( {D11} {A01} {D02} {A02} {D03} {A03} {D04} {A04} {D05} {A05} {D06} {A06} {D07} {A07} {D08} {A08} {D09} ) {P01} {P02} {P03}'      
            new_born2 = f'{O11} {O12} {O13} ( {D01} {A11} {D12} {A12} {D13} {A13} {D14} {A14} {D15} {A15} {D16} {A16} {D17} {A17} {D18} {A18} {D19} ) {P11} {P12} {P13}'
        if random.random() < crossoverrate:
            O01, O02, O03, D01, A01, D02, A02, D03, A03, D04, A04, D05, A05, D06, A06, D07, A07, D08, A08, D09, P01, P02, P03 = GENE_split(new_born1)
            O11, O12, O13, D11, A11, D12, A12, D13, A13, D14, A14, D15, A15, D16, A16, D17, A17, D18, A18, D19, P11, P12, P13 = GENE_split(new_born2)
            new_born1 = f'{O01} {O02} {O03} ( {D01} {A11} {D12} {A02} {D03} {A03} {D04} {A04} {D05} {A05} {D06} {A06} {D07} {A07} {D08} {A08} {D09} ) {P01} {P02} {P03}'      
            new_born2 = f'{O11} {O12} {O13} ( {D11} {A01} {D02} {A12} {D13} {A13} {D14} {A14} {D15} {A15} {D16} {A16} {D17} {A17} {D18} {A18} {D19} ) {P11} {P12} {P13}'
        if random.random() < crossoverrate:
            O01, O02, O03, D01, A01, D02, A02, D03, A03, D04, A04, D05, A05, D06, A06, D07, A07, D08, A08, D09, P01, P02, P03 = GENE_split(new_born1)
            O11, O12, O13, D11, A11, D12, A12, D13, A13, D14, A14, D15, A15, D16, A16, D17, A17, D18, A18, D19, P11, P12, P13 = GENE_split(new_born2)
            new_born1 = f'{O01} {O02} {O03} ( {D01} {A01} {D02} {A12} {D13} {A03} {D04} {A04} {D05} {A05} {D06} {A06} {D07} {A07} {D08} {A08} {D09} ) {P01} {P02} {P03}'      
            new_born2 = f'{O11} {O12} {O13} ( {D11} {A11} {D12} {A02} {D03} {A13} {D14} {A14} {D15} {A15} {D16} {A16} {D17} {A17} {D18} {A18} {D19} ) {P11} {P12} {P13}'
        if random.random() < crossoverrate:
            O01, O02, O03, D01, A01, D02, A02, D03, A03, D04, A04, D05, A05, D06, A06, D07, A07, D08, A08, D09, P01, P02, P03 = GENE_split(new_born1)
            O11, O12, O13, D11, A11, D12, A12, D13, A13, D14, A14, D15, A15, D16, A16, D17, A17, D18, A18, D19, P11, P12, P13 = GENE_split(new_born2)
            new_born1 = f'{O01} {O02} {O03} ( {D01} {A01} {D02} {A02} {D03} {A13} {D14} {A04} {D05} {A05} {D06} {A06} {D07} {A07} {D08} {A08} {D09} ) {P01} {P02} {P03}'      
            new_born2 = f'{O11} {O12} {O13} ( {D11} {A11} {D12} {A12} {D13} {A03} {D04} {A14} {D15} {A15} {D16} {A16} {D17} {A17} {D18} {A18} {D19} ) {P11} {P12} {P13}'
        if random.random() < crossoverrate:
            O01, O02, O03, D01, A01, D02, A02, D03, A03, D04, A04, D05, A05, D06, A06, D07, A07, D08, A08, D09, P01, P02, P03 = GENE_split(new_born1)
            O11, O12, O13, D11, A11, D12, A12, D13, A13, D14, A14, D15, A15, D16, A16, D17, A17, D18, A18, D19, P11, P12, P13 = GENE_split(new_born2)
            new_born1 = f'{O01} {O02} {O03} ( {D01} {A01} {D02} {A02} {D03} {A03} {D04} {A14} {D15} {A05} {D06} {A06} {D07} {A07} {D08} {A08} {D09} ) {P01} {P02} {P03}'      
            new_born2 = f'{O11} {O12} {O13} ( {D11} {A11} {D12} {A12} {D13} {A13} {D14} {A04} {D05} {A15} {D16} {A16} {D17} {A17} {D18} {A18} {D19} ) {P11} {P12} {P13}'
        if random.random() < crossoverrate:
            O01, O02, O03, D01, A01, D02, A02, D03, A03, D04, A04, D05, A05, D06, A06, D07, A07, D08, A08, D09, P01, P02, P03 = GENE_split(new_born1)
            O11, O12, O13, D11, A11, D12, A12, D13, A13, D14, A14, D15, A15, D16, A16, D17, A17, D18, A18, D19, P11, P12, P13 = GENE_split(new_born2)
            new_born1 = f'{O01} {O02} {O03} ( {D01} {A01} {D02} {A02} {D03} {A03} {D04} {A04} {D05} / {D06} {A06} {D07} {A07} {D08} {A08} {D09} ) {P01} {P02} {P03}'      
            new_born2 = f'{O11} {O12} {O13} ( {D11} {A11} {D12} {A12} {D13} {A13} {D14} {A14} {D15} / {D16} {A16} {D17} {A17} {D18} {A18} {D19} ) {P11} {P12} {P13}'
        if random.random() < crossoverrate:
            O01, O02, O03, D01, A01, D02, A02, D03, A03, D04, A04, D05, A05, D06, A06, D07, A07, D08, A08, D09, P01, P02, P03 = GENE_split(new_born1)
            O11, O12, O13, D11, A11, D12, A12, D13, A13, D14, A14, D15, A15, D16, A16, D17, A17, D18, A18, D19, P11, P12, P13 = GENE_split(new_born2)
            new_born1 = f'{O01} {O02} {O03} ( {D01} {A01} {D02} {A02} {D03} {A03} {D04} {A04} {D05} {A05} {D06} {A16} {D17} {A07} {D08} {A08} {D09} ) {P01} {P02} {P03}'
            new_born2 = f'{O11} {O12} {O13} ( {D11} {A11} {D12} {A12} {D13} {A13} {D14} {A14} {D15} {A15} {D16} {A06} {D07} {A17} {D18} {A18} {D19} ) {P11} {P12} {P13}'
        if random.random() < crossoverrate:
            O01, O02, O03, D01, A01, D02, A02, D03, A03, D04, A04, D05, A05, D06, A06, D07, A07, D08, A08, D09, P01, P02, P03 = GENE_split(new_born1)
            O11, O12, O13, D11, A11, D12, A12, D13, A13, D14, A14, D15, A15, D16, A16, D17, A17, D18, A18, D19, P11, P12, P13 = GENE_split(new_born2)
            new_born1 = f'{O01} {O02} {O03} ( {D01} {A01} {D02} {A02} {D03} {A03} {D04} {A04} {D05} {A05} {D06} {A06} {D07} {A17} {D18} {A08} {D09} ) {P01} {P02} {P03}'      
            new_born2 = f'{O11} {O12} {O13} ( {D11} {A11} {D12} {A12} {D13} {A13} {D14} {A14} {D15} {A15} {D16} {A16} {D17} {A07} {D08} {A18} {D19} ) {P11} {P12} {P13}'
        if random.random() < crossoverrate:
            O01, O02, O03, D01, A01, D02, A02, D03, A03, D04, A04, D05, A05, D06, A06, D07, A07, D08, A08, D09, P01, P02, P03 = GENE_split(new_born1)
            O11, O12, O13, D11, A11, D12, A12, D13, A13, D14, A14, D15, A15, D16, A16, D17, A17, D18, A18, D19, P11, P12, P13 = GENE_split(new_born2)
            new_born1 = f'{O01} {O02} {O03} ( {D01} {A01} {D02} {A02} {D03} {A03} {D04} {A04} {D05} {A05} {D06} {A06} {D07} {A07} {D08} {A18} {D19} ) {P01} {P02} {P03}'      
            new_born2 = f'{O11} {O12} {O13} ( {D11} {A11} {D12} {A12} {D13} {A13} {D14} {A14} {D15} {A15} {D16} {A16} {D17} {A17} {D18} {A08} {D09} ) {P11} {P12} {P13}'
        expression_list.append(new_born1)
        expression_list.append(new_born2)
    return expression_list
def mutate(expression_list, mutationrate = 0.01, intensity = 0.5):
    for index, expression in enumerate(expression_list):
        if random.random() < mutationrate:
            O0, O1, O2, D1, A1, D2, A2, D3, A3, D4, A4, D5, A5, D6, A6, D7, A7, D8, A8, D9, P1, P2, P3 = GENE_split(expression)
            new_mutation = f'{O0} {O1} {O2} ( {D1} {A1} {D2} {A2} {D3} {A3} {D4} {A4} {D5} {A5} {D6} {A6} {D7} {A7} {D8} {A8} {D9} ) {P1} {P2} {P3}' # 這兩行其實可以直接寫成 new_mutation = GENE_split(expression)
            Os = [random.choice(O) for _ in range(3)]
            Ps = [random.choice(P) for _ in range(3)]
            Ds = [random.choice(D) for _ in range(9)]
            As = [random.choice(A) for _ in range(8)]
            if random.random() < intensity:
                O0, O1, O2, D1, A1, D2, A2, D3, A3, D4, A4, D5, A5, D6, A6, D7, A7, D8, A8, D9, P1, P2, P3 = GENE_split(new_mutation)
                new_mutation = f'{Os[0]} {O1} {O2} ( {D1} {A1} {D2} {A2} {D3} {A3} {D4} {A4} {D5} {A5} {D6} {A6} {D7} {A7} {D8} {A8} {D9} ) {P1} {P2} {Ps[0]}'
            if random.random() < intensity:
                O0, O1, O2, D1, A1, D2, A2, D3, A3, D4, A4, D5, A5, D6, A6, D7, A7, D8, A8, D9, P1, P2, P3 = GENE_split(new_mutation)
                new_mutation = f'{O0} {Os[1]} {O2} ( {D1} {A1} {D2} {A2} {D3} {A3} {D4} {A4} {D5} {A5} {D6} {A6} {D7} {A7} {D8} {A8} {D9} ) {P1} {Ps[1]} {P3}'
            if random.random() < intensity:
                O0, O1, O2, D1, A1, D2, A2, D3, A3, D4, A4, D5, A5, D6, A6, D7, A7, D8, A8, D9, P1, P2, P3 = GENE_split(new_mutation)
                new_mutation = f'{O0} {O1} {Os[2]} ( {D1} {A1} {D2} {A2} {D3} {A3} {D4} {A4} {D5} {A5} {D6} {A6} {D7} {A7} {D8} {A8} {D9} ) {Ps[2]} {P2} {P3}'
            if random.random() < intensity:
                O0, O1, O2, D1, A1, D2, A2, D3, A3, D4, A4, D5, A5, D6, A6, D7, A7, D8, A8, D9, P1, P2, P3 = GENE_split(new_mutation)
                new_mutation = f'{O0} {O1} {O2} ( {Ds[8]} {A1} {D2} {A2} {D3} {A3} {D4} {A4} {D5} {A5} {D6} {A6} {D7} {A7} {D8} {A8} {D9} ) {P1} {P2} {P3}'
            if random.random() < intensity:
                O0, O1, O2, D1, A1, D2, A2, D3, A3, D4, A4, D5, A5, D6, A6, D7, A7, D8, A8, D9, P1, P2, P3 = GENE_split(new_mutation)
                new_mutation = f'{O0} {O1} {O2} ( {D1} {As[0]} {Ds[0]} {A2} {D3} {A3} {D4} {A4} {D5} {A5} {D6} {A6} {D7} {A7} {D8} {A8} {D9} ) {P1} {P2} {P3}'
            if random.random() < intensity:
                O0, O1, O2, D1, A1, D2, A2, D3, A3, D4, A4, D5, A5, D6, A6, D7, A7, D8, A8, D9, P1, P2, P3 = GENE_split(new_mutation)
                new_mutation = f'{O0} {O1} {O2} ( {D1} {A1} {D2} {As[1]} {Ds[1]} {A3} {D4} {A4} {D5} {A5} {D6} {A6} {D7} {A7} {D8} {A8} {D9} ) {P1} {P2} {P3}'
            if random.random() < intensity:
                O0, O1, O2, D1, A1, D2, A2, D3, A3, D4, A4, D5, A5, D6, A6, D7, A7, D8, A8, D9, P1, P2, P3 = GENE_split(new_mutation)
                new_mutation = f'{O0} {O1} {O2} ( {D1} {A1} {D2} {A2} {D3} {As[2]} {Ds[2]} {A4} {D5} {A5} {D6} {A6} {D7} {A7} {D8} {A8} {D9} ) {P1} {P2} {P3}'
            if random.random() < intensity:
                O0, O1, O2, D1, A1, D2, A2, D3, A3, D4, A4, D5, A5, D6, A6, D7, A7, D8, A8, D9, P1, P2, P3 = GENE_split(new_mutation)
                new_mutation = f'{O0} {O1} {O2} ( {D1} {A1} {D2} {A2} {D3} {A3} {D4} {As[3]} {Ds[3]} {A5} {D6} {A6} {D7} {A7} {D8} {A8} {D9} ) {P1} {P2} {P3}'
            if random.random() < intensity:
                O0, O1, O2, D1, A1, D2, A2, D3, A3, D4, A4, D5, A5, D6, A6, D7, A7, D8, A8, D9, P1, P2, P3 = GENE_split(new_mutation)
                new_mutation = f'{O0} {O1} {O2} ( {D1} {A1} {D2} {A2} {D3} {A3} {D4} {A4} {D5} / {Ds[4]} {A6} {D7} {A7} {D8} {A8} {D9} ) {P1} {P2} {P3}'
            if random.random() < intensity:
                O0, O1, O2, D1, A1, D2, A2, D3, A3, D4, A4, D5, A5, D6, A6, D7, A7, D8, A8, D9, P1, P2, P3 = GENE_split(new_mutation)
                new_mutation = f'{O0} {O1} {O2} ( {D1} {A1} {D2} {A2} {D3} {A3} {D4} {A4} {D5} {A5} {D6} {As[5]} {Ds[5]} {A7} {D8} {A8} {D9} ) {P1} {P2} {P3}'
            if random.random() < intensity:
                O0, O1, O2, D1, A1, D2, A2, D3, A3, D4, A4, D5, A5, D6, A6, D7, A7, D8, A8, D9, P1, P2, P3 = GENE_split(new_mutation)
                new_mutation = f'{O0} {O1} {O2} ( {D1} {A1} {D2} {A2} {D3} {A3} {D4} {A4} {D5} {A5} {D6} {A6} {D7} {As[6]} {Ds[6]} {A8} {D9} ) {P1} {P2} {P3}'
            if random.random() < intensity:
                O0, O1, O2, D1, A1, D2, A2, D3, A3, D4, A4, D5, A5, D6, A6, D7, A7, D8, A8, D9, P1, P2, P3 = GENE_split(new_mutation)
                new_mutation = f'{O0} {O1} {O2} ( {D1} {A1} {D2} {A2} {D3} {A3} {D4} {A4} {D5} {A5} {D6} {A6} {D7} {A7} {D8} {As[7]} {Ds[7]} ) {P1} {P2} {P3}'
            expression_list[index] = new_mutation
        else:
            expression_list[index] = expression
    return expression_list



