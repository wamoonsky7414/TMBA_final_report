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

### dataset資料(3)
個股可投資上限比率 = pd.read_pickle(rf'{target_folder_path}/data_daily/個股可投資上限比率.pkl')
個股買賣超股數 = pd.read_pickle(rf'{target_folder_path}/data_daily/個股買賣超股數.pkl')
合計持股數 = pd.read_pickle(rf'{target_folder_path}/data_daily/合計持股數(千股).pkl')
合計持股數市值 = pd.read_pickle(rf'{target_folder_path}/data_daily/合計持股數市值(百萬).pkl')
合計持股率 = pd.read_pickle(rf'{target_folder_path}/data_daily/合計持股率%.pkl')
合計買賣超 = pd.read_pickle(rf'{target_folder_path}/data_daily/合計買賣超(千股).pkl')
合計買賣超市值 = pd.read_pickle(rf'{target_folder_path}/data_daily/合計買賣超市值(百萬).pkl')
外資總投資市值 = pd.read_pickle(rf'{target_folder_path}/data_daily/外資總投資市值(百萬).pkl')
外資總投資比率 = pd.read_pickle(rf'{target_folder_path}/data_daily/外資總投資比率%-TSE.pkl')
外資總投資股數 = pd.read_pickle(rf'{target_folder_path}/data_daily/外資總投資股數.pkl')
外資總投資股率 = pd.read_pickle(rf'{target_folder_path}/data_daily/外資總投資股率%.pkl')
外資買賣超 = pd.read_pickle(rf'{target_folder_path}/data_daily/外資買賣超(千股).pkl')
外資買進張數 = pd.read_pickle(rf'{target_folder_path}/data_daily/外資買進張數.pkl')
外資賣出張數 = pd.read_pickle(rf'{target_folder_path}/data_daily/外資賣出張數.pkl')
外資週轉率 = pd.read_pickle(rf'{target_folder_path}/data_daily/外資週轉率%.pkl')
尚可投資比率 = pd.read_pickle(rf'{target_folder_path}/data_daily/尚可投資比率-TSE.pkl')
尚可投資股數 = pd.read_pickle(rf'{target_folder_path}/data_daily/尚可投資股數-TSE.pkl')
投信持股數 = pd.read_pickle(rf'{target_folder_path}/data_daily/投信持股數(千股).pkl')
投信持股數市值 = pd.read_pickle(rf'{target_folder_path}/data_daily/投信持股數市值(百萬).pkl')
投信持股率 = pd.read_pickle(rf'{target_folder_path}/data_daily/投信持股率%.pkl')
投信買賣超 = pd.read_pickle(rf'{target_folder_path}/data_daily/投信買賣超(千股).pkl')
投信買賣超市值 = pd.read_pickle(rf'{target_folder_path}/data_daily/投信買賣超市值(百萬).pkl')
投信買進張數 = pd.read_pickle(rf'{target_folder_path}/data_daily/投信買進張數.pkl')
投信賣出張數 = pd.read_pickle(rf'{target_folder_path}/data_daily/投信賣出張數.pkl')
投信週轉率 = pd.read_pickle(rf'{target_folder_path}/data_daily/投信週轉率%.pkl')
總外資市值_點 = pd.read_pickle(rf'{target_folder_path}/data_daily/總外資市值%(TSE,OTC).pkl')
總外資市值_加 = pd.read_pickle(rf'{target_folder_path}/data_daily/總外資市值%(TSE+OTC).pkl')
自營商自行買進張數 = pd.read_pickle(rf'{target_folder_path}/data_daily/自營商自行買進張數.pkl')
自營商自行賣出張數 = pd.read_pickle(rf'{target_folder_path}/data_daily/自營商自行賣出張數.pkl')
自營商買進張數 = pd.read_pickle(rf'{target_folder_path}/data_daily/自營商買進張數.pkl')
自營商賣出張數 = pd.read_pickle(rf'{target_folder_path}/data_daily/自營商賣出張數.pkl')
自營商週轉率 = pd.read_pickle(rf'{target_folder_path}/data_daily/自營商週轉率%.pkl')
自營商避險買進張數 = pd.read_pickle(rf'{target_folder_path}/data_daily/自營商避險買進張數.pkl')
自營商避險賣出張數 = pd.read_pickle(rf'{target_folder_path}/data_daily/自營商避險賣出張數.pkl')
自營持股數 = pd.read_pickle(rf'{target_folder_path}/data_daily/自營持股數(千股).pkl')
自營持股數市值 = pd.read_pickle(rf'{target_folder_path}/data_daily/自營持股數市值(百萬).pkl')
自營持股率 = pd.read_pickle(rf'{target_folder_path}/data_daily/自營持股率%.pkl')
自營自行買賣超_張數 = pd.read_pickle(rf'{target_folder_path}/data_daily/自營自行買賣超(千股).pkl')
自營自行買賣超_百萬 = pd.read_pickle(rf'{target_folder_path}/data_daily/自營自行買賣超(百萬).pkl')
自營買賣超 = pd.read_pickle(rf'{target_folder_path}/data_daily/自營買賣超(千股).pkl')
自營買賣超市值 = pd.read_pickle(rf'{target_folder_path}/data_daily/自營買賣超市值(百萬).pkl')
自營避險買賣超_張數 = pd.read_pickle(rf'{target_folder_path}/data_daily/自營避險買賣超(千股).pkl')
自營避險買賣超_百萬 = pd.read_pickle(rf'{target_folder_path}/data_daily/自營避險買賣超(百萬).pkl')

### dataset資料(4)
累計營業利益 = pd.read_pickle(rf'{target_folder_path}/data_Month/累計營業利益(千元)_full.pkl')
創N月新高低 = pd.read_pickle(rf'{target_folder_path}/data_Month/創 N月新高低 (月數)_full.pkl')
合併累計營收 = pd.read_pickle(rf'{target_folder_path}/data_Month/合併累計營收(千元)_full.pkl') 
以合併為主單月營收 = pd.read_pickle(rf'{target_folder_path}/data_Month/以合併為主單月營收(千元)_full.pkl')
單月營業利益率 = pd.read_pickle(rf'{target_folder_path}/data_Month/單月營業利益率％_full.pkl')
單月每股稅後盈餘_WA = pd.read_pickle(rf'{target_folder_path}/data_Month/單月每股稅後盈餘(WA)_full.pkl')
單月每股稅後盈餘_元 = pd.read_pickle(rf'{target_folder_path}/data_Month/單月每股稅後盈餘(元)_full.pkl')
近3月累計營收成長率 = pd.read_pickle(rf'{target_folder_path}/data_Month/近3月累計營收成長率_full.pkl')
淨值 = pd.read_pickle(rf'{target_folder_path}/data_Month/淨值(千元)_full.pkl')
近3月累計營收_千元 = pd.read_pickle(rf'{target_folder_path}/data_Month/近 3月累計營收(千元)_full.pkl')
歷史最低單月營收_千元 = pd.read_pickle(rf'{target_folder_path}/data_Month/歷史最低單月營收(千元)_full.pkl')
以合併為主單月營收與上月比 = pd.read_pickle(rf'{target_folder_path}/data_Month/以合併為主單月營收與上月比％_full.pkl')
合併去年累計營收_千元 = pd.read_pickle(rf'{target_folder_path}/data_Month/合併去年累計營收(千元)_full.pkl')
去年近12月累計營收 = pd.read_pickle(rf'{target_folder_path}/data_Month/去年近12月累計營收_full.pkl')
單月營收成長率 = pd.read_pickle(rf'{target_folder_path}/data_Month/單月營收成長率％_full.pkl')
近3月累計營收與上月比 = pd.read_pickle(rf'{target_folder_path}/data_Month/近3月累計營收與上月比％_full.pkl')
預估稅後盈餘 = pd.read_pickle(rf'{target_folder_path}/data_Month/預估稅後盈餘(千元)_full.pkl')
以合併總損益為主累計稅後盈餘 = pd.read_pickle(rf'{target_folder_path}/data_Month/以合併總損益為主累計稅後盈餘(千元)_full.pkl')
合併單月營業利益 = pd.read_pickle(rf'{target_folder_path}/data_Month/合併單月營業利益(千元)_full.pkl')
# = pd.read_pickle(rf'{target_folder_path}/data_Month/歷史最低單月營收-年月_full.pkl')
單月每股營收 = pd.read_pickle(rf'{target_folder_path}/data_Month/單月每股營收(元)_full.pkl')
近12月累計營收 = pd.read_pickle(rf'{target_folder_path}/data_Month/近12月累計營收(千元)_full.pkl')
去年近_3月累計營收 = pd.read_pickle(rf'{target_folder_path}/data_Month/去年近 3月累計營收_full.pkl')
合併累計營收成長率 = pd.read_pickle(rf'{target_folder_path}/data_Month/合併累計營收成長率％_full.pkl')
近12月累計營收成長率 = pd.read_pickle(rf'{target_folder_path}/data_Month/近12月累計營收成長率_full.pkl')
以合併為主單月營收成長率 = pd.read_pickle(rf'{target_folder_path}/data_Month/以合併為主單月營收成長率％_full.pkl')
合併單月營收與上月比 = pd.read_pickle(rf'{target_folder_path}/data_Month/合併單月營收與上月比％_full.pkl')
累計每股稅後盈餘 = pd.read_pickle(rf'{target_folder_path}/data_Month/累計每股稅後盈餘(WA)_full.pkl')
歸屬母公司累計稅後盈餘 = pd.read_pickle(rf'{target_folder_path}/data_Month/歸屬母公司累計稅後盈餘(千元)_full.pkl')
流通在外股數 = pd.read_pickle(rf'{target_folder_path}/data_Month/流通在外股數(千股)_full.pkl')
以合併為主累計稅前盈餘= pd.read_pickle(rf'{target_folder_path}/data_Month/以合併為主累計稅前盈餘(千元)_full.pkl')
去年單月營收 = pd.read_pickle(rf'{target_folder_path}/data_Month/去年單月營收(千元)_full.pkl')
合併單月稅前盈餘 = pd.read_pickle(rf'{target_folder_path}/data_Month/合併單月稅前盈餘(千元)_full.pkl')
合併累計營業利益率 = pd.read_pickle(rf'{target_folder_path}/data_Month/合併累計營業利益率％_full.pkl')
與歷史最低單月營收比 = pd.read_pickle(rf'{target_folder_path}/data_Month/與歷史最低單月營收比%_full.pkl')
累計每股稅後盈餘 = pd.read_pickle(rf'{target_folder_path}/data_Month/累計每股稅後盈餘(元)_full.pkl')
單月營收與上月比_成長月數 = pd.read_pickle(rf'{target_folder_path}/data_Month/單月營收與上月比％-成長月數_full.pkl')
單月營收 = pd.read_pickle(rf'{target_folder_path}/data_Month/單月營收(千元)_full.pkl')
去年累計營收= pd.read_pickle(rf'{target_folder_path}/data_Month/去年累計營收(千元)_full.pkl')
合併單月營業利益率 = pd.read_pickle(rf'{target_folder_path}/data_Month/合併單月營業利益率％_full.pkl')
累計營收 = pd.read_pickle(rf'{target_folder_path}/data_Month/累計營收(千元)_full.pkl')
單月營業利益 = pd.read_pickle(rf'{target_folder_path}/data_Month/單月營業利益(千元)_full.pkl')
預估稅前盈餘 = pd.read_pickle(rf'{target_folder_path}/data_Month/預估稅前盈餘(千元)_full.pkl')
與歷史最高單月營收比 = pd.read_pickle(rf'{target_folder_path}/data_Month/與歷史最高單月營收比%_full.pkl')
預估營收 = pd.read_pickle(rf'{target_folder_path}/data_Month/預估營收(千元)_full.pkl')
合併單月營收與上月比_成長月數 = pd.read_pickle(rf'{target_folder_path}/data_Month/合併單月營收與上月比％-成長月數_full.pkl')
單月每股稅前盈餘 = pd.read_pickle(rf'{target_folder_path}/data_Month/單月每股稅前盈餘(元)_full.pkl')
單月每股稅前盈餘 = pd.read_pickle(rf'{target_folder_path}/data_Month/單月每股稅前盈餘(WA)_full.pkl')
母公司資金貸放佔淨值比 = pd.read_pickle(rf'{target_folder_path}/data_Month/母公司資金貸放佔淨值比_full.pkl')
近3月每股營收 = pd.read_pickle(rf'{target_folder_path}/data_Month/近 3月每股營收_full.pkl')
以合併總損益為主單月稅後盈餘 = pd.read_pickle(rf'{target_folder_path}/data_Month/以合併總損益為主單月稅後盈餘(千元)_full.pkl')
歷史最高單月營收_年月= pd.read_pickle(rf'{target_folder_path}/data_Month/歷史最高單月營收-年月_full.pkl')
合併累計營業利益= pd.read_pickle(rf'{target_folder_path}/data_Month/合併累計營業利益(千元)_full.pkl')
合併單月營收成長率 = pd.read_pickle(rf'{target_folder_path}/data_Month/合併單月營收成長率％_full.pkl')
廣義逾放比率 = pd.read_pickle(rf'{target_folder_path}/data_Month/廣義逾放比率_full.pkl')
以合併為主累計營收成長率 = pd.read_pickle(rf'{target_folder_path}/data_Month/以合併為主累計營收成長率％_full.pkl')
合併去年單月營收 = pd.read_pickle(rf'{target_folder_path}/data_Month/合併去年單月營收(千元)_full.pkl')
狹義逾放比率 = pd.read_pickle(rf'{target_folder_path}/data_Month/狹義逾放比率_full.pkl')
累計營收成長率 = pd.read_pickle(rf'{target_folder_path}/data_Month/累計營收成長率％_full.pkl')
每股淨值 = pd.read_pickle(rf'{target_folder_path}/data_Month/每股淨值(元)_full.pkl')
累計每股稅前盈餘 = pd.read_pickle(rf'{target_folder_path}/data_Month/累計每股稅前盈餘(元)_full.pkl')
近12月每股營收 = pd.read_pickle(rf'{target_folder_path}/data_Month/近12月每股營收_full.pkl')
歸屬母公司單月稅後盈餘 = pd.read_pickle(rf'{target_folder_path}/data_Month/歸屬母公司單月稅後盈餘(千元)_full.pkl')
單月營收與上月比 = pd.read_pickle(rf'{target_folder_path}/data_Month/單月營收與上月比％_full.pkl')
以合併為主單月營收 = pd.read_pickle(rf'{target_folder_path}/data_Month/應予觀察放款比率_full.pkl')
應予觀察放款比率 = pd.read_pickle(rf'{target_folder_path}/data_Month/母公司背書保証佔淨值比_full.pkl')
以合併為主單月稅前盈餘 = pd.read_pickle(rf'{target_folder_path}/data_Month/以合併為主單月稅前盈餘(千元)_full.pkl')
累計每股營收 = pd.read_pickle(rf'{target_folder_path}/data_Month/累計每股營收(元)_full.pkl')
歷史最高單月營收 = pd.read_pickle(rf'{target_folder_path}/data_Month/歷史最高單月營收(千元)_full.pkl')
背書保証餘額_千元_母公司 = pd.read_pickle(rf'{target_folder_path}/data_Month/背書保証餘額(千元)－母公司_full.pkl')
合併創新高低_歷史 = pd.read_pickle(rf'{target_folder_path}/data_Month/合併創新高低(歷史)_full.pkl')
累計每股稅前盈餘 = pd.read_pickle(rf'{target_folder_path}/data_Month/累計每股稅前盈餘(WA)_full.pkl')
資金貸放餘額_千元_母公司 = pd.read_pickle(rf'{target_folder_path}/data_Month/資金貸放餘額(千元)－母公司_full.pkl')
合併單月營收 = pd.read_pickle(rf'{target_folder_path}/data_Month/合併單月營收(千元)_full.pkl')
以合併為主累計營收 = pd.read_pickle(rf'{target_folder_path}/data_Month/以合併為主累計營收(千元)_full.pkl')
近3月累計營收變動率 = pd.read_pickle(rf'{target_folder_path}/data_Month/近3月累計營收變動率％_full.pkl')
合併累計稅前盈餘 = pd.read_pickle(rf'{target_folder_path}/data_Month/合併累計稅前盈餘(千元)_full.pkl')
累計營業利益率 = pd.read_pickle(rf'{target_folder_path}/data_Month/累計營業利益率％_full.pkl')


data_list = ['Open', 'Close', 'High', 'Low', 'Volume',
    '一以下', '一到五', '五到十', '十到十五', '十五到二十', '二十到三十', '三十到四十', '四十到五十',
    '五十到一百', '一百到二百', '二百到四百', '四百到六百', '六百到八百', '八百到一千', '一千以上',

    "個股可投資上限比率", "個股買賣超股數", "合計持股數", "合計持股數市值", "合計持股率", "合計買賣超", "合計買賣超市值",
    "外資總投資市值", "外資總投資比率", "外資總投資股數", "外資總投資股率", "外資買賣超", "外資買進張數", "外資賣出張數",
    "外資週轉率", "尚可投資比率", "尚可投資股數", "投信持股數", "投信持股數市值", "投信持股率", "投信買賣超",
    "投信買賣超市值", "投信買進張數", "投信賣出張數", "投信週轉率", "總外資市值_點", "總外資市值_加", "自營商自行買進張數",
    "自營商自行賣出張數", "自營商買進張數", "自營商賣出張數", "自營商週轉率", "自營商避險買進張數", "自營商避險賣出張數",
    "自營持股數", "自營持股數市值", "自營持股率", "自營自行買賣超_張數", "自營自行買賣超_百萬", "自營買賣超", "自營買賣超市值",
    "自營避險買賣超_張數", "自營避險買賣超_百萬",

    "累計營業利益", "創N月新高低", "合併累計營收", "以合併為主單月營收", "單月營業利益率", 
    "單月每股稅後盈餘_WA", "單月每股稅後盈餘_元", "近3月累計營收成長率", "淨值", "近3月累計營收_千元", 
    "歷史最低單月營收_千元", "以合併為主單月營收與上月比", "合併去年累計營收_千元", "去年近12月累計營收", 
    "單月營收成長率", "近3月累計營收與上月比", "預估稅後盈餘", "以合併總損益為主累計稅後盈餘", "合併單月營業利益", 
    "單月每股營收", "近12月累計營收", "去年近_3月累計營收", "合併累計營收成長率", "近12月累計營收成長率", 
    "以合併為主單月營收成長率", "合併單月營收與上月比", "累計每股稅後盈餘", "歸屬母公司累計稅後盈餘", 
    "流通在外股數", "以合併為主累計稅前盈餘", "去年單月營收", "合併單月稅前盈餘", "合併累計營業利益率", 
    "與歷史最低單月營收比", "累計每股稅後盈餘", "單月營收與上月比_成長月數", "單月營收", "去年累計營收", 
    "合併單月營業利益率", "累計營收", "單月營業利益", "預估稅前盈餘", "與歷史最高單月營收比", "預估營收", 
    "合併單月營收與上月比_成長月數", "單月每股稅前盈餘", "母公司資金貸放佔淨值比", "近3月每股營收", 
    "以合併總損益為主單月稅後盈餘", "歷史最高單月營收_年月", "合併累計營業利益", "合併單月營收成長率", 
    "廣義逾放比率", "以合併為主累計營收成長率", "合併去年單月營收", "狹義逾放比率", "累計營收成長率", 
    "每股淨值", "累計每股稅前盈餘", "近12月每股營收", "歸屬母公司單月稅後盈餘", "單月營收與上月比", 
    "應予觀察放款比率", "以合併為主單月稅前盈餘", "累計每股營收", "歷史最高單月營收", "背書保証餘額_千元_母公司", 
    "合併創新高低_歷史", "累計每股稅前盈餘", "資金貸放餘額_千元_母公司", "合併單月營收", "以合併為主累計營收", 
    "近3月累計營收變動率", "合併累計稅前盈餘", "累計營業利益率"
]

arithmetic_list = ['+', '-', '*', '/']
ts_operators_list = ['ts_delta(', 'ts_max(', 'ts_min(', 'ts_rank(', 'ts_stddev(', 'ts_sum(', 'ts_decay(', 'ts_mean(', 'ts_product(']
random_numbers = [random.randint(1, 504) for _ in range(504)]
period_list = [rf',{num})' for num in random_numbers] # 1是for decay 不變
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
        trash_filter_list = pd.read_pickle(rf'{target_folder_path}/data_DM/WR_trash_filter_list.pkl')
    except FileNotFoundError:
        trash_filter_list = []
    return trash_filter_list

def first_layer_generate(trash_filter_list:pd.Series, sample_num = 10000):
    expression_list = []
    for _ in range(sample_num):
        TorC = random.random()
        if TorC < 0.9:
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



