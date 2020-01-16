import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd

STD_Period = 20
RSI_Periods = 14


def check20SMA(df):
    cols = df.columns.values
    for i in range(df.shape[1]):
        if cols[i] == "20SMA":
            return i
    if SMAindex == -1:
        add20SMA(df)
        return df.shape[1]-1

def addRSI(df):
    gains = 0
    losses = 0
    df.loc[:,"RSI"] = 0
    for i in range(RSI_Periods):
        if df.iloc[i, 0] < df.iloc[i+1, 0]:
            gains += df.iloc[i+1, 0] - df.iloc[i, 0]
        else:
            losses += df.iloc[i, 0] - df.iloc[i+1, 0]
    RS = gains/losses
    df.iloc[14, df.shape[1]-1] = 100 - (100/(1+RS))
    gains = gains / 14
    losses = losses / 14
    for i in range(RSI_Periods+1, len(df)):
        if df.iloc[i-1, 0] < df.iloc[i, 0]:
            gains *= 13
            gains += df.iloc[i, 0] - df.iloc[i-1, 0]
            gains /= 14
            losses = losses * 13 /14
        else:
            losses *= 13
            losses += df.iloc[i-1, 0] - df.iloc[i, 0]
            losses /= 14
            gains = gains * 13 / 14
        RS = gains/losses
        df.iloc[i, df.shape[1]-1] = 100 - (100/(1+RS))


def add20SMA(df):
    df.loc[:,"20SMA"] = 0
    for i in range(19,len(df)):
        sum20 = 0
        for j in range(20):
            sum20 += df.iloc[i-19+j, 0]
        df.iloc[i, df.shape[1]-1] = sum20/20
            
def addBolingerBands(df):
    SMAindex = check20SMA(df)

    df.loc[:,"BBLower"] = 0
    df.loc[:,"BBUpper"] = 0
    for i in range (19, len(df)):
        last20Close = np.zeros(20)
        for j in range(20):
            last20Close[j] = df.iloc[i-j, 0]
        std = np.std(last20Close)
        df.iloc[i, df.shape[1]-2] = df.iloc[i, SMAindex] - 2 * std
        df.iloc[i, df.shape[1]-1] = df.iloc[i, SMAindex] + 2 * std

def addSTD(df):
    df.loc[:,"STD"] = 0
    for i in range (STD_Period-1, len(df)):
        lastPeriodClose = np.zeros(STD_Period)
        for j in range(STD_Period):
            lastPeriodClose[j] = df.iloc[i-j, 0]
        df.iloc[i, df.shape[1]-1] = np.std(lastPeriodClose)

    
def addZScore(df):
    SMAindex = check20SMA(df)
    
    df.loc[:,"ZScore"] = 0
    for i in range (19, len(df)):
        last20Close = np.zeros(20)
        for j in range(20):
            last20Close[j] = df.iloc[i-j, 0]
        std = np.std(last20Close)
        df.iloc[i, df.shape[1]-1] = (df.iloc[i, 0] - df.iloc[i, SMAindex]) / std

        
def addMACDandSignal(df):
    EMA12 = np.zeros(len(df))
    sum = 0
    for i in range(12):
       	sum += df.iloc[i, 0]
    avg = sum / 12
    EMA12[11] = avg
    for i in range(12, len(df)):
       	avg = df.iloc[i, 0] * 2 / 13 + avg * (1 - 2 / 13)
       	EMA12[i] = avg

    EMA26 = np.zeros(len(df))
    sum = 0
    for i in range(26):
       	sum += df.iloc[i, 0]
    avg = sum / 26
    EMA26[25] = avg
    for i in range(26, len(df)):
        avg = df.iloc[i, 0] * 2 / 27 + avg * (1 - 2 / 27)
        EMA26[i] = avg
        
    df.loc[:,"MACD"] = 0
    for i in range(25, len(df)):
       	df.iloc[i, df.shape[1]-1] = EMA12[i] - EMA26[i]

    signal = np.zeros(len(df))
    sum = 0
    for i in range(9):
       	sum += df.iloc[i+25, df.shape[1]-1]
    avg = sum / 9
    signal[33] = avg
    for i in range(34, len(df)):
       	avg = df.iloc[i, df.shape[1]-1] * 2 / 10 + avg * (1 - 2 / 10)
       	signal[i] = avg
    df.loc[:,"MACD Signal Line"] = 0
    for i in range(len(df)):
       	df.iloc[i, df.shape[1]-1] = signal[i]

def addNextClose(df):
    df.loc[:, "Next Close"] = 0
    for i in range(0, len(df)-1):
        df.iloc[i, df.shape[1]-1] = df.iloc[i+1, 0]


ticker = input("Filename: ")
df = pd.read_csv(ticker + '.csv', index_col='Date', parse_dates=True, usecols=['Date', 'Close', 'Volume'], na_values='nan')

df = df.rename(columns={'Close': ticker})
df.dropna(inplace=True)

addNextClose(df)
addRSI(df)
add20SMA(df)
addBolingerBands(df)
addSTD(df)
addZScore(df)
addMACDandSignal(df)

ticker = ticker + '_features.csv'
df.to_csv(ticker)
ax = df.plot(title = '{} Price'.format(ticker))
ax.set_xlabel('Date')
ax.set_ylabel('RSI')

ax.grid()
plt.show()
