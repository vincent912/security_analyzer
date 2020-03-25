import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import argparse
import sys



#Defines how many days to use to calculate stdevs
STD_Period = 20
RSI_Periods = 14

#The ticker symbol of the security to be analyzed.
ticker = ""

def check20SMA(df):
    cols = df.columns.values
    for i in range(df.shape[1]):
        if cols[i] == "20SMA":
            return i
    if SMAindex == -1:
        add20SMA(df)
        return df.shape[1]-1

#Adds the Relative Strength Indicator
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

#Adds a 20 day simple moving average.
def add20SMA(df):
    df.loc[:,"20SMA"] = 0
    for i in range(19,len(df)):
        sum20 = 0
        for j in range(20):
            sum20 += df.iloc[i-19+j, 0]
        df.iloc[i, df.shape[1]-1] = sum20/20

#Adds bolinger bands, a range that tracks 2 stdevs above and 2 stdevs below the current 20 day simple moving average
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

#Adds the security's price's 20 day standard deviation
def addSTD(df):
    df.loc[:,"STD"] = 0
    for i in range (STD_Period-1, len(df)):
        lastPeriodClose = np.zeros(STD_Period)
        for j in range(STD_Period):
            lastPeriodClose[j] = df.iloc[i-j, 0]
        df.iloc[i, df.shape[1]-1] = np.std(lastPeriodClose)


#Adds the security's price's z score using the 20 day SMA and stdevs
def addZScore(df):
    SMAindex = check20SMA(df)
    df.loc[:,"ZScore"] = 0
    for i in range (STD_Period-1, len(df)):
        last20Close = np.zeros(STD_Period)
        for j in range(STD_Period):
            last20Close[j] = df.iloc[i-j, 0]
        std = np.std(last20Close)
        df.iloc[i, df.shape[1]-1] = (df.iloc[i, 0] - df.iloc[i, SMAindex]) / std

#Adds Moving Average Convergenve Divergence Indicator and its position to its signal line
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
    df.loc[:,"MACD_Signal_Line"] = 0
    for i in range(len(df)):
       	df.iloc[i, df.shape[1]-1] = signal[i]

#Adds the price the security will close at tomorrow, a potential label.
def addNextClose(df):
    df.loc[:, "Next_Close"] = 0
    for i in range(0, len(df)-1):
        df.iloc[i, df.shape[1]-1] = df.iloc[i+1, 0]

#Adds the amount the security will change by tomorrow, a potential labe.
def addDaysChange(df):
    df["Days_Change"] = df["Next_Close"] - df["Close"]

#Adds a boolean feature as to whether the price of the security will go up
#by the next day's close. Potential label for classification.
def addGreenRed(df):
    df["GreenRed"] = df["Days_Change"].apply(isPositive)

def isPositive(n):
    if n > 0:
        return 1
    else:
        return 0


#Adds all features to the dataframe. Some features are based on others, so
#the order these functions are called are somewhat important.
def addAllFeatures(df):
    df.dropna(inplace=True)
    df.rename(columns={ticker: "Close"})
    addNextClose(df)
    addRSI(df)
    add20SMA(df)
    addBolingerBands(df)
    addSTD(df)
    addZScore(df)
    addMACDandSignal(df)
    addDaysChange(df)
    addGreenRed(df)

def plot(df):
    ax = df.plot(title = '{} Price'.format(ticker))
    ax.set_xlabel('Date')
    ax.set_ylabel('RSI')
    ax.grid()
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("Incorrect usage, please pass in ticker name as only argument.")
        sys.exit(1)
    
    ticker = sys.argv[1]
    df = pd.read_csv(ticker + '.csv', index_col='Date', parse_dates=True, usecols=['Date', 'Close', 'Volume'], na_values='nan')

    addAllFeatures(df)
    
#    plot(df)

    df.to_csv(ticker + '_features.csv')
    print("All features generated and exported as csv for " + ticker + " as " + ticker + '_features.csv' + ".")

main()




