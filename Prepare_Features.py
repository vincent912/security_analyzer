import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
import sys


#Augments the feature MACD signal difference, the essential element of this technical indicator
def addMACDSignalDifference(df):
    df["MACD_Signal_Diff"] = df["MACD"] - df["MACD_Signal_Line"]

def dataPipline(df):
    num_features = ["Volume", "RSI", "STD", "ZScore", "MACD_Signal_Diff"]
    num_pipeline = Pipeline([('std_scaler', StandardScaler())])
    full_pipeline = ColumnTransformer([("num", num_pipeline, num_features)])
    return full_pipeline.fit_transform(df)

#Saves targests in a separate array
#Potential targets are wheter or not the price goes up or down, days change, and tomorrow's close price
def createTargets(df):
    targets = df[["Next_Close", "Days_Change", "GreenRed"]]
    return targets


def main():
    if len(sys.argv) != 2:
        print("Incorrect usage, input features.csv filename as only argument.")
        sys.exit(1)
    
    ticker = sys.argv[1]
    df = pd.read_csv(ticker + ".csv", index_col="Date", parse_dates=True)


    #Drops instances with incomplete data(the first 33 days and the very last day)
    df = df[df.MACD_Signal_Line != 0]
    df = df[df.Next_Close > 0]

    addMACDSignalDifference(df)

    targets = createTargets(df)

    #Drops redundant features, features whose information is contained in other features.
    df = df.drop("BBLower", axis=1)
    df = df.drop("BBUpper", axis=1)
    df = df.drop("MACD", axis=1)
    df = df.drop("MACD_Signal_Line", axis=1)
    df = df.drop("20SMA", axis=1)

    df_prepared = pd.DataFrame(data=dataPipline(df))

    print (df_prepared.shape, targets.shape)

    df_prepared.to_csv(ticker + "_prepared.csv")
    targets.to_csv(ticker + "_targets.csv")
    print("All processed features exported as csv for " + ticker + " as " + ticker + "_prepared.csv")
    print("All targets exported as csv for " + ticker + " as " + ticker + "_targets.csv")


main()


