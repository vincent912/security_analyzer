import pandas as pd

filename = input("Filename: ")
df = pd.read_csv(filename + ".csv", index_col = "Date", parse_dates=True)

df.loc[:,"MACD Signal"] = 0


MACD_index = -1
cols = df.columns.values
for i in range(df.shape[1]):
    if cols[i] == "MACD":
        MACD_index = i

Signal_index = -1
cols = df.columns.values
for i in range(df.shape[1]):
    if cols[i] == "MACD Signal Line":
        Signal_index = i


for i in range(33, len(df)):
    df.iloc[i, df.shape[1]-1] = df.iloc[i, MACD_index] - df.iloc[i, Signal_index]

df = df.drop("BBLower", axis=1)
df = df.drop("BBUpper", axis=1)
df = df.drop("MACD", axis=1)
df = df.drop("MACD Signal Line", axis=1)


Next_Close_index = -1
cols = df.columns.values
for i in range(df.shape[1]):
    if cols[i] == "Next Close":
        Next_Close_index = i

df.loc[:,"Day Change"] = 0
for i in range(0, df.shape[0]-1):
    df.iloc[i, df.shape[1]-1] = df.iloc[i, Next_Close_index] - df.iloc[i, 0]


#cols = df.columns.values
#for i in range(1, df.shape[1]):
#    if cols[i] != "Last Close":
#        print(cols[i])
#        for j in range(df.shape[0], 1):
#            df.iloc[j, i] = df.iloc[j-1, i]

df.to_csv(filename + "_prepared_lin.csv")
