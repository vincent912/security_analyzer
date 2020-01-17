# security_analyzer

Hello!

Thanks for your interest in my project.
This project was developed locally, and over the course of the next few days I will try my best to commit each part for all to see.

Regards,

Vincent Li

1/15/2020
Added the following files:
      Generate_Features.py - Takes a csv and generates technical indicators(simple/exponential averages, standard deviation of prices, MACD, RSI) based off of past prices.
      VOO.csv - An example of the type of csv that Generate_Features.py operates on. Must include Date, Close(price), and volume. This example contains 2 years of daily data for VOO, which is an exchange traded mutual fund that imitates the companies in the S&P 500 index.
      VOO_features.csv - An example of the type of csv that Generate_Features.py outputs.

1/16/2020
Added the following files:
      Prepare_LinR.py - Takes a *_features.csv file generate by Generate_Features.py and reformats the csv to be fed into a linear regression model.
      VOO_features_prepared_lin.csv - An example of the type of csv that Prepare_LinR.py outputs.