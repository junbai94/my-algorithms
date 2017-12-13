import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
from matplotlib.finance import candlestick_ohlc

###############################################################################

aapl = pd.read_csv("C:/Users/j291414/Desktop/AAPL.csv")
aapl['Date'] = pd.to_datetime(aapl['Date'], format="%Y-%m-%d")
aapl.index = aapl['Date']

###############################################################################

#plt.plot(aapl['Date'], aapl['Adj Close'])
#plt.grid()
#plt.xticks(rotation='vertical')
#plt.show()

###############################################################################

"""
    :param dat: pandas DataFrame object with datetime64 index, and float columns "Open", "High", "Low", and "Close", likely created via DataReader from "yahoo"
    :param stick: A string or number indicating the period of time covered by a single candlestick. Valid string inputs include "day", "week", "month", and "year", ("day" default), and any numeric input indicates the number of trading days included in a period
    :param otherseries: An iterable that will be coerced into a list, containing the columns of dat that hold other series to be plotted as lines
     
    This will show a Japanese candlestick plot for stock data stored in dat, also plotting other series if passed.
""" 
def pandas_candlestick_ohlc(dat, stick = "day", otherseries = None):
   
    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    alldays = DayLocator()              # minor ticks on the days
    dayFormatter = DateFormatter('%d')      # e.g., 12
 
    # Create a new DataFrame which includes OHLC data for each period specified by stick input
    transdat = dat.loc[:,["Open", "High", "Low", "Close"]]
    if (type(stick) == str):
        if stick == "day":
            plotdat = transdat
            stick = 1 # Used for plotting
        elif stick in ["week", "month", "year"]:
            if stick == "week":
                transdat["week"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[1]) # Identify weeks
            elif stick == "month":
                transdat["month"] = pd.to_datetime(transdat.index).map(lambda x: x.month) # Identify months
            transdat["year"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[0]) # Identify years
            grouped = transdat.groupby(list(set(["year",stick]))) # Group by year and other appropriate variable
            plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
            for name, group in grouped:
                plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],
                                            "High": max(group.High),
                                            "Low": min(group.Low),
                                            "Close": group.iloc[-1,3]},
                                           index = [group.index[0]]))
            if stick == "week": stick = 5
            elif stick == "month": stick = 30
            elif stick == "year": stick = 365
 
    elif (type(stick) == int and stick >= 1):
        transdat["stick"] = [np.floor(i / stick) for i in range(len(transdat.index))]
        grouped = transdat.groupby("stick")
        plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
        for name, group in grouped:
            plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],
                                        "High": max(group.High),
                                        "Low": min(group.Low),
                                        "Close": group.iloc[-1,3]},
                                       index = [group.index[0]]))
 
    else:
        raise ValueError('Valid inputs to argument "stick" include the strings "day", "week", "month", "year", or a positive integer')
 
 
    # Set plot parameters, including the axis object ax used for plotting
     # Set plot parameters, including the axis object ax used for plotting
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    if plotdat.index[-1] - plotdat.index[0] < pd.Timedelta('730 days'):
        weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(alldays)
    else:
        weekFormatter = DateFormatter('%b %d, %Y')
    ax.xaxis.set_major_formatter(weekFormatter)
 
    ax.grid(True)
 
    # Create the candelstick chart
    candlestick_ohlc(ax, list(zip(list(date2num(plotdat.index.tolist())), plotdat["Open"].tolist(), plotdat["High"].tolist(),
                      plotdat["Low"].tolist(), plotdat["Close"].tolist())),
                      colorup = "black", colordown = "red", width = stick * .4)
 
    # Plot other series (such as moving averages) as lines
    if otherseries != None:
        if type(otherseries) != list:
            otherseries = [otherseries]
        dat.loc[:,otherseries].plot(ax = ax, lw = 1.3, grid = True)
 
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
 
    plt.show()
        
###############################################################################

data = EMA(aapl, 5)
data = EMA(data, 20)
#data = data[data['Date']>'2017-06-01']
plt.plot(data['Close'])
plt.plot(data['EMA_20'], 'y', label='EMA(20)')
plt.plot(data['EMA_5'], 'r', label='EMA(5)')
plt.legend()
plt.grid()
plt.show()

###############################################################################

# generate double crossover signal of EMA
"""
EMA Crossover signal
position column. when positive signal happens, all entry 1. If negative signal 
happens, clear position.
"""
test = data[['Date', 'Close', 'EMA_20', 'EMA_5']]
test['Position'] = np.nan
for i in np.arange(1, len(test['Position'])):
    if test['EMA_5'][i] >= test['EMA_20'][i] and test['EMA_5'][i-1] < test['EMA_20'][i-1]:
        test['Position'][i] = 1
    if test['EMA_5'][i] <= test['EMA_20'][i] and test['EMA_5'][i-1] > test['EMA_20'][i-1]:
        test['Position'][i] = 0
test['Position'] = test['Position'].fillna(method='pad')
test['Position'] = test['Position'].fillna(0)

# Two subplots, the axes array is 1-d
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(test['Date'], test['Close'])
axarr[0].plot(data['EMA_20'], 'y', label='EMA(10)')
axarr[0].plot(data['EMA_5'], 'r', label='EMA(5)')
axarr[0].legend()
axarr[0].grid()
axarr[0].set_title('Sharing X axis')
axarr[1].plot(test['Date'], test['Position'])
axarr[1].grid()
plt.show()

# total earning calculation for crossover signal strategy
def earning(df):
    exposure = False
    if df['Position'][0] != 0:
        exposure = True
        
    profit = 0
    enter = 0
    transaction = []
    
    for i in np.arange(1, len(df['Position'])):
        if df['Position'][i] == 1 and df['Position'][i-1] == 0:
            enter = df['Close'][i]
        if df['Position'][i] == 0 and df['Position'][i-1] == 1:
            profit += df['Close'][i] - enter
            transaction.append(profit)
            enter = 0
    if enter != 0:
        profit += df['Close'][-1] - enter
        transaction.append(profit)
    return profit, transaction

###############################################################################

"""
Stochastic Oscillator Indicator KD (10, 3, 3)
"""
#test = FULL_STOK(aapl)
#test = FULL_STO(test, 3)
#test = test[test.index > '2017-06-01']
#plt.plot(test['Date'], test['SO%k'], label='k')
#plt.plot(test['Date'], test['SO%d'], label='d')
#plt.plot(test['Date'], 20*np.ones((1,len(test['Date']))).transpose(), color='r', linestyle='--')
#plt.plot(test['Date'], 80*np.ones((1,len(test['Date']))).transpose(), color='r', linestyle='--')
#plt.grid()
#plt.legend()
#plt.show()


###############################################################################

'''
Distribution of Loss
'''
def earning_hist(df):
    exposure = False
    if df['Position'][0] != 0:
        exposure = True
        
    profit = 0
    enter = 0
    earning = []
    
    for i in np.arange(1, len(df['Position'])):
        if df['Position'][i] == 1 and df['Position'][i-1] == 0:
            enter = df['Close'][i]
        if df['Position'][i] == 0 and df['Position'][i-1] == 1:
            profit += df['Close'][i] - enter
            earning.append(df['Close'][i]-enter)
            enter = 0
    if enter != 0:
        profit += df['Close'][-1] - enter
        earning.append(df['Close'][i]-enter)
    return earning