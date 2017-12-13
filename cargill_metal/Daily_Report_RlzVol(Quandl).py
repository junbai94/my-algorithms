# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 00:39:52 2017

@author: k913238
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
import quandl

'''
 Params set-up
'''
td = datetime.date.strftime(datetime.date.today(),'%Y-%m-%d')
ReportPdf = PdfPages('Realized Volatulity Daily Report '+ td +'.pdf')
pagesize = (20,30)
fmt = '%3.0f%%'
yticks = FormatStrFormatter(fmt)
years = mdates.YearLocator()   
months = mdates.MonthLocator()  
days = mdates.DayLocator()
hours = mdates.HourLocator(12) #if you want ticks every 12 hrs, you can pass 12 to this function
minutes = mdates.MinuteLocator() 
DateFmt = mdates.DateFormatter('%y/%m/%d')


def RlzVol(df,p,window,yearday,standard=False):
    '''
    standard:
        it is true means using standard deviation as volatility.false means using
        average of sum of square as the variance, which means assuming the expected logreturn is 0
    ''' 
    logreturn = np.log(df[p]/df[p].shift(1))
    if standard:
#        return np.sqrt(yearday)* pd.rolling_std(logreturn,window)
        return np.sqrt(yearday)* pd.Series(logreturn).rolling(window).std()
    else:
        square_r = logreturn*logreturn
        return np.sqrt(yearday)* np.sqrt( pd.Series(square_r).rolling(window).sum() / window)

def VolPlot(df,prod,convention,tail,ax,title):
    df = df[df[prod]!=0]
    df['5d_vol'] = 100.0*RlzVol(df,prod,5,convention)
    df['10d_vol'] = 100.0*RlzVol(df,prod,10,convention)
    df['20d_vol'] = 100.0*RlzVol(df,prod,20,convention)
    df['30d_vol'] = 100.0*RlzVol(df,prod,30,convention)
    voldf = df[['5d_vol','10d_vol','20d_vol','30d_vol']]

    df_vol = df.tail(tail)[['5d_vol','10d_vol','20d_vol','30d_vol']]
    dateindex = df_vol.index.to_pydatetime()
    ax.plot(dateindex, df_vol)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_major_formatter(yticks)
    ax.yaxis.grid(True,which='minor')
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(DateFmt)
    ax.xaxis.grid(True,which='major')
    ax.legend(['5d_vol','10d_vol','20d_vol','30d_vol'])
    ax.set_title(title)
    return voldf

def VolSpreadPlot(df1,df2,tail,ax,title):
    df = df1 - df2
    
    df_vol = df.tail(tail)[['5d_vol','10d_vol','20d_vol','30d_vol']]
    df_vol = df_vol.dropna(axis = 0, how = 'all')
    dateindex = df_vol.index.to_pydatetime()
    ax.plot(dateindex, df_vol)
    ax.yaxis.set_major_formatter(yticks)
    ax.yaxis.grid(True)
    ax.xaxis.set_major_formatter(DateFmt)
    ax.xaxis.grid(True,which='major')
    ax.legend(['5d_vol','10d_vol','20d_vol','30d_vol'])
    ax.set_title(title)
    return


'''
Data Importing
'''
quandl.ApiConfig.api_key='JyoyDac81TmTDZMTEuQu'

iodce = quandl.get(['DCE/IF2016.4','DCE/IK2016.4','DCE/IU2016.4','DCE/IF2017.4','DCE/IK2017.4','DCE/IU2017.4','DCE/IF2018.4','DCE/IK2018.4'])

rbshfe = quandl.get(['SHFE/RBF2017.5','SHFE/RBK2017.5','SHFE/RBV2017.5','SHFE/RBF2018.5','SHFE/RBK2018.5'])

hcshfe = quandl.get(['SHFE/HCF2017.5','SHFE/HCK2017.5','SHFE/HCV2017.5','SHFE/HCF2018.5','SHFE/HCK2018.5'])

iosgx = quandl.get(['SGX/FEFF2017.5','SGX/FEFJ2017.5','SGX/FEFK2017.5','SGX/FEFQ2017.5','SGX/FEFU2017.5','SGX/FEFV2017.5','SGX/FEFX2017.5','SGX/FEFZ2017.5','SGX/FEFF2018.5','SGX/FEFG2018.5','SGX/FEFH2018.5','SGX/FEFJ2018.5','SGX/FEFK2018.5'])

jdce = quandl.get(['DCE/JF2017.4','DCE/JK2017.4','DCE/JU2017.4','DCE/JF2018.4','DCE/JK2018.4'])

FX = quandl.get(['CME/CNHK2017.6','CME/CNHU2017.6','CME/CNHF2018.6','CME/CNHK2018.6'],start_date='2016-01-01')



'''
Create PDF file and first page 
'''

fig,axes = plt.subplots(nrows=5,ncols=2,figsize=pagesize)

i01 = VolPlot(iodce,"DCE/IF2018 - Close",244.0,150,axes[0,0],'i1801 Realized Vol')

i02 = VolPlot(iodce,"DCE/IK2018 - Close",244.0,80,axes[0,1],'i1805 Realized Vol')

rb01 = VolPlot(rbshfe,"SHFE/RBF2018 - Close",244.0,150,axes[1,0],'RB1801 Realized Vol')

rb02 = VolPlot(rbshfe,"SHFE/RBK2018 - Close",244.0,80,axes[1,1],'RB1805 Realized Vol')

hc01 = VolPlot(hcshfe,"SHFE/HCF2018 - Close",244.0,150,axes[2,0],'HC1801 Realized Vol')

hc02 = VolPlot(hcshfe,"SHFE/HCK2018 - Close",244.0,80,axes[2,1],'HC1805 Realized Vol')

'''
SGX_Oct
'''
sg = VolPlot(iosgx,"SGX/FEFV2017 - Settle",250.0,150,axes[3,0],'SGX Oct17 Realized Vol (Settle)')

'''
SGX_Nov
'''
sg = VolPlot(iosgx,"SGX/FEFX2017 - Settle",250.0,150,axes[3,1],'SGX Nov17 Realized Vol (Settle)')

'''
SGX_Dec
'''
sg = VolPlot(iosgx,"SGX/FEFZ2017 - Settle",250.0,150,axes[4,0],'SGX Dec17 Realized Vol (Settle)')

'''
SGX_Jan
'''
sg = VolPlot(iosgx,"SGX/FEFF2018 - Settle",250.0,150,axes[4,1],'SGX Jan18 Realized Vol (Settle)')

'''
Save first page
'''
fig.suptitle('Realized Volatulity Daily Report '+ td,fontsize = 25)
ReportPdf.savefig(fig)
del axes

'''
Second Page
'''
fig = plt.figure(figsize = pagesize)
gs = gridspec.GridSpec(5,2)

'''
Vol spread i vs. rb
'''
ax = plt.subplot(gs[0,0])
VolSpreadPlot(i01,rb01,150,ax,'Vol Spread: i1801 vs. RB1801')
ax = plt.subplot(gs[0,1])
VolSpreadPlot(i02,rb02,80,ax,'Vol Spread: i1805 vs. RB1805')

'''
Vol spread rb vs. hc
'''
ax = plt.subplot(gs[1,0])
VolSpreadPlot(rb01,hc01,150,ax,'Vol Spread: RB1801 vs. HC1801')
ax = plt.subplot(gs[1,1])
VolSpreadPlot(rb02,hc02,80,ax,'Vol Spread: RB1805 vs. HC1805')


'''
RB Profit : original equation
'''
jlist = ['DCE/JK2017 - Close','DCE/JU2017 - Close','DCE/JF2018 - Close','DCE/JK2018 - Close']
iolist = ['DCE/IK2017 - Close','DCE/IU2017 - Close','DCE/IF2018 - Close','DCE/IK2018 - Close']
RBlist = ['SHFE/RBK2017 - Close','SHFE/RBV2017 - Close','SHFE/RBF2018 - Close','SHFE/RBK2018 - Close']
rblegend = ['1705','1710','1801','1805']
rbprofit = iodce[[iolist[0],iolist[1]]]
rbprofit = rbprofit.tail(250)

for i in range(0,4):
    rbprofit[RBlist[i][5:12]] = \
        rbshfe[RBlist[i]] - iodce[iolist[i]]*1.582 - jdce[jlist[i]]*0.461 - 1109.54
    
df_profit = rbprofit
del df_profit[iolist[0]]
del df_profit[iolist[1]]
ax = plt.subplot(gs[3:,0:])
dateindex = df_profit.index.to_pydatetime()
ax.plot(dateindex, df_profit)
ax.yaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_minor_locator(MultipleLocator(50))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
ax.yaxis.grid(True,which='minor')
#ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(DateFmt)
ax.xaxis.grid(True,which='major')
ax.legend(rblegend,fontsize=15)
ax.set_title('Rebar Margin: RB - 1.582 i - 0.461 j - 1109.54',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

'''
RB profit vol 1
'''
df=df_profit
ticker = RBlist[2][5:12]
df[ticker] = df[ticker] + 400.0
ax = plt.subplot(gs[2,0])

rbp = VolPlot(df,ticker,244.0,80,ax,ticker+' Margin Realized Vol')

'''
RB profit vol 2
'''
df=df_profit
ticker = RBlist[3][5:12]
df[ticker] = df[ticker] + 400.0
ax = plt.subplot(gs[2,1])

rbp = VolPlot(df,ticker,244.0,60,ax,ticker+' Margin Realized Vol')

'''
Save second page
'''
ReportPdf.savefig(fig)


'''
Third Page
'''
fig = plt.figure(figsize = pagesize)
gs = gridspec.GridSpec(3,2)

'''
RB Profit : simplified equation
'''
rbprofit = iodce[[iolist[0],iolist[1]]]
rbprofit = rbprofit.tail(250)

for i in range(0,4):
    rbprofit[RBlist[i][5:12]] = \
        rbshfe[RBlist[i]] - iodce[iolist[i]]*1.6
    
df_profit = rbprofit
del df_profit[iolist[0]]
del df_profit[iolist[1]]
ax = plt.subplot(gs[0,:])
dateindex = df_profit.index.to_pydatetime()
ax.plot(dateindex, df_profit)
ax.yaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_minor_locator(MultipleLocator(50))
ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
ax.yaxis.grid(True,which='minor')
ax.xaxis.set_major_formatter(DateFmt)
ax.xaxis.grid(True,which='major')
ax.legend(rblegend,fontsize=15)
ax.set_title('Rebar Margin: RB - 1.6 i',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

'''
IO DCE calendar spread
'''
spreadlist = ['1601-1605','1605-1609','1609-1701','1701-1705','1705-1709','1709-1801','1801-1805']
iodce[spreadlist[0]] = iodce['DCE/IF2016 - Close'] - iodce['DCE/IK2016 - Close']
iodce[spreadlist[1]] = iodce['DCE/IK2016 - Close'] - iodce['DCE/IU2016 - Close']
iodce[spreadlist[2]] = iodce['DCE/IU2016 - Close'] - iodce['DCE/IF2017 - Close']
iodce[spreadlist[3]] = iodce['DCE/IF2017 - Close'] - iodce['DCE/IK2017 - Close']
iodce[spreadlist[4]] = iodce['DCE/IK2017 - Close'] - iodce['DCE/IU2017 - Close']
iodce[spreadlist[5]] = iodce['DCE/IU2017 - Close'] - iodce['DCE/IF2018 - Close']
iodce[spreadlist[6]] = iodce['DCE/IF2018 - Close'] - iodce['DCE/IK2018 - Close']
#iodce[spreadlist[7]] = iodce['DCE/IK2018 - Close'] - iodce['DCE/IU2018 - Close']

for i in spreadlist:
    T = []
    for dat in iodce.index:
        tau = datetime.datetime.strptime('20'+i[:4]+'30','%Y%m%d') - dat
        T = T + [tau.days]
    Tau = pd.Series(T,index = iodce.index)
    iodce['Tau'+i] = Tau

ax = plt.subplot(gs[1,:])
ax.set_xlim(left=275,right=0)
ax.yaxis.tick_right()
for i in spreadlist[:-2]:
    plt.plot(iodce['Tau'+i],iodce[i],alpha=0.3)
for i in spreadlist[-2:]:
    plt.plot(iodce['Tau'+i],iodce[i],alpha=1.0,linewidth = 3.0)
ax.yaxis.set_major_locator(MultipleLocator(20))
ax.yaxis.set_minor_locator(MultipleLocator(10))
ax.xaxis.set_major_locator(MultipleLocator(30))
ax.xaxis.set_minor_locator(MultipleLocator(15))
plt.grid(which='minor')
ax.set_title('DCE IO Calendar Spread',fontsize=20)
ax.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

'''
IO SGX-DCE spread in USD
'''

SDlist=['1705','1709','1801','1805']
DCEmatlist = ['K2017','U2017','F2018','K2018']
SGXmatlist = ['J2017','Q2017','Z2017','J2018']
SDspread = iodce[['DCE/I'+DCEmatlist[0]+' - Close','DCE/I'+DCEmatlist[0]+' - Close']]
for i in range(0,4):
    SDspread[SDlist[i]] = iosgx['SGX/FEF'+SGXmatlist[i]+' - Settle'] - \
                          (iodce['DCE/I'+DCEmatlist[i]+' - Close'] - 30.0/0.92)/1.17/FX['CME/CNH'+DCEmatlist[i]+' - Settle']
                           

SDspread = SDspread.tail(250)[SDlist]
SDspread = SDspread.dropna(axis = 0, how = 'all')
ax = plt.subplot(gs[2,:])
SDspread.plot(grid=True, ax = ax)
ax.yaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.grid(True,which='minor')
ax.set_title('IO SGX-DCE Spread (in USD)',fontsize=20)
ax.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)




ReportPdf.savefig(fig)

ReportPdf.close()


    

