# -*- coding: utf-8 -*-
"""
Created on Tue Dec 05 14:55:30 2017

@author: junbai

EastMoney stock data scooping
"""

import urllib
import re
import pandas as pd
import datetime
import time



proxies = {'http':'http://j291414:Battleship1!@10.252.22.102:4200', 
           'https':'https://j291414:Battleship1!@10.252.22.102:4200'}
URL = 'http://quote.eastmoney.com/stocklist.html'


def getHtml(url):
    html = urllib.urlopen(url, proxies=proxies).read()
    html = html.decode('gbk')
    return html

def getStackCode(html):
    s = r'<li><a target="_blank" href="http://quote.eastmoney.com/\S\S(.*?).html">'
    pat = re.compile(s)
    code = pat.findall(html)
    return code

#codes = getStackCode(getHtml(URL))

def eastmoney_data_scooping(code_list, start_date=None, end_date=None, proxy=None):
    print('----------------------------------------------------------------------')
    print('loading commencing')
    print('----------------------------------------------------------------------')
    start = time.time()
    if not end_date:
        end_date = datetime.datetime.now().strftime("%Y%m%d")
    if not start_date:
        start_date = datetime.datetime(2017, 1, 1).strftime("%Y%m%d")
    if not proxy:
        proxy = {}   
        
    for code in code_list:
        code = str(code)
        if code[0] == '0' or code[0] == '3':
            url = 'http://quotes.money.163.com/service/chddata.html?code=1'+code+'&start='+start_date+'&end='+end_date
        elif code[0] == '6':
            url = 'http://quotes.money.163.com/service/chddata.html?code=0'+code+'&start='+start_date+'&end='+end_date
        else:
            print (code + 'does not start with 6 or 0')
            continue
        
        # scoop data
        content = urllib.urlopen(url, proxies=proxy).read()
        reader = content.split('\n')
        print (code + ' loaded')
            
        
            
    end = time.time()
    print('----------------------------------------------------------------------')
    print('loading completed. total run time: %.2f' % (end-start))
    print('----------------------------------------------------------------------')
    return reader

#if __name__ == '__main__':
#    reader = eastmoney_data_scooping(['0003333',], end='20171204', proxy=True)
