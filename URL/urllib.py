# -*- coding: utf-8 -*-
"""
Created on Tue Dec 05 10:08:08 2017

@author: j291414

urllib2
"""

import urllib
import requests
import re

#proxies = {'http':'http://j291414:Battleship1!@10.252.22.101:4300', 
#           'https':'https://j291414:Battleship1!@10.252.22.101:4300'}
proxies = {'http':'http://j291414:Battleship1!@10.252.22.102:4200', 
           'https':'https://j291414:Battleship1!@10.252.22.102:4200'}

#response = urllib.urlopen('http://python.org', proxies=proxies)
#html = response.read()
# r is a Response object
#payload = {'key1': 'value1', 'key2':'value2'}
# pass in keys in to get function
# r = request.get('http://httpbin.org/get', params=payload)
#r = requests.get('http://api.github.com/events', proxies=proxies)
#print (r.url)
#print 
##print r.text
#print r.encoding
#print r.content

#url = 'http://nufm.dfcfw.com/EM_Finance2014NumericApplication/JS.aspx?&type=CT&sty=GB20GFBTC&st=z&js=((x))&token=4f1862fc3b5e77c150a2b985b12db0fd&cb=jQuery1830059569148545979056_1512442607630&cmd=3001012&_=1512442627721'
##r = requests.get(url, proxies=proxies)
###print r.content
##print r.text
#
#url2 = 'http://nufm.dfcfw.com/EM_Finance2014NumericApplication/JS.aspx?&type=CT&sty=GB20GFBTC&st=z&js=((x))&token=4f1862fc3b5e77c150a2b985b12db0fd&cb=jQuery183029288183941331236_1512444104562&cmd=3001012&_=1512444206141'
#
#url3 = 'http://nufm.dfcfw.com/EM_Finance2014NumericApplication/JS.aspx?type=CT&cmd=C.BK04481&sty=FDCS&st=C&sr=-1&p=1&ps=5&lvl=&cb=&js=var%20jspy=[(x)];&token=4f1862fc3b5e77c150a2b985b12db0fd&v=0.600632293543534&_=1512444241121'
#
#
#url4 = 'http://mdfm.eastmoney.com/EM_UBG_MinuteApi/Js/Get?dtype=25&style=tail&check=st&dtformat=HH:mm:ss&cb=jQuery183029288183941331236_1512444104562&id=3001012&num=9&_=1512444214121'
#
#
#_url = 'http://nufm.dfcfw.com/EM_Finance2014NumericApplication/JS.aspx?type=CT&cmd=3001012&sty=CTBF&st=z&sr=&p=&ps=&cb=var%20pie_data=&js=(x)&token=28758b27a75f62dc3065b81f7facb365&_=1497258906443'
#_url1 = 'http://nufm.dfcfw.com/EM_Finance2014NumericApplication/JS.aspx?type=CT&cmd=0008582&sty=CTBF&st=z&sr=&p=&ps=&cb=var%20pie_data=&js=(x)&token=28758b27a75f62dc3065b81f7facb365&_=1497258906443'
#r = requests.get(_url1, proxies=proxies)
URL = 'http://quote.eastmoney.com/stocklist.html'#东方财富网股票数据连接地址
def getHTML(url):
    html = urllib.urlopen(url, proxies=proxies).read()
    html = html.decode('gbk')
    return html

def getStackCode(html):
    s = r'<li><a target="_blank" href="http://quote.eastmoney.com/\S\S(.*?).html">'
    pat = re.compile(s)
    code = pat.findall(html)
    return code


def get_CSI300_constituents():
    txt = urllib.urlopen('http://www.csindex.com.cn/uploads/file/autofile/cons/000300cons.xls', proxies=proxies).read()
    return txt