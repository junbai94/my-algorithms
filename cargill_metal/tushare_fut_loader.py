# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:10:05 2017

@author: junbai

Using tushare package to load SHFE and DCE future data into database.
"""

import tushare
import sqlite3

DATABASE = "C:\\Users\\j291414\\Desktop\\market_data.db"

def tushare_fut_loader():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    
    
    
    conn.commit()
    conn.close()
    
    return 