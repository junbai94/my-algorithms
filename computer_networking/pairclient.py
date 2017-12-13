# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 09:08:32 2017

@author: j291414
"""

import zmq
import random
import sys
import time

port = "5556"
context = zmq.Context()
socket2 = context.socket2(zmq.PAIR)
socket2.connect("tcp://localhost:%s" % port)

while True:
    msg = socket2.recv()
    print msg
    socket2.send("client message to server1")
    socket2.send("client message to server2")
    time.sleep(1)