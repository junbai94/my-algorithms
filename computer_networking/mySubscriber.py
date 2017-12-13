import zmq
import time
import random

context = zmq.Context()
socket = context.socket(zmq.SUB)

topicFilter = ['2', '3']
for e in topicFilter:
    socket.setsockopt(zmq.SUBSCRIBE, e)

socket.connect("tcp://localhost:8000")
print("I am gonna to receive")

for i in range(10):
    string = socket.recv()
    topic, message = string.split()
    print topic, message
    
print "completed"