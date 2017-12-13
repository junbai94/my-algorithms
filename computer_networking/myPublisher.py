import zmq
import time
import random

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:8000")

while True:
   topic = random.randrange(1, 10)
   message = random.randrange(10000, 20000)
   print ("topic:{} message:{}".format(topic, message))
   socket.send("{} {}".format(topic, message))
   time.sleep(1)