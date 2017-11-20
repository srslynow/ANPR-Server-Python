import os
import numpy as np
import cv2
import time
import socket
import tempfile
from threading import Thread
from queue import Queue
from Predict import Predict
import logging

class Server:
    predictor = Predict()
    port = 7000
    q = Queue(20)
    logger = logging.getLogger(__name__)

    def __init__(self):
        self.logger.setLevel(logging.DEBUG)
        t = Thread(target=self.worker)
        t.daemon = True
        t.start()
        try:
            self.start_server()
        except (KeyboardInterrupt, SystemExit):
            exit()

    def recv_basic(self, the_socket):
        total_data = b''
        while True:
            data = the_socket.recv(4096)
            if not data: break
            total_data += data
        return total_data

    def start_server(self):
        sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.bind(('localhost',self.port))
        sock.listen(self.port)
        self.logger.info('Started on port ' + str(self.port))
        while True:
            try:
                newsock, addr = sock.accept()
                result = self.recv_basic(newsock)
                if len(result) == 0: continue
                result = np.frombuffer(result, dtype=np.uint8)
                img = cv2.imdecode(result, cv2.IMREAD_GRAYSCALE)
                self.q.put(img)
                self.logger.info("New image put to queue")
            except:
                continue

    
    def worker(self):
        while True:
            img = self.q.get()
            start = time.time()
            plates = self.predictor.predict(img)
            for plate in plates:
                self.logger.info(plate)
            self.logger.info("Done in %.2f s." % (time.time() - start))
            self.q.task_done()


if __name__== '__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%H:%M:%S %d-%m-%Y')
    Server()
