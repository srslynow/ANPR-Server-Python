import io
import socket
import struct
from time import sleep

import numpy as np

import cv2

UDP_IP = 'localhost'
UDP_PORT = 6112

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap = cv2.VideoCapture('2.mp4')
    continue_reading = True
    while continue_reading:
        (continue_reading, image_file) = cap.read()
        (_, data) = cv2.imencode('.jpg', image_file)
        datalen = int(len(data))

        sock.sendto(struct.pack("<I", datalen), (UDP_IP, UDP_PORT))
        chunks = [data[x:x + 1472] for x in range(0, len(data), 1472)]
        for chunk in chunks:
            sock.sendto(chunk, (UDP_IP, UDP_PORT))
            # sleep(0.001)
except Exception as e:
    print(e)
    print("Failed")
