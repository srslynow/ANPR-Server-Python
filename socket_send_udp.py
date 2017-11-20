import socket
import numpy as np
import io
import struct
from time import sleep

UDP_IP = 'localhost'
UDP_PORT = 6112

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    with io.FileIO("1.png", 'rb') as image_file:
        data = image_file.readall()
        datalen = int(len(data))

        sock.sendto(struct.pack("<I", datalen), (UDP_IP, UDP_PORT))
        chunks = [data[x:x+1472] for x in range(0, len(data), 1472)]
        for chunk in chunks:
            sock.sendto(chunk, (UDP_IP, UDP_PORT))
            #sleep(0.001)
except Exception as e:
    print(e)
    print("Failed")
