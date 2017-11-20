import socket
import numpy as np
import io

try:
    sock = socket.create_connection(('localhost', 7000))
    with io.FileIO("4.jpg", 'r') as image_file:
        data = image_file.readall()
        sock.sendall(data)
    sock.close()
except Exception:
    print("Failed")
