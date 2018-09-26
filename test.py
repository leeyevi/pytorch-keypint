from socket import *
import numpy as np
import cv2

address = '0.0.0.0'
port = 5001             #监听自己的哪个端口

buffsize = 1000000          #接收从客户端发来的数据的缓存区大小

s = socket(AF_INET, SOCK_STREAM)
s.bind((address, port))
s.listen(1)     #最大连接数

#
while True:
    clientsock, clientaddress = s.accept()
    print('connect from:', clientaddress)

    #
    recvdata = clientsock.recv(buffsize)
    clientsock.close()
    print('recvdata:', recvdata)

    #
    date1 = 0
    date2 = 0
    date3 = 0
    date4 = 0

    for i in range(4):
        if (0 == i):
            date1 = recvdata[i]
        if (1 == i):
            date2 = recvdata[i]
        if (2 == i):
            date3 = recvdata[i]
        if (3 == i):
            date4 = recvdata[i]

    rows = date2 * 256 + date1
    cols = date4 * 256 + date3

    image = np.zeros((rows, cols, 1), dtype=np.uint8)

    iLen = len(recvdata)
    for i in range(rows):
        for j in range(cols):
            iPos = i*cols + j
            image[i][j] = recvdata[iPos + 4]

    #
    cv2.imshow("img_trans", image)
    cv2.waitKey()



s.close()
