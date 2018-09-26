import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import json
import torch.nn.parallel
import torch.optim
from networks import network
from model.test_config import cfg
from utils.imutils import im_to_numpy, im_to_torch
#from dataloader.mscocoMulti import MscocoMulti
from dataloader.Multi import MscocoMulti
from utils.osutils import mkdir_p, isdir
from dataloader.dataset import getData, to_torch

from socket import *
import time

def hm_show(heatmap, image):
    plt.figure()
    #plt.figure(num='astronaut', figsize=(8, 8))
    #plt.subplot(2, 4, 8)
    #plt.imshow(image)

    #heatmap.transpose((0, 1, 3, 2))

    for i in range(8):
        if i < 7:

            plt.subplot(2, 4, 1 + i)
            plt.title('predict heatmap #{}'.format(i))
            plt.imshow(heatmap[0, i, :, :])
        else:
            plt.subplot(2, 4, 8)
            plt.imshow(image)

    plt.show()

def kp_inference(image):
    #start = time.clock()
    # create model
    model = network.__dict__[cfg.model](cfg.output_shape, cfg.num_class, pretrained=False)
    start = time.clock()
    model = torch.nn.DataParallel(model).cuda()
    c1 = time.clock()
    print('create model:%s s' % (c1-start))


    inputs, meta = getData(image)
    c2 = time.clock()
    print('load data:%s s' % (c2 - c1))

    # load trainning weights
    checkpoint_file = os.path.join(cfg.checkpoint, 'epoch500checkpoint.pth.tar')  # 'model',  args.test+'.pth.tar')
    checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))
    c3 = time.clock()
    print('load weights:%s s' % (c3 - c2))
    # change to evaluation mode
    model.eval()

    print('testing...')
    full_result = []

    result = ''
    #a = test_loader[0]

    #for i, (inputs, meta) in tqdm(enumerate(test_loader)):#E:\pWorkSpace\pytorch-cpn\model\checkpoint\epoch500checkpoint.pth.tar
        # print(i)
        # if i > 3:
        #     break
    c4 = time.clock()
    print('enumerate:%s s' % (c4 - c3))
    with torch.no_grad():
        input_var = torch.autograd.Variable(inputs.cuda())
        if cfg.flip == True:
            flip_inputs = inputs.clone()
            for i, finp in enumerate(flip_inputs):
                finp = im_to_numpy(finp)
                finp = cv2.flip(finp, 1)
                flip_inputs[i] = im_to_torch(finp)
            flip_input_var = torch.autograd.Variable(flip_inputs.cuda())
        c6 = time.clock()
        print('flip :%s s' % (c6 - c4))
            # compute output
            #print('predict...')
        global_outputs, refine_output = model(input_var)
        c7 = time.clock()
        print('predict :%s s' % (c7 - c6))
        score_map = refine_output.data.cpu()
        score_map = score_map.numpy()
        cfg.flip = False
        if cfg.flip == True:
            flip_global_outputs, flip_output = model(flip_input_var)
            flip_score_map = flip_output.data.cpu()
            flip_score_map = flip_score_map.numpy()

            for i, fscore in enumerate(flip_score_map):
                fscore = fscore.transpose((1, 2, 0))
                fscore = cv2.flip(fscore, 1)
                fscore = list(fscore.transpose((2, 0, 1)))
                for (q, w) in cfg.symmetry:
                    fscore[q], fscore[w] = fscore[w], fscore[q]
                fscore = np.array(fscore)
                score_map[i] += fscore
                score_map[i] /= 2

        ids = meta['imgID']  # hehaojie.numpy()
        det_scores = meta['det_scores']
        for b in range(inputs.size(0)):

                # if b > 3:
                #     break

            details = meta['augmentation_details']
            details = details[np.newaxis, :]
            details = to_torch(details)
            single_result_dict = {}
            single_result = []

            single_map = score_map[b]
            r0 = single_map.copy()
            r0 /= 255
            r0 += 0.5
            v_score = np.zeros(7)
            for p in range(7):
                single_map[p] /= np.amax(single_map[p])
                border = 10
                dr = np.zeros((cfg.output_shape[0] + 2 * border, cfg.output_shape[1] + 2 * border))
                dr[border:-border, border:-border] = single_map[p].copy()
                dr = cv2.GaussianBlur(dr, (21, 21), 0)
                lb = dr.argmax()
                y, x = np.unravel_index(lb, dr.shape)
                dr[y, x] = 0
                lb = dr.argmax()
                py, px = np.unravel_index(lb, dr.shape)
                y -= border
                x -= border
                py -= border + y
                px -= border + x
                ln = (px ** 2 + py ** 2) ** 0.5
                delta = 0.25
                if ln > 1e-3:
                    x += delta * px / ln
                    y += delta * py / ln
                x = max(0, min(x, cfg.output_shape[1] - 1))
                y = max(0, min(y, cfg.output_shape[0] - 1))
                resy = float((4 * y + 2) / cfg.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                resx = float((4 * x + 2) / cfg.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
                v_score[p] = float(r0[p, int(round(y) + 1e-10), int(round(x) + 1e-10)])
                single_result.append(resx)
                single_result.append(resy)
                single_result.append(1)
                    #输出string
                result = result + str(resx) + ',' + str(resy) + ','



            if len(single_result) != 0:

                    #img = cv2.imread(
                    #    'E:/pWorkSpace/pytorch-cpn/data/20180906_KeyPoint/val/' + ids[b] + '.jpg')
                    # '/media/hehaojie/KBD_Work/2_数据集制作/20180831_关键点_直行箭头/20180831_KeyPoint/val/' +ids[b] + '.jpg')
                ig = 0
                for hh in range(0, len(single_result), 3):
                    cv2.circle(image, (int(single_result[hh]), int(single_result[hh + 1])), radius=1,
                                   color=(255, 0, 255), thickness=-1)
                    cv2.putText(image, str(ig), (int(single_result[hh]) + 2, int(single_result[hh + 1]) + 2),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 255), 1)
                    ig = ig + 1

                    # inputs('请输入任意键继续：')

                #single_result_dict['image_id'] = ids[b]
                #single_result_dict['category_id'] = 1
                #single_result_dict['keypoints'] = single_result
                #single_result_dict['score'] = float(det_scores[b]) * v_score.mean()

                #full_result.append(single_result_dict)

                # cv2.namedWindow('keypoint', 0)
                # cv2.resizeWindow('keypoint', 1000, 1000)

                # cv2.imshow('keypoint', img)
                # cv2.waitKey(3000)
                #plt.imshow(image)
                #plt.show()
                hm_show(score_map, image)

    return result
    #result_path = args.result
    #if not isdir(result_path):
    #    mkdir_p(result_path)
    #result_file = os.path.join(result_path, 'result.json')
    #with open(result_file, 'w') as wf:
    #    json.dump(full_result, wf)

    # evaluate on COCO
    '''eval_gt = COCO(cfg.ori_gt_path)
    eval_dt = eval_gt.loadRes(result_file)
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()'''

def get_tcp():

    address = '0.0.0.0'
    port = 5001  # 监听自己的哪个端口

    buffsize = 1000000  # 接收从客户端发来的数据的缓存区大小

    s = socket(AF_INET, SOCK_STREAM)
    s.bind((address, port))
    s.listen(1)

    while True:
        clientsock, clientaddress = s.accept()
        print('connect from:', clientaddress)

        #
        recvdata = clientsock.recv(buffsize)
        print('recvdata:', recvdata)
        print('recvdata type:', type(recvdata))

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
                iPos = i * cols + j
                image[i][j] = recvdata[iPos + 4]

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        #
        result = kp_inference(image)

        result = result.encode(encoding="utf-8")

        #cv2.imshow("img_trans", image)
        #cv2.waitKey()

        clientsock.send(result)
        clientsock.close()

    s.close()



if __name__ == '__main__':

    #image = cv2.imread('E:\\pWorkSpace\\pytorch-cpn\\data\\20180906_KeyPoint\\val\\300000444.jpg')
    #iamge = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #start = time.clock()
    #re = kp_inference(image)
    #end = time.clock()
    #a = re[0]
    #print('inference :%s s' % (end - start))
    get_tcp()