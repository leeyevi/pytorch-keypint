import os
import os.path
import sys
import numpy as np

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        
class Config:

    cur_dir = os.path.dirname(os.path.abspath(__file__))#.split('\\')[:-1]#[:-2]
    this_dir_name = cur_dir.split('\\')[-1]#this_dir_name = cur_dir.split('/')[-1]

    #root_dir = os.path.join(cur_dir, '..')
    root_dir = cur_dir[:-7]#os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

    model = 'CPN50' # option 'CPN50', 'CPN101'

    num_class = 7 #hehaojie17
    img_path ='E:/pWorkSpace/pytorch-cpn/data/20180906_KeyPoint/val'#

    #symmetry = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    symmetry = [(1, 2), (3, 4), (5, 6)]
    bbox_extend_factor = (0.1, 0.15) # x, y

    pixel_means = np.array([122.7717, 115.9465, 102.9801]) # RGB
    data_shape = (384, 288)#data_shape = (256, 192)
    output_shape = (96, 72)#output_shape = (64, 48)

    batch = 1
    num_gpus = 1
    workers = 12
    flip = False#True
    checkpoint = 'checkpoint'

    use_GT_bbox = True
    if use_GT_bbox:
        gt_path = 'E:/pWorkSpace/pytorch-cpn/data/20180906_KeyPoint/annotations/val.json'#

    else:
        # if False, make sure you have downloaded the val_dets.json and place it into annotation folder
        gt_path = os.path.join(root_dir, 'data', 'COCO2017', 'annotations', 'val_dets.json')
    #ori_gt_path = '/media/hehaojie/KBD_Work/3_网络模型/6_CPN/COCO2017/annotations/person_keypoints_val2017.json' #hehaojie os.path.join(root_dir, 'data', 'COCO2017', 'annotations', 'person_keypoints_val2017.json')

cfg = Config()
add_pypath(cfg.root_dir)
add_pypath(os.path.join(cfg.root_dir, 'cocoapi/PythonAPI'))