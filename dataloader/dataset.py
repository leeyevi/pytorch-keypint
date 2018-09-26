import numpy as np
import cv2
import torch
#from utils.transforms import *
from model.test_config import cfg

def color_normalize(x, mean):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)
    mean /= 255
    for t, m in zip(x, mean):
        t.sub_(m)
    return x

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    #mg = img[np.newaxis, :]#
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img

def augmentationCropImage(img, bbox, joints=None):
    height, width = cfg.data_shape#self.inp_res[0], self.inp_res[1]
    bbox = np.array(bbox).reshape(4, ).astype(np.float32)
    add = max(img.shape[0], img.shape[1])
    mean_value = cfg.pixel_means
    bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT, value=mean_value.tolist())
    objcenter = np.array([(bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.])
    bbox += add
    objcenter += add
    is_train = False
    if is_train:
        joints[:, :2] += add
        inds = np.where(joints[:, -1] == 0)
        joints[inds, :2] = -1000000 # avoid influencing by data processing
    crop_width = (bbox[2] - bbox[0]) * (1 + cfg.bbox_extend_factor[0] * 2)
    crop_height = (bbox[3] - bbox[1]) * (1 + cfg.bbox_extend_factor[1] * 2)
    if is_train:
        crop_width = crop_width * (1 + 0.25)
        crop_height = crop_height * (1 + 0.25)
    if crop_height / height > crop_width / width:
        crop_size = crop_height
        min_shape = height
    else:
        crop_size = crop_width
        min_shape = width

    crop_size = min(crop_size, objcenter[0] / width * min_shape * 2. - 1.)
    crop_size = min(crop_size, (bimg.shape[1] - objcenter[0]) / width * min_shape * 2. - 1)
    crop_size = min(crop_size, objcenter[1] / height * min_shape * 2. - 1.)
    crop_size = min(crop_size, (bimg.shape[0] - objcenter[1]) / height * min_shape * 2. - 1)

    min_x = int(objcenter[0] - crop_size / 2. / min_shape * width)
    max_x = int(objcenter[0] + crop_size / 2. / min_shape * width)
    min_y = int(objcenter[1] - crop_size / 2. / min_shape * height)
    max_y = int(objcenter[1] + crop_size / 2. / min_shape * height)

    x_ratio = float(width) / (max_x - min_x)
    y_ratio = float(height) / (max_y - min_y)

    if is_train:
        joints[:, 0] = joints[:, 0] - min_x
        joints[:, 1] = joints[:, 1] - min_y

        joints[:, 0] *= x_ratio
        joints[:, 1] *= y_ratio
        label = joints[:, :2].copy()
        valid = joints[:, 2].copy()

    img = cv2.resize(bimg[min_y:max_y, min_x:max_x, :], (width, height))
    details = np.asarray([min_x - add, min_y - add, max_x - add, max_y - add]).astype(np.float)

    if is_train:
        return img, joints, details
    else:
        return img, details

def getData(image):
    gt_bbox = [0, 0, image.shape[1], image.shape[0]]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, details = augmentationCropImage(image, gt_bbox)
    #image = image[np.newaxis, :]
    #image = color_normalize(image, cfg.pixel_means)
    img = im_to_torch(image)
    img = color_normalize(img, cfg.pixel_means)
    img = to_numpy(img)
    img = img[np.newaxis, :]
    img = to_torch(img)

    meta = {'index': 1, 'imgID': 1,
            'GT_bbox': np.array([gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]]),
            'img_path': 'xxx', 'augmentation_details': details}

    meta['det_scores'] = 1
    return img, meta