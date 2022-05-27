

import numpy as np
import cv2
import time


def preprocess_factory(image, target):
    rescale_t = None
    rescale_t, scale = rescaleAbsolute(image, target)


    padded_image, pad = CenterPadTight(rescale_t, target)

    return padded_image, scale, pad


def _scale(image, target_w, target_h):
    h, w, c = image.shape
    image = cv2.resize(image, (int(target_w), int(target_h)), interpolation=cv2.INTER_LINEAR)
    x_scale = (image.shape[1] - 1) / (w - 1)
    y_scale = (image.shape[0]- 1) / (h -1 )
    return image, (x_scale, y_scale)

def rescaleAbsolute(image, target):
    src_h, src_w, c = image.shape
    dst_w = target[1];
    dst_h = target[0]; 
    w = dst_w;
    h = dst_w * src_h / src_w
    if h > dst_h:
        h = dst_h;
        w = h * src_w / src_h   
    return _scale(image, w, h)
    
        
def CenterPadTight(image, target):
    start_time = time.time()
    image, pad = center_pad(image, target)
    # print(np.asarray(image))
    return image, pad

def center_pad(image, target):
    h, w, c = image.shape

    target_width = target[1]
    target_height = target[0]

    left, top, right, bottom = 0, 0, 0, 0
    if w < target_width:
        left = int((target_width - w) / 2.0)
        right = target_width - left - w
    else:
        top = int((target_height - h) / 2.0)
        bottom = target_height - top - h
    ltrb = (left, top, right, bottom)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return image, ltrb

# if __name__ == "__main__":
#     image = cv2.imread("D:\\ThangLM\\Images\\5.jpeg")
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     print("image size: ", image.shape)
#     result, scale, ltrb = preprocess_factory(image, (385, 673))

#     result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
#     cv2.imwrite("scale.jpg", result)
#     print("shape: ", result.shape)
#     print("scale: ", scale)
#     print("ltrb: ", ltrb)
