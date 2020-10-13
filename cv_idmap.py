import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
import torchvision.transforms as transform
import cv2 as cv
'''
sky : [252, 151, 146]
ground_1 : [252, 229, 176]
ground_2 : [248, 156, 79]
ground_3 : [183, 140, 93]
ground_4 : [159, 233, 168]
ground_5 : [137, 157, 197]
ground_6 : [223, 75, 38]
ground_7 : [102, 230, 252]
ground_8 : [229, 117, 213]
ground_9 : [168, 88, 248]
ground_10 : [129, 236, 90]
ground_11 : [73, 161, 247]
rock_1 : [175, 232, 86] //jang ground
gate : [210, 93, 134]
classfication : sky + ground + janground + gate + others 4class
'''
map_dic = {0 : [0,0,0], 1 : [146, 151, 252], 2:[[176, 229, 252], [79, 156, 248], [93, 140, 183], [168, 233, 159], [197, 157, 137],
           [38, 75, 223], [252, 230, 102], [213, 117, 229], [248, 88, 168], [90, 236, 129],
           [247, 161, 73], [86, 232, 175]], 3:[134, 93, 210]}


sky = [146, 151, 252]
grounds = [[176, 229, 252], [79, 156, 248], [93, 140, 183], [168, 233, 159],
           [197, 157, 137], [38, 75, 223], [252, 230, 102], [213, 117, 229],
           [248, 88, 168], [90, 236, 129], [247, 161, 73], [86, 232, 175]]
gate = [134, 93, 210]

def rgb2id(img, sky, grounds, gate) :
    id_seg = np.zeros((img.shape[:2]), dtype=np.int)
    id_seg[(img==sky).all(axis=2)] = 0
    for i, id in enumerate(grounds) :
        id_seg[(img==id).all(axis=2)] = 1
    id_seg[(img==gate).all(axis=2)] = 2
    #0 -> other, 70 -> sky ,140 -> ground, 255 -> gate
    return id_seg

type = 'spline'
Map = 'Soccer_Medium'

def makeTargetMap():
    files_spline = glob.glob('ws/dataset1/val/spline/'+Map+'/'+Map+'_left_seg/*.png')
    files_onebyone = glob.glob('ws/dataset1/val/onebyone/'+Map+'/'+Map+'_left_seg/*.png')
    print("spline:",len(files_spline))
    print("onebyone:",len(files_onebyone))
    for i in range(len(files_spline)):
        left_img_spline = cv.imread('ws/dataset1/val/spline/'+Map+'/'+Map+'_left_seg/left_'+str(i)+'.png')
        right_img_spline = cv.imread('ws/dataset1/val/spline/'+Map+'/'+Map+'_right_seg/right_'+str(i)+'.png')
        left_id_seg_spline = rgb2id(left_img_spline, sky, grounds, gate)
        right_id_seg_spline = rgb2id(right_img_spline, sky, grounds, gate)
        left_id_spline = left_id_seg_spline.astype('uint8')
        right_id_spline = right_id_seg_spline.astype('uint8')
        cv.imwrite('ws/dataset1/val/spline/'+Map+'/'+Map+'_left_label/left_'+str(i)+'.png', left_id_spline)
        cv.imwrite('ws/dataset1/val/spline/'+Map+'/'+Map+'_right_label/right_'+str(i)+'.png', right_id_spline)

    for i in range(len(files_onebyone)):
        left_img_onebyone = cv.imread(
            'ws/dataset1/val/onebyone/' + Map + '/' + Map + '_left_seg/left_' + str(i) + '.png')
        right_img_onebyone = cv.imread(
            'ws/dataset1/val/onebyone/' + Map + '/' + Map + '_right_seg/right_' + str(i) + '.png')
        left_id_seg_onebyone = rgb2id(left_img_onebyone, sky, grounds, gate)
        right_id_seg_onebyone = rgb2id(right_img_onebyone, sky, grounds, gate)
        left_id_onebyone = left_id_seg_onebyone.astype('uint8')
        right_id_onebyone = right_id_seg_onebyone.astype('uint8')
        cv.imwrite('ws/dataset1/val/onebyone/' + Map + '/' + Map + '_left_label/left_' + str(i) + '.png',
                   left_id_onebyone)
        cv.imwrite('ws/dataset1/val/onebyone/' + Map + '/' + Map + '_right_label/right_' + str(i) + '.png',
                   right_id_onebyone)

    order = 0
    # for file in files:
    #     img = cv.imread(file)
    #     id_seg = rgb2id(img, sky, grounds, gate)
    #     img = id_seg.astype('uint8')
    #     cv.imwrite('ws/dataset1/val/spline/Soccer_Medium/Soccer_Medium_right_label/right_'+str(order)+'.png', img)
    #     order = order + 1

def main():
    img = cv.imread('left_0.png')
    id_seg = rgb2id(img, sky, grounds, gate)
    id_seg = id_seg.astype('uint8')
    cv.imshow('id_seg',id_seg)
    cv.waitKey(0)

if __name__ == '__main__':
    main()
