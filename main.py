import os
import argparse

import numpy as np
import torch.utils.data as data
from PIL import Image
from albumentations import Compose, OneOf
from natsort import natsorted
from utilities import find_occ_mask
from utilities import readPFM
import cv2
'''
   A demo to generate Sceneflow dataset occlusion map

'''
#if first run,automatically create folders,cost some time
first_run = True

def generate_occ_mask_map(args,split_folder = 'TRAIN'):

    DATA_DIR = os.path.join(args.DATA_PATH,'flyingthings3d')
    directory = os.path.join(DATA_DIR,'frames_cleanpass',split_folder)
    #A,B,C
    sub_folders = [os.path.join(directory, subset) for subset in os.listdir(directory) if
                       os.path.isdir(os.path.join(directory, subset))]
    #For seq scene
    seq_folders = []
    for sub_folder in sub_folders:
        seq_folders += [os.path.join(sub_folder, seq) for seq in os.listdir(sub_folder) if
                        os.path.isdir(os.path.join(sub_folder, seq))]
    left_data = []
    for seq_folder in seq_folders:
        left_data += [os.path.join(seq_folder, 'left', img) for img in
                            os.listdir(os.path.join(seq_folder, 'left'))]
    left_data = natsorted(left_data)

    #make occ_dirs
    occ_dirs = [os.path.join(i,'left') for i in seq_folders] + [os.path.join(i,'right') for i in seq_folders]
    occ_dirs = [i.replace('frames_cleanpass','occlusion') for i in occ_dirs]
    if first_run:
        for i in occ_dirs:
            if not os.path.exists(i):
                os.makedirs(i,exist_ok=True)

    Count = 0
    img_num = len(left_data)
    for left_fname in left_data:
        right_fname = left_fname.replace('left', 'right')
        disp_left_fname = left_fname.replace('frames_cleanpass', 'disparity').replace('.png', '.pfm')
        disp_right_fname = right_fname.replace('frames_cleanpass', 'disparity').replace('.png', '.pfm')
        
        disp_left, _ = readPFM(disp_left_fname)
        disp_right, _ = readPFM(disp_right_fname)

        occ_left_name = left_fname.replace('frames_cleanpass', 'occlusion')
        occ_right_name = right_fname.replace('frames_cleanpass', 'occlusion')
        print(occ_left_name)
        #generate occ_map
        left_occ,right_occ = find_occ_mask(disp_left,disp_right)
        left_occ = left_occ.astype(np.uint8)*255.
        right_occ = right_occ.astype(np.uint8)*255.

        cv2.imwrite(occ_left_name,left_occ)
        cv2.imwrite(occ_right_name,right_occ)
        print(" rate of advance : {} / {}    {} % ".format(Count+1,img_num,Count/img_num*100))
        Count += 1
    

def generate_occ_mask_map_for_others(args,sub_dir = 'Monkaa'):

    DIR = os.path.join(args.DATA_PATH,sub_dir)
    directory = os.path.join(DIR,'frames_cleanpass')
    #A,B,C
    sub_folders = [os.path.join(directory, subset) for subset in os.listdir(directory) if
                       os.path.isdir(os.path.join(directory, subset))]

    left_data = []
    for sub_folder in sub_folders:
        left_data += [os.path.join(sub_folder, 'left', img) for img in
                            os.listdir(os.path.join(sub_folder, 'left'))]
    left_data = natsorted(left_data)

    #make occ_dirs
    occ_dirs = [os.path.join(i,'left') for i in sub_folders] + [os.path.join(i,'right') for i in sub_folders]
    occ_dirs = [i.replace('frames_cleanpass','occlusion') for i in occ_dirs]
    if first_run:
        for i in occ_dirs:
            if not os.path.exists(i):
                os.makedirs(i,exist_ok=True)

    Count = 0
    img_num = len(left_data)
    for left_fname in left_data:
        right_fname = left_fname.replace('left', 'right')
        disp_left_fname = left_fname.replace('frames_cleanpass', 'disparity').replace('.png', '.pfm')
        disp_right_fname = right_fname.replace('frames_cleanpass', 'disparity').replace('.png', '.pfm')
        
        disp_left, _ = readPFM(disp_left_fname)
        disp_right, _ = readPFM(disp_right_fname)

        occ_left_name = left_fname.replace('frames_cleanpass', 'occlusion')
        occ_right_name = right_fname.replace('frames_cleanpass', 'occlusion')
        print(occ_left_name)
        #generate occ_map
        left_occ,right_occ = find_occ_mask(disp_left,disp_right)
        left_occ = left_occ.astype(np.uint8)*255.
        right_occ = right_occ.astype(np.uint8)*255.

        cv2.imwrite(occ_left_name,left_occ)
        cv2.imwrite(occ_right_name,right_occ)
        print(" rate of advance : {} / {}    {} % ".format(Count+1,img_num,Count/img_num*100))
        Count += 1

def main():
    generate_occ_mask_map(split_folder='TRAIN',args)
    generate_occ_mask_map(split_folder='TEST',args)
    if(args.all_dataset):
        generate_occ_mask_map_for_others(args,'Monkaa')
        generate_occ_mask_map_for_others(args,'Driving')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_PATH',default = '/data3/StereoMatching/SceneFlow/',
                         help = 'directory to your SceneFlow Dataset')

    parser.add_argument('--all_dataset',action='store_true',
                        help = 'For all Sceneflow dataset or only for flyingthings3d')

    args = parser.parse_args()
    main()