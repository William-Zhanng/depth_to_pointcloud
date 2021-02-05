import numpy as np
import re


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def find_occ_mask(disp_left, disp_right):
    """
    find occlusion map
    1 indicates occlusion
    disp range [0,w]
    """
    w = disp_left.shape[-1]

    # # left occlusion
    # find corresponding pixels in target image
    
    coord = np.linspace(0, w - 1, w)[None,]  # 1xW
    right_shifted = coord - disp_left

    # 1. negative locations and out of bounds locations will be occlusion
    occ_mask_neg = (right_shifted <= 0) 
    occ_mask_out = (right_shifted >= w)

    occ_mask_l = np.logical_or(occ_mask_neg,occ_mask_out)

    # 2. wrong matches will be occlusion
    
    right_shifted[occ_mask_l] = 0  # set negative locations to 0
    right_shifted = right_shifted.astype(np.int)
    disp_right_selected = np.take_along_axis(disp_right, right_shifted,
                                             axis=1)  # find tgt disparity at src-shifted locations
    wrong_matches = np.abs(disp_right_selected - disp_left) > 1  # theoretically, these two should match perfectly
    wrong_matches[disp_right_selected <= 0.0] = False
    wrong_matches[disp_left <= 0.0] = False

    # produce final occ
    wrong_matches[occ_mask_l] = True  # apply case 1 occlusion to case 2
    occ_mask_l = wrong_matches
    
    # # right occlusion
    # find corresponding pixels in target image
    coord = np.linspace(0, w - 1, w)[None,]  # 1xW
    left_shifted = coord + disp_right

    # 1. negative locations negative locations and out of bounds locations will be occlusionwill be occlusion
    occ_mask_out = left_shifted >= w
    occ_mask_neg = left_shifted <= 0
    occ_mask_r = np.logical_or(occ_mask_neg,occ_mask_out)
    # 2. wrong matches will be occlusion
    left_shifted[occ_mask_r] = 0  # set negative locations to 0
    left_shifted = left_shifted.astype(np.int)
    #print(left_shifted[left_shifted<0])
    disp_left_selected = np.take_along_axis(disp_left, left_shifted,axis=1)  # find tgt disparity at src-shifted locations
   # disp_left_selected = np.take(disp_left,left_shifted,axis=1)

    #print("same",disp_left_selected == disp_left_selected_new)
    wrong_matches = np.abs(disp_left_selected - disp_right) > 1  # theoretically, these two should match perfectly
    wrong_matches[disp_left_selected <= 0.0] = False
    wrong_matches[disp_right <= 0.0] = False

    # produce final occ
    wrong_matches[occ_mask_r] = True  # apply case 1 occlusion to case 2
    occ_mask_r = wrong_matches

    return occ_mask_l, occ_mask_r

