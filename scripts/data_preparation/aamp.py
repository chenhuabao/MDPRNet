# ------------------------------------------------------------------------
# Modified from NAFNet (https://github.com/megvii-research/NAFNet)
# ------------------------------------------------------------------------
import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm

from basicsr.utils.create_lmdb import create_lmdb_for_AAPM


def main():
    opt = {}

    opt['lq_img_path'] = './datdsets/AAPM/train/input'
    opt['hq_img_path'] = './datdsets/AAPM/train/gt'
    opt['lq_lmdb_path'] = './datdsets/AAPM/train/input.lmdb'
    opt['hq_lmdb_path'] = './datdsets/AAPM/train/gt.lmdb'


    create_lmdb_for_SIDD(opt)


if __name__ == '__main__':
    main()