from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
from PIL import Image
import pdb
from scipy.stats import binom
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import copy
import data_manager
from dataset_loader import ImageDataset, ImageDatasetLazy
import transforms as T
import models
from losses import CrossEntropyLabelSmooth, DeepSupervision
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate
from optimizers import init_optim

import enum
parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
# Datasets
parser.add_argument('--root', type=str, default='data', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='market1501',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--split-id', type=int, default=0, help="split index")
# CUHK03-specific setting
parser.add_argument('--cuhk03-labeled', action='store_true',
                    help="whether to use labeled images, if false, detected images are used (default: False)")
parser.add_argument('--cuhk03-classic-split', action='store_true',
                    help="whether to use classic split by Li et al. CVPR'14 (default: False)")
parser.add_argument('--use-metric-cuhk03', action='store_true',
                    help="whether to use cuhk03-metric (default: False)")
# Optimization options
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")
parser.add_argument('--max-epoch', default=60, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=32, type=int, help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=20, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
# Miscs
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

f_rate = 60
Temporal_corr_prob = np.zeros((8,8,30))
Spatial_corr_prob = np.array([
[0.8638795986619184,0.05317725752506583,0.0,0.0,0.0020066889632100312,0.0,0.0006688963210700104,0.005016722408025078,0.07525083612037617],
[0.05122349102771575,0.8753670473080342,0.028711256117445773,0.0,0.020228384991836792,0.0,0.0006525285481237675,0.002936378466556954,0.02088091353996056],
[0.0,0.07226738934049479,0.8148148148140788,0.0776874435410319,0.01626016260161133,0.0,0.0,0.0,0.018970189701879882],
[0.0,0.0,0.04969366916266189,0.8869979577938142,0.0013614703880181337,0.0,0.0,0.0,0.06194690265482509],
[0.005278592375363473,0.047507331378271254,0.014076246334302595,0.002932551319646374,0.8709677419349731,0.03988269794719068,0.005865102639292748,0.0035190615835756487,0.00997067448679767],
[0.0,0.0,0.0,0.0,0.022562240663894565,0.9079356846470674,0.020746887966799597,0.0,0.04875518672197906],
[0.0007401924500364617,0.0,0.0,0.0,0.013323464100656312,0.09548482605470357,0.8356772760911653,0.0503330866024794,0.00444115470021877],
[0.03530819868340197,0.0023937761819255573,0.0,0.0,0.003590664272888336,0.0,0.06343506882102727,0.8414123279468334,0.05385996409332504],
[0.048892490545637796,0.01566720691517675,0.005402485143164397,0.02025931928686649,0.0056726094003226165,0.042679632630998734,0.005942733657480837,0.04511075094542272,0.8103727714746596],
#[0.06698741672832458,0.021465581051065333,0.007401924500367357,0.02775721687637759,0.007772020725385725,0.058475203552902116,0.008142116950404092,0.06180606957806743,0.7401924500367357],
#[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2]
#[0.15806724, 0.16203214, 0.05852189, 0.07765912, 0.09013533, 0.20384859, 0.07142102, 0.08833791, 0.08997674]
])
corr_matrix_prob = [
    [0.,         0.87362637, 0.,         0.,         0.03296703, 0., 0.01098901,  0.08241758],
    [0.49371069, 0.,         0.27672956, 0.,         0.19496855, 0., 0.00628931, 0.02830189],
    [0.,         0.43478261, 0.,         0.4673913,  0.09782609, 0., 0.,         0.,        ],
    [0.,         0.,         0.97333333, 0.,         0.02666667, 0., 0.,         0.,        ],
    [0.04433498, 0.39901478, 0.1182266,  0.02463054, 0.,         0.33497537, 0.04926108, 0.02955665],
    [0.,         0.,         0.,         0.,         0.52095808, 0., 0.47904192, 0.,        ],
    [0.00462963, 0.,         0.,         0.,         0.08333333, 0.59722222, 0.,         0.31481481],
    [0.33714286, 0.02285714, 0.,         0.,         0.03428571, 0.,0.60571429, 0.        ]
]
corr_matrix = [
    [0, 1],
    [0, 1, 2, 4],
    [1, 2, 3],
    [2, 3],
    [1, 2, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [0, 6, 7]
]
start_times = [
    [ 0,  5,  0,  0, 40,  0, 35, 20],
    [10,  0,  0,  0,  5,  0,  0, 10],
    [ 0,  0,  0,  5,  0,  0,  0,  0],
    [ 0,  0,  5,  0, 15,  0,  0,  0],
    [30,  5,  0, 20,  0,  5,  0, 15],
    [ 0,  0,  0,  0,  5,  0,  5,  0],
    [40,  0,  0,  0,  0,  0,  0, 10],
    [10,  5,  0,  0, 10,  0, 10,  0]
]
end_times = [
    [ 6, 80,  0,  0, 60,  0, 55, 55],
    [45,  6, 10,  0, 30,  0,  0, 30],
    [ 0, 15,  6, 40, 10,  0,  0,  0],
    [ 0,  0, 30,  6, 30,  0,  0,  0],
    [65, 55, 50, 30,  6, 50, 10, 35],
    [ 0,  0,  0,  0, 30,  6, 15,  0],
    [65,  0,  0,  0, 15, 55,  6, 30],
    [55, 20,  0,  0, 40,  0,150,  6]
]

start_times_ = [
    [ 0, 12,  0,  0, 52,  0,  0, 28,],
    [23,  0, 11,  0, 11,  0,  0, 17,],
    [ 0, 11,  0, 11, 10,  0,  0,  0,],
    [ 0,  0, 21,  0,  0,  0,  0,  0,],
    [36, 12,  6, 33,  0, 17,  6, 29,],
    [ 0,  0,  0,  0, 23,  0, 15,  0,],
    [ 0,  0,  0,  0, 11, 11,  0, 18,],
    [29,  0,  0,  0, 29,  0, 19,  0,]
]
end_times_ = [
    [  0, 166,   0,   0,  64,   0,   0,  69,],
    [106,   0, 109,   0,  41,   0,   0,  55,],
    [  0,  31,   0,  35,  32,   0,   0,   0,],
    [  0,   0,  56,   0,   0,   0,   0,   0,],
    [ 60, 149,  27,  41,   0,  81,  18,  34,],
    [  0,   0,   0,   0, 803,   0, 392,   0,],
    [  0,   0,   0,   0,  18, 233,   0,  33,],
    [ 52,   0,   0,   0,  42,   0,  39,   0,]
]

start_times = [[f_rate * x for x in y] for y in start_times]
end_times = [[f_rate * x for x in y] for y in end_times]
cam_offsets = [5542, 3606, 27243, 31181, 0, 22401, 18967, 46765]
skip_frame = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]}
last_frame = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0}
frame_window_size = 840
threshold_value = np.array([0.735,0.735,0.735,0.735,0.735,0.735,0.735,0.735])
Random_Prob = 0.0
has_been_scanned = np.zeros(227544)
#np.array([1,1,1,1,1,1,1,1])
#np.array([0.735,0.735,0.735,0.735,0.735,0.735,0.735,0.735])

def baseline(paths,sorted_frame_list):
    tot_count = 0
    fail_count = 0
    result_package ={}
    for key_path in list(paths.keys()):
        query_person = int(key_path)
        Find = 0
        cur_f_end = max_frame + 1;
        cur_f_start = cur_f_end - 6000;
        while not Find :
            if cur_f_end <= 0:
                fail_count += 1
                break

            for ind in range(8):
                for frame in sorted_frame_list[ind]:
                    if frame[0] >= cur_f_start and frame[0] < cur_f_end:
                        for z in range(len(frame)):
                            if z == 0:
                                continue
                            if frame[z] == query_person:
                                Find = 1
                                result_package[query_person] = frame[0]
                                tot_count += 1
                                break
                            else :
                                tot_count += 1

                    if Find == 1:
                        break
                if Find == 1:
                    break

            cur_f_end -= 6000
            cur_f_start -= 6000

    print("Tot _ count : ",tot_count)
    print("Fail _ count :",fail_count)
    #print(result_package)
def update(ind, cur_f_start, cur_f_end):
    for index in corr_matrix[ind]:
        skip_frame[index].append((cur_f_start - start_times[index][ind], cur_f_end - end_times[index][ind]))
def Do_not_check(frame,ind):
    '''global last_frame
    if last_frame[ind] == 0:
        last_frame[ind] = frame
    else :
        if last_frame[ind] < frame + 1000:
            #last_frame[ind] = frame
            return False
        else:
            last_frame[ind] = frame
            return True
    '''


    r1 = 0
    r2 = 0
    for (s_s, s_e) in skip_frame[ind]:
        if frame >= s_s and frame < s_e:
            r1 = 1
            break

    r2 = binom.rvs(1,0.88)

    return r2
    #if not r2:
    #    return r1
    #else: 
    #    return True

    #if r1 or r2:
    #    return True
    #else:
    #    return False
def version1(paths,sorted_frame_list):
    tot_count = 0
    fail_count = 0
    tot_people = len(list(paths.keys()))
    pn = 0
    result_package = {}
    for key_path in list(paths.keys()):
        pn += 1
        global skip_frame
        global last_frame 
        last_frame = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0}
        skip_frame = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]}
        print("Cur query people : ", pn, "/ ", tot_people)
        query_person = int(key_path)
        Find = 0
        cur_f_end = max_frame + 1;
        cur_f_start = cur_f_end - 6000;
        while not Find :
            if cur_f_end <= 0 : 
                fail_count += 1
                break

            for ind in range(8):
                for frame in sorted_frame_list[ind]:
                    if frame[0] >= cur_f_start and frame[0] < cur_f_end and not Do_not_check(frame[0],ind):
                        for z in range(len(frame)):
                            if z == 0:
                                continue
                            if frame[z] == query_person:
                                Find = 1
                                result_package[query_person] = frame[0]
                                tot_count += 1
                                #print("Find ~~")
                                #print(result_package)
                                #input()
                                break
                            else :
                                tot_count += 1
                    if Find == 1:
                        break
                update(ind,cur_f_start,cur_f_end)
                if Find == 1:
                    break

            cur_f_end -= 6000
            cur_f_start -= 6000

    print("Tot _ count : ",tot_count)
    print("Fail _ count : ",fail_count)
    #print(result_package)
in_update_time1 = 0
in_update_time2 = 0
in_update_time3 = 0
in_update_time4 = 0
def init(sorted_frame_list_all):
    len_s = len(sorted_frame_list_all)
    for i in range(len_s):
        sorted_frame_list_all[i][0] = 0
        sorted_frame_list_all[i][1] = 1
def retry_update(index,cam_id,frame_id,frame_list):
    count = 0
    cur_frame_id = frame_id
    i = 0
    while i < 500:
        cur_index = index + i + 1
        if cur_index >= len(frame_list):
            break
        if frame_list[cur_index][2] == cam_id:
            if (frame_list[cur_index][1] - frame_id) % 120 == 0:
                del frame_list[cur_index]
                index -= 1
        i += 1
def update_version2(index,cam_id,frame_id,frame_list,ad_frame,sample_frame,count_frame):
    count = 0
    cur_frame_id = frame_id
    #ii = 0
    add = count_frame[index] #0.000000000000001
    if sample_frame.get(index) != None :
        for frame_index in sample_frame[index]:
            frame_list[frame_index][0] = 1

    if ad_frame.get(index) != None:
        for frame_index in ad_frame[index]:
            add = 1
            if count_frame[index] != 0:
                add = 1 - corr_matrix_prob[cam_id][frame_list[frame_index][3]] / count_frame[index]
            else :
                add = 1
            #print(add,count_frame[index])
            #input()
            frame_list[frame_index][1] *= add
            if frame_list[frame_index][1] < 0.93:
                frame_list[frame_index][0] = 1
    '''
    ii = 0
    tot_count = 0
    while ii < 500:
        cur_index = index + ii + 1
        if cur_index >= len(frame_list):
            break
        if frame_list[cur_index][3] != cam_id:
            if frame_list[cur_index][3] in corr_matrix[cam_id]:
                if frame_list[cur_index][2] > frame_id + start_times[cam_id][frame_list[cur_index][3]] and frame_list[cur_index][2] <= frame_id + end_times[cam_id][frame_list[cur_index][3]]:
                    tot_count += 1
        ii += 1
    print(count_f,tot_count)
    input()
    del_count = 0
    add = 0
    if tot_count == 0:
        add = 1
    else :
        add = 1 - (1/tot_count)
    i = 0
    while i < 500:
        cur_index = index + i + 1
        if cur_index >= len(frame_list):
            break
        if frame_list[cur_index][3] == cam_id:
            if (frame_list[cur_index][2] - frame_id) % 120 == 0:
                frame_list[cur_index][0] = 1
                del_count += 1
                index -= 1
        else:
            if frame_list[cur_index][3] in corr_matrix[cam_id]:
                if frame_list[cur_index][2] > frame_id + start_times[cam_id][frame_list[cur_index][3]] and frame_list[cur_index][2] <= frame_id + end_times[cam_id][frame_list[cur_index][3]]:
                    frame_list[cur_index][1] *= add
                    if frame_list[cur_index][1] < 0.93:
                        frame_list[cur_index][0] = 1
                        del_count += 1
                        index -= 1
        i += 1
    print("Del Count ",del_count)
    '''
def before_search(sorted_frame_list_all):
    ad_frame = {}
    sample_frame = {}
    count_frame = {}
    all_frame_len = len(sorted_frame_list_all)
    for i in range(all_frame_len):
        count_frame[i] = 0
        cam_id = sorted_frame_list_all[i][3]
        frame_id = sorted_frame_list_all[i][2]
        for j in range(500):
            cur_index = i + j + 1
            if cur_index >= all_frame_len:
                break
            if sorted_frame_list_all[cur_index][3] == cam_id:
                if (sorted_frame_list_all[cur_index][2] - frame_id) % 120 == 0:
                    if sample_frame.get(i) == None:
                        sample_frame[i] = []
                    sample_frame[i].append(cur_index)

            else:
                if sorted_frame_list_all[cur_index][3] in corr_matrix[cam_id]:
                    if sorted_frame_list_all[cur_index][2] > frame_id + start_times[cam_id][sorted_frame_list_all[cur_index][3]] and sorted_frame_list_all[cur_index][2] <= frame_id + end_times[cam_id][sorted_frame_list_all[cur_index][3]]:
                        if ad_frame.get(i) == None:
                            ad_frame[i] = []
                        ad_frame[i].append(cur_index)
                        count_frame[i] += corr_matrix_prob[cam_id][sorted_frame_list_all[cur_index][3]]
                        #if sorted_frame_list_all[cur_index] all in sorted_frame_list_all[i]_person :



        #if count_frame[i] == 0:
        #    count_frame[i] = 1
        #else:
        #    count_frame[i] = 1 - (1.0/count_frame[i])

    return ad_frame,sample_frame,count_frame
def retry(new_sorted_frame_list_all,query_person):
    tot_count = 0
    temp_frame_list = copy.deepcopy(new_sorted_frame_list_all)
    Find = 0
    cur_index = 0
    while not Find :
        if cur_index >= len(temp_frame_list):
            print("Still Fail!")
            return (False,tot_count)
        for z, term_frame in enumerate(temp_frame_list[cur_index]):
            if z == 0 or z == 1 or z == 2:
                continue
            if term_frame == query_person :              
                Find = 1
                tot_count += 1
                break
            else :
                tot_count += 1
            if Find == 1:
                break
            retry_update(cur_index,temp_frame_list[cur_index][2],temp_frame_list[cur_index][1],temp_frame_list)
        if Find == 1:
            break
        cur_index += 1

    if Find == 1:
        print("Retry Find !")
        return (True,tot_count)
def Do_not_check_version2(ind,frame_id,checked_set):
    included = False
    count = 0
    for i in corr_matrix[ind]:
        if i == ind:
            for j in range(121):
                if checked_set[ind,frame_id - j] != 0:
                    included = True
                    count = checked_set[ind,frame_id - j]



    if included and count <= 3:
        checked_set[ind,frame_id] = count + 1
        return True
    else :
        return False
def version_false(paths,sorted_frame_list,max_frame):
    tot_count = 0
    fail_count = 0
    tot_people = len(list(paths.keys()))
    pn = 0
    result_package = {}
    for key_path in list(paths.keys()):
        pn += 1 
        checked_set = np.zeros((8,max_frame + 1))#{0:np.zeros((max_frame+1)),1:np.zeros((max_frame+1)),2:np.zeros((max_frame+1)),3:np.zeros((max_frame+1)),4:np.zeros((max_frame+1)),5:np.zeros((max_frame+1)),6:np.zeros((max_frame+1)),7:np.zeros((max_frame+1))}
        print("Cur query people : ", pn, "/ ", tot_people)
        query_person = int(key_path)
        Find = 0
        cur_f_end = 120
        cur_f_start = 0
        while not Find :
            if cur_f_end > max_frame: 
                fail_count += 1
                break

            for ind in range(8):
                for frame in sorted_frame_list[ind]:
                    if frame[0] >= cur_f_start and frame[0] < cur_f_end and not Do_not_check_version2(ind,frame[0],checked_set):
                        for z in range(len(frame)):
                            if z == 0:
                                continue
                            if frame[z] == query_person:
                                Find = 1
                                result_package[query_person] = frame[0]
                                tot_count += 1
                                #print("Find ~~")
                                #print(result_package)
                                #input()
                                break
                            else :
                                tot_count += 1
                    if Find == 1:
                        break
                update_version2(ind,frame[0],checked_set)
                if Find == 1:
                    break

            cur_f_end += 120
            cur_f_start += 120

    print("Tot _ count : ",tot_count)
    print("Fail _ count : ",fail_count)
    #print(result_package)
def version2(paths,sorted_frame_list_all,max_frame,fitt):
    tot_count = 0
    fail_count = 0
    tot_people = len(list(paths.keys()))
    pn = 0
    result_package = {}
    #temp_frame_list = []
    #sorted_frame_list_all
    new_sorted_frame_list_all = []

    for term in sorted_frame_list_all:
        temp = copy.deepcopy(term)
        temp.insert(0,1.0)
        temp.insert(0,0)
        new_sorted_frame_list_all.append(temp)

    ad_frame,sample_frame,count_frame = before_search(new_sorted_frame_list_all)
    temp_frame_list = new_sorted_frame_list_all
    #len_temp_frame_list = len(temp_frame_list)
    for key_path in list(paths.keys()):
        #start0 = time.time()
        #start1 = time.time()
        #temp_frame_list = copy.deepcopy(new_sorted_frame_list_all)
        init(temp_frame_list)
        pn += 1 
        #checked_set = np.zeros((8,max_frame + 1))#{0:np.zeros((max_frame+1)),1:np.zeros((max_frame+1)),2:np.zeros((max_frame+1)),3:np.zeros((max_frame+1)),4:np.zeros((max_frame+1)),5:np.zeros((max_frame+1)),6:np.zeros((max_frame+1)),7:np.zeros((max_frame+1))}
        print("Cur query people : ", pn, "/ ", tot_people)
        query_person = int(key_path)
        Find = 0
        cur_index = 0
        find_count = 0
        find_time = 0
        update_time = 0
        while not Find :
            find_count += 1
            #print(cur_index)
            #start2 = time.time()
            r2 = binom.rvs(1,fitt)
            if r2  == 1:
                cur_index += 1
                continue
            if cur_index >= len(temp_frame_list):
                fail_count += 1 
                break
            if temp_frame_list[cur_index][0] == 1:
                cur_index += 1
                continue

                '''
                print("Begin retry")
                can_find, retry_count = retry(new_sorted_frame_list_all,query_person)
                if can_find:
                    tot_count += retry_count
                    break
                else:
                    tot_count += retry_count
                    fail_count += 1
                    break'''
            #for z in temp_frame_list[cur_index]:
            for z, term_frame in enumerate(temp_frame_list[cur_index]):
                if z == 0 or z == 1 or z == 2 or z == 3:
                    continue
                if term_frame == query_person :              
                    Find = 1
                    #result_package[query_person] = temp_frame_list[cur_index][1]
                    tot_count += 1
                    break
                else :
                    tot_count += 1

                #if Find == 1:
                #    break
                #print(temp_frame_list[cur_index][0],temp_frame_list[cur_index][1],temp_frame_list[cur_index][2])
                #start3 = time.time()
                #update_version2(cur_index,temp_frame_list[cur_index][3],temp_frame_list[cur_index][2],temp_frame_list,ad_frame,sample_frame,count_frame)
                #end3 = time.time()
                #update_time += (end3 - start3)
                #print("Update time : ",end3 - start3)
            #if Find == 1:
            #    break
            cur_index += 1
            #end2 = time.time()
            #find_time += (end2 - start2)
            #print("Pre iteration time : ",end2 - start2)
            #input()
        #print("Find count : ",find_count)
        #print("Find time : ",find_time)
        #print("Update time :",update_time)
        #end0 = time.time()
        #print("Total time : ",end0 - start0)
        #print("In update time1 : ",in_update_time1)
        #print("In update time2 : ",in_update_time2)
        #print("In update time3 : ",in_update_time3)
        #print("In update time4 : ",in_update_time4)
        #input()
    print("Tot _ count : ",tot_count)
    print("Fail _ count : ",fail_count)
    #print(result_package)
def update_rule(cur_prob,index_set):
    #include_set = []
    #for i in range(10):
    #    if i not in index_set:
    #        include_set.append(i)
    #print("Include set :",include_set)
    tot_prob = 0.0
    tmp_prob = np.zeros(9)
    for t in index_set:
        cur_prob[8] += cur_prob[t]
        cur_prob[t] = 0
    #print("Before update : ",cur_prob)

    #print("--------------------------------------------")
    #print("cur prob : ",cur_prob)
    #print("include_set : ",include_set)
    #print("_+_________________________________________+_")
    new_corr_prob = np.zeros((9,9,20))
    cur_prob = np.zeros((9,20))


    for i in range(9):
        tmp_prob[i] = 0.0
        tmp2 = 0.0
        for index in range(9):
            for t_j in range(20):
                tmp_prob[i] += new_corr_prob[index][i][t_j]*cur_prob[index][t_j]
            #print(tmp_prob[i])

    cur_prob = tmp_prob.copy()
    #print("Scan set : ",index_set)
    #print("prob : ",cur_prob)
    #input()
    return cur_prob
    #cur_prob = cur_prob/tot_prob
    #print("Cur_prob " , cur_prob)
    #input()
def update_rule_temporal(cur_prob,index_set):

    new_cur_prob = np.zeros((9,30))
    #cur_prob[:,0:-1] = 0
    for t in index_set:
        cur_prob[t,-1] = 0
    cur_prob[8,-1] = 0
    cur_prob[8,-1] = 1 - np.sum(cur_prob[:,-1])

    #if cur_prob[8,-1] < 0 :
    #    print("May exist error !")
    #    input()

    new_cur_prob[:,0:-1] = cur_prob[:,1:].copy() #shift 
    
    #print("Before Update : ",np.sum(cur_prob[:,:]))

    #for i in range(9):
    #    for j in range(9):
    #        for z in range(30):
    #            new_cur_prob[j,-1] += Temporal_corr_prob[i][j][z] * cur_prob[i,z]
    #a = 0
    #for i in range(9):
    #    for z in range(30):
    #        np.sum(Temporal_corr_prob[i,:,z]) * cur_prob[i,z]

    new_cur_prob[:,-1] = np.sum(np.sum(Temporal_corr_prob.transpose((1,0,2)) * cur_prob,axis = -1),axis = -1)
    #print("Index set : ",index_set)
    #print("Cur prob : ",new_cur_prob[:,-1])
    #input()
    #print("Sum : ",np.sum(new_cur_prob[:,:]))
    #input()
    return new_cur_prob
def update_rule_block(cur_prob,index_set):

    new_cur_prob = np.zeros((9,5))
    for t in index_set:
        cur_prob[t,-1] = 0
    #cur_prob[8,-1] = 0
    #cur_prob[8,-1] = 1 - np.sum(cur_prob[:,-1])
    cur_prob[8,-1] = 1
    new_cur_prob[:,0:-1] = cur_prob[:,1:].copy() #shift 
    new_cur_prob[:,-1] = np.sum(np.sum(Temporal_corr_prob.transpose((1,0,2)) * cur_prob,axis = -1),axis = -1)
    new_cur_prob[:,-1] = np.clip(new_cur_prob[:,-1], 0,1)
    return new_cur_prob

def update_frame_window(frame_window_start,frame_window_end):
    return frame_window_start + frame_window_size, frame_window_end + frame_window_size
def update_version3(index,cam_id,frame_id,frame_list,ad_frame,sample_frame,count_frame,person_count):
    count = 0
    cur_frame_id = frame_id
    add = count_frame[index] 
    if sample_frame.get(index) != None :
        for frame_index in sample_frame[index]:
            frame_list[frame_index][0] = 1

    if ad_frame.get(index) != None:
        for frame_index in ad_frame[index]:
            add = 1

            if count_frame[index] != 0:
                add = 1 - corr_matrix_prob[cam_id][frame_list[frame_index][3]] / count_frame[index]
            else :
                add = 1
            frame_list[frame_index][1] *= add
            if frame_list[frame_index][1] < 0.93:
                frame_list[frame_index][0] = 1
_turn = 0
gallery_base = []
frame_base = []

def select_and_search_camera(query_person,cur_prob,frame_window_start,frame_window_end,temp_frame_list,cur_index):
    global _turn,has_been_scanned
    global gallery_base,frame_base
    _turn += 1
    search_count = 0
    cam_index = []
    if cur_index >= len(temp_frame_list):
        return 0,cam_index,search_count,cur_index
    th_value = np.max(cur_prob[0:8,-1])
    no_image_frame = np.zeros(8)
    scan_image = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]}


    while temp_frame_list[cur_index][2] <= frame_window_end:

        no_image_frame[temp_frame_list[cur_index][3]] = 1

        if cur_prob[temp_frame_list[cur_index][3],-1] >= threshold_value[temp_frame_list[cur_index][3]]:# or (temp_frame_list[cur_index][3] in [7] and _turn >= 3):
            

            #print(th_value)
            #print(cur_prob[:,-1])
            check = 0
            for scan_term in scan_image[temp_frame_list[cur_index][3]]:
                if (temp_frame_list[cur_index][2] - scan_term)%120 == 0:
                    check = 1
                    break
            if check == 1:
                cur_index += 1
                continue
            scan_image[temp_frame_list[cur_index][3]].append(temp_frame_list[cur_index][2])

            has_been_scanned[temp_frame_list[cur_index][2]] = 1

            for z, term_frame in enumerate(temp_frame_list[cur_index]):
                if z == 0 or z == 1 or z == 2 or z == 3:
                    continue
                if term_frame in gallery_base:
                    kkk = 1

                if term_frame == query_person :              
                    return 1,[],search_count,0
                else:
                    gallery_base.append(term_frame)
                    frame_base.append(cur_index)

            search_count += 1
            #if temp_frame_list[cur_index][3] in [7] and _turn >=3:
            #    _turn = 0
            #else:
            cam_index.append(temp_frame_list[cur_index][3])
        cur_index += 1
        if cur_index >= len(temp_frame_list):
            return 0,cam_index,search_count,cur_index
        #input()

    #print("Before cam _ index : ",cam_index)
    for i in range(8):
        if no_image_frame[i] == 0 and cur_prob[i,-1] >= threshold_value[temp_frame_list[cur_index][3]]:
            cam_index.append(i)

    #print("Cur prob : ",cur_prob[:,-1])
    #print("Cur frame window end : ",frame_window_end)
    #print("Cur frame : ",temp_frame_list[cur_index][2])
    #print("After cam _ index : ",cam_index)
    #input()

    return 0,cam_index,search_count,cur_index


def navie_select_and_search_camera(query_person,cur_prob,frame_window_start,frame_window_end,temp_frame_list,cur_index):
    search_count = 0
    cam_index = []
    if cur_index >= len(temp_frame_list):
        return 0,cam_index,search_count,cur_index
    th_value = np.max(cur_prob[0:8,-1])
    no_image_frame = np.zeros(8)
    scan_image = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]}


    while temp_frame_list[cur_index][2] <= frame_window_end:

        check = 0
        for scan_term in scan_image[temp_frame_list[cur_index][3]]:
            if (temp_frame_list[cur_index][2] - scan_term)%120 == 0:
                check = 1
                break
        if check == 1:
            cur_index += 1
            continue
        scan_image[temp_frame_list[cur_index][3]].append(temp_frame_list[cur_index][2])

        for z, term_frame in enumerate(temp_frame_list[cur_index]):
            if z == 0 or z == 1 or z == 2 or z == 3:
                continue
            if term_frame == query_person :              
                return 1,[],search_count,0
        search_count += 1

        cur_index += 1
        if cur_index >= len(temp_frame_list):
            return 0,cam_index,search_count,cur_index

    return 0,cam_index,search_count,cur_index



def Random_select_and_search_camera(query_person,cur_prob,frame_window_start,frame_window_end,temp_frame_list,cur_index):
    global _turn
    global gallery_base,frame_base
    _turn += 1
    search_count = 0
    cam_index = []
    if cur_index >= len(temp_frame_list):
        return 0,cam_index,search_count,cur_index
    th_value = np.max(cur_prob[0:8,-1])
    no_image_frame = np.zeros(8)
    scan_image = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]}
    while temp_frame_list[cur_index][2] <= frame_window_end:

        no_image_frame[temp_frame_list[cur_index][3]] = 1
        p1 = binom.rvs(1,1 - Random_Prob)
        if p1:
            for z, term_frame in enumerate(temp_frame_list[cur_index]):
                if z == 0 or z == 1 or z == 2 or z == 3:
                    continue

                if term_frame == query_person :              
                    return 1,[],search_count,0

            search_count += 1
            cam_index.append(temp_frame_list[cur_index][3])
        cur_index += 1
        if cur_index >= len(temp_frame_list):
            return 0,cam_index,search_count,cur_index


    return 0,cam_index,search_count,cur_index
play_count = 10
def Research_play(query_person,temp_frame_list):
    global has_been_scanned
    skip_count = 0
    scan_count = 0
    for term in temp_frame_list:
        if has_been_scanned[term[2]]:
            continue
        skip_count += 1
        if skip_count == play_count:
            for z, term_frame in enumerate(term):
                if z == 0 or z == 1 or z == 2 or z == 3:
                    continue

                if term_frame == query_person :              
                    return True,scan_count

            scan_count += 1
            skip_count = 0
    return False,scan_count


paths = 0
sorted_frame_list_all = 0
max_frame = 0
def version3():
    global paths,sorted_frame_list_all,max_frame
    tot_count = 0
    fail_count = 0
    tot_people = len(list(paths.keys()))
    pn = 0
    result_package = {}
    #temp_frame_list = []
    #sorted_frame_list_all
    new_sorted_frame_list_all = []
    #global Temporal_corr_prob
    #Temporal_corr_prob[:,:,-1] = Spatial_corr_prob
    for term in sorted_frame_list_all:
        temp = copy.deepcopy(term)
        temp.insert(0,1.0)
        temp.insert(0,0)
        new_sorted_frame_list_all.append(temp)

    ad_frame,sample_frame,count_frame = before_search(new_sorted_frame_list_all)
    temp_frame_list = new_sorted_frame_list_all
    len_temp_frame_list = len(temp_frame_list)
    #print("Frame Len : ",len_temp_frame_list)
    cur_frame = temp_frame_list[0][2]
    #{0:0.1,1:0.1,2:0.1,3:0.1,4:0.1,5:0.1,6:0.1,7:0.1}
    #init_prob = np.zeros(8)#{0:0.1,1:0.1,2:0.1,3:0.1,4:0.1,5:0.1,6:0.1,7:0.1}
    

    for key_path in list(paths.keys()):
        global gallery_base,frame_base
        global has_been_scanned
        has_been_scanned = np.zeros(227544)
        gallery_base = []
        frame_base = []
        init(temp_frame_list)
        pn += 1 
        cur_prob = np.zeros((9,5))
        cur_prob[:,-1] = Temporal_corr_prob[8,:,-1].copy()    
        #print(cur_prob)
        #input()
        #print("Cur query people : ", pn, "/ ", tot_people)
        query_person = int(key_path)
        frame_window_start = 49700
        frame_window_end = frame_window_start + frame_window_size
        search_count = 0
        cur_index = 0
        Find = 0
        while not Find :
            #Find,cam_index,tmp_search_count,cur_index = select_and_search_camera(query_person,cur_prob,frame_window_start,frame_window_end,temp_frame_list,cur_index)
            
            Find,cam_index,tmp_search_count,cur_index = Random_select_and_search_camera(query_person,cur_prob,frame_window_start,frame_window_end,temp_frame_list,cur_index)
            #Find,cam_index,tmp_search_count,cur_index = navie_select_and_search_camera(query_person,cur_prob,frame_window_start,frame_window_end,temp_frame_list,cur_index)
            #print(cam_index)
            if Find:
                break

            
            if cur_index >= len(temp_frame_list):
                fail_count += 1
                break

            '''if cur_index >= len(temp_frame_list):
                anwser,reply_count = Research_play(query_person,temp_frame_list)
                if not anwser:
                    fail_count += 1
                else:
                    tot_count += reply_count
                break'''
            tot_count += tmp_search_count
            cur_prob = update_rule_block(cur_prob,cam_index)
            frame_window_start, frame_window_end = update_frame_window(frame_window_start,frame_window_end) 
            #print(frame_window_start,frame_window_end)



        
    #print("Tot _ count : ",tot_count)
    #print("Fail _ count : ",fail_count)
    return tot_count, fail_count
    #print(result_package)
def wrapper(j):#t0,t1,t2,t3,t4,t5,t6,t7):
    #global threshold_value,play_count
    #play_count = k
    #th = np.array([j,j,j,j,j,j,j,j])
    #threshold_value = th
    global Random_Prob
    Random_Prob = j
    t,f = version3()
    print(t,f)
    t = (5327723 - t)/5327723
    f = f / 702
    #t = (5171085 - t)/5171085
    #f = f / 702
    #print(t,f)
    return  t,f#- ((t - 400000)*(t - 400000)/(5171085*5171085) + (f - 10)*(f - 10)/(702 * 702))
def main():
    global Temporal_corr_prob
    global paths,sorted_frame_list_all,max_frame
    
    Temporal_corr_prob = np.zeros((9,9,5))

    Temporal_corr_prob = np.load('./t_5_corr.npy')
    Temporal_corr_prob[8,:,-1] = [0.51415094, 0.20754717,0.07075472,0.25,0.08962264,0.45283019,0.0754717,0.49528302,1]
    #Temporal_corr_prob[8,:,-1] = [0.96295517, 0.96295517, 0.64197011, 0.96295517, 0.77036413, 0.96295517,0.64197011, 0.96295517, 1]
    #800 Temporal_corr_prob[8,:,-1] = [0.10991234, 0.03641268, 0.01213756, 0.04517869, 0.01416049, 0.09844909,0.01213756, 0.10249494,1]
    #480 Temporal_corr_prob[8,:,-1] = [0.34993271 , 0.12651413 , 0.04306864,  0.16689098 , 0.05114401,  0.30417227, 0.04306864, 0.32570659,1]
    #720 Temporal_corr_prob[8,:,-1] = [0.45967742, 0.1733871,  0.06451613, 0.22983871, 0.07258065, 0.40725806, 0.06451613, 0.43951613,1]
    #600 Temporal_corr_prob[8,:,-1] = [0.4169468,0.15467384,0.0537996,0.19166106,0.0672495,0.35642233,0.05716207,0.39340955,1]
    #1080 Temporal_corr_prob[8,:,-1] = [0.55533199, 0.23541247, 0.09054326, 0.28973843, 0.11468813, 0.52515091,0.09657948, 0.58551308,1]
    #Temporal_corr_prob[8,:,-1] = [0.49708912, 0.17913121, 0.06717421, 0.2507837,  0.08060905, 0.43439319, 0.07165249, 0.46574116, 1]
    #Temporal_corr_prob[8,2,29] += 0.1
    #Temporal_corr_prob[8,8,29] -= 0.1
    #Temporal_corr_prob[:,:,0:-1] = 0
    #Temporal_corr_prob[:,:,-1] = Spatial_corr_prob
    #Temporal_corr_prob[8,:,-1] = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    dataset = data_manager.init_img_dataset(
        root=args.root, name=args.dataset, split_id=args.split_id,
        cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
    )

    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory =  False

    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        batch_size=args.train_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    #gallery = ImageDatasetLazy(dataset.gallery, transform=transform_test)
    galleryloader = DataLoader( ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    paths = dict()
    life_frequency = []
    stand_frequency = { 0 : [], 1:[],2:[],3:[],4:[],5:[],6:[],7:[]}

    frame_list = { 0 : dict(), 1:dict(),2:dict(),3:dict(),4:dict(),5:dict(),6:dict(),7:dict()}

    change = np.zeros((9,9))
    cal_v = np.zeros(9)
    total_number = 0
    min_frame = 10000000
    max_frame = -1
    with torch.no_grad():
        for batch_idx, (names, imgs, pids, camids, fids) in enumerate(trainloader):
            for s_index in range(len(names)):
                if paths.get(int(pids[s_index])) == None:
                    paths[int(pids[s_index])] = []

                fid_ = fids[s_index] + cam_offsets[camids[s_index]]
                #if paths[int(pids[s_index])] != []:
                #    if paths
                paths[int(pids[s_index])].append((fid_ ,camids[s_index]))

                if int(fid_) >= max_frame:
                    max_frame = int(fid_)
                if int(fid_) <= min_frame:
                    min_frame = int(fid_)

                if frame_list[int(camids[s_index])].get(int(fid_)) == None:
                    frame_list[int(camids[s_index])][int(fid_)] = []

                frame_list[int(camids[s_index])][int(fid_)].append(int(pids[s_index]))


                cal_v[camids[s_index]] += 1
                total_number += 1
    print("Max frame and min frame : ", max_frame, min_frame)
    print("")
    sorted_frame_list_all = []#{ 0 : list(), 1:list(),2:list(),3:list(),4:list(),5:list(),6:list(),7:list()}
    for ind in range(8):
        for key in list(frame_list[ind].keys()):
            tmp_list = frame_list[ind][key]
            tmp_list.insert(0,ind)
            tmp_list.insert(0,key)

            #sorted_frame_list[ind].append(tmp_list)
            sorted_frame_list_all.append(tmp_list)
        #sorted_frame_list[ind] = sorted(sorted_frame_list[ind], key=lambda x: x[0])
        #print("Sorted Index : ",ind, "With term : ",len(sorted_frame_list[ind]))
    sorted_frame_list_all = sorted(sorted_frame_list_all, key=lambda x: x[0])
    #input()
    print("Sorted Index : ",ind, "With term : ",len(sorted_frame_list_all))
    #227540 49700



    #version1(paths,sorted_frame_list)
    #baseline(paths,sorted_frame_list)

    '''global frame_window_size
    global threshold_value
    a = np.zeros(100)
    b = np.zeros(100)
    for i in range(100):
        a[i] = 10*i + 10
        b[i] = pow(0.42,i)

    result = []
    for i in range(100):
        for j in range(100):
            frame_window_size = a[i]
            threshold_value = b[j]
            t,f = version3(paths,sorted_frame_list_all,max_frame)
            print("Time : ",i,j,"with tot : ",t, " and f : ",f, " setting : ",frame_window_size,threshold_value)

            if f < 50 and t < 600000:
                result.append((t,f))
    print(result)'''
    from bayes_opt import BayesianOptimization

    # Bounded region of parameter space
    
    #pbounds = {'t0':(0.6,1.5),'t1':(0.6,1.5),'t2':(0.2,0.8),'t3':(0.6,1.2),'t4':(0.2,0.8),'t5':(0.5,1.2),'t6':(0.2,0.8),'t7':(0.3,1.2),}
    

    #wrapper = 

    for i in [0.96,0.965,0.97,0.975,0.98,0.99]:#1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.8,3,4,5,6,10,100,1e5,1e6]:#0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.735,0.80,0.85,0.9,
        #for k in [7,13,17,20,23,27,31,71,103,143]:
            j = i
            t,f = wrapper(j)
            print(j,",",t,",",f)

        #version2(paths,sorted_frame_list_all,max_frame,j)

    '''optimizer = BayesianOptimization(
        f=wrapper,
        pbounds=pbounds,
        random_state=1,
    )
    optimizer.maximize(
    init_points = 100,
    n_iter=500,
    )
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print(optimizer.max)'''
    #print(t,f)

if __name__ == '__main__':
    main()
