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
import data_manager
from dataset_loader import ImageDataset, ImageDatasetLazy,ImageDataset_forBeijing
import transforms as T
import models
from losses import CrossEntropyLabelSmooth, DeepSupervision
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate
from optimizers import init_optim
import enum
class CameraCheck(enum.Enum):
    primary = 1
    skipped = 2
    all = 3

parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
# Datasets
parser.add_argument('--root', type=str, default='data', help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='beijing',
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
parser.add_argument('--evaluate', action='store_true',default = True, help="evaluation only")
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
def load_corr_and_hp(m):
    ts = np.load('corr_matrix_porto.npy')
    corr_matrix = list(ts)
    start_times = list(np.load('start_times_porto.npy'))
    end_times = np.load('end_times_porto.npy')
    for i in range(end_times.shape[0]):
        start_times[i][i] = 0
        end_times[i][i] = 120
    print(end_times.shape)
    fallback_times = np.zeros(ts.shape[0])#list(np.load('fallback_times.npy'))
    for i in range(ts.shape[0]):
        fallback_times[i] = np.max(end_times[i,:])
    print(fallback_times)
    exit_times = fallback_times.copy()
    cam_offsets = np.zeros(ts.shape[0])
    f_rate = 60
    return corr_matrix,start_times,end_times,fallback_times,exit_times,cam_offsets,f_rate
def main(check_model,mm = 1):

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
    '''trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        batch_size=args.train_batch, shuffle=True, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )'''
    query = ImageDataset(dataset.query, transform=transform_test)
    if args.dataset == 'beijing':
        query = ImageDataset_forBeijing(dataset.query, transform=transform_test)
   
    #gallery = ImageDatasetLazy(dataset.gallery, transform=transform_test)
    gallery = ImageDataset(dataset.gallery, transform=transform_test)
    if args.dataset == 'beijing':
        gallery = ImageDataset_forBeijing(dataset.gallery, transform=transform_test)
    

    if args.evaluate:
        #print("Evaluate only")
        if mm == 1:
            cost,recall,precision = test(query, gallery,check_model,mm)
            return cost,recall,precision
        else:
            cost,recall,precision,delay = test(query, gallery,check_model,mm)
            return cost,recall,precision,delay
wid_rate = 0.25
def check_exit_retry(f_rate, camid, s_lower_b, s_upper_b, fallback_times, exit_times, cam_check):
    to_exit = False
    check_next = True
    if cam_check == CameraCheck.primary:
        if s_upper_b >= fallback_times[camid]:
            #print("now checking OTHER cameras!")
            cam_check = CameraCheck.skipped
            s_lower_b = 0.
            s_upper_b = f_rate * wid_rate#0.1#0.25#2.
            check_next = False
    elif cam_check == CameraCheck.skipped:
        if s_upper_b >= exit_times[camid]:
            #print("could not find person, giving up!")
            to_exit = True
            check_next = False
        elif s_upper_b >= fallback_times[camid]:
            #print("now checking ALL cameras!")
            cam_check = CameraCheck.all
    elif cam_check == CameraCheck.all:
        if s_upper_b >= exit_times[camid]:
            #print("could not find person, giving up!")
            to_exit = True
            check_next = False

    if check_next:
        s_lower_b = s_upper_b
        s_upper_b += (f_rate * wid_rate)#0.1)#0.25)#2.0)

    return to_exit, s_lower_b, s_upper_b, cam_check

    
def test(queryl, gallery, check_model, m = 1,ranks=[1, 5, 10, 20]):

    batch_time = AverageMeter()
    #f_rate = 2.5
    #dist_thresh = 160.
    #cam_offsets = [5542, 3606, 27243, 31181, 0, 22401, 18967, 46765]
    #cam_offsets = [0,0,0,0,0,0,0,0]
    #corr_matrix,start_times,end_times,fallback_times,exit_times,cam_offsets,f_rate = load_corr_and_hp()
    #f_rate = 60.
    f_rate = 60.
    #wid_rate = 0.25
    cam_offsets = [5542, 3606, 27243, 31181, 0, 22401, 18967, 46765]
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
    corr_matrix = [
        [0,1,7],
        [0,1,2,4,7],
        [1,2,3,4],
        [2,3,5],
        [1,2,4,5,6,7],
        [4,5,6],
        [4,5,6,7],
        [0,1,4,6,7]

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
    end_times = [[f_rate * x for x in y] for y in end_times]
    #print("end times", end_times)

    fallback_times = [
        80,
        45,
        40,
        30,
        55,
        30,
        55,
       150
    ]
    exit_times = [
        80,
        45,
        40,
        30,
        65,
        30,
        65,
       150
    ]
    fallback_times = [x * f_rate for x in fallback_times]
    exit_times = [x * f_rate for x in exit_times]
    #print('fallback_times', fallback_times)
    #print('exit_times', exit_times)
    corr_matrix,start_times,end_times, fallback_times,exit_times,cam_offsets,f_rate = load_corr_and_hp(m)
    dist_thresh = 160.
    #print('fallback_times', fallback_times)
    #print('exit_times', exit_times)
    # process query images
    qf, q_pids, q_camids, q_fids, q_names = [], [], [], [], []
    for object_ in queryl:
        name, img, pid, camid, fid = object_
        fid += cam_offsets[camid]
        q_pids.append(pid)
        q_camids.append(camid)
        q_names.append(name)
        q_fids.append(fid)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    q_fids = np.asarray(q_fids)
    q_names = np.asarray(q_names)
        #print("query imgs", q_names)
        #print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
    # process gallery images
    dtype = [('name',int),('img',int),('index',int), ('cam_id', int),('frame', int)]
    #dtype = [('name',str),('index',int), ('cam_id', int),('frame', int)]
    gallery = np.array(gallery,dtype = dtype)
    for i in range(gallery.shape[0]):
        gallery[i][-1] += cam_offsets[gallery[i][-2]]
    gallery = np.sort(gallery, order='frame')
    #gallery = np.array(gallery).dtype = int
    #print(gallery.shape)
    query_frame_index = []
    for i in range(gallery.shape[0]):
        query_frame_index.append(gallery[i][-1])
    query_frame_index = np.array(query_frame_index)
    #print(np.searchsorted(query_frame_index,60, side='left'))
    #input()
    all_cmc = []
    all_AP = []
    num_valid_q = 0.

    tot_img_seen = 0
    tot_img_elim = 0
    tot_num_inst = 0
    tot_match_found = 0
    tot_match_pres = 0
    tot_delay = 0.

    tot_t_pos = 0
    tot_f_pos = 0
    tot_t_neg = 0
    tot_f_neg = 0


    # execute queries
    for q_idx, (q_pid, q_camid, q_fid, q_name) in enumerate(list(zip(q_pids, q_camids, q_fids, q_names))[:100]):

        #print("\nnew query person ------------------------------------ ")
        #print("query id: ", q_idx, "pid: ", q_pid, "camid: ", q_camid,
        #    "frameid: ", q_fid, "name: ", q_name)

        # query vars
        q_iter = 0
        s_lower_b = 0.
        s_upper_b = f_rate * wid_rate#0.1#0.25#2.

        cam_check = CameraCheck.all
        if check_model == 1:
            cam_check = CameraCheck.primary

        # query features
        #qf_orig = qf[q_idx].unsqueeze(0)
        #qf_i = qf_orig

        # query stats
        q_img_seen = 0
        q_img_elim = 0
        q_match_found = 0
        q_match_pres = 0
        q_delay = 0.

        q_img_seen_arr = []
        q_img_elim_arr = []
        q_delay_arr = []

        t_pos = 0.
        f_pos = 0.
        t_neg = 0.
        f_neg = 0.

        num_inst = 0.
        # count total num. of pos. examples
        his_gallery = []
        for idx in range(0, len(gallery)):
            img_name,img,pid, camid, fid = gallery[idx]
            #fid += cam_offsets[camid]
            if pid == q_pid and fid > q_fid:
                num_inst += 1
                his_gallery.append((img_name, img, pid, camid, fid))
        find_people_times = 0
        probability1 =  0.7602589205132292#0.812#0.6878 #0.6935#   |  0.04795#0.5501#   |  0.02561#0.6878#   |  0.021590.5501#0.575#0.591#0.7374
        probability2 =  0.06395328874237172#0.0432159#0.02532#0.02561#0.021590#0.787#0.515#0.1123
        while q_iter >= 0:
            #print("\nquery: (", q_idx, ",", q_iter, ")","pid: ", q_pid, "camid: ", q_camid, "frameid: ", q_fid,
            #    "\twin: [", s_lower_b / f_rate, ",", s_upper_b / f_rate, "]")
            #print("search mode: ", cam_check)

            img_elim = 0
            ture_person = 0
            ture_camid = -1
            ture_name = 0
            ture_fid = 0
            min_dist = 1e10


            gf, g_pids, g_camids, g_fids, g_names = [], [], [], [], []
            g_a_pids, g_a_camids = [], []

            # load gallery

            low_bound = q_fid + s_lower_b
            up_bound = q_fid + s_upper_b
            l_index = np.searchsorted(query_frame_index,low_bound,side = 'right')
            u_index = np.searchsorted(query_frame_index,up_bound, side = 'left')
            if u_index + 1 < gallery.shape[0]:
                u_index += 1
            idx_list = list(np.arange(l_index,u_index))
            ture_find_it = 0
            #print("Gallery len : ",len(idx_list))
            for idx in idx_list:
                img_name,img,pid, camid, fid = gallery[idx]
                # adjust frame id
                #fid += cam_offsets[camid]

                if fid > (q_fid + s_lower_b) and fid <= (q_fid + s_upper_b):
                    check_frame = False
                    included = fid <= (q_fid + end_times[q_camid][camid]) and \
                        fid >= (q_fid + start_times[q_camid][camid])

                    if cam_check == CameraCheck.all:
                        # baseline: check all
                        if True:
                            check_frame = True
                        else:
                            img_elim += 1
                    elif cam_check == CameraCheck.skipped:
                        # special case: hist. search on skipped cameras
                        if camid not in corr_matrix[q_camid]:
                            if included:
                                check_frame = True
                                img_elim -= 1
                    elif cam_check == CameraCheck.primary:
                        # pruned search
                        if camid in corr_matrix[q_camid] and included:
                            check_frame = True
                        else:
                            img_elim += 1

                    if check_frame:
                        g_names.append(img_name)
                        g_pids.append(pid)
                        g_camids.append(camid)
                        if pid == q_pids[q_idx] and fid - q_fid < min_dist:
                            ture_person = pid
                            ture_camid = camid
                            ture_fid = fid
                            ture_name = img_name
                            ture_find_it = 1
                            min_dist = fid - q_fid

                        g_fids.append(fid)

                    g_a_pids.append(pid)
                    g_a_camids.append(camid)
            # load images
            # update delay
            if len(q_delay_arr) <= q_iter:
                q_delay_arr.append(0)

            if cam_check == CameraCheck.skipped:
                q_delay += wid_rate#0.1#0.25#2.
                q_delay_arr[q_iter] += wid_rate#0.1#0.25#2.

            # handle no candidate case

            if len(g_names) == 0:
                #print("no candidates detected, skipping")

                # check exit / retry
                exit, s_lower_b, s_upper_b, cam_check = check_exit_retry(f_rate=f_rate, camid=q_camid,
                    s_lower_b=s_lower_b, s_upper_b=s_upper_b, fallback_times=fallback_times, exit_times=exit_times,
                    cam_check=cam_check)
                if exit:
                    #print("\nframes tracked: ", q_fids[q_idx], "-", q_fid)
                    break
                else:
                    continue

            # gallery features
            

            g_a_pids = np.asarray(g_a_pids)
            g_a_camids = np.asarray(g_a_camids)
            g_pids = np.asarray(g_pids)
            g_camids = np.asarray(g_camids)
            g_names = np.asarray(g_names)
            g_fids = np.asarray(g_fids)
            # gallery pruning stats
            #print("eliminated: ", img_elim)
            #print("new gallery size: ", len(g_names))
            q_img_seen += len(g_names)
            q_img_elim += img_elim
            if len(q_img_seen_arr) <= q_iter:
                q_img_seen_arr.append(0)
                q_img_elim_arr.append(0)
            q_img_seen_arr[q_iter] += len(g_names)
            q_img_elim_arr[q_iter] += img_elim

            #print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, len(g_names)))

            # compute dist matrix
            #m, n = qf_i.size(0), gf.size(0)
            #distmat = torch.pow(qf_i, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            #          torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            #distmat.addmm_(1, -2, qf_i, gf.t())
            #distmat = distmat.numpy()
            find_it = 0
            find_it_index = -1
            have_true = 0
            for i in range(len(g_names)):
                if g_pids[i] == q_pids[q_idx]:
                    have_true = 1
                    if find_it == 0:
                        p1 = binom.rvs(1,probability1)
                        if p1:
                            find_it = 1
                            find_it_index = i


            if not have_true:
                p2 = binom.rvs(1,probability2)
                if p2:
                    find_it = 1
                    find_it_index = np.random.randint(0,len(g_names))

            '''
            for i in range(len(g_names)):
                if g_pids[i] == q_pids[q_idx]:
                    p1 = binom.rvs(1,probability1)#74)
                    if p1:
                        find_it = 1
                        find_it_index = i
                        break
                    else:
                        continue
                else:
                    p2 = binom.rvs(1,probability2)
                    if p2:
                        find_it = 1
                        find_it_index = i
                    else:
                        continue'''

            #print("Computing CMC and mAP")
            #cmc, AP, valid, f, p = evaluate(distmat, np.expand_dims(q_pid, axis=0), g_pids, np.expand_dims(q_camid, axis=0), g_camids,
            #    use_metric_cuhk03=args.use_metric_cuhk03, img_names=g_names, g_a_pids=g_a_pids, g_a_camids=g_a_camids)

            #if valid == 1:
            #    all_cmc.append(cmc[0])
            #    all_AP.append(AP[0])
            #    num_valid_q += valid
            #    q_match_found += f
            #    q_match_pres  += p

            #print("mAP (so far): {:.1%}".format(np.mean(all_AP)))
            #print("img seen (so far): {}".format(q_img_seen))
            #print("img tot. (so far): {}".format(q_img_seen + q_img_elim))
            #print("matches found (so far): {}".format(q_match_found))
            #print("matches pres. (so far): {}".format(q_match_pres))
            #print("delay (so far): {}".format(q_delay))
            #print("t_pos {}, f_neg {}".format(t_pos, f_neg))
            #print("t_pos {}, f_pos {}".format(t_pos, f_pos))
            #p1 = binom.rvs(1,0.5)
            #p1 = 0
            # check for match
            #indices = np.argsort(distmat, axis=1)
            #if ture_camid == -1 or p1:
            if find_it and find_people_times >= 5 and g_pids[find_it_index] != q_pids[q_idx]:
                find_it = 0
                find_people_times = 0
                break
            if find_it == 0:
            #if p1 == 0:#distmat[0][indices[0][0]] > dist_thresh:
                #print("not close enough, waiting...", distmat[0][indices[0][0]])
                # set accuracy stats
                #p2 = binom.rvs(1,0.05601)
                #if p2:#q_pids[q_idx] in g_pids[indices][0]:
                #    f_neg += 1.
                #else:
                #    t_neg += 1.
                if ture_find_it == 1:
                    f_neg += 1
                else:
                    t_neg += 1
                # check exit / retry
                exit, s_lower_b, s_upper_b, cam_check = check_exit_retry(f_rate=f_rate, camid=q_camid,
                    s_lower_b=s_lower_b, s_upper_b=s_upper_b, fallback_times=fallback_times, exit_times=exit_times,
                    cam_check=cam_check)
                if exit:
                    #print("\nframes tracked: ", q_fids[q_idx], "-", q_fid)

                    break
                else:
                    continue

            else:
                #print("match declared:", distmat[0][indices[0][0]])
                # set accuracy stats
                #p3 = binom.rvs(1,0.7934)
                if g_pids[find_it_index] == q_pids[q_idx]:# and g_fids[find_it_index] == ture_fid:
                    #q_pids[q_idx] == g_pids[indices][0][0]:
                    q_pid = g_pids[find_it_index]
                    q_camid = g_camids[find_it_index] #g_camids[indices][0][0]
                    q_fid = g_fids[find_it_index]  #g_fids[indices][0][0]
                    q_name = ture_name  #g_names[indices][0][0]
                    find_people_times = 0
                    t_pos += 1.
                else:
                    q_pid = g_pids[find_it_index]
                    q_camid = g_camids[find_it_index]
                    q_fid = g_fids[find_it_index]
                    q_name = g_names[find_it_index]
                    probability2 *= 1
                    find_people_times += 1
                    f_pos += 1.

                # update delay
                if cam_check == CameraCheck.skipped:
                    lag = (fallback_times[q_camid] - s_upper_b) / f_rate
                    #print("Now resuming tracking, adding", lag ,"seconds delay")
                    q_delay += lag
                    q_delay_arr[q_iter] += lag

                # reset window, flag
                s_lower_b = 0.
                s_upper_b = f_rate * wid_rate#0.1#0.25#2.
                cam_check = CameraCheck.all
                if check_model == 1:
                    cam_check = CameraCheck.primary
                #cam_check = CameraCheck.primary

                # find next query img
                q_iter += 1
                #q_pid = g_pids[indices][0][0]
                #q_camid = g_camids[indices][0][0]
                #q_fid = g_fids[indices][0][0]
                #q_name = g_names[indices][0][0]
                #print("Next query (name, pid, cid, fid): ", q_name, q_pid, q_camid, q_fid)

                # extract next img features
                #ori_w = 0.5
                #run_w = 0.
                #new_w = 0.5
                #with torch.no_grad():
                    #next_path = osp.normpath("data/dukemtmc-reid/DukeMTMC-reID/bounding_box_test/" + q_name)
                    #next_img = read_image(next_path)
                    #if use_gpu: next_img = next_img.cuda()
                    #features = model(next_img.unsqueeze(0))
                    #qf_i = (ori_w * qf_orig) + (run_w * qf_i) + (new_w * features.data.cpu())
        '''
        print("\nFinal query {} stats ----------".format(q_idx))
        print("img seen: {}".format(sum(q_img_seen_arr[:-1])))
        print("img tot.: {}".format(sum(q_img_seen_arr[:-1] + q_img_elim_arr[:-1])))
        print("num inst: {}".format(num_inst))
        #print("matches found: {}".format(q_match_found))
        #print("matches pres.: {}".format(q_match_pres))
        print("delay: {}".format(sum(q_delay_arr[:-1])))
        print("acc. (recall) {:1.3f}".format(t_pos / (1e-8 + num_inst)))
        print("acc. (precis) {:1.3f}".format(t_pos / (1e-8 + t_pos + f_pos)))'''

        # update aggregate stats
        tot_img_seen += sum(q_img_seen_arr[:-1])
        tot_img_elim += sum(q_img_elim_arr[:-1])
        tot_num_inst += num_inst
        #tot_match_found += q_match_found
        #tot_match_pres  += q_match_pres
        tot_delay += sum(q_delay_arr[:-1])
        tot_t_pos += t_pos
        tot_f_pos += f_pos
        tot_t_neg += t_neg
        tot_f_neg += (num_inst - t_pos)

        '''
        print("\nAggregate results ----------")
        print("img seen: {}".format(tot_img_seen))
        print("img tot.: {}".format(tot_img_seen + tot_img_elim))
        print("num inst.: {}".format(tot_num_inst))
        #print("matches found: {}".format(tot_match_found))
        #print("matches pres.: {}".format(tot_match_pres))
        print("delay (avg.): {}".format(tot_delay / (q_idx + 1)))
        print("mAP: {:.1%}".format(np.mean(all_AP)))
        print("acc. (recall) {}".format(tot_t_pos / (1e-8 + tot_t_pos + tot_f_neg)))
        print("acc. (precis) {}".format(tot_t_pos / (1e-8 + tot_t_pos + tot_f_pos)))
        print(tot_t_pos,tot_f_pos,tot_t_neg,tot_f_neg)'''

    #min_len = min(map(len, all_cmc))
    #all_cmc = [cmc[:min_len] for cmc in all_cmc]
    #all_cmc = np.asarray(all_cmc).astype(np.float32)
    #cmc = all_cmc.sum(0) / num_valid_q

    #print("CMC curve")
    #for r in ranks:
    #    if r-1 < len(cmc):
    #        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    #print("------------------")
    cost = tot_img_seen
    recall = tot_t_pos / (1e-8 + tot_t_pos + tot_f_neg)
    precison = tot_t_pos / (1e-8 + tot_t_pos + tot_f_pos)
    if m == 1:
        return cost,recall,precison
    else:
        return cost,recall,precison,tot_delay
def run(cam_n,check_model,mm = 1):
    if mm == 1:
        cost,recall,precision = main(check_model,mm)
        return cost,recall,precision
    else:
        cost,recall,precision,delay = main(check_model,mm)
        return cost,recall,precision,delay


if __name__ == '__main__':
    main()