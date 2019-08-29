import pandas as pd
import numpy as np
import datetime as dt
import os
from xml.dom.minidom import parse
import xml.dom.minidom
import pandas as pd
import numpy as np

import folium
from folium import plugins
import webbrowser
import geopandas as gp

import datetime as dt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import csv
inter_x = 0.005
inter_y = 0.005
times = 0.25
#cam_corr_x = [[39.9775,39.9825,39.9825,39.9775,39.9775],[39.9737,]]
#cam_corr_y = [[116.34, 116.34,116.35,116.35,116.34],[116.31,]]
index_k_x = [-1,+1,+1,-1,-1]
index_k_y = [-1,-1,+1,+1,-1]
cam_number = 180
center_cam_x = [39.9750, 39.9750,39.9850, 39.9850,  39.9800, 39.9750,39.9840]
center_cam_y = [116.3475, 116.335,116.3475, 116.3325, 116.32, 116.31,116.31]
#center_cam_x = [39.98224537,39.98622918,39.97487002,39.97522396,39.98018901,39.98500,39.9875,39.98125,39.97375]
#center_cam_y = [116.32700423,116.30635033,116.33146695,116.30902603,116.3444744,116.340000,116.32,116.305,116.345]
#center_cam_x = [39.98231682,39.97793089,39.96243912,39.99480623,39.93675056,39.95087347,39.97460507,39.94860765,40.00619461,39.98811576,39.99179522,39.97106438,40.00250373,39.93669579,39.96946726,39.97031962,39.96468977,39.9354272,39.98544602,39.98156024]
#center_cam_y = [116.3466115,116.30864797,116.32772928,116.32755192,116.32292862,116.31336302,116.33190811,116.36064387,116.31992667,116.30355491,116.36226491,116.29881618,116.28737684,116.30007509,116.31625778,116.36152543,116.34769739,116.34244199,116.31843923,116.32841243]
center_cam_x = np.array(center_cam_x)
center_cam_y = np.array(center_cam_y)
bound_boxs_x = []
bound_boxs_y = []

amount = 5000
change = np.zeros((cam_number + 1 ,cam_number + 1))
import matplotlib.animation as animation

camera_manually = []
def on_press(event):
    print("Len : ",len(camera_manually))
    print("my position:" ,event.button,event.xdata, event.ydata)
    camera_manually.append((event.xdata,event.ydata))
    save_tmp = np.array(camera_manually)
    np.save('./newBeijing_camera_xy_200.npy',save_tmp)
def print_map():
    #if print_flag:
    #print_flag = 0
    #print(use_for_print_trans_time)
    #color  = ['s','p','*','h','+','x','d']#,'2','3','s','p','*','h','+','x','d','2','3','s','p']
    #for i in range(cam_number):
    #  plt.plot(bound_boxs_x[i],bound_boxs_y[i],marker=color[i])

    #schools_map = folium.Map(location=[x.mean(), y.mean()], zoom_start=10)
    #marker_cluster = plugins.MarkerCluster().add_to(schools_map)
    #t_c = []
    #for i in range(x.shape[0]):
    #  t_c.append((x[i],y[i]))

    #folium.PolyLine(t_c).add_to(schools_map)
    #schools_map.save('schools_map.html')
    #input()
    #for ii in range(x.shape[0]):
    #  folium.Marker([x[ii], y[ii]]).add_to(marker_cluster)
    #display(schools_map) 
    ###plt.plot(x,y,linewidth=1)
    #plt.show()
    pass
def check_in_camera(corr,index):
  if corr[0] >= center_cam_x[index] - times*inter_x and corr[0] <= center_cam_x[index] + times*inter_x:
    if (corr[1] >= (center_cam_y[index] - times*inter_y) and (corr[1] <= center_cam_y[index] + times*inter_y)):
      return True
  return False
def print_sp_corr(change,cam_number):
  print(change)
  for i in range(cam_number + 1):
    total = np.sum(change[i,:])
    change[i,:] = change[i,:]/total
  print(change)

  for i in range(cam_number):
    x_list = [1,2,3,4,5,6,7,8,'Other']
    y_list = change[i,:]
    plt.figure('Sc fig' + str(i))
    ax = plt.gca()
    ax.set_xlabel('Destination camera')
    ax.set_ylabel('Traffic percentage (%)')
    xticks = np.arange(1, len(x_list)+1)
    bar_width=0.5
    ax.bar(xticks, y_list, width=bar_width, edgecolor='none')
    ax.set_xticks(xticks)
    ax.set_xticklabels(x_list)
    ax.set_xlim(0,len(xticks))
    plt.savefig("./Porto_Cam_8/Sc_"+str(i + 1)+"_.jpg")
def print_tp_corr(statistics_time,cam_number):
  for i in range(cam_number):
    for j in range(cam_number):
      statistics_time[i][j] = sorted(statistics_time[i][j])
      
  tot_number = 200
  mult_n = 8
  x_list = []
  y_list = []
  for i in range(tot_number):
    x_list.append(i * mult_n)
  for i in range(cam_number):
    plt.figure('Line fig' + str(i))
    ax = plt.gca()
    ax.set_xlabel('Travel time (sec.)')
    ax.set_ylabel('Amount')
    for j in range(cam_number):
      y_list = []
      statistics_time[j][i] = sorted(statistics_time[j][i])
      tot_t = len(statistics_time[j][i])
      if tot_t <= 5:
        continue
      cur_t = 0
      for z in range(tot_number):
        c_z = 0
        while cur_t < tot_t and statistics_time[j][i][cur_t] >= z*mult_n  and statistics_time[j][i][cur_t] < (z+1)*mult_n:
            c_z += 1
            cur_t += 1
        y_list.append(c_z)
      y_list[-1] += (tot_t - cur_t)
      ax.plot(x_list, y_list,label=str(j + 1)) 
    plt.legend()
    plt.savefig("./Porto_Cam_8/Tc_"+str(i + 1)+"_.jpg")
def saving_sp_corr_geo_prox(path_corr,change,cam_number):
  corr_matrix = []
  for i in range(cam_number):
    tmp = []
    for j in range(cam_number):
      dist = (center_cam_x[i] - center_cam_x[j])*(center_cam_x[i] - center_cam_x[j]) +  (center_cam_y[i] - center_cam_y[j])* (center_cam_y[i] - center_cam_y[j])
      if dist < (times*inter_x*3)*(times*inter_x*3):
        tmp.append(j)
    corr_matrix.append(tmp)
  np.save(path_corr + "/corr_matrix_porto.npy",np.array(corr_matrix))
def saving_sp_corr(path_corr,change,cam_number,th = 0.05):
  #print(change)
  for i in range(cam_number + 1):
    total = np.sum(change[i,:])
    change[i,:] = change[i,:]/total
  #print(change)
  corr_matrix = []
  for i in range(cam_number):
    tmp = []
    tmp.append(i)
    for j in range(cam_number):
      if change[i][j] >= th:
        tmp.append(j)
    corr_matrix.append(tmp)
  np.save(path_corr + "/corr_matrix_porto.npy",np.array(corr_matrix))
def saving_tp_corr(path_corr,statistics_time,cam_number,th = 0.01):
  for i in range(cam_number):
    for j in range(cam_number):
      statistics_time[i][j] = sorted(statistics_time[i][j])

  start_times = np.zeros((cam_number,cam_number))
  end_times = np.zeros((cam_number,cam_number))

  for i in range(cam_number):
    end_times[i][i] = 16
    for j in range(cam_number):
      tot_number = len(statistics_time[i][j])
      if tot_number < 2:
        continue
      last_index = int(tot_number * (1 - th))
      if last_index == tot_number:
        last_index -= 1
      start_times[i][j] = statistics_time[i][j][0]
      end_times[i][j] = statistics_time[i][j][last_index]
  np.save(path_corr + "/start_times_porto.npy",start_times)
  np.save(path_corr + "/end_times_porto.npy",end_times)
def check_overlap_camera(corr,camera_x,camera_y,th):
  for i in range(len(camera_x)):
    dist = (corr[0] - camera_x[i])*(corr[0] - camera_x[i]) + (corr[1] - camera_y[i])*(corr[1] - camera_y[i])
    if dist < th*th:
      return True
  return False
def process_camera_xy(path):
  dist_th = 2*inter_x*times

  global center_cam_x,center_cam_y,bound_boxs_x,bound_boxs_y
  center_cam_x = []
  center_cam_y = []
  bound_boxs_x = []
  bound_boxs_y = []
  t = np.load(path)
  center_point = (-8.62,41.16)
  todo_list = []
  for i in range(t.shape[0]):
    dist = (t[i][0] + 8.62)*(t[i][0] + 8.62) + (t[i][1] - 41.16)*(t[i][1] - 41.16)
    todo_list.append((dist,i))
  todo_list = sorted(todo_list,key= lambda x:x[0])
  #sorted_frame_list_all = sorted(sorted_frame_list_all, key=lambda x: x[0])
  select_number = 0
  cur_index = 0
  while select_number < cam_number:
    if not check_overlap_camera(t[todo_list[cur_index][1]],center_cam_x,center_cam_y,dist_th):
      center_cam_x.append(t[todo_list[cur_index][1]][0])
      center_cam_y.append(t[todo_list[cur_index][1]][1])
      select_number += 1
    cur_index += 1
      #camera_list.append(t[todo_list[cur_index][1]][0],t[todo_list[cur_index][1]][1])

  #np.random.shuffle(t)
  #print("Tot cam : ",t.shape[0])
  for i in range(cam_number):
    center_cam_x.append(t[i][0])
    center_cam_y.append(t[i][1])
    #print("Camera center x y : ",t[i][0],t[i][1])

  for i in range(cam_number):
    bound_box_x = []
    bound_box_y = []
    for j in range(5):
      bound_box_x.append(center_cam_x[i] + times * index_k_x[j] * inter_x)
      bound_box_y.append(center_cam_y[i] + times * index_k_y[j] * inter_y)

    bound_boxs_x.append(bound_box_x)
    bound_boxs_y.append(bound_box_y)
  bound_boxs_x = np.array(bound_boxs_x)
  bound_boxs_y = np.array(bound_boxs_y)
def Beijing_data():
  print_flag = 0
  traj_count = 0
  point_count = 0
  in_point_count = 0
  pdf = PdfPages('Beijing.pdf')
  ac = plt.figure()
  corr_list = []
  instance_index = -1
  statistics_time = {}
  data_m = np.load("./geolife_total_3975_1166.npy")
  use_for_print_trans_time = 0
  #color  = ['s','p','*','h','+','x','d']#,'2','3','s','p','*','h','+','x','d','2','3','s','p']
  process_camera_xy("./beijing_camera_xy_100.npy")
  for i in range(cam_number):
    plt.plot(bound_boxs_x[i],bound_boxs_y[i])
  for i in range(cam_number):
      statistics_time[i] = {}
      for j in range(cam_number):
          statistics_time[i][j] = []
  last_cam_id = -1
  last_second = -1
  last_index = -1
  x = []
  y = []
  print("Total : ",data_m.shape[0])
  for image_id in range(data_m.shape[0]):
    if image_id % 100000 == 0:
      print("Image Id : ",image_id,"/ ",data_m.shape[0])
    cur_index = data_m[image_id][0]
    corr = (data_m[image_id][1],data_m[image_id][2])
    second_time = data_m[image_id][3]
    if last_index == -1:
      last_index = cur_index
    if last_index != cur_index:
      #change[last_cam_id][cam_number] += 1
      last_cam_id = -1
      last_second = -1
      x = np.array(x)
      y = np.array(y)
      plt.plot(x,y,linewidth = 0.1)
      #__Call_print_map_function__
      last_index = cur_index
      x = []
      y = []
    '''for id_ in range(cam_number):
      if check_in_camera(corr,id_):
        if last_cam_id == -1 and last_second == -1:
          last_cam_id = id_
          last_second = second_time
        elif last_cam_id == id_:
          last_second = second_time
        else:
          if last_cam_id != id_:
            change[last_cam_id][id_] += 1
            trans_time = second_time - last_second
            if last_cam_id == 1 and id_ == 5:
              print_flag = 1
              use_for_print_trans_time = trans_time
            if trans_time < 0:
              trans_time = 86399 - second_time + last_second
            statistics_time[last_cam_id][id_].append(trans_time)
            last_cam_id = id_'''
    x.append(corr[0])
    y.append(corr[1])

  tot_valid_sample = 0
  ac.canvas.mpl_connect('button_press_event', on_press)
  plt.show()
  pdf.savefig()
  plt.close()
  pdf.close()
  csvFile.close()
def analy_cam_data(path,path_corr):
  instance_index = -1
  statistics_time = {}
  data_m = np.load(path)
  #color  = ['s','p','*','h','+','x','d']#,'2','3','s','p','*','h','+','x','d','2','3','s','p']
  for i in range(cam_number):
      statistics_time[i] = {}
      for j in range(cam_number):
          statistics_time[i][j] = []
  last_cam_id = -1
  last_second = -1
  last_index = -1
  x = []
  y = []
  print("Total : ",data_m.shape[0])
  for image_id in range(data_m.shape[0]):
    if image_id % 5000000 == 0:
      print("Image Id : ",image_id,"/ ",data_m.shape[0])
    cur_index = data_m[image_id][0]
    corr = (data_m[image_id][1],data_m[image_id][2])
    second_time = int(data_m[image_id][3])
    cur_cam_id = int(data_m[image_id][4])
    if last_index == -1:
      last_index = cur_index
      last_cam_id = cur_cam_id
      last_second = int(second_time)
    if last_index != cur_index:
      #print(last_cam_id,cam_number,last_index,cur_index)
      change[last_cam_id][cam_number] += 1
      last_cam_id = -1
      last_second = -1
      last_index = cur_index
      last_cam_id = cur_cam_id
      last_second = second_time
      continue
    if last_cam_id == cur_cam_id:
      last_second = second_time
    else:
      change[last_cam_id][cur_cam_id] += 1
      trans_time = second_time - last_second
      if trans_time < 0:
        trans_time = 86399 - second_time + last_second
      statistics_time[last_cam_id][cur_cam_id].append(trans_time)
      last_cam_id = cur_cam_id
      last_second = second_time

  saving_sp_corr(path_corr,change,cam_number,0.05)
  #saving_sp_corr_geo_prox(path_corr,change,cam_number,0.05)
  saving_tp_corr(path_corr,statistics_time,cam_number,0.01)
def print_porto_data(path):
  pdf = PdfPages('porto_cam_d.pdf')
  data_m = np.load("./porto.npy")
  ac = plt.figure()
  last_index = -1
  x_list = []
  y_list = []
  process_camera_xy(path)
  for i in range(cam_number):
    plt.plot(bound_boxs_x[i],bound_boxs_y[i])
  #print("Total len : ",data_m.shape[0])
  for i in range(data_m.shape[0]):
    if i % 1000 == 0:
      print(i)
    if i >= 150000:
      break
    cur_index,x,y,times = data_m[i]
    if last_index == -1:
      last_index = cur_index
    if last_index != cur_index:
      x_list = np.array(x_list)
      y_list = np.array(y_list)
      plt.plot(x_list,y_list,linewidth = 0.1)
      x_list = []
      y_list = []
      last_index = cur_index
    else:
      if not (x > -8.70 and x < -8.55):
        continue
      if not (y > 41.1 and y < 41.2):
        continue
      flag = False
      cam_id = -1
      #for i in range(cam_number):
      #  flag = check_in_camera((x,y),i)
      #  if flag == True:
      #    cam_id = i
      #if cam_id == -1:
      #  continue

      x_list.append(x)
      y_list.append(y)

  #ac.canvas.mpl_connect('button_press_event', on_press)
  #plt.show()
  pdf.savefig()
  plt.close()
  pdf.close()
def convert_to_camera_data(load_path,cam_path,save_path):
  data_m = np.load(load_path)
  save_list = []
  #camera_xy = np.load("./porto_camera_xy.npy")
  process_camera_xy(cam_path)
  #print("Total len : ",data_m.shape[0])
  last_index = -1
  last_second = -1
  last_cam  = -1
  todo_save_list = []
  can_save = False
  for i in range(data_m.shape[0]):
    if i % 1000000 == 0:
      print(i,len(save_list))
    if i >= 1000000:
      break
    #print(data_m[i])
    cur_index,x,y,times = data_m[i]
    times = int(int(times) % 86400)
    if last_index == -1:
      last_index = cur_index
      last_second = times
    elif last_index == cur_index:
      times = last_second + 15
      last_second = times
    elif last_index != cur_index:
      last_index = cur_index
      last_second = times
      last_cam = -1
      todo_save_list = []
      can_save = False

    flag = False
    cam_id = -1

    for i in range(cam_number):

      flag = check_in_camera((x,y),i)
      if flag == True:
        cam_id = i
        if last_cam == -1:
          last_cam = cam_id
        elif last_cam != cam_id:
          can_save = True
          save_list.extend(todo_save_list)
          todo_save_list = []

    if cam_id == -1:
      continue
    if can_save:
      save_list.append((cur_index,x,y,times,cam_id))
    else:
      todo_save_list.append((cur_index,x,y,times,cam_id))

  np.save(save_path,np.array(save_list))
  #print(min_t,max_t)
  '''if last_index == -1:
      last_index = cur_index
      last_second = 
    elif last_index == cur_index:
      times '''
def bike_data():
  plt.figure()
  data_dir = '/Users/zhangxun/Downloads/A collection of sport activity ﬁles for data analysis and data mining(运动路线数据)/Sport'
  dirlist = os.listdir(data_dir)
  count = 0
  for i in dirlist:
     print(i)
     if not i == 'Rider3':
        continue
     tmp_dir = os.listdir(data_dir + '/' + i)
     print(i)
     for j in tmp_dir:
        count += 1
        if count >= 2:
           break
        fn = data_dir + '/' + i + '/' + j
        print(fn)
        DOMTree = xml.dom.minidom.parse(fn)
        collection = DOMTree.documentElement
        trajs = collection.getElementsByTagName("trkpt")
        x = []
        y = []
        corrs = []
        for corr in trajs:
           x.append(corr.attributes["lat"].value)
           y.append(corr.attributes["lon"].value)
           corrs.append((corr.attributes["lat"].value,corr.attributes["lon"].value))
        x = np.array(x)
        y = np.array(y)
        plt.plot(x,y,linewidth=0.0001)

  plt.savefig("./easyplot.jpg")
  #pdf.savefig()
  plt.close()
def convert_Beijing_data(for_save_cost = False):
  process_camera_xy()
  traj_count = 0
  point_count = 0
  in_point_count = 0
  data_dir = 'geolife/Data'
  corr_list = []
  dirlist = os.listdir(data_dir)
  for_save = []
  instance_index = -1
  for i in dirlist:
    if i[0] == '.':
      continue
    tmp_dir = os.listdir(data_dir + '/' + i + '/Trajectory')
    print(i)
    #if traj_count >= amount:
    #  break
    for j in tmp_dir:
      if j == 'labels.txt':
        continue
      traj_count += 1
      fn = data_dir + '/' + i + '/Trajectory' + '/' + j
      fp1 = open(fn,'r+')
      lines = fp1.readlines()
      l_list = lines[6:]
      instance_index += 1
      x = []
      y = []
      last_corr = []
      flag = 0
      point_count += len(l_list)
      #Begin a trajectory .......
      last_cam_id = -1
      last_second = -1
      for z in l_list:
        traj = z.split(',')
        corr = (float(traj[0]),float(traj[1]))
        time_ = traj[-1][0:-1].split(':')
        cam_id = -1
        for id_ in range(cam_number):
          if check_in_camera(corr,id_):
            cam_id = id_

        second_time = int(time_[0]) * 3600 + int(time_[1]) * 60 + int(time_[2])
        if corr[0] < 39.75 or corr[0] > 40.1:
          continue
        if corr[1] < 116.30 or corr[1] > 116.60:
          continue
        in_point_count += 1
        if last_corr == []:
          last_corr = corr
        else:
          dist = (last_corr[0] - corr[0])*(last_corr[0] - corr[0]) + (last_corr[1] - corr[1])*(last_corr[1] - corr[1])
          if dist > 0.00001:
            #print(last_corr,corr)
            last_corr = []
            instance_index += 1
            x = []
            y = []
            continue
        #for_save.append((int(instance_index),float(corr[0]),float(corr[1]),int(second_time),int(i)))
        if cam_id != -1:
          for_save.append((int(instance_index),float(corr[0]),float(corr[1]),int(second_time),int(cam_id)))
        last_corr = corr
  for_save = np.array(for_save)
  np.save("./geolife_total_3975_1166_cam_100.npy",for_save)
  print(traj_count,point_count,in_point_count,instance_index)
def convert_porto_data():
  csvFile = open("./train.csv", "r")
  reader = csv.reader(csvFile,delimiter=',')
  for_save = []
  result = {}
  count = 0
  max_x = 52.900803
  max_y = 51.037119
  min_x = -36.913779
  min_y = 31.992111
  cur_index = -1
  for item in reader:
      if count == 0:
          count += 1
          continue
      count+= 1
      if count % 1000 == 0:
          print(count)
      corrs = item[-1][1:-1].split(',')
      times = int(item[-4])
      flag = 0
      true_corrs  = []
      x = []
      y = []
      tmp_x = 0
      skip = 0
      for i in corrs:
          if flag == 0:
              #print(i)
              if i == '':
                  flag  = 1 
                  continue
              true_corr = i[1:]
              tmp_x = float(true_corr)
              if tmp_x < -8.8 or tmp_x > - 8.4:
                  skip = 1
                  flag = 1
                  continue
              flag = 1
          elif flag  == 1:
              if skip == 1:
                  skip = 0
                  flag = 0
                  continue
              true_corr = i[0:-1]
              if float(true_corr) < 41 or float(true_corr) > 41.5:
                  #skip = 1
                  flag = 0
                  continue

              if len(true_corrs) > 0:
                  if (true_corrs[-1][0] - tmp_x)*(true_corrs[-1][0] - tmp_x) + (true_corrs[-1][1] - float(true_corr))*(true_corrs[-1][1] - float(true_corr)) > 0.00001:
                      x = np.array(x)
                      y = np.array(y)
                      true_corrs = []
                      cur_index += 1
                      for t in range(x.shape[0]):
                        for_save.append((cur_index,x[t],y[t],times))
                      x = []
                      y = []

              x.append(tmp_x)
              y.append(float(true_corr))
              true_corrs.append([tmp_x,float(true_corr)])
              if float(true_corr) > max_y:
                  max_y = float(true_corr)
              if float(true_corr) < min_y:
                  min_y = float(true_corr)
              flag  = 0

      x = np.array(x)
      y = np.array(y)
      cur_index += 1
      for t in range(x.shape[0]):
        for_save.append((cur_index,x[t],y[t],times))

  np.save("./porto.npy",np.array(for_save))
  #ac.canvas.mpl_connect('button_press_event', on_press)
  #plt.show()
  csvFile.close()
  print(max_y,min_y,max_x,min_x)
def run(cam_n):
  global cam_number,change
  cam_number = cam_n
  change = np.zeros((cam_number+1,cam_number+1))
  print("processing raw data to camera data.....")
  #data_path = convert_to_camera_porto_data("./porto_cam_xy_total.npy")
  ###For Beijing....
  #load_path = "geolife_total_3975_1166.npy"
  #load_train_path = "geolife_total_3975_1166_train.npy"
  #load_test_path = "geolife_total_3975_1166_test.npy"
  #cam_path = "beijing_camera_xy_total.npy"
  #save_train_path = "beijing_datapoint_incamera.npy"
  #save_test_path = "beijing_datapoint_incamera_test.npy"
  ###For porto.....
  load_train_path = "porto_train.npy"
  load_test_path = "porto_test.npy"
  cam_path = "porto_cam_xy_total.npy"
  save_train_path = "porto_data_point_cam.npy"
  save_test_path = "porto_data_point_cam_test.npy"
  convert_to_camera_data(load_train_path,cam_path,save_train_path)
  convert_to_camera_data(load_test_path,cam_path,save_test_path)
  print("generating corretation.....")
  reid_path = "/Users/zhangxun/Documents/Chicago/Code/Deep-Person-ReId"
  #Correlation
  analy_cam_data(save_train_path,reid_path)


if __name__ == '__main__':
  #convert_Beijing_data()
  Beijing_data()
  #analy_cam_data()
  #convert_to_camera_porto_data()
  #print_porto_data("./porto_cam_f.npy")
  #From numpy data to cam data
  #data_path = convert_to_camera_porto_data("./porto_cam_xy_total.npy")
  #reid_path = "/Users/zhangxun/Documents/Chicago/Code/Deep-Person-ReId"
  #Correlation
  #analy_cam_data(data_path,reid_path)