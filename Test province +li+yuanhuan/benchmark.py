import csv
import numpy as np
import string
import time
from cleanlabel.cleanlabels import arrange_labels
from shuju import prodata
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import os
from scipy.interpolate import make_interp_spline
import re
import math
from clear import clearv
clearv()

matplotlib.use('Agg')

csv_reader = csv.reader(
    open('C:/Users/lenovo/Documents/WeChat Files/wxid_ynlzus6rb66j21/FileStorage/File/2020-05/LeiJ.csv', encoding='utf-8'))
numofprovince = len(list(csv_reader))       #省份个数
# files = os.listdir(path)
# files.sort(key=lambda x: int(x.split('.')[0]))
num = []
alllabel = []
count1 = -1
count2 = 0  # 交点约束
path = open('C:/Users/lenovo/Desktop/Test province/all Test.txt',encoding='utf-8')
for index, line in enumerate(path):  # 省份
    line = line.strip('\n')
    label = re.findall(r'[\u4E00-\u9FA5]+', line)
    time = re.findall(r'[,](\d+)', line)
    if(time != []):  
        num.append(index)   # 相交位置的在txt中的index
    if(label != []):
        alllabel.append(label)
path = open('C:/Users/lenovo/Desktop/Test province/all Test.txt',
            encoding='utf-8')
for index, line in enumerate(path):
    line = line.strip('\n')
    label = re.findall(r'[\u4E00-\u9FA5]+', line)
    time = re.findall(r'[,](\d+)', line)
    if (time != []):
        date = time[0]
    else:
        date = line
    allj = []
    newj = []
    f = open('C:/Users/lenovo/Desktop/Test province/shuzu.txt',  # 全局约束
             encoding='UTF-8')
    for i, lin in enumerate(f):  # x,y
        lin = lin.strip('\n')
        li = re.findall(r"[+-]?\d+\.?\d*", lin)
        allj.append(li)
    if(index <= num[-1]):  # 最后一个相交位置以内
        if(index == 0):
            v = 0
            vp = 0
            box = []
            [newj.append(allj[count2][i:i+2]) for i in range(0, len(allj[count2]), 2)]
            label = alllabel[count2]
        if(index < num[count2] and index != 0):  #下一个交点之前
            allv = []
            allbox = []
            allvp = []
            newv = []
            newbbox = []
            newvp = []
            f = open('C:/Users/lenovo/Desktop/Test province/v.txt',  # 力约束
                     encoding='UTF-8')
            for i, lin in enumerate(f):  # x,y
                lin = lin.strip('\n')
                if(i % 2 == 1):
                    li = re.findall(r"[+-]?\d+\.?\d*", lin)
                    allv.append(li)
                if(i % 2 == 0):
                    li = re.findall(r"[+-]?\d+\.?\d*", lin)
                    allbox.append(li)
            [newv.append(allv[i:i+numofprovince]) for i in range(0, len(allv), numofprovince)]
            [newbbox.append(allbox[i:i+numofprovince]) for i in range(0, len(allbox), numofprovince)]
            v = newv[count1]
            box = newbbox[count1]
            # print(v,box)
            fi = open('C:/Users/lenovo/Desktop/Test province/vp.txt',  # Vp
                      encoding='UTF-8')
            for i, lin in enumerate(fi):  # x,y
                lin = lin.strip('\n')
                li = re.findall(r"[+-]?\d+\.?\d*", lin)
                allvp.append(li)
            [newvp.append(allvp[i:i+numofprovince]) for i in range(0, len(allvp), numofprovince)]
            vp = newvp[count1]
            [newj.append(allj[count2][i:i+2]) for i in range(0, len(allj[count2]), 2)]
            label = alllabel[count2]
        if(index == num[count2]):       #交点时
            allv = []
            allbox = []
            allvp = []
            newv = []
            newbbox = []
            newvp = []
            f = open('C:/Users/lenovo/Desktop/Test province/v.txt',  # 力约束
                    encoding='UTF-8')
            for i, lin in enumerate(f):  # x,y
                lin = lin.strip('\n')
                if(i % 2 == 1):
                    li = re.findall(r"[+-]?\d+\.?\d*", lin)
                    allv.append(li)
                if(i % 2 == 0):
                    li = re.findall(r"[+-]?\d+\.?\d*", lin)
                    allbox.append(li)
            [newv.append(allv[i:i+numofprovince]) for i in range(0, len(allv), numofprovince)]
            [newbbox.append(allbox[i:i+numofprovince]) for i in range(0, len(allbox), numofprovince)]
            v = newv[count1]
            box = newbbox[count1]
            # print(v,box)
            fi = open('C:/Users/lenovo/Desktop/Test province/vp.txt',  # Vp
                    encoding='UTF-8')
            for i, lin in enumerate(fi):  # x,y
                lin = lin.strip('\n')
                li = re.findall(r"[+-]?\d+\.?\d*", lin)
                allvp.append(li)
            [newvp.append(allvp[i:i+numofprovince]) for i in range(0, len(allvp), numofprovince)]
            vp = newvp[count1]
            [newj.append(allj[count2][i:i+2]) for i in range(0, len(allj[count2]), 2)]
            count2 += 1  # 下一个交点

    else:       #再也没有交点
        allv = []
        allbox = []
        allvp = []
        newv = []
        newbbox = []
        newvp = []
        f = open('C:/Users/lenovo/Desktop/Test province/v.txt',  # 力约束
                 encoding='UTF-8')
        for i, lin in enumerate(f):  # x,y
            lin = lin.strip('\n')
            if(i % 2 == 1):
                li = re.findall(r"[+-]?\d+\.?\d*", lin)
                allv.append(li)
            if(i % 2 == 0):
                li = re.findall(r"[+-]?\d+\.?\d*", lin)
                allbox.append(li)
        [newv.append(allv[i:i+numofprovince]) for i in range(0, len(allv), numofprovince)]
        [newbbox.append(allbox[i:i+numofprovince]) for i in range(0, len(allbox), numofprovince)]
        v = newv[count1]
        box = newbbox[count1]
        # print(v,box)
        fi = open('C:/Users/lenovo/Desktop/Test province/vp.txt',  # Vp
                  encoding='UTF-8')
        for i, lin in enumerate(fi):  # x,y
            lin = lin.strip('\n')
            li = re.findall(r"[+-]?\d+\.?\d*", lin)
            allvp.append(li)
        [newvp.append(allvp[i:i+numofprovince]) for i in range(0, len(allvp), numofprovince)]
        vp = newvp[count1]
        newj = []       #全局约束为0

    hpro = []
    hidepro = []
    fi = open('C:/Users/lenovo/Desktop/Test province/score.txt')    #hide label
    for i, lin in enumerate(fi):  # x,y
        lin = lin.strip('\n')
        hide = re.findall(r'\d+', lin)
        hpro.append(hide)
    if(hpro!=[]):
        for i in range(len(hpro[0])):
            hidepro.append(int(hpro[0][i]))

    hprofe = []
    hideprofe = []
    fi = open('C:/Users/lenovo/Desktop/Test province/score-f.txt')  # hide label
    for i, lin in enumerate(fi):  # x,y
        lin = lin.strip('\n')
        hide = re.findall(r'\d+', lin)
        hprofe.append(hide)
    if(hprofe != []):
        for i in range(len(hprofe[0])):
            hideprofe.append(int(hprofe[0][i]))

    data = pd.read_csv(
        'C:/Users/lenovo/Desktop/Test province/province.csv', encoding='utf-8')
    labels = data['省份']
    LeiJ, XinZ, SiW, ZhiY = prodata(int(date))
    ax = plt.gca()
    delete = ax.scatter(LeiJ, XinZ, color='r')
    delete.remove()
    plt.xlabel('The cumulative confirmed', fontsize=15)
    plt.ylabel('The new diagnosis' , fontsize=15)
    # plt.title(date, fontsize=15)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    ys1 = newj
    arrange_labels(LeiJ, XinZ, SiW, ZhiY, labels, date,
                   label, ys1, v, vp, box, newj,hidepro,hideprofe,ax=ax)
    # mng = plt.get_current_fig_manager()
    # mng.window.state("zoomed")
    # plt.axis('tight')
    fig = plt.gcf()
    fig.set_size_inches(15.36, 8.07)
    plt.pause(0.1)
    plt.close()
    # fig.savefig('D:/Google Downloads/yiqing/0617'+'/'+date+'.png')
    fig.savefig('D:/Google Downloads/yiqing/1022'+'/'+date+'.png')
    count1 += 1

