from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools

import numpy as np
from matplotlib import pyplot as plt
import math
import re
from cleanlabel.c_geometry import line_intersection


def fast_overlap(box, qtree):
    intersection = qtree.intersect(box.points)
    return sum(map(lambda x: box.overlap(x), intersection))


def energy_componentwise(label_bboxes, anchors, non_label_bbox_qtree, axes_bbox,text_strings,text_objects,ys1,label):
    guding1 = []
    guding2 = []
    index = []
    gudingdelta = []
    gudingangle = []
    gudingbbox = []
    gudinganchors = []
    gudinglabel_bboxes = []
    mx = []
    my = []
    px = []
    py = []
    d = []
    a = []
    numpro = []
    for k in range(len(label)):
        dk = [i for i, x in enumerate(text_strings) if(
            x == label[k])]
        numpro.append(dk[0])
    yueshubbox = []
    k = 0
    for i in range(len(numpro)):    #约束label的初始位置
        yueshubbox.append(ys1[k])
        k+=1
    allbbox2 = []  # 约束label的bbox
    allanchors = []
    for i in range(len(numpro)):
        allbbox2.append(label_bboxes[numpro[i]])
    # print('allbbox2:',allbbox2)
    for i in range(len(numpro)):
        allanchors.append(anchors[numpro[i]])
    x2 = 0
    y2 = 0
    quanju1 = 0
    quanju2 = 0
    quanju = 0
    for k in range(len(allbbox2)):
        x2 = abs(
            float(yueshubbox[k][0])-(allbbox2[k].midpoint()[0]-allanchors[k][0]))
        quanju1 += x2
    # print(quanju1)
    for k in range(len(allbbox2)):
        y2 = abs(
            float(yueshubbox[k][1])-(allbbox2[k].midpoint()[1]-allanchors[k][1]))
        quanju2 += y2
    quanju = quanju1+quanju2

    overlap_area = 0
    n_overlaps = 0
    n_intersecting_lines = 0
    out_of_axes_area = 0
    for (bbox, anchor), (other_bbox, other_anchor) in itertools.combinations(
            zip(label_bboxes, anchors), 2):
        # penalty for bbox overlap 标签之间重叠
        overlap = bbox.overlap(other_bbox)
        overlap_area += overlap
        # print('overlap:', overlap_area)
    
        if overlap > 0:
            n_overlaps += 1    #重叠标签数量

        # out of axis area  超过坐标轴
        out_of_axes_area += bbox.area_outside_of_box(axes_bbox)

        anchor_x, anchor_y = anchor
        # closest_x, closest_y = bbox.nearest_point(anchor)
        closest_x, closest_y = bbox.midpoint()
        other_anchor_x, other_anchor_y = other_anchor
        other_closest_x, other_closest_y = other_bbox.midpoint()
        # penalty for intersecting lines
        if line_intersection(anchor_x, anchor_y,
                             closest_x, closest_y,
                             other_anchor_x, other_anchor_y,
                             other_closest_x, other_closest_y):
            n_intersecting_lines += 1  #引线相交的个数

    distances_to_anchor = 0
    non_label_overlap_area = 0
    orient = 0
    for bbox, anchor in zip(label_bboxes, anchors):
        # compute penalty for distance away from anchor  标签到点的距离
        distances_to_anchor += pow((bbox.midpoint()[0]-anchor[0]), 2)+pow(
            (bbox.midpoint()[1]-anchor[1]), 2)
        # compute penalty for overlap with other bboxes:  点与标签的重叠
        non_label_overlap_area += fast_overlap(bbox, non_label_bbox_qtree)
        #orient 
        dx = bbox.midpoint()[0]-anchor[0]
        dy = bbox.midpoint()[1]-anchor[1]
        if(dx >= 0 and dy >= 0):
            orient_score = 0
        if(dx <= 0 and dy >= 0):
            orient_score = 100
        if(dx <= 0 and dy <= 0):
            orient_score = 200
        if(dx >= 0 and dy <= 0):
            orient_score = 300
        orient += orient_score
        # print(anchor)
        # print(bbox.midpoint())
        # print('orient:', orient_score)
    return {'overlap_area': overlap_area,
            'n_overlaps': n_overlaps,
            'n_intersecting_lines': n_intersecting_lines,
            'out_of_axes_area': out_of_axes_area,
            'distances_to_anchor': distances_to_anchor,
            'non_label_overlap_area': non_label_overlap_area,
            'orient_score':orient,
            'quanju': quanju
            }
    
def energy(label_bboxes,
           anchors,
           non_label_bbox_qtree,
           axes_bbox,
           text_strings,
           text_objects,
           ys1,
           label,
           w_line_length=200,
           w_intersecting_lines=1,
           w_n_overlaps=200,
           w_bbox_overlap=2400,
           w_non_label_bbox_overlap=600,
           w_out_of_axes=320,
           w_orient=5,
           b=10000
           ):
    """
    Energy function,
    Heavily inspired by https://github.com/tinker10/D3-Labeler

    :param label_bboxes:
    :param anchors:
    :param non_label_bbox_qtree:
    :param w_line_length:
    :param w_intersecting_lines:
    :param w_bbox_overlap:
    :param w_non_label_bbox_overlap:
    :return:
    """

    components = energy_componentwise(label_bboxes,
                                      anchors,
                                      non_label_bbox_qtree,
                                      axes_bbox,text_strings,text_objects,ys1,label)

    ener = 0.0

    ener += components['n_overlaps'] * w_n_overlaps #重叠标签个数
    ener += components['out_of_axes_area'] * w_out_of_axes  #超过轴的面积

    ener += components['overlap_area'] * w_bbox_overlap #标签重叠面积
    ener += components['n_intersecting_lines'] * w_intersecting_lines #引线相交

    ener += components['distances_to_anchor'] * w_line_length  #点与标签距离
    ener += components['non_label_overlap_area'] * w_non_label_bbox_overlap #点与标签重叠
    ener += components['orient_score'] * w_orient  # 方向
    # ener += b*(components['quanju'])  # 第一帧

    # print('noverlaps:', components['n_overlaps'] * w_n_overlaps)
    # print('axes:', components['out_of_axes_area'] * w_out_of_axes)
    # print('overlap:', components['overlap_area'] * w_bbox_overlap)
    # print('interscet:',
    #       components['n_intersecting_lines'] * w_intersecting_lines)
    # print('yinxian:', components['distances_to_anchor'] * w_line_length)
    # print('non:', components['non_label_overlap_area']
    #       * w_non_label_bbox_overlap)
    # print('orient:', components['orient_score'] * w_orient)

    return ener
   
