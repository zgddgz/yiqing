import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import basinhopping
from scipy.optimize import minimize
from simanneal import Annealer
import itertools

# from cleanlabel.energy import energy_qianhou, energy_qh
from cleanlabel.energy import energy_componentwise, energy
from cleanlabel.c_geometry import bbox_overlap
from cleanlabel.energy import fast_overlap
from cleanlabel.c_geometry import line_intersection

from cleanlabel.geometry import Box, make_qtree
from tqdm import tqdm

import networkx as nx
import math
import matplotlib.patches as mpathes

import os
import re
from matplotlib.offsetbox import AnchoredText

import numpy 

def _adjust_bboxes(state, bboxes, previous_state=None, previous_bboxes=None):
    state = state.reshape(-1, 2)

    if previous_state is not None and previous_bboxes is not None:
        previous_deltas = previous_state.reshape(-1, 2)  # 两列
    else:
        previous_deltas = None

    new_bboxes = []
    for i, (delta, bbox) in enumerate(zip(state, bboxes)):

        if previous_deltas is not None and (delta == previous_deltas[i]).all():
            # If delta hasn't changed get ox from cache
            new_box = previous_bboxes[i]
        else:
            new_box = bbox.translate(*delta)

        new_bboxes.append(new_box)
    # print(new_bboxes)
    return new_bboxes


def spring_layout_initialisation(anchors_transformed,
                                 k=0.001,
                                 iterations=100):

    # based on http://stackoverflow.com/a/34697108
    G = nx.DiGraph()
    data_nodes = []
    init_pos = {}
    for i, (xi, yi) in enumerate(anchors_transformed):
        data_str = 'data_{0}'.format(i)
        G.add_node(data_str)

        data_nodes.append(data_str)
        init_pos[data_str] = (xi, yi)

        label_str = 'label_{0}'.format(i)
        G.add_node(label_str)
        G.add_edge(label_str, data_str)
        init_pos[label_str] = (xi, yi)

    pos = nx.spring_layout(
        G, pos=init_pos, fixed=data_nodes, k=k, iterations=iterations)

    # undo spring_layout's rescaling
    pos_after = np.vstack([pos[d] for d in data_nodes])
    pos_before = np.vstack([init_pos[d] for d in data_nodes])
    scale, shift_x = np.polyfit(pos_after[:, 0], pos_before[:, 0], 1)
    scale, shift_y = np.polyfit(pos_after[:, 1], pos_before[:, 1], 1)
    shift = np.array([shift_x, shift_y])
    for key, val in pos.items():
        pos[key] = (val * scale) + shift

    deltas = np.empty((len(anchors_transformed), 2), dtype=float)

    for label, data_str in G.edges():
        i = int(label[6:])
        xy = pos[data_str]
        xytext = pos[label]

        delta = xytext - xy
        deltas[i] = delta
    # print(deltas)
    return np.array(deltas).ravel()  # 多维变一维


def randomly_rotate(state, bboxes, anchors,
                    rotation_probability=0.05,
                    random_state=None,
                    **kwargs):

    random_state = np.random.RandomState(random_state)

    adjusted_bboxes = _adjust_bboxes(state, bboxes, **kwargs)
    state = state.reshape(-1, 2)
    new_deltas = []
    for delta, original, adjusted, anchor in zip(state, bboxes, adjusted_bboxes, anchors):
        rotate = random_state.rand() < rotation_probability
        if not rotate:
            new_delta = delta
        else:
            random_angle = random_state.rand() * 2.0 * np.pi
            random_angle -= np.pi

            d = np.linalg.norm(adjusted.vector_midpoint_to_point(anchor))
            new_bbox = adjusted.move_midpoint_to_angle(anchor, random_angle, d)

            new_delta = new_bbox.mins - original.mins

        new_deltas.append(new_delta)

    return np.array(new_deltas).ravel()

def point_bounding_box(point,x,padding, ax):
    if(x == 0.0):
        r = 0
    else:
        r = math.log(math.pow(10,x),2)
    point0 = point-7-r
    point1 = point+7+r
    return Box(point0[0], point0[1], point1[0], point1[1])

def collect_text_bboxes(text_objects,
                        renderer,
                        padding):

    bboxes = [t.get_window_extent(renderer)
              for t in text_objects]
    boxes = []
    # print(bboxes)
    ax = plt.gca()
    for bbox in bboxes:
        expanded = bbox.padded(p=1)
        b = Box(*expanded.get_points().ravel())
        # print(b)
        # boxes.append(Box(b.x_min, b.y_min+2, b.midpoint()[0]-1, b.y_max-1))
        boxes.append(Box(b.x_min, b.y_min+4, b.x_max-2, b.y_max))
        # print(Box(b.x_min, b.y_min+4, b.x_max-2, b.y_max))
    return boxes


def collect_point_bboxes(ax, points, padding):  # 点的矩形box
    trans_data = ax.transData
    return [point_bounding_box(trans_data.transform(point),point[0],padding, ax
                               ) for point in points]


def collect_point_r(ax, points, padding):  # r
    return [0 if point[0] == 0.0 else math.log(math.pow(10, point[0]), 2) for point in points]

def cosVector(x, y):
    if(len(x) != len(y)):
        print('error input,x and y is not in the same space')
        return
    result1 = 0.0
    result2 = 0.0
    result3 = 0.0
    for i in range(len(x)):
        result1 += x[i]*y[i]  # sum(X*Y)
        result2 += x[i]**2  # sum(X*X)
        result3 += y[i]**2  # sum(Y*Y)
    #print(result1)
    #print(result2)
    #print(result3)
    return float(result1/((result2*result3)**0.5))  # cos

class _Objective(object):

    def __init__(self,
                 bboxes, anchors, other_bboxes_qtree, axes_bbox, text_strings, text_objects, ys1, label,
                 **energy_kwargs
                 ):
        self.bboxes = bboxes
        self.anchors = anchors
        self.other_bboxes_qtree = other_bboxes_qtree
        self.axes_bbox = axes_bbox
        self.text_strings = text_strings
        self.text_objects = text_objects
        self.ys1 = ys1
        self.label = label
        self.energy_kwargs = energy_kwargs
        self._previous_state = None
        self._previous_bboxes = None

    def adjust_bboxes(self, new_state):
        adjusted = _adjust_bboxes(new_state, self.bboxes,
                                  self._previous_state, self._previous_bboxes)

        self._previous_state = new_state
        self._previous_bboxes = adjusted

        return adjusted

    def func(self, state):
        adjusted = self.adjust_bboxes(state)
        return energy(adjusted, self.anchors, self.other_bboxes_qtree, self.axes_bbox, self.text_strings, self.text_objects, self.ys1,
                      self.label, **self.energy_kwargs)

    def func_componentwise(self, state):
        adjusted = self.adjust_bboxes(state)
        return energy_componentwise(adjusted, self.anchors, self.other_bboxes_qtree, self.axes_bbox, self.text_strings, self.text_objects, self.ys1, self.label)

    def funcqianhou(self, state):
        adjusted = self.adjust_bboxes(state)
        return energy_qianhou(adjusted, self.anchors, self.other_bboxes_qtree, self.axes_bbox, self.text_strings, self.text_objects,
                              **self.energy_kwargs)

    def funcqianhou_componentwise(self, state):
        adjusted = self.adjust_bboxes(state)
        return energy_qh(adjusted, self.anchors, self.other_bboxes_qtree, self.axes_bbox, self.text_strings, self.text_objects)


def _find_label_offsets(label_bboxes, anchors_transformed,
                        other_bboxes, axes_bbox, text_strings, text_objects, ys1, label,
                        minimise_callback=None,
                        maxfev=None,
                        n_iter=10,
                        temperature=100,
                        ):
    other_bboxes_qtree = make_qtree(other_bboxes)
    initial_offsets = np.zeros(len(label_bboxes) * 2, dtype=float)
    obj = _Objective(label_bboxes, anchors_transformed, other_bboxes_qtree,
                     axes_bbox, text_strings, text_objects, ys1, label)
    # func = obj.func
    # func_componentwise = obj.func_componentwise

    func = obj.func
    func_componentwise = obj.func_componentwise

    progressbar = tqdm(desc='Finding best placements')

    def callback(x):
        progressbar.update()
        if minimise_callback is not None:
            return minimise_callback(x,
                                     func_componentwise(x),
                                     func(x))

    if maxfev is None:
        # maxfev = len(label_bboxes) * 200
        maxfev = 1000
    if minimise_callback:
        minimise_callback(initial_offsets,
                          func_componentwise(initial_offsets),
                          func(initial_offsets))

    def step_function(x): return randomly_rotate(
        x, label_bboxes, anchors_transformed)
    # initial_offsets = step_function(initial_offsets)

    initial_offsets = spring_layout_initialisation(anchors_transformed)
    # ener = basinhopping(func, initial_offsets,
    #                    T=temperature,
    #                    niter=n_iter,
    #                    take_step=step_function,
    #                    minimizer_kwargs=dict(method='powell', options=dict(maxfev=maxfev)),
    #                    )
    ener = minimize(func, initial_offsets,
                    method='powell',
                    options=dict(maxfev=maxfev),
                    callback=callback)

    best = ener.x
    # This a bit roundabout way of getting offsets back is here so one could change
    # adjust_bboxes function and still get the correct behaviour
    adjusted_bboxes = _adjust_bboxes(best, label_bboxes)
    offsets = np.zeros((len(label_bboxes), 2))
    for i, (bbox, adjusted_bbox) in enumerate(zip(label_bboxes, adjusted_bboxes)):
        offsets[i] = np.array(
            [(adjusted_bbox.x_min - bbox.x_min), (adjusted_bbox.y_min - bbox.y_min)])
    return offsets, ener


def arrange_labels(LeiJ, XinZ,
                   SiW, ZhiY,
                   labels,
                   date,
                   label,
                   ys1,
                   v,vp,box,newj,
                   hidepro,
                   hideprofe,
                   padding=1,
                   ax=None,
                   **kwargs):
    if ax is None:
        ax = plt.gca()

    anchors, anchors_transformed, \
        axes_bbox, label_bboxes, other_bboxes, text_strings, text_objects,r = _initialise(ax, LeiJ,
                                                                                        XinZ, labels,date,
                                                                                        padding)
    if (v == 0):
        offsets, __ = _find_label_offsets(label_bboxes,
                                        anchors_transformed,
                                        other_bboxes,
                                        axes_bbox, text_strings, text_objects, ys1, label, **kwargs)
        _draw_annotations(date, SiW, ZhiY, label, v, vp, box, newj, r, hidepro,hideprofe, ax, anchors, offsets, text_strings,
                          label_bboxes, other_bboxes, padding=padding)
    else:
        offsets = 0
        _draw_annotations(date, SiW, ZhiY, label, v, vp, box, newj, r, hidepro, hideprofe, ax, anchors, offsets, text_strings,
                          label_bboxes, other_bboxes, padding=padding)


def _draw_annotations(date, SiW, ZhiY, label, v, vp, box, newj, r, hidepro, hideprofe, ax, anchors, offsets, text_strings,
                      label_bboxes, other_bboxes, padding, draw_bboxes=True, **kwargs):
    # print(r)
    # print(label)
    newbbox = []
    nanchors = []
    distance1 = 0
    distance2 = 0
    distance3 = 0
    distance4 = 0
    distance5 = 0
    t = 0.1
    deltatime = 1
    trans = ax.transData.inverted() + ax.transData
    numpro = []
    for k in range(len(label)):
        dk = [i for i, x in enumerate(text_strings) if(
            x == label[k])]
        numpro.append(dk[0])
    text_strings = ['AnHui', 'AoMen', 'BeiJing', 'FuJian', 'GanSu', 'GuangDong', 'GuangXi', 'GuiZhou', 'HaiNan', 'HeBei', 'HeNan', 'HeiLongJiang', 'HuBei', 'HuNan', 'JiLin', 'JiangSu', 'JiangXi', 'LiaoNing', 'NeiMengGu', 'NingXia', 'QingHai', 'ShanDong', 'ShanXi',
                    'ShanXi', 'ShangHai', 'SiChuan', 'TaiWan', 'TianJin', 'XiZang', 'XiangGang', 'XinJiang', 'YunNan', 'ZheJiang', 'ChongQing']
    if(v == 0):
        for text, anchor, offset, bbox, radius, siw, zhiy,otherbox in zip(text_strings, anchors, offsets, label_bboxes, r, SiW, ZhiY, other_bboxes):
            bbox = bbox.translate(*offset)
            nanchors.append(ax.transData.transform(anchor))
            ax.annotate(text,
                    xy=anchor,
                    xytext=(bbox.x_min+2, bbox.midpoint()[1]-2),
                    xycoords= 'data',
                    textcoords=trans,
                    arrowprops=dict(arrowstyle='-',
                                    connectionstyle="arc3",
                                    color='black',
                                    alpha=.3,
                                    ),
                    fontsize=12,
                    )
            newbbox.append(bbox)   
            if draw_bboxes:
                patch = bbox.rectangle_patch(fill=True, color='r',
                                             transform=trans, alpha=0)
                ax.add_patch(patch)

                # patch = otherbox.rectangle_patch(fill=True, color='r',
                #                              transform=trans, alpha=0)
                # ax.add_patch(patch)
                if(siw == 0.0 and zhiy == 0.0):
                    a3 = bbox.Wedge_patch(
                        ax.transData.transform(anchor)[0], ax.transData.transform(anchor)[1], 6+radius, 0, 360, (6+radius)/2, color='blue', transform=trans)
                    ax.add_patch(a3)
                else:
                    thelta1 = 360*(siw/(siw+zhiy))
                    thelta2 = 360*(zhiy/(siw+zhiy))
                    # print(thelta1,thelta2)
                    a1 = bbox.Wedge_patch(
                        ax.transData.transform(anchor)[0], ax.transData.transform(anchor)[1], 6+radius, 0, thelta1, (6+radius)/2, color='crimson', transform=trans)
                    a2 = bbox.Wedge_patch(
                        ax.transData.transform(anchor)[0], ax.transData.transform(anchor)[1], 6+radius, thelta1, 360,(6+radius)/2, color='lightgreen', transform=trans)
                    ax.add_patch(a1)
                    ax.add_patch(a2)

    else:
        bboxes = []
        for i in range(len(box)):
            bboxes.append(Box(float(box[i][0]), float(box[i][1]), float(box[i][2]), float(box[i][3])))
        for text, anchor, offset, bbox, radius, siw, zhiy, otherbox in zip(text_strings, anchors, v, bboxes, r, SiW, ZhiY,other_bboxes):
            bbox = bbox.translate(float(offset[0])*t, float(offset[1])*t)
            nanchors.append(ax.transData.transform(anchor))
            newbbox.append(bbox)

        #显示不隐藏的与隐藏的点
        showtext_strings = []
        showanchors = []
        showv = []
        showbboxes = []
        showr = []
        showSiW = []
        showZhiY = []
        showother_bboxes = []
        hideshowtext_strings = []
        hideshowanchors = []
        hideshowv = []
        hideshowbboxes = []
        hideshowr = []
        hideshowSiW = []
        hideshowZhiY = []
        hideshowother_bboxes = []

        if(hidepro != []):                           
            for i in range(len(text_strings)):
                if (i in hidepro) and (i not in hideprofe):
                    hideshowtext_strings.append(text_strings[i])
                    hideshowanchors.append(anchors[i])
                    hideshowv.append(v[i])
                    hideshowbboxes.append(bboxes[i])
                    hideshowr.append(r[i])
                    hideshowSiW.append(SiW[i])
                    hideshowZhiY.append(ZhiY[i])
                    hideshowother_bboxes.append(other_bboxes[i])
                if i not in hidepro:
                    showtext_strings.append(text_strings[i])
                    showanchors.append(anchors[i])
                    showv.append(v[i])
                    showbboxes.append(bboxes[i])
                    showr.append(r[i])
                    showSiW.append(SiW[i])
                    showZhiY.append(ZhiY[i])
                    showother_bboxes.append(other_bboxes[i])

            #显示feature
            for text, anchor, offset, bbox, radius, siw, zhiy, otherbox in zip(text_strings, anchors, v, bboxes, r, SiW, ZhiY, other_bboxes):
                bbox = bbox.translate(float(offset[0])*t, float(offset[1])*t)
                if draw_bboxes:
                    patch = bbox.rectangle_patch(fill=True, color='r',
                                                 transform=trans, alpha=0)
                    ax.add_patch(patch)
                    # patch = otherbox.rectangle_patch(fill=True, color='b',
                    #                                 transform=trans, alpha=0)
                    # ax.add_patch(patch)
                    if(siw == 0.0 and zhiy == 0.0):
                        a3 = bbox.Wedge_patch(
                            ax.transData.transform(anchor)[0], ax.transData.transform(anchor)[1], 6+radius, 0, 360, (6+radius)/2, color='blue', transform=trans)
                        ax.add_patch(a3)
                    else:
                        thelta1 = 360*(siw/(siw+zhiy))
                        thelta2 = 360*(zhiy/(siw+zhiy))
                        a1 = bbox.Wedge_patch(
                            ax.transData.transform(anchor)[0], ax.transData.transform(anchor)[1], 6+radius, 0, thelta1, (6+radius)/2, color='crimson', transform=trans)
                        a2 = bbox.Wedge_patch(
                            ax.transData.transform(anchor)[0], ax.transData.transform(anchor)[1], 6+radius, thelta1, 360, (6+radius)/2, color='lightgreen', transform=trans)
                        ax.add_patch(a1)
                        ax.add_patch(a2)
            #显示不隐藏的
            for text, anchor, offset, bbox, radius, siw, zhiy, otherbox in zip(showtext_strings, showanchors, showv, showbboxes, showr, showSiW, showZhiY, showother_bboxes):
                bbox = bbox.translate(float(offset[0])*t, float(offset[1])*t)
                ax.annotate(text,
                            xy=ax.transData.transform(anchor),
                            xytext=(bbox.x_min+2, bbox.midpoint()[1]-2),
                            xycoords=trans,
                            textcoords=trans,
                            arrowprops=dict(arrowstyle='-',
                                            connectionstyle="arc3",
                                            color='black',
                                            alpha=.3,
                                            ),
                            fontsize=12,
                            )
                if draw_bboxes:
                    patch = bbox.rectangle_patch(fill=True, color='r',
                                                transform=trans, alpha=0)
                    ax.add_patch(patch)
                    patch = otherbox.rectangle_patch(fill=True, color='b',
                                                    transform=trans, alpha=0)
                    ax.add_patch(patch)
                    if(siw == 0.0 and zhiy == 0.0):
                        a3 = bbox.Wedge_patch(
                            ax.transData.transform(anchor)[0], ax.transData.transform(anchor)[1], 6+radius, 0, 360, (6+radius)/2, color='blue', transform=trans)
                        ax.add_patch(a3)
                    else:
                        thelta1 = 360*(siw/(siw+zhiy))
                        thelta2 = 360*(zhiy/(siw+zhiy))
                        a1 = bbox.Wedge_patch(
                            ax.transData.transform(anchor)[0], ax.transData.transform(anchor)[1], 6+radius, 0, thelta1, (6+radius)/2, color='crimson', transform=trans)
                        a2 = bbox.Wedge_patch(
                            ax.transData.transform(anchor)[0], ax.transData.transform(anchor)[1], 6+radius, thelta1, 360, (6+radius)/2, color='lightgreen', transform=trans)
                        ax.add_patch(a1)
                        ax.add_patch(a2)
        else:
            for text, anchor, offset, bbox, radius, siw, zhiy, otherbox in zip(text_strings, anchors, v, bboxes, r, SiW, ZhiY, other_bboxes):
                bbox = bbox.translate(float(offset[0])*t, float(offset[1])*t)
                ax.annotate(text,
                            xy=ax.transData.transform(anchor),
                            xytext=(bbox.x_min+2, bbox.midpoint()[1]-2),
                            xycoords=trans,
                            textcoords=trans,
                            arrowprops=dict(arrowstyle='-',
                                            connectionstyle="arc3",
                                            color='black',
                                            alpha=.3,
                                            ),
                            fontsize=12,
                            )
                if draw_bboxes:
                    patch = bbox.rectangle_patch(fill=True, color='r',
                                                transform=trans, alpha=0)
                    ax.add_patch(patch)
                    # patch = otherbox.rectangle_patch(fill=True, color='b',
                    #                                 transform=trans, alpha=0)
                    # ax.add_patch(patch)
                    if(siw == 0.0 and zhiy == 0.0):
                        a3 = bbox.Wedge_patch(
                            ax.transData.transform(anchor)[0], ax.transData.transform(anchor)[1], 6+radius, 0, 360, (6+radius)/2, color='blue', transform=trans)
                        ax.add_patch(a3)
                    else:
                        thelta1 = 360*(siw/(siw+zhiy))
                        thelta2 = 360*(zhiy/(siw+zhiy))
                        a1 = bbox.Wedge_patch(
                            ax.transData.transform(anchor)[0], ax.transData.transform(anchor)[1], 6+radius, 0, thelta1, (6+radius)/2, color='crimson', transform=trans)
                        a2 = bbox.Wedge_patch(
                            ax.transData.transform(anchor)[0], ax.transData.transform(anchor)[1], 6+radius, thelta1, 360, (6+radius)/2, color='lightgreen', transform=trans)
                        ax.add_patch(a1)
                        ax.add_patch(a2)

    overlap_area = 0
    n_intersecting_lines = 0
    for (bbox, anchor), (other_bbox, other_anchor) in itertools.combinations(
            zip(newbbox, nanchors), 2):
        overlap = bbox.overlap(other_bbox)
        overlap_area += overlap

        anchor_x, anchor_y = anchor
        closest_x, closest_y = bbox.x_min+2, bbox.midpoint()[1]-2
        other_anchor_x, other_anchor_y = other_anchor
        other_closest_x, other_closest_y = other_bbox.x_min+2, other_bbox.midpoint()[1]-2
        # penalty for intersecting lines
        if line_intersection(anchor_x, anchor_y,
                             closest_x, closest_y,
                             other_anchor_x, other_anchor_y,
                             other_closest_x, other_closest_y):
            n_intersecting_lines += 1  # 引线相交的个数

    distances_to_anchor = 0
    non_label_overlap_area = 0
    other_bboxes_qtree = make_qtree(other_bboxes)

    for bbox, anchor in zip(newbbox, nanchors):
        # compute penalty for distance away from anchor  标签到点的距离
        distances_to_anchor += pow((bbox.midpoint()[0]-anchor[0]), 2)+pow(
            (bbox.midpoint()[1]-anchor[1]), 2)
        non_label_overlap_area += fast_overlap(bbox, other_bboxes_qtree)
    print(date)
    # with open('D:/Google Downloads/yiqing/evaluate/li.txt', 'a')as fi:
    #     fi.write(str(overlap_area)+','+str(non_label_overlap_area)+','+str(distances_to_anchor)+','+str(n_intersecting_lines)+'\n')

    # mx = []
    # my = []
    # px = []
    # py = []
    # for i in range(len(newbbox)):
    #     px.append(nanchors[i][0])
    #     py.append(nanchors[i][1])
    #     mx.append(newbbox[i].midpoint()[0])
    #     my.append(newbbox[i].midpoint()[1])
    # for i in range(len(newbbox)):
    #     with open('D:/Google Downloads/yiqing/evaluate/smooth li.txt', 'a')as fi:
    #         fi.write(str(round(mx[i]-px[i], 2))+','+str(round(my[i]-py[i], 2))+',')
    # with open('D:/Google Downloads/yiqing/evaluate/smooth li.txt', 'a')as fi:
    #     fi.write('\n')


    m1 = 30     #label-label
    m2 = 17     #feature-label  
    m3 = 150    # feature future
    mdis = 18   #mpull
    clabel = 40     
    cpoint = 50
    cpull = 20
    cjiao = 0
    cclosed = 15   
    ctime = 1
    # clabel = 1
    # cpoint = 1
    # cpull = 20
    # cjiao = 20
    # cclosed = 15
    # ctime = 1
    cf = 6
    k = 0
    R = 6               #初始圆环半径
    priority = []       #优先级
    priorindex = []     #优先级索引

    #priority
    for i in range(len(newbbox)):
        if(v != 0 ):
            deltaLeiJ = abs((nanchors[i][0]-float(vp[i][0]))/nanchors[i][0])
            vSiW = SiW[i]/nanchors[i][0]
            nLeiJ = nanchors[i][0]
            priority.append({'index':i,'pro':text_strings[i],'sum':deltaLeiJ+vSiW+(1-(1/nLeiJ))})   
    sort = sorted(priority, key=lambda x: x['sum'],reverse=True)
    if(v != 0):
        for i in range(len(newbbox)):
            index = sorted(priority, key=lambda x: x['sum'], reverse=True)[i]['index']
            priorindex.append(index)

    hide = []
    hidefeature = []
    labelscore = []
    for i in range(len(newbbox)):
        label_score = 0
        for j in range(len(newbbox)):
            if(j != i and v != 0):
                # 判断label_label overlap
                overlap1 = newbbox[i].overlap(newbbox[j])
                if overlap1 > 50 and(priorindex.index(i) > priorindex.index(j)):
                    # print(text_strings[newbbox.index(bbox)],text_strings[newbbox.index(other_bbox)])
                    label_score += 1
                # 判断label_feature overlap
                overlap2 = newbbox[i].overlap(other_bboxes[j])
                if overlap2 > 50:
                    label_score += 1
        labelscore.append(label_score)

    featurescore = []
    for i in range(len(other_bboxes)):
        feature_score = 0
        for j in range(len(other_bboxes)):
            if(j != i and v != 0):
                # 判断feature_label overlap
                overlap1 = other_bboxes[i].overlap(newbbox[j])
                # if overlap1 > 1 and(priorindex.index(i) > priorindex.index(j)):
                if overlap1 > 50:
                    # print(text_strings[newbbox.index(bbox)],text_strings[newbbox.index(other_bbox)])
                    feature_score += 1
                    # print(date,text_strings[i],text_strings[j])
        featurescore.append(feature_score)

    for i in range(len(labelscore)):
        if(labelscore[i] > 0 or featurescore[i] > 0):
            hide.append(i)
        if(featurescore[i] > 0):
            hidefeature.append(i)

    newhide = []
    newhidefe = []
    for i in hide:
        if i not in newhide:
            newhide.append(i)
    for i in hidefeature:
        if i not in newhidefe:
            newhidefe.append(i)
    if(hide!= []):
        with open('C:/Users/lenovo/Desktop/Test province/score.txt', 'w')as fi:
            fi.write(str(newhide))
    else:
        with open('C:/Users/lenovo/Desktop/Test province/score.txt', "r+") as f:
            read_data = f.read()
            f.seek(0)
            f.truncate()
    if(hidefeature != []):
        with open('C:/Users/lenovo/Desktop/Test province/score-f.txt', 'w')as fi:
            fi.write(str(newhidefe))
    else:
        with open('C:/Users/lenovo/Desktop/Test province/score-f.txt', "r+") as f:
            read_data = f.read()
            f.seek(0)
            f.truncate()

    for i in range(len(newbbox)):  # collision
        F1 = [0, 0] #label collision
        F2 = [0, 0] #feature collision
        F3 = [0, 0] #pull
        F4 = [0, 0] #jiaodian pull
        F5 = [0, 0] #friction
        F6 = [0, 0] #ftime
        F7 = [0, 0] #point closed collision
        vector1 = [0, 0]
        vector2 = [0, 0]
        vector3 = [0, 0]
        vector4 = [0, 0]
        vector5 = [0, 0]
        vector6 = [0, 0]
        hide = []
        for j in range(len(newbbox)):
            if(j != i):                
                if(v == 0):
                    distance1 = np.linalg.norm(newbbox[j].midpoint()-newbbox[i].midpoint())-(
                        newbbox[i].height+newbbox[j].height)/2
                    vector1 = (newbbox[i].midpoint() -
                                newbbox[j].midpoint())/np.linalg.norm(newbbox[i].midpoint() -
                                                                    newbbox[j].midpoint())
                    F1 += min((distance1/m1)-1,0)*(-vector1)
                else:
                    if(priorindex.index(i) < priorindex.index(j)):
                        distance1 = np.linalg.norm(newbbox[j].midpoint()-newbbox[i].midpoint())-(
                            newbbox[i].height+newbbox[j].height)/2
                        vector1 = (newbbox[i].midpoint() -
                                    newbbox[j].midpoint())/np.linalg.norm(newbbox[i].midpoint() -
                                                                        newbbox[j].midpoint())
                        # print(min((distance1/m1)-1, 0)*(-vector1))
                        F1 += min((distance1/m1)-1,0)*(-vector1)
                    else:
                        F1 += numpy.array([0.0, 0.0])

                distance2 = np.linalg.norm(newbbox[i].midpoint()-nanchors[j])-(newbbox[i].height)/2
                vector2 = (newbbox[i].midpoint()-nanchors[j]) / \
                    np.linalg.norm(newbbox[i].midpoint()-nanchors[j])
                F2 += min((distance2/(m2+R+r[i]))-1,0)*(-vector2)

         # pull
        distance3 = np.linalg.norm(newbbox[i].midpoint()-nanchors[i])-(newbbox[i].height)/2
        vector3 = (newbbox[i].midpoint()-nanchors[i]) / \
            np.linalg.norm(newbbox[i].midpoint()-nanchors[i])
        if(distance3 - mdis <= 0):
            F3 = 0
        else:
            F3 += - \
                (math.log(
                    np.linalg.norm(newbbox[i].midpoint()-nanchors[i])-(newbbox[i].height)/2-mdis+1))*vector3

        # point closed collision
        distance6 = (newbbox[i].width)/2+(R+r[i])-np.linalg.norm(newbbox[i].midpoint()-nanchors[i])
        # print(distance6)
        if(distance6 > 0):
            F7 += (math.log(
                np.linalg.norm(abs(newbbox[i].midpoint()-nanchors[i])-((newbbox[i].width)/2+(R+r[i])))+1))*vector3
            # F7 = 0 
        else:
            F7 = 0

        if(i not in numpro):  # jiao dian pull
            F4 = 0
        else:
            midx = nanchors[numpro[k]][0]+float(newj[k][0])
            midy = nanchors[numpro[k]][1]+float(newj[k][1])
            distance4 = max(
                abs(newbbox[numpro[k]].midpoint()-[midx, midy]))-(newbbox[numpro[k]].height)/2
            vector4 = (newbbox[numpro[k]].midpoint()-[midx, midy]) / \
                np.linalg.norm(newbbox[numpro[k]].midpoint() -
                            [midx, midy])
            # F4 = 0
            F4 += - \
                (math.log(
                    np.linalg.norm(abs(newbbox[numpro[k]].midpoint()-[midx, midy])-(newbbox[numpro[k]].height)/2)+1))*vector4
            k += 1
        if(v == 0):
            F5 = 0
            F = clabel*F1+cpoint*F2+cpull*F3+cjiao*F4+cf*F5
            V = 0+F*t
            with open('C:/Users/lenovo/Desktop/Test province/v.txt', 'a')as fi:
                fi.write(str(newbbox[i])+'\n'+str(np.round(V,3))+'\n')
            with open('C:/Users/lenovo/Desktop/Test province/vp.txt', 'a')as fi:
                fi.write(str(nanchors[i])+'\n')
        else:
            # f
            deltapi = np.array(nanchors[i])-[float(vp[i][0]), float(vp[i][1])]
            f5 = -([float(v[i][0]), float(v[i][1])]-(deltapi/t))
            F5 += f5
            # ftime
            for j in range(len(newbbox)):
                if(j != i):
                    deltapj = np.array(
                        nanchors[j])-[float(vp[j][0]), float(vp[j][1])]
                    vi = np.linalg.norm(deltapi/t)
                    vj = np.linalg.norm(deltapj/t)
                    newanchors = nanchors[j] + (deltapj/t)*deltatime
                    distance5 = np.linalg.norm(
                        newbbox[i].midpoint()-newanchors)-(newbbox[i].height)/2
                    vector5 = (newbbox[i].midpoint() - newanchors) / \
                        np.linalg.norm(newbbox[i].midpoint() - newanchors)
                    cosangle = cosVector(deltapi/t, deltapj/t)
                    # print('angle',cosangle)
                    F6 += vector5 * math.log(max(abs(vj/vi), abs(vi/vj))+1) * deltatime*max(1-(distance5/m3), 0)*(math.pow(math.e,cosangle))
                    # F6 = 0

                    with open('C:/Users/lenovo/Desktop/Test province/deltat.txt', 'a')as fi:
                        fi.write(
                            str(date)+' '+str(text_strings[i])+str(text_strings[j])+str(F6)+'\n')
            F = clabel*F1+cpoint*F2+cpull*F3+cjiao*F4+cf*F5+ctime*F6+cclosed*F7
            V = [float(v[i][0]), float(v[i][1])]+F*t
            with open('C:/Users/lenovo/Desktop/Test province/v.txt', 'a')as fi:
                fi.write(str(newbbox[i])+'\n'+str(np.round(V,3))+'\n')
            with open('C:/Users/lenovo/Desktop/Test province/vp.txt', 'a')as fi:
                fi.write(str(nanchors[i])+'\n')
        # print(text_strings[i], clabel*F1, cpoint*F2, cpull*F3, cjiao*F4, cf*F5, ctime*F6,cclosed*F7)
        # print(i+1, 'Flabel,Fpoint,Fpull,Fj,Ff,F', clabel*F1, cpoint*F2, cpull*F3, cpull*F4,cf*F5,F)
    # mx = []
    # my = []
    # px = []
    # py = []
    # yueshu = []
    # # pro =[1,2]
    # # for i in range(len(pro)):
    # #     px.append(nanchors[pro[i]][0])
    # #     py.append(nanchors[pro[i]][1])
    # #     mx.append(newbbox[pro[i]].midpoint()[0])
    # #     my.append(newbbox[pro[i]].midpoint()[1])

    # # for i in range(len(pro)):
    # #     with open('C:/Users/lenovo/Desktop/Test province/shuzu.txt', 'a')as fi:
    # #         fi.write(str(round(mx[i]-px[i], 2))+',' +
    # #                  str(round(my[i]-py[i], 2))+',')

    # for i in range(len(numpro)):
    #     px.append(nanchors[numpro[i]][0])
    #     py.append(nanchors[numpro[i]][1])
    #     mx.append(newbbox[numpro[i]].midpoint()[0])
    #     my.append(newbbox[numpro[i]].midpoint()[1])

    # for i in range(len(numpro)):
    #     with open('C:/Users/lenovo/Desktop/Test province/shuzu.txt', 'a')as fi:
    #         fi.write(str(round(mx[i]-px[i],2))+','+str(round(my[i]-py[i],2))+',')

def _initialise(ax, anchor_x, anchor_y, labels, time, padding):
    renderer = ax.get_figure().canvas.get_renderer()

    # force a draw to get accurate axis bound
    ax.draw(renderer)
    axes_bbox = Box(*ax.get_window_extent(renderer).get_points().ravel())

    text_objects = []
    text_strings = []
    anchors = []
    t_o = []
    other_points = []
    for x, y, label in zip(anchor_x, anchor_y, labels):
        point = np.array([x, y])
        if label:
            anchor = np.array([x, y])
            anchors.append(anchor)
            text_objects.append(ax.text(x, y, label, size=12))
            text_strings.append(label)
        else:
            other_points.append(point)

    # Trigger draw event so text boxes get correct locations
    ax.draw(renderer)
    trans_data = ax.transData
    anchors_transformed = []
    for anchor in anchors:
        anchors_transformed.append(trans_data.transform(anchor))

    label_bboxes = collect_text_bboxes(text_objects, renderer, padding=padding)
    # Delete all the text objects, we don't need them no more for now #删除textobjects
    for text_obj in text_objects:
        text_obj.remove()

    # Collect other bboxes to avoid
    other_bboxes = collect_point_bboxes(ax,
                                        anchors + other_points,
                                        padding=padding)
    r = collect_point_r(ax,
                        anchors + other_points,
                        padding=padding)
    
    return anchors, anchors_transformed, axes_bbox, label_bboxes, other_bboxes, text_strings, text_objects,r
