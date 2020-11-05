from functools import lru_cache

import numpy as np
import pyqtree
from matplotlib.patches import Rectangle as RectanglePatch
from matplotlib.patches import Wedge as WedgePatch
from cleanlabel.c_geometry import bbox_overlap
import pyximport

class Box(object):
    __slots__ = ('x_min', 'y_min', 'x_max', 'y_max', '_midpoint',
                 '_scipy_rectangle', '_anchoring_position_cache', 'mins', 'maxes',
                 '_nearest_point_cache',
                 '_area')

    def __init__(self, x_min, y_min, x_max, y_max):

        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

        # min/maxes
        self.mins = np.array([self.x_min, self.y_min])
        self.maxes = np.array([self.x_max, self.y_max])

        # Be lazy and do not compute if not needed
        self._midpoint = None
        self._scipy_rectangle = None
        self._anchoring_position_cache = {}
        self._area = None
        self._nearest_point_cache = {}

    def __repr__(self):
        return '{}({}, {}, {}, {})'.format(self.__class__.__name__,
                                           self.x_min, self.y_min, self.x_max, self.y_max)

    @property
    def width(self):
        return self.x_max - self.x_min

    @property
    def height(self):
        return self.y_max - self.y_min

    @property
    def points(self):
        return self.x_min, self.y_min, self.x_max, self.y_max

    def area(self):
        if self._area is None:
            self._area = self.width * self.height
        return self._area

    def midpoint(self):  # 标签中点
        if self._midpoint is None:
            x0, y0, x1, y1 = self.points
            self._midpoint = np.array(
                [x0 + (x1 - x0) / 2.0, y0 + (y1 - y0) / 2.0])
        return self._midpoint
    def vector_midpoint_to_point(self, point):
        midpoint = self.midpoint()
        return midpoint - point
    def angle_to_point(self, point):  # 标签中点到点的角度
        vec = self.vector_midpoint_to_point(point)
        return np.arctan2(vec[1], vec[0])
    def position_score(self, point):
        a = self.angle_to_point(point)
        if 0 <= a <= np.pi / 2:
            return 0
        elif np.pi / 2 < a <= np.pi:
            return 1
        elif -np.pi <= a < -np.pi / 2:
            return 2
        else:
            return 3

    def distance_to_point(self, point):#标签与点的距离
        closest_point = self.nearest_point(point)
        # print(closest_point,point)
        vector = closest_point - point
        return np.linalg.norm(vector) #求2范数即长度

    def nearest_point(self, point):
        """
        Inspired by http://stackoverflow.com/a/20453634
        :return:
        """
        key = tuple(point)
        # print(key)
        try:
            return self._nearest_point_cache[key]
        except KeyError:
            self._nearest_point_cache[key] = ener = np.maximum(self.mins,
                                                                np.minimum(self.maxes, point))#逐次比较选择最大
            return ener

            # return _nearest_point(*self.points, x_point=point[0], y_point=point[1])

    def anchoring_position(self, point):
        """
        Returns the anchoring position to the point based
        on the verticalalign and horizontalalign properties in matplotlib

        :param point: point to anchor to
        :return: tuple (closest_point, horizontalalign, verticalalign)
        """
        try:
            return self._anchoring_position_cache[tuple(point)]
        except KeyError:

            x_point, y_point = point

            x_min, y_min, x_max, y_max = self.points
            x_mid, y_mid = self.midpoint()

            x, x_anchor = min(zip([x_min, x_max, x_mid], ['left', 'right', 'center']),
                              key=lambda xx: np.abs(xx[0] - x_point))#按维度找最小值

            y, y_anchor = min(zip([y_min, y_max, y_mid], ['bottom', 'top', 'middle']),
                              key=lambda yy: np.abs(yy[0] - y_point))

            closest = np.array([x, y])
            ener = closest, x_anchor, y_anchor
            print(ener)
            return ener

    
    def overlap(self, other_box):
        # from matplotlib.transforms.Bbox.intersection

        self_xmin = self.x_min
        self_xmax = self.x_max
        self_ymax = self.y_max
        self_ymin = self.y_min

        other_ymax = other_box.y_max
        other_ymin = other_box.y_min
        other_xmax = other_box.x_max
        other_xmin = other_box.x_min

        return bbox_overlap(self_xmin, self_ymin, self_xmax, self_ymax,
                            other_xmin, other_ymin,
                            other_xmax, other_ymax)

    def __eq__(self, other):
        if not isinstance(other, Box):
            return False

        return self.points == other.points

    def translate(self, delta_x, delta_y):
        x0, y0, x1, y1 = self.points
        return Box(x0 + delta_x, y0 + delta_y,
                   x1 + delta_x, y1 + delta_y)
    
    def expand(self,padding):
        x0,y0,x1,y1 = self.x_min,self.y_min,self.x_max,self.y_max
        # print(Box(x0-padding, y0-padding, x1+padding, y1+padding))
        return Box(x0-padding,y0-padding,x1+padding,y1+padding)
    def rectangle_patch(self, **kwargs):
        x0, y0 = self.x_min, self.y_min
        width, height = self.width, self.height
        return RectanglePatch((x0, y0), width, height, **kwargs)
        
    def Wedge_patch(self, x, y, r, theta1, theta2, width, **kwargs):
        return WedgePatch((x, y), r, theta1, theta2, width, ** kwargs)

    def rotate_around_point(self, point, angle_radians, return_delta=False):

        midpoint = self.midpoint().copy()

        # Translate so the rotation point is 0, 0
        midpoint -= point

        sin_ = np.sin(angle_radians)
        cos_ = np.cos(angle_radians)

        new_midpoint = np.array([cos_ * midpoint[0] - sin_ * midpoint[1],
                                 sin_ * midpoint[0] + cos_ * midpoint[1]])

        # check how much we had to move
        delta = new_midpoint - midpoint

        if return_delta:
            return delta
        else:
            return self.translate(*delta)

    def move_midpoint_to(self, new_midpoint):
        half_width = self.width / 2.0
        half_height = self.height / 2.0
        offset = np.array([half_width, half_height])
        point_a = new_midpoint - offset
        point_b = new_midpoint + offset
        return Box(point_a[0], point_a[1], point_b[0], point_b[1])

    def move_midpoint_to_angle(self, anchor_point, new_angle, new_distance):
        anchor_point = np.asarray(anchor_point)
        offset = np.array([np.cos(new_angle) * new_distance, np.sin(new_angle) * new_distance])
        new_midpoint = anchor_point + offset

        new_box = self.move_midpoint_to(new_midpoint)
        return new_box

    def flip(self, val, axis=0):
        x0, y0, x1, y1 = self.points

        if axis == 0:
            x0 = x0 - val
            x1 = x1 - val

            x0 *= -1.0
            x1 *= -1.0

            x0 += val
            x1 += val

            x0, x1 = x1, x0

        elif axis == 1:
            y0 = y0 - val
            y1 = y1 - val

            y0 *= -1.0
            y1 *= -1.0

            y0 += val
            y1 += val

            y0, y1 = y1, y0

        return Box(x0, y0, x1, y1)

    def __hash__(self):
        return hash(self.points)

    def area_outside_of_box(self, box):
        return self.area() - self.overlap(box)


# def bbox_overlap(xmin0, ymin0, xmax0,ymax0,xmin1,ymin1,xmax1,ymax1):
#     #标签重叠面积(矩形)
#     intersects = not (xmin1 > xmax0 or
#                         xmax1 < xmin0 or
#                         ymin1 > ymax0 or
#                         ymax1 < ymin0)
#     ener = 0.0
#     if intersects:
#         x0 = max(xmin0, xmin1)
#         x1 = min(xmax0, xmax1)
#         y0 = max(ymin0, ymin1)
#         y1 = min(ymax0, ymax1)
#         ener = (x1 - x0) * (y1 - y0)
#     return ener
        
def make_qtree(boxes, **kwargs):
    xmin = min(map(lambda x: x.x_min, boxes))
    xmax = max(map(lambda x: x.x_max, boxes))
    ymin = min(map(lambda x: x.y_min, boxes))
    ymax = max(map(lambda x: x.y_max, boxes))

    bbox = (xmin, ymin, xmax, ymax)

    qtree = pyqtree.Index(bbox, **kwargs)
    for box in boxes:
        qtree.insert(box, box.points)
        # print(qtree.insert(box, box.points))
    return qtree
