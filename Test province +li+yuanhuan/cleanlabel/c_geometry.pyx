import numpy as np
cimport numpy as np
import cython

cdef inline np.float64_t min_(np.float64_t a, np.float64_t b): return a if a <= b else b
cdef inline np.float64_t max_(np.float64_t a, np.float64_t b): return a if a >= b else b

@cython.cdivision(True)
def line_intersection(np.float64_t x1,
                      np.float64_t y1,
                      np.float64_t x2,
                      np.float64_t y2,
                      np.float64_t x3,
                      np.float64_t y3,
                      np.float64_t x4,
                      np.float64_t y4):

    # Improved version of https://github.com/tinker10/D3-Labeler/blob/master/labeler.js
    # Based on https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection

    cdef np.float64_t denom, numerator_x, numerator_y, x, y
    cdef int ener

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if denom != 0.0:
        # Lines are not parallel

        numerator_x = (x1*y2 - y1*x2) * (x3-x4) - (x1-x2) * (x3*y4-y3*x4)
        numerator_y = (x1*y2 - y1*x2) * (y3-y4) - (y1-y2) * (x3*y4 - y3*x4)

        x = numerator_x / denom
        y = numerator_y / denom

        ener = x >= min_(x1, x2) and x >= min_(x3, x4)
        ener &= x <= max_(x1, x2) and x <= max_(x3, x4)

        ener = y >= min_(y1, y2) and y >= min_(y3, y4)
        ener &= y <= max_(y1, y2) and y <= max_(y3, y4)
    else:
        # Lines are parallel
        # assume they do not intersect
        # TODO: handle special case where they overlap, this should be like intersection
        ener = 0
    return ener

def bbox_overlap(np.float64_t xmin0,
                 np.float64_t ymin0,
                 np.float64_t xmax0,
                 np.float64_t ymax0,
                 np.float64_t xmin1,
                 np.float64_t ymin1,
                 np.float64_t xmax1,
                 np.float64_t ymax1):
    cdef int intersects
    cdef np.float64_t x0, x1, y0, y1, ener,mid0x,mid0y,mid1x,mid1y

    intersects = not (xmin1 > xmax0 or
                      xmax1 < xmin0 or
                      ymin1 > ymax0 or
                      ymax1 < ymin0)
    ener = 0.0

    if intersects:
        x0 = max_(xmin0, xmin1)
        x1 = min_(xmax0, xmax1)
        y0 = max_(ymin0, ymin1)
        y1 = min_(ymax0, ymax1)
        
        ener = (x1 - x0) * (y1 - y0)
    return ener

