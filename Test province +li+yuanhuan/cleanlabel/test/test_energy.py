import unittest

from cleanlabel.cleanlabels import point_bounding_box
from cleanlabel.energy import fast_overlap
from cleanlabel.geometry import Box, make_qtree
import numpy as np


class TestFastOverlap(unittest.TestCase):

    def test_fast_overlap_calculation(self):

        box = Box(1, 1, 3, 3)
        random = np.random.RandomState(42)

        points = random.randn(50, 2) + 2
        point_bboxes = []
        for i in range(len(points)):
            point_bboxes.append(point_bounding_box(points[i], padding=0.05))

        expected_overlap = 0
        for point_bbox in point_bboxes:
            expected_overlap += box.overlap(point_bbox)

        qtree = make_qtree(point_bboxes)
        overlap = fast_overlap(box, qtree)

        self.assertAlmostEqual(expected_overlap, overlap)
