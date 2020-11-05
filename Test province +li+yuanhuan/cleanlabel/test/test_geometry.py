import unittest
import numpy as np
from scipy.spatial import Rectangle
from numpy.testing import assert_array_equal

from cleanlabel.c_geometry import line_intersection
from cleanlabel.geometry import Box
import itertools

class TestBox(unittest.TestCase):

    def test_width(self):

        box = Box(-1.3, -5.4, 2.4, -3.2)

        expected_width = 2.4 - (-1.3)
        self.assertEqual(expected_width, box.width)

    def test_height(self):
        box = Box(-1.3, -5.4, 2.4, -3.2)

        expected_height = -3.2 - (-5.4)
        self.assertEqual(expected_height, box.height)

    def test_midpoint(self):
        box = Box(-1.3, -5.4, 2.4, -3.2)

        expected_midpoint = np.array([-1.3 + (2.4-(-1.3))/2.0,
                                      -5.4 + (-3.2 - (-5.4))/2.0])

        assert_array_equal(box.midpoint(), expected_midpoint)

    def test_move_midpoint(self):
        box = Box(-1.3, -5.4, 2.4, -3.2)

        new_midpoint = np.array([-3.1, 17.0])

        new_box = box.move_midpoint_to(new_midpoint)

        assert_array_equal(new_midpoint, new_box.midpoint())
        self.assertAlmostEqual(box.width, new_box.width)
        self.assertAlmostEqual(box.height, new_box.height)

    def test_area(self):
        box = Box(-1.3, -5.4, 2.4, -3.2)

        area = box.width * box.height

        self.assertEqual(area, box.area())

    def test_distance_to_point(self):
        box = Box(-1.3, -5.4, 2.4, -3.2)
        point = np.array([-10, 13])

        closest_point = np.array([box.x_min,
                                  box.y_max])

        expected_distance = np.linalg.norm(closest_point - point)

        self.assertEqual(expected_distance, box.distance_to_point(point))

    def test_distance_to_point_inside_box(self):
        box = Box(0, 0, 1, 1)
        point = np.array([0.1, 0.3])

        expected_distance = 0
        self.assertEqual(expected_distance, box.distance_to_point(point))

    def test_angle_to_point(self):

        point = [0, 0]

        ur = Box(1, 1, 2, 2)
        angle_ur = np.pi / 4

        ul = Box(-2, 1, -1, 2)
        angle_ul = 3/4 * np.pi

        ll = Box(-2, -2, -1, -1)
        angle_ll = -3/4 * np.pi

        lr = Box(1, -2, 2, -1)
        angle_lr = -np.pi / 4

        self.assertAlmostEqual(angle_ur, ur.angle_to_point(point))
        self.assertAlmostEqual(angle_ul, ul.angle_to_point(point))
        self.assertAlmostEqual(angle_ll, ll.angle_to_point(point))
        self.assertAlmostEqual(angle_lr, lr.angle_to_point(point))


    def test_anchoring_position(self):
        step = np.pi/16

        positions = list(range(17)) + list(range(-15, 0))

        point = np.array([1, -2])
        box = Box(0, 0, 1, 1)

        for position in positions:
            angle = step * position
            new_box = box.move_midpoint_to_angle(point, angle, new_distance=3)

            combinations = itertools.product([(new_box.x_min, 'left'),
                                              (new_box.x_max, 'right'),
                                              (new_box.midpoint()[0], 'center'),
                                              ],
                                             [(new_box.y_min, 'bottom'),
                                              (new_box.y_max, 'top'),
                                              (new_box.midpoint()[1], 'middle'),
                                              ],
                                             )

            best_combination = None
            best_distance = np.inf

            for (x, x_align), (y, y_align) in combinations:
                distance = np.linalg.norm(np.array([x, y]) - point)
                if distance < best_distance:
                    best_combination = (x, y), x_align, y_align
                    best_distance = distance

            actual_anchor = new_box.anchoring_position(point)

            self.assertEqual(best_combination[1:], actual_anchor[1:],
                             'Wrong alignment for pos {}: {!r} != {!r}'.format(position,
                                                                                 best_combination,
                                                                                 actual_anchor))

            assert_array_equal(best_combination[0], actual_anchor[0],
                               'Wrong ans for pos {}: {!r} != {!r}'.format(position,
                                                                           best_combination,
                                                                           actual_anchor))

    def test_overlap(self):
        box1 = Box(-5, -3, 6, 1)
        box2 = Box(-2, 0, 1, 3)

        box3 = Box(-7, -5, -3, -2)

        intersection12 = Box(-2, 0, 1, 1).area()
        intersection13 = Box(-5, -3, -3, -2).area()

        intersection23 = 0.0

        self.assertEqual(box1.overlap(box2), intersection12)
        self.assertEqual(box2.overlap(box1), intersection12)

        self.assertEqual(box1.overlap(box3), intersection13)
        self.assertEqual(box3.overlap(box1), intersection13)

        self.assertEqual(box2.overlap(box3), intersection23)
        self.assertEqual(box3.overlap(box2), intersection23)

        self.assertEqual(box1.overlap(box1), box1.area())

    def test_line_intersection(self):
        line1 = [[-2, -1], [3, 2]]
        line2 = [[1, 2], [3, 1]]
        line3 = [[1, -1], [3, -2]]

        def _lines_to_args(a, b):
            a = np.asarray(a)
            b = np.asarray(b)

            return list(a.ravel()) + list(b.ravel())

        self.assertTrue(line_intersection(*_lines_to_args(line1, line2)))
        self.assertFalse(line_intersection(*_lines_to_args(line1, line3)))

    def test_line_intersection_parallel_lines(self):

        line1 = [[1, 1], [2, 1]]
        line2 = [[1, 2], [2, 2]]
        line3 = [[0, 1], [2, 1]]

        def _lines_to_args(a, b):
            a = np.asarray(a)
            b = np.asarray(b)

            return list(a.ravel()) + list(b.ravel())

        self.assertFalse(line_intersection(*_lines_to_args(line1, line2)))
        # TODO: this case not handled yet
        #self.assertTrue(line_intersection(*_lines_to_args(line1, line3)))
        self.assertFalse(line_intersection(*_lines_to_args(line2, line3)))




