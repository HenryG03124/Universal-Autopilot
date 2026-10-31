import unittest
from unittest.mock import patch

import numpy as np

from MyAutoPilot.auto_brakes import AutoBrakes
from MyAutoPilot.lane_refiner import LaneRefiner


class LaneRefinerTests(unittest.TestCase):
    def test_lanes_are_unavailable_before_first_frame(self):
        refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
        self.assertEqual(refiner.lane_available , {"current": False , "L": False , "R": False})

    def test_missing_boundaries_reset_each_lane_available(self):
        cases = [
            ([100] , {"current": True , "L": False , "R": True}) ,
            ([540] , {"current": True , "L": True , "R": False}) ,
            ([100 , 220] , {"current": False , "L": False , "R": True}) ,
            ([420 , 540] , {"current": False , "L": True , "R": False}) ,
            ([100 , 220 , 420 , 540] , {"current": False , "L": False , "R": False})
        ]
        for missing_x_values , expected in cases:
            with self.subTest(missing_x_values = missing_x_values):
                refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
                mask = np.zeros((352 , 640) , dtype = np.uint8)
                for x in (100 , 220 , 420 , 540):
                    mask[200 : 321 , x] = 3
                refiner.refine(mask , 2 , 10 , 30)
                self.assertEqual(refiner.lane_available , {"current": True , "L": True , "R": True})

                for x in missing_x_values:
                    mask[: , x] = 0
                refiner.refine(mask , 2 , 10 , 30 , return_sample = True)
                self.assertEqual(refiner.lane_available , expected)

    def test_neighbour_lanes_must_cover_far_target(self):
        refiner = LaneRefiner(200 , 320 , 10 , 260 , 300)
        mask = np.zeros((352 , 640) , dtype = np.uint8)
        for x in (100 , 220 , 420 , 540):
            mask[200 : 321 , x] = 3
        _ , _ , nl_lane_points , nr_lane_points = refiner.refine(mask , 2 , 10 , 30)
        self.assertTrue(nl_lane_points)
        self.assertTrue(nr_lane_points)
        self.assertEqual(refiner.lane_available , {"current": True , "L": False , "R": False})

    def test_crossed_fitted_boundaries_are_unavailable(self):
        refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
        mask = np.zeros((352 , 640) , dtype = np.uint8)
        cases = [
            ([[220 , 200] , [430 , 248]] , [[420 , 200] , [420 , 248]] , [[100 , 200] , [100 , 248]] , [[540 , 200] , [540 , 248]] , {"current": False , "L": True , "R": True}) ,
            ([[220 , 200] , [220 , 248]] , [[420 , 200] , [420 , 248]] , [[100 , 200] , [220 , 248]] , [[540 , 200] , [540 , 248]] , {"current": True , "L": False , "R": True}) ,
            ([[220 , 200] , [220 , 248]] , [[420 , 200] , [420 , 248]] , [[100 , 200] , [100 , 248]] , [[540 , 200] , [400 , 248]] , {"current": True , "L": True , "R": False})
        ]
        for cl_lane_points , cr_lane_points , nl_lane_points , nr_lane_points , expected in cases:
            with self.subTest(expected = expected):
                with patch.object(refiner , "fit_lane_points" , side_effect = [cl_lane_points , cr_lane_points , nl_lane_points , nr_lane_points]):
                    refiner.refine(mask , 2 , 10 , 30)
                self.assertEqual(refiner.lane_available , expected)

    def test_cached_width_and_brake_lanes_do_not_restore_available(self):
        refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
        brakes = AutoBrakes()
        mask = np.zeros((352 , 640) , dtype = np.uint8)
        mask[200 : 321 , 220] = 3
        mask[200 : 321 , 420] = 3
        for frame in range(5):
            cl_lane_points , cr_lane_points , _ , _ = refiner.refine(mask , 2 , 10 , 30)
            refiner.calculate_center_points(cl_lane_points , cr_lane_points)
            brakes.update_lanes(cl_lane_points , cr_lane_points , frame * 0.1 , refiner.lane_available["current"])

        mask[: , 220] = 0
        cl_lane_points , cr_lane_points , _ , _ = refiner.refine(mask , 2 , 10 , 30)
        center_points = refiner.calculate_center_points(cl_lane_points , cr_lane_points)
        brakes.update_lanes(cl_lane_points , cr_lane_points , 0.5 , refiner.lane_available["current"])
        self.assertTrue(center_points)
        self.assertTrue(refiner.prev_lane_width)
        self.assertEqual(brakes.lane_status , "held")
        self.assertTrue(brakes.cl_lane_points)
        self.assertFalse(refiner.lane_available["current"])
        self.assertFalse(refiner.lane_available["L"])

    def test_width_updates_after_five_tracking_frames(self):
        refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
        mask = np.zeros((352 , 640) , dtype = np.uint8)
        mask[200 : 321 , 200] = 3
        mask[200 : 321 , 440] = 3

        for frame in range(1 , 6):
            cl_lane_points , cr_lane_points , _ , _ = refiner.refine(mask , 2 , 10 , 30)
            center_points = refiner.calculate_center_points(cl_lane_points , cr_lane_points)
            self.assertEqual(refiner.tracking_frames , frame)
            self.assertEqual(center_points , [[320 , y] for y in range(320 , 199 , -1)])
            if frame < 5:
                self.assertEqual(refiner.prev_lane_width , [])
            else:
                self.assertEqual(refiner.prev_lane_width , [[240 , y] for y in range(320 , 199 , -1)])

        mask[: , 440] = 0
        mask[200 : 321 , 460] = 3
        cl_lane_points , cr_lane_points , _ , _ = refiner.refine(mask , 2 , 10 , 30 , return_sample = True)[4 : ]
        refiner.calculate_center_points(cl_lane_points , cr_lane_points)
        self.assertEqual(refiner.tracking_frames , 6)
        self.assertEqual(refiner.prev_lane_width , [[260 , y] for y in range(320 , 199 , -1)])

    def test_fitted_curvature_uses_absolute_second_derivative(self):
        for coefficient in (-0.002 , 0.002):
            with self.subTest(coefficient = coefficient):
                refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
                sample_points = [[220 + coefficient * (y - 260) ** 2 , y] for y in range(200 , 321 , 4)]
                lane_points = refiner.fit_lane_points(sample_points , 352 , 640 , 2 , "CL")
                self.assertTrue(lane_points)
                self.assertAlmostEqual(refiner.lane_curvature["CL"] , 0.004)
                np.testing.assert_allclose(refiner.lane_2nd_derivative["CL"] , 2 * coefficient)

    def test_failed_fit_clears_only_current_lane_curvature(self):
        refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
        self.assertEqual(refiner.lane_curvature , {"CL" : None , "CR" : None , "NL" : None , "NR" : None})
        self.assertEqual(refiner.lane_2nd_derivative , {"CL" : None , "CR" : None , "NL" : None , "NR" : None})
        self.assertIsNone(refiner.lane_2nd_derivative_diff)
        refiner.lane_curvature["CR"] = 0.003
        refiner.lane_2nd_derivative["CR"] = np.array([0.003])
        sample_points = [[220 + 0.002 * (y - 260) ** 2 , y] for y in range(200 , 321 , 4)]
        for missing_points in ([] , [[220 , 250] , [221 , 260] , [222 , 270]]):
            with self.subTest(missing_points = missing_points):
                refiner.fit_lane_points(sample_points , 352 , 640 , 2 , "CL")
                self.assertAlmostEqual(refiner.lane_curvature["CL"] , 0.004)
                refiner.lane_2nd_derivative_diff = 0.001
                self.assertEqual(refiner.fit_lane_points(missing_points , 352 , 640 , 2 , "CL") , [])
                self.assertIsNone(refiner.lane_curvature["CL"])
                self.assertIsNone(refiner.lane_2nd_derivative["CL"])
                self.assertIsNone(refiner.lane_2nd_derivative_diff)
                self.assertEqual(refiner.lane_curvature["CR"] , 0.003)
                np.testing.assert_array_equal(refiner.lane_2nd_derivative["CR"] , [0.003])

        refiner.lane_2nd_derivative_diff = 0.001
        self.assertEqual(refiner.calculate_center_points([] , []) , [])
        self.assertIsNone(refiner.lane_2nd_derivative_diff)

    def test_second_derivative_difference_supports_linear_and_cubic_fits(self):
        for degree , coefficient in ((1 , 0) , (3 , 0.00001)):
            with self.subTest(degree = degree):
                refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
                cl_sample_points = [[200 + 0.5 * (y - 260) + coefficient * (y - 260) ** 3 , y] for y in range(200 , 321 , 4)]
                cr_sample_points = [[400 + 0.5 * (y - 260) - coefficient * (y - 260) ** 3 , y] for y in range(200 , 321 , 4)]
                cl_lane_points = refiner.fit_lane_points(cl_sample_points , 352 , 640 , degree , "CL")
                cr_lane_points = refiner.fit_lane_points(cr_sample_points , 352 , 640 , degree , "CR")
                expected = 6 * coefficient * (np.arange(200 , 321) - 260)
                np.testing.assert_allclose(refiner.lane_2nd_derivative["CL"] , expected , atol = 1e-12)
                np.testing.assert_allclose(refiner.lane_2nd_derivative["CR"] , -expected , atol = 1e-12)
                refiner.calculate_center_points(cl_lane_points , cr_lane_points)
                self.assertAlmostEqual(refiner.lane_2nd_derivative_diff , 12 * coefficient * 60)

    def test_width_accepts_narrowing_and_rounding_platforms(self):
        refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
        refiner.tracking_frames = 5
        cl_lane_points = [[200 , y] for y in range(320 , 199 , -1)]
        cr_lane_points = [[400 - (320 - y) // 3 , y] for y in range(320 , 199 , -1)]
        refiner.update_lane_width(cl_lane_points , cr_lane_points)
        self.assertFalse(refiner.abnormal_state)
        self.assertEqual(refiner.prev_lane_width , [[200 - (320 - y) // 3 , y] for y in range(320 , 199 , -1)])

    def test_width_anomaly_counts_adjacent_increases_above_ten_percent(self):
        cases = [
            ([200 , 201 , 200 , 199 , 198 , 197 , 196 , 195 , 194 , 193 , 192] , False) ,
            ([200 , 201 , 202 , 201 , 200 , 199 , 198 , 197 , 196 , 195 , 194] , True) ,
            ([200 , 201 , 200 , 199 , 198 , 197 , 196 , 195 , 194 , 193] , True)
        ]
        for widths , expected in cases:
            with self.subTest(widths = widths):
                refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
                refiner.tracking_frames = 5
                previous_width = [[240 , 320 - i] for i in range(len(widths))]
                refiner.prev_lane_width = previous_width[:]
                cl_lane_points = [[200 , 320 - i] for i in range(len(widths))]
                cr_lane_points = [[200 + lane_width , 320 - i] for i , lane_width in enumerate(widths)]
                refiner.update_lane_width(cl_lane_points , cr_lane_points)
                self.assertEqual(refiner.abnormal_state , expected)
                if expected:
                    self.assertEqual(refiner.prev_lane_width , previous_width)
                else:
                    self.assertEqual(refiner.prev_lane_width , [[lane_width , 320 - i] for i , lane_width in enumerate(widths)])

    def test_intersection_prediction_selects_minimum_or_maximum_curvature_side(self):
        cases = [
            ("min" , 0.001 , 0.004 , "CL") ,
            ("min" , 0.004 , 0.001 , "CR") ,
            ("max" , 0.001 , 0.004 , "CR") ,
            ("max" , 0.004 , 0.001 , "CL")
        ]
        for selection , cl_curvature , cr_curvature , selected_side in cases:
            with self.subTest(selection = selection , selected_side = selected_side):
                refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
                refiner.tracking_frames = 5
                previous_width = [[200 , y] for y in range(320 , 199 , -1)]
                refiner.prev_lane_width = previous_width[:]
                refiner.lane_curvature["CL"] = cl_curvature
                refiner.lane_curvature["CR"] = cr_curvature
                refiner.lane_2nd_derivative["CL"] = np.array([cl_curvature])
                refiner.lane_2nd_derivative["CR"] = np.array([cr_curvature])
                cl_lane_points = [[200 , y] for y in range(320 , 199 , -1)]
                cr_lane_points = [[400 + 320 - y , y] for y in range(320 , 199 , -1)]
                with patch("MyAutoPilot.lane_refiner.select_lane_curvature" , selection):
                    center_points = refiner.calculate_center_points(cl_lane_points , cr_lane_points)
                if selected_side == "CL":
                    self.assertEqual(center_points , [[300 , y] for y in range(320 , 199 , -1)])
                else:
                    self.assertEqual(center_points , [[300 + 320 - y , y] for y in range(320 , 199 , -1)])
                self.assertTrue(refiner.abnormal_state)
                self.assertEqual(refiner.prev_lane_width , previous_width)
                self.assertEqual(refiner.status["center"] , "predicted")

    def test_intersection_prediction_requires_derivative_difference_and_abnormal_state(self):
        cases = [(False , 0.004) , (True , 0.001) , (True , 0.003)]
        for abnormal_state , curvature in cases:
            with self.subTest(abnormal_state = abnormal_state , curvature = curvature):
                refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
                refiner.tracking_frames = 5
                previous_width = [[180 , y] for y in range(320 , 199 , -1)]
                refiner.prev_lane_width = previous_width[:]
                refiner.lane_curvature["CL"] = 0.001
                refiner.lane_curvature["CR"] = curvature
                refiner.lane_2nd_derivative["CL"] = np.array([0.001])
                refiner.lane_2nd_derivative["CR"] = np.array([curvature])
                cl_lane_points = [[200 , y] for y in range(320 , 199 , -1)]
                if abnormal_state:
                    cr_lane_points = [[400 + 320 - y , y] for y in range(320 , 199 , -1)]
                    expected = [[(600 + 320 - y) // 2 , y] for y in range(320 , 199 , -1)]
                else:
                    cr_lane_points = [[400 , y] for y in range(320 , 199 , -1)]
                    expected = [[300 , y] for y in range(320 , 199 , -1)]
                center_points = refiner.calculate_center_points(cl_lane_points , cr_lane_points)
                self.assertEqual(refiner.abnormal_state , abnormal_state)
                self.assertEqual(center_points , expected)
                self.assertEqual(refiner.status["center"] , "accurate")
                self.assertEqual(refiner.prev_lane_width , previous_width)

    def test_derivative_difference_preserves_signs(self):
        cases = [(0.004 , -0.004 , True) , (0.004 , 0.005 , True) , (0.004 , 0.005 , False)]
        for cl_derivative , cr_derivative , abnormal_state in cases:
            with self.subTest(cr_derivative = cr_derivative , abnormal_state = abnormal_state):
                refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
                refiner.tracking_frames = 5
                refiner.prev_lane_width = [[180 , y] for y in range(320 , 199 , -1)]
                refiner.lane_curvature["CL"] = abs(cl_derivative)
                refiner.lane_curvature["CR"] = abs(cr_derivative)
                refiner.lane_2nd_derivative["CL"] = np.array([cl_derivative])
                refiner.lane_2nd_derivative["CR"] = np.array([cr_derivative])
                cl_lane_points = [[200 , y] for y in range(320 , 199 , -1)]
                if abnormal_state:
                    cr_lane_points = [[400 + 320 - y , y] for y in range(320 , 199 , -1)]
                else:
                    cr_lane_points = [[400 , y] for y in range(320 , 199 , -1)]
                center_points = refiner.calculate_center_points(cl_lane_points , cr_lane_points)
                self.assertAlmostEqual(refiner.lane_2nd_derivative_diff , abs(cl_derivative - cr_derivative))
                if cr_derivative < 0:
                    self.assertEqual(center_points , [[290 , y] for y in range(320 , 199 , -1)])
                    self.assertEqual(refiner.status["center"] , "predicted")
                else:
                    self.assertEqual(refiner.status["center"] , "accurate")
                    if not abnormal_state:
                        self.assertEqual(center_points , [[300 , y] for y in range(320 , 199 , -1)])
                        self.assertEqual(refiner.prev_lane_width , [[200 , y] for y in range(320 , 199 , -1)])

    def test_intersection_without_previous_width_has_no_center(self):
        refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
        refiner.tracking_frames = 5
        refiner.lane_curvature["CL"] = 0.001
        refiner.lane_curvature["CR"] = 0.004
        refiner.lane_2nd_derivative["CL"] = np.array([0.001])
        refiner.lane_2nd_derivative["CR"] = np.array([0.004])
        cl_lane_points = [[200 , y] for y in range(320 , 199 , -1)]
        cr_lane_points = [[400 + 320 - y , y] for y in range(320 , 199 , -1)]
        self.assertEqual(refiner.calculate_center_points(cl_lane_points , cr_lane_points) , [])
        self.assertEqual(refiner.status["center"] , "missing")
        self.assertEqual(refiner.prev_lane_width , [])

    def test_intersection_recovers_before_history_can_update(self):
        refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
        refiner.tracking_frames = 2
        previous_width = [[200 , y] for y in range(320 , 199 , -1)]
        refiner.prev_lane_width = previous_width[:]
        refiner.lane_curvature["CL"] = 0.001
        refiner.lane_curvature["CR"] = 0.004
        refiner.lane_2nd_derivative["CL"] = np.array([0.001])
        refiner.lane_2nd_derivative["CR"] = np.array([0.004])
        cl_lane_points = [[200 , y] for y in range(320 , 199 , -1)]
        cr_lane_points = [[400 + 320 - y , y] for y in range(320 , 199 , -1)]
        center_points = refiner.calculate_center_points(cl_lane_points , cr_lane_points)
        self.assertTrue(refiner.abnormal_state)
        self.assertEqual(center_points , [[300 , y] for y in range(320 , 199 , -1)])
        self.assertEqual(refiner.prev_lane_width , previous_width)

        refiner.tracking_frames = 3
        refiner.lane_curvature["CR"] = 0.002
        refiner.lane_2nd_derivative["CR"] = np.array([0.002])
        cr_lane_points = [[440 , y] for y in range(320 , 199 , -1)]
        center_points = refiner.calculate_center_points(cl_lane_points , cr_lane_points)
        self.assertFalse(refiner.abnormal_state)
        self.assertEqual(center_points , [[320 , y] for y in range(320 , 199 , -1)])
        self.assertEqual(refiner.status["center"] , "accurate")
        self.assertEqual(refiner.prev_lane_width , previous_width)

        refiner.tracking_frames = 5
        refiner.calculate_center_points(cl_lane_points , cr_lane_points)
        self.assertEqual(refiner.prev_lane_width , [[240 , y] for y in range(320 , 199 , -1)])

    def test_fitted_branch_does_not_pollute_width_before_intersection(self):
        refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
        refiner.tracking_frames = 5
        cl_sample_points = [[200 , y] for y in range(200 , 321 , 4)]
        cr_sample_points = [[360 + y - 200 , y] for y in range(200 , 321 , 4)]
        cl_lane_points = refiner.fit_lane_points(cl_sample_points , 352 , 640 , 2 , "CL")
        cr_lane_points = refiner.fit_lane_points(cr_sample_points , 352 , 640 , 2 , "CR")
        previous_center = refiner.calculate_center_points(cl_lane_points , cr_lane_points)
        previous_width = [point[:] for point in refiner.prev_lane_width]
        self.assertTrue(previous_width)

        refiner.tracking_frames = 6
        cr_sample_points = [[360 + y - 200 + 0.002 * (y - 260) ** 2 , y] for y in range(200 , 321 , 4)]
        cl_lane_points = refiner.fit_lane_points(cl_sample_points , 352 , 640 , 2 , "CL")
        cr_lane_points = refiner.fit_lane_points(cr_sample_points , 352 , 640 , 2 , "CR")
        refiner.calculate_center_points(cl_lane_points , cr_lane_points)
        self.assertAlmostEqual(refiner.lane_2nd_derivative_diff , 0.004)
        self.assertFalse(refiner.abnormal_state)
        self.assertEqual(refiner.status["center"] , "accurate")
        self.assertEqual(refiner.prev_lane_width , previous_width)

        refiner.tracking_frames = 7
        cr_sample_points = [[480 + 0.004 * (y - 320) ** 2 , y] for y in range(200 , 321 , 4)]
        cl_lane_points = refiner.fit_lane_points(cl_sample_points , 352 , 640 , 2 , "CL")
        cr_lane_points = refiner.fit_lane_points(cr_sample_points , 352 , 640 , 2 , "CR")
        center_points = refiner.calculate_center_points(cl_lane_points , cr_lane_points)
        self.assertTrue(refiner.abnormal_state)
        self.assertEqual(refiner.status["center"] , "predicted")
        self.assertEqual(refiner.prev_lane_width , previous_width)
        self.assertEqual(center_points , previous_center)

    def test_missing_lane_restarts_count_and_preserves_width(self):
        for missing in ("CL" , "CR" , "both"):
            with self.subTest(missing = missing):
                refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
                mask = np.zeros((352 , 640) , dtype = np.uint8)
                mask[200 : 321 , 200] = 3
                mask[200 : 321 , 440] = 3
                for frame in range(5):
                    cl_lane_points , cr_lane_points , _ , _ = refiner.refine(mask , 2 , 10 , 30)
                    refiner.calculate_center_points(cl_lane_points , cr_lane_points)
                previous_width = [point[:] for point in refiner.prev_lane_width]

                missing_mask = mask.copy()
                if missing in ("CL" , "both"):
                    missing_mask[: , 200] = 0
                if missing in ("CR" , "both"):
                    missing_mask[: , 440] = 0
                cl_lane_points , cr_lane_points , _ , _ = refiner.refine(missing_mask , 2 , 10 , 30)
                center_points = refiner.calculate_center_points(cl_lane_points , cr_lane_points)
                self.assertEqual(refiner.tracking_frames , 0)
                self.assertEqual(refiner.prev_lane_width , previous_width)
                if missing == "both":
                    self.assertEqual(center_points , [])
                else:
                    self.assertEqual(center_points , [[320 , y] for y in range(320 , 199 , -1)])

                mask[: , 440] = 0
                mask[200 : 321 , 460] = 3
                for frame in range(1 , 6):
                    cl_lane_points , cr_lane_points , _ , _ = refiner.refine(mask , 2 , 10 , 30)
                    refiner.calculate_center_points(cl_lane_points , cr_lane_points)
                    self.assertEqual(refiner.tracking_frames , frame)
                    if frame < 5:
                        self.assertEqual(refiner.prev_lane_width , previous_width)
                    else:
                        self.assertEqual(refiner.prev_lane_width , [[260 , y] for y in range(320 , 199 , -1)])


if __name__ == "__main__":
    unittest.main()
