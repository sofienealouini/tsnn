import unittest
from tsnn.models.deepsense import DeepSense


class TestDeepSense(unittest.TestCase):

    def test_DeepSense_should_have_correct_tensor_shapes(self):

        # Given
        expected_shapes = [(None, 1000, 3),
                           (None, 1000, 5),
                           (None, 125, 8, 3),
                           (None, 125, 8, 5),
                           (None, 125, 8, 3, 1),
                           (None, 125, 8, 5, 1),
                           (None, 125, 7, 1, 64),
                           (None, 125, 7, 1, 64),
                           (None, 125, 7, 64),
                           (None, 125, 7, 64),
                           (None, 125, 5, 64),
                           (None, 125, 5, 64),
                           (None, 125, 4, 64),
                           (None, 125, 4, 64),
                           (None, 125, 256),
                           (None, 125, 256),
                           (None, 125, 256, 1),
                           (None, 125, 256, 1),
                           (None, 125, 256, 2),
                           (None, 125, 256, 2, 1),
                           (None, 125, 255, 1, 64),
                           (None, 125, 255, 64),
                           (None, 125, 253, 64),
                           (None, 125, 252, 64),
                           (None, 125, 16128),
                           (None, 125, 32),
                           (None, 32),
                           (None, 1)]

        # When
        model = DeepSense(sensor_dims_list=[3, 5], sequence_length=1000, time_window_tau=8)
        computed_shapes = [l.get_output_shape_at(node_index=0) for l in model.layers]

        # Check
        self.assertEqual(computed_shapes, expected_shapes)

    def test_DeepSense_should_call_layers_in_order(self):
        pass
