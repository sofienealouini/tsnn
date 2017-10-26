import unittest
from tsnn.models.deeprec import DeepRecurrent


class TestDeepRecurrentModel(unittest.TestCase):

    def test_DeepRecurrent_should_have_correct_tensor_shapes(self):

        # Given
        expected_shapes = [(None, 168, 321), (None, 168, 128), (None, 64), (None, 5)]

        # When
        model = DeepRecurrent((168, 321), [128, 64], output_dim=5)
        computed_shapes = [l.get_output_shape_at(node_index=0) for l in model.layers]

        # Check
        self.assertEqual(computed_shapes, expected_shapes)

    def test_DeepRecurrent_should_call_layers_in_order(self):
        pass
