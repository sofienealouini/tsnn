import unittest
from tsnn.models.deeprec import DeepRecurrent


class TestDeepRecurrentModel(unittest.TestCase):

    def test_DeepRecurrent_should_have_correct_tensor_shapes_with_no_attention(self):

        # Given
        expected_shapes = [(None, 168, 321), (None, 168, 128), (None, 168, 64), (None, 32), (None, 5)]

        # When
        model = DeepRecurrent((168, 321), [128, 64, 32], output_dim=5, attention_dim="")
        computed_shapes = [l.get_output_shape_at(node_index=0) for l in model.layers]

        # Check
        self.assertEqual(computed_shapes, expected_shapes)

    def test_DeepRecurrent_should_have_correct_tensor_shapes_with_timesteps_attention(self):

        # Given
        expected_shapes = [(None, 168, 321),
                           (None, 168, 128),
                           (None, 128, 168),
                           (None, 128, 168),
                           (None, 128, 168),
                           (None, 168, 128),
                           (None, 168, 64),
                           (None, 64, 168),
                           (None, 64, 168),
                           (None, 64, 168),
                           (None, 168, 64),
                           (None, 32),
                           (None, 32),
                           (None, 32),
                           (None, 5)]

        # When
        model = DeepRecurrent((168, 321), [128, 64, 32], output_dim=5, attention_dim="timesteps")
        computed_shapes = [l.get_output_shape_at(node_index=0) for l in model.layers]

        # Check
        self.assertEqual(computed_shapes, expected_shapes)

    def test_DeepRecurrent_should_have_correct_tensor_shapes_with_features_attention(self):

        # Given
        expected_shapes = [(None, 168, 321),
                           (None, 168, 128),
                           (None, 168, 128),
                           (None, 168, 128),
                           (None, 168, 64),
                           (None, 168, 64),
                           (None, 168, 64),
                           (None, 32),
                           (None, 32),
                           (None, 32),
                           (None, 5)]

        # When
        model = DeepRecurrent((168, 321), [128, 64, 32], output_dim=5, attention_dim="features")
        computed_shapes = [l.get_output_shape_at(node_index=0) for l in model.layers]

        # Check
        self.assertEqual(computed_shapes, expected_shapes)