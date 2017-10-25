import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_equal
import keras.backend as K
from tsnn.models.lstnet import autoreg_prep, gru_skip_prep, LSTNet


class TestLSTNetModel(unittest.TestCase):

    def test_autoreg_prep_should_return_correct_tensor(self):

        # Given
        batch = np.zeros((64, 168, 3))
        for i in range(batch.shape[0]):
            for j in range(batch.shape[1]):
                for k in range(batch.shape[2]):
                    batch[i, j, k] = 10000 * (i + 1) + (j + 1) + (k + 1) / 1000
        ar_window = 24
        interest_vars = [0, 2]

        expected_batch = np.zeros((128, 24))
        for i in range(expected_batch.shape[0]):
            for j in range(expected_batch.shape[1]):
                if i % 2 == 0:
                    expected_batch[i, j] = 10000 * (i + 2) // 2 + (168 - 24 + j + 1) + 1 / 1000
                else:
                    expected_batch[i, j] = 10000 * (i + 1) // 2 + (168 - 24 + j + 1) + 3 / 1000

        # When
        computed_batch = autoreg_prep(batch, ar_window, interest_vars).eval(session=K.get_session())

        # Check
        assert_equal(computed_batch, expected_batch)

    def test_gru_skip_prep_should_return_correct_tensor(self):

        # Given
        batch = np.zeros((64, 163, 100))
        for i in range(batch.shape[0]):
            for j in range(batch.shape[1]):
                for k in range(batch.shape[2]):
                    batch[i, j, k] = 10000 * (i + 1) + (j + 1) + (k + 1) / 1000
        gru_skip_step = 24

        c = 0
        expected_batch = np.zeros((1536, 6, 100))
        for i in range(expected_batch.shape[0]):
            c +=1
            for j in range(expected_batch.shape[1]):
                for k in range(expected_batch.shape[2]):
                    expected_batch[i, j, k] = 10000 * (i // 24 + 1) + (163 % 24 + j * 24 + c) + (k + 1) / 1000
            if c == 24:
                c = 0

        # When
        computed_batch = gru_skip_prep(batch, gru_skip_step).eval(session=K.get_session())

        # Check
        assert_equal(computed_batch, expected_batch)

    def test_LSTNet_should_have_correct_tensor_shapes(self):

        # Given
        expected_shapes = [(None, 168, 321),
                           (None, 168, 321, 1),
                           (None, 163, 1, 100),
                           (None, 163, 1, 100),
                           (None, 163, 100),
                           (None, 6, 100),
                           (None, 5),
                           (None, 100),
                           (None, 120),
                           (None, 24),
                           (None, 220),
                           (None, 1),
                           (None, 5),
                           (None, 5),
                           (None, 5)]

        # When
        model = LSTNet((168, 321), [0, 6, 11, 36, 315])
        computed_shapes = [l.get_output_shape_at(node_index=0) for l in model.layers]

        # Check
        self.assertEqual(computed_shapes, expected_shapes)

    def test_LSTNet_should_call_layers_in_order(self):
        pass
