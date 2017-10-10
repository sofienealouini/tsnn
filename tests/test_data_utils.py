import unittest
import pandas as pd
from numpy.testing import assert_equal, assert_almost_equal
from tsnn.data_utils import stats, scale_standard, scale_maxabs, scale_minmax, \
    reverse_standard, reverse_maxabs, reverse_minmax


class TestDataUtilsFunctions(unittest.TestCase):

    def test_stats(self):
        data = pd.DataFrame({'A': [1., 1., 1., 1., 1.],
                             'B': [-5., -3., -2., -1., 1.],
                             'C': [-20., -8., -11., -12., -14.]})

        expected_stats = pd.DataFrame({'min': [1., -5., -20.],
                                       'max': [1., 1., -8.],
                                       'mean': [1., -2., -13.],
                                       'std': [0., 2., 4.],
                                       'maxabs': [1., 5., 20.]},
                                      index=['A', 'B', 'C'],
                                      columns=['min', 'max', 'mean', 'std', 'maxabs'])

        stats_df = stats(data)
        assert_equal(stats_df.values, expected_stats.values)

    def test_scale_standard(self):
        data = pd.DataFrame({'A': [1., 1., 1., 1., 1.],
                             'B': [-5., -3., -2., -1., 1.],
                             'C': [-20., -8., -11., -12., -14.]})

        expected_scaled = pd.DataFrame({'A': [0., 0., 0., 0., 0.],
                                        'B': [-1.5, -0.5, 0., 0.5, 1.5],
                                        'C': [-1.75, 1.25, 0.5, 0.25, -0.25]})

        computed_scaled, _ = scale_standard(data)
        assert_almost_equal(computed_scaled.values, expected_scaled.values)

    def test_scale_maxabs(self):
        data = pd.DataFrame({'A': [1., 1., 1., 1., 1.],
                             'B': [-5., -3., -2., -1., 1.],
                             'C': [-20., -8., -11., -12., -14.]})

        expected_scaled = pd.DataFrame({'A': [1., 1., 1., 1., 1.],
                                        'B': [-1., -0.6, -0.4, -0.2, 0.2],
                                        'C': [-1., -0.4, -0.55, -0.6, -0.7]})

        computed_scaled, _ = scale_maxabs(data)
        assert_almost_equal(computed_scaled.values, expected_scaled.values)

    def test_scale_minmax(self):
        data = pd.DataFrame({'A': [1., 1., 1., 1., 2.],
                             'B': [-5., -3., -2., -1., 5.],
                             'C': [-20., 0., -11., -12., -14.]})

        expected_scaled = pd.DataFrame({'A': [0., 0., 0., 0., 1.],
                                        'B': [0., 0.2, 0.3, 0.4, 1.],
                                        'C': [0., 1., 0.45, 0.4, 0.3]})

        computed_scaled, _ = scale_minmax(data)
        assert_almost_equal(computed_scaled.values, expected_scaled.values)

    def test_reverse_standard(self):
        pass

    def test_reverse_maxabs(self):
        pass

    def test_reverse_minmax(self):
        pass
