import unittest
import pandas as pd
from numpy.testing import assert_equal
from tsnn.data_utils import stats, scale, reverse_scaling


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