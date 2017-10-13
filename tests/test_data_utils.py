import unittest
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_almost_equal
from tsnn.data_utils import stats, scale_standard, scale_maxabs, scale_minmax, \
    reverse_standard, reverse_maxabs, reverse_minmax, inputs_targets_split, train_val_split, \
    colnames_to_colindices, sample_gen_rnn


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
        data = pd.DataFrame({'A': [0., 0., 0., 0., 0.],
                             'B': [-1.5, -0.5, 0., 0.5, 1.5],
                             'C': [-1.75, 1.25, 0.5, 0.25, -0.25]}).values

        stats_df = pd.DataFrame({'min': [1., -5., -20.],
                                 'max': [1., 1., -8.],
                                 'mean': [1., -2., -13.],
                                 'std': [0., 2., 4.],
                                 'maxabs': [1., 5., 20.]},
                                index=['A', 'B', 'C'],
                                columns=['min', 'max', 'mean', 'std', 'maxabs'])

        expected_reversed_all = pd.DataFrame({'A': [1., 1., 1., 1., 1.],
                                              'B': [-5., -3., -2., -1., 1.],
                                              'C': [-20., -8., -11., -12., -14.]}).values

        expected_reversed_part = expected_reversed_all[:3, 1:3]

        computed_reversed_all = reverse_standard(data, [0, 1, 2], stats_df)
        computed_reversed_part = reverse_standard(data[:3, 1:3], [1, 2], stats_df)

        assert_almost_equal(computed_reversed_all, expected_reversed_all)
        assert_almost_equal(computed_reversed_part, expected_reversed_part)

    def test_reverse_maxabs(self):
        data = pd.DataFrame({'A': [1., 1., 1., 1., 1.],
                             'B': [-1., -0.6, -0.4, -0.2, 0.2],
                             'C': [-1., -0.4, -0.55, -0.6, -0.7]}).values

        stats_df = pd.DataFrame({'min': [1., -5., -20.],
                                 'max': [1., 1., -8.],
                                 'mean': [1., -2., -13.],
                                 'std': [0., 2., 4.],
                                 'maxabs': [1., 5., 20.]},
                                index=['A', 'B', 'C'],
                                columns=['min', 'max', 'mean', 'std', 'maxabs'])

        expected_reversed_all = pd.DataFrame({'A': [1., 1., 1., 1., 1.],
                                              'B': [-5., -3., -2., -1., 1.],
                                              'C': [-20., -8., -11., -12., -14.]}).values

        expected_reversed_part = expected_reversed_all[:3, 1:3]

        computed_reversed_all = reverse_maxabs(data, [0, 1, 2], stats_df)
        computed_reversed_part = reverse_maxabs(data[:3, 1:3], [1, 2], stats_df)

        assert_almost_equal(computed_reversed_all, expected_reversed_all)
        assert_almost_equal(computed_reversed_part, expected_reversed_part)

    def test_reverse_minmax(self):
        data = pd.DataFrame({'A': [0., 0., 0., 0., 1.],
                             'B': [0., 0.2, 0.3, 0.4, 1.],
                             'C': [0., 1., 0.45, 0.4, 0.3]}).values

        stats_df = pd.DataFrame({'min': [1., -5., -20.],
                                 'max': [2., 5., 0.],
                                 'mean': [1.2, -1.2, -11.4],
                                 'std': [0.4, 3.370460, 6.499231],
                                 'maxabs': [2., 5., 20.]},
                                index=['A', 'B', 'C'],
                                columns=['min', 'max', 'mean', 'std', 'maxabs'])

        expected_reversed_all = pd.DataFrame({'A': [1., 1., 1., 1., 2.],
                                              'B': [-5., -3., -2., -1., 5.],
                                              'C': [-20., 0., -11., -12., -14.]}).values
        expected_reversed_part = expected_reversed_all[:3, 1:3]

        computed_reversed_all = reverse_minmax(data, [0, 1, 2], stats_df)
        computed_reversed_part = reverse_minmax(data[:3, 1:3], [1, 2], stats_df)

        assert_almost_equal(computed_reversed_all, expected_reversed_all)
        assert_almost_equal(computed_reversed_part, expected_reversed_part)

    def test_inputs_targets_split(self):
        data = pd.DataFrame({'A': [1., 1., 1., 1., 1., 14., 20., -10., 12., 1., 3., -2., -1., 1., 1.],
                             'B': [-5., -3., -2., -1., 1., 1., 0., 10., 1., 1., -3., 0., 10., 12., 14],
                             'C': [-20., -8., -11., -12., -14., 0., 0., 0., 0., 0., 7., -20., -8., -11., -12.],
                             'D': [-2., 3., 6., 7., 18., 1., 2., 3., 4., 5., -12., -5., -3., -2., -1.],
                             'E': [10., 0., 10., 12., 14., 10., 0., 0., 0., 0., 0., 1., 1., 1., 1.]})
        input_cols = []
        target_cols = ['B', 'D', 'E']
        samples_length = 5
        pred_delay = 3
        pred_length = 1
        computed_inp, computed_tar = inputs_targets_split(data, input_cols, target_cols,
                                                          samples_length, pred_delay, pred_length)
        expected_inp = data.iloc[:-3]
        expected_tar = data[['B', 'D', 'E']].iloc[7:]

        assert_equal(computed_inp.values, expected_inp.values)
        assert_equal(computed_tar.values, expected_tar.values)
        assert_equal(computed_tar[['D']].iloc[-1, 0], -1.)
        self.assertEqual(computed_inp[['B']].iloc[7, 0], computed_tar[['B']].iloc[0, 0])
        self.assertEqual(len(computed_inp), 12)
        self.assertEqual(len(computed_tar), 8)

    def test_train_val_split(self):
        target = pd.DataFrame({'B': [10., 1., 1., -3., 0., 10., 12., 14],
                               'D': [3., 4., 5., -12., -5., -3., -2., -1.],
                               'E': [0., 0., 0., 0., 1., 1., 1., 1.]})
        train_ratio = 0.6
        val_ratio = 0.2
        expected_limits = ((0, 5), (5, 7), (7, 8))
        computed_limits = train_val_split(target, train_ratio, val_ratio)

        self.assertEqual(computed_limits, expected_limits)

    def test_colnames_to_colindices(self):
        origin = pd.DataFrame({'A': [1., 1., 1., 1., 1., 14., 20., -10., 12., 1., 3., -2.],
                               'B': [-5., -3., -2., -1., 1., 1., 0., 10., 1., 1., -3., 0.],
                               'C': [-20., -8., -11., -12., -14., 0., 0., 0., 0., 0., 7., -20.],
                               'D': [-2., 3., 6., 7., 18., 1., 2., 3., 4., 5., -12., -5.],
                               'E': [10., 0., 10., 12., 14., 10., 0., 0., 0., 0., 0., 1.]})

        target_names = ['B', 'C', 'E']
        expected_result = [1, 2, 4]
        computed_result = colnames_to_colindices(target_names, origin)
        self.assertEqual(computed_result, expected_result)

    def test_sample_gen_rnn(self):
        inputs = pd.DataFrame({'A': [1., 1., 1., 1., 1., 14., 20., -10., 12., 1., 3., -2.],
                               'B': [-5., -3., -2., -1., 1., 1., 0., 10., 1., 1., -3., 0.],
                               'C': [-20., -8., -11., -12., -14., 0., 0., 0., 0., 0., 7., -20.],
                               'D': [-2., 3., 6., 7., 18., 1., 2., 3., 4., 5., -12., -5.],
                               'E': [10., 0., 10., 12., 14., 10., 0., 0., 0., 0., 0., 1.]})

        targets = pd.DataFrame({'B': [10., 1., 1., -3., 0., 10., 12., 14],
                                'D': [3., 4., 5., -12., -5., -3., -2., -1.],
                                'E': [0., 0., 0., 0., 1., 1., 1., 1.]})

        gen = sample_gen_rnn(inputs, targets,
                             limits=(0, 5),
                             samples_length=5,
                             sampling_step=1,
                             batch_size=2)

        computed_xbatch, computed_ybatch = next(gen)

        expected_xbatch = np.array([[[1., -5., -20., -2., 10.],
                                     [1., -3., -8., 3., 0.],
                                     [1., -2., -11., 6., 10.],
                                     [1., -1., -12., 7., 12.],
                                     [1., 1., -14., 18., 14.]],
                                    [[1., -3., -8., 3., 0.],
                                     [1., -2., -11., 6., 10.],
                                     [1., -1., -12., 7., 12.],
                                     [1., 1., -14., 18., 14.],
                                     [14., 1., 0., 1., 10.]]])
        expected_ybatch = np.array([[10., 3., 0.],
                                    [1., 4., 0.]])

        last_batch_x, last_batch_y = None, None
        for i in range(2):
            last_batch_x, last_batch_y = next(gen)

        assert_equal(computed_xbatch, expected_xbatch)
        assert_equal(computed_ybatch, expected_ybatch)
        self.assertEqual(len(last_batch_x), 1)
        self.assertEqual(len(last_batch_y), 1)
