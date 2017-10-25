import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_almost_equal
from tsnn.data_utils import stats, scale_standard, scale_maxabs, scale_minmax, scaling, \
    reverse_standard, reverse_maxabs, reverse_minmax, reverse_scaling,\
    inputs_targets_split, train_val_split, colnames_to_colindices, sample_gen_rnn


class TestDataUtilsFunctions(unittest.TestCase):

    def test_stats_should_return_correct_values(self):

        # Given
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

        # When
        stats_df = stats(data)

        # Check
        assert_equal(stats_df.values, expected_stats.values)

    @patch('sklearn.preprocessing.StandardScaler.fit_transform')
    def test_scale_standard_should_call_sklearn_StandardScaler(self, mock):

        # Given
        data = pd.DataFrame({'A': [1., 1., 1., 1., 1.],
                             'B': [-5., -3., -2., -1., 1.],
                             'C': [-20., -8., -11., -12., -14.]})

        mock.return_value = pd.DataFrame(np.random.rand(data.shape[0], data.shape[1]))

        # When
        computed_scaled, _ = scale_standard(data)

        # Check
        mock.assert_called_once()

    @patch('sklearn.preprocessing.MaxAbsScaler.fit_transform')
    def test_scale_maxabs_should_call_sklearn_MaxAbsScaler(self, mock):

        # Given
        data = pd.DataFrame({'A': [1., 1., 1., 1., 1.],
                             'B': [-5., -3., -2., -1., 1.],
                             'C': [-20., -8., -11., -12., -14.]})

        mock.return_value = pd.DataFrame(np.random.rand(data.shape[0], data.shape[1]))

        # When
        computed_scaled, _ = scale_maxabs(data)

        # Check
        mock.assert_called_once()

    @patch('sklearn.preprocessing.MinMaxScaler.fit_transform')
    def test_scale_minmax_should_call_sklearn_MinMaxScaler(self, mock):

        # Given
        data = pd.DataFrame({'A': [1., 1., 1., 1., 1.],
                             'B': [-5., -3., -2., -1., 1.],
                             'C': [-20., -8., -11., -12., -14.]})

        mock.return_value = pd.DataFrame(np.random.rand(data.shape[0], data.shape[1]))

        # When
        computed_scaled, _ = scale_minmax(data)

        # Check
        mock.assert_called_once()

    @patch('tsnn.data_utils.scale_standard')
    def test_scaling_should_call_scale_standard_if_method_is_standard(self, mock):

        # Given
        data = pd.DataFrame({'A': [1., 1., 1., 1., 1.],
                             'B': [-5., -3., -2., -1., 1.],
                             'C': [-20., -8., -11., -12., -14.]})
        method = "standard"
        mock.return_value = (0, 42)

        # When
        scaling(data, method)

        # Check
        mock.assert_called_once()

    @patch('tsnn.data_utils.scale_maxabs')
    def test_scaling_should_call_scale_maxabs_if_method_is_maxabs(self, mock):

        # Given
        data = pd.DataFrame({'A': [1., 1., 1., 1., 1.],
                             'B': [-5., -3., -2., -1., 1.],
                             'C': [-20., -8., -11., -12., -14.]})
        method = "maxabs"
        mock.return_value = (0, 42)

        # When
        scaling(data, method)

        # Check
        mock.assert_called_once()

    @patch('tsnn.data_utils.scale_minmax')
    def test_scaling_should_call_scale_minmax_if_method_is_minmax(self, mock):

        # Given
        data = pd.DataFrame({'A': [1., 1., 1., 1., 1.],
                             'B': [-5., -3., -2., -1., 1.],
                             'C': [-20., -8., -11., -12., -14.]})
        method = "minmax"
        mock.return_value = (0, 42)

        # When
        scaling(data, method)

        # Check
        mock.assert_called_once()

    def test_reverse_standard_should_return_correct_values_when_predicting_all_features(self):

        # Given
        predicted_data = pd.DataFrame({'A': [0., 0., 0., 0., 0.],
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

        # When
        computed_reversed_all = reverse_standard(predicted_data, [0, 1, 2], stats_df)

        # Check
        assert_almost_equal(computed_reversed_all, expected_reversed_all)

    def test_reverse_standard_should_return_correct_values_when_predicting_one_feature(self):

        # Given
        predicted_data = np.array([-1.5, -0.5, 0.])

        stats_df = pd.DataFrame({'min': [1., -5., -20.],
                                 'max': [1., 1., -8.],
                                 'mean': [1., -2., -13.],
                                 'std': [0., 2., 4.],
                                 'maxabs': [1., 5., 20.]},
                                index=['A', 'B', 'C'],
                                columns=['min', 'max', 'mean', 'std', 'maxabs'])

        expected_reversed_part = np.array([-5., -3., -2.])

        # When
        computed_reversed_part = reverse_standard(predicted_data, [1], stats_df)

        # Check
        assert_almost_equal(computed_reversed_part, expected_reversed_part)

    def test_reverse_maxabs_should_return_correct_values_when_predicting_all_features(self):

        # Given
        predicted_data = pd.DataFrame({'A': [1., 1., 1., 1., 1.],
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

        # When
        computed_reversed_all = reverse_maxabs(predicted_data, [0, 1, 2], stats_df)

        # Check
        assert_almost_equal(computed_reversed_all, expected_reversed_all)

    def test_reverse_maxabs_should_return_correct_values_when_predicting_one_feature(self):

        # Given
        predicted_data = np.array([-1., -0.6, -0.4])

        stats_df = pd.DataFrame({'min': [1., -5., -20.],
                                 'max': [1., 1., -8.],
                                 'mean': [1., -2., -13.],
                                 'std': [0., 2., 4.],
                                 'maxabs': [1., 5., 20.]},
                                index=['A', 'B', 'C'],
                                columns=['min', 'max', 'mean', 'std', 'maxabs'])

        expected_reversed_part = np.array([-5., -3., -2.])

        # When
        computed_reversed_part = reverse_maxabs(predicted_data, [1], stats_df)

        # Check
        assert_almost_equal(computed_reversed_part, expected_reversed_part)

    def test_reverse_minmax_should_return_correct_values_when_predicting_all_features(self):

        # Given
        predicted_data = pd.DataFrame({'A': [0., 0., 0., 0., 1.],
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

        # When
        computed_reversed_all = reverse_minmax(predicted_data, [0, 1, 2], stats_df)

        # Check
        assert_almost_equal(computed_reversed_all, expected_reversed_all)

    def test_reverse_minmax_should_return_correct_values_when_predicting_one_feature(self):

        # Given
        predicted_data = np.array([0., 0.2, 0.3])

        stats_df = pd.DataFrame({'min': [1., -5., -20.],
                                 'max': [2., 5., 0.],
                                 'mean': [1.2, -1.2, -11.4],
                                 'std': [0.4, 3.370460, 6.499231],
                                 'maxabs': [2., 5., 20.]},
                                index=['A', 'B', 'C'],
                                columns=['min', 'max', 'mean', 'std', 'maxabs'])

        expected_reversed_part = np.array([-5., -3., -2.])

        # When
        computed_reversed_part = reverse_minmax(predicted_data, [1], stats_df)

        # Check
        assert_almost_equal(computed_reversed_part, expected_reversed_part)

    @patch('tsnn.data_utils.reverse_standard')
    def test_reverse_scaling_should_call_reverse_standard_if_method_is_standard(self, mock):

        # Given
        predicted_data = pd.DataFrame({'A': [0., 0., 0., 0., 1.],
                                       'B': [0., 0.2, 0.3, 0.4, 1.],
                                       'C': [0., 1., 0.45, 0.4, 0.3]}).values

        stats_df = pd.DataFrame({'min': [1., -5., -20.],
                                 'max': [2., 5., 0.],
                                 'mean': [1.2, -1.2, -11.4],
                                 'std': [0.4, 3.370460, 6.499231],
                                 'maxabs': [2., 5., 20.]},
                                index=['A', 'B', 'C'],
                                columns=['min', 'max', 'mean', 'std', 'maxabs'])

        method = "standard"
        interest_vars = [0, 1, 2]
        mock.return_value = pd.DataFrame(np.random.rand(predicted_data.shape[0], predicted_data.shape[1]))

        # When
        _ = reverse_scaling(predicted_data, interest_vars, stats_df, method)

        # Check
        mock.assert_called_once()

    @patch('tsnn.data_utils.reverse_maxabs')
    def test_reverse_scaling_should_call_reverse_maxabs_if_method_is_maxabs(self, mock):

        # Given
        predicted_data = pd.DataFrame({'A': [0., 0., 0., 0., 1.],
                                       'B': [0., 0.2, 0.3, 0.4, 1.],
                                       'C': [0., 1., 0.45, 0.4, 0.3]}).values

        stats_df = pd.DataFrame({'min': [1., -5., -20.],
                                 'max': [2., 5., 0.],
                                 'mean': [1.2, -1.2, -11.4],
                                 'std': [0.4, 3.370460, 6.499231],
                                 'maxabs': [2., 5., 20.]},
                                index=['A', 'B', 'C'],
                                columns=['min', 'max', 'mean', 'std', 'maxabs'])

        method = "maxabs"
        interest_vars = [0, 1, 2]
        mock.return_value = pd.DataFrame(np.random.rand(predicted_data.shape[0], predicted_data.shape[1]))

        # When
        _ = reverse_scaling(predicted_data, interest_vars, stats_df, method)

        # Check
        mock.assert_called_once()

    @patch('tsnn.data_utils.reverse_minmax')
    def test_reverse_scaling_should_call_reverse_minmax_if_method_is_minmax(self, mock):

        # Given
        predicted_data = pd.DataFrame({'A': [0., 0., 0., 0., 1.],
                                       'B': [0., 0.2, 0.3, 0.4, 1.],
                                       'C': [0., 1., 0.45, 0.4, 0.3]}).values

        stats_df = pd.DataFrame({'min': [1., -5., -20.],
                                 'max': [2., 5., 0.],
                                 'mean': [1.2, -1.2, -11.4],
                                 'std': [0.4, 3.370460, 6.499231],
                                 'maxabs': [2., 5., 20.]},
                                index=['A', 'B', 'C'],
                                columns=['min', 'max', 'mean', 'std', 'maxabs'])

        method = "minmax"
        interest_vars = [0, 1, 2]
        mock.return_value = pd.DataFrame(np.random.rand(predicted_data.shape[0], predicted_data.shape[1]))

        # When
        _ = reverse_scaling(predicted_data, interest_vars, stats_df, method)

        # Check
        mock.assert_called_once()

    def test_train_val_split_should_split_correctly(self):

        # Given
        target = pd.DataFrame({'B': [10., 1., 1., -3., 0., 10., 12., 14],
                               'D': [3., 4., 5., -12., -5., -3., -2., -1.],
                               'E': [0., 0., 0., 0., 1., 1., 1., 1.]})
        train_ratio = 0.6
        val_ratio = 0.2
        expected_limits = ((0, 5), (5, 7), (7, 8))

        # When
        computed_limits = train_val_split(target, train_ratio, val_ratio)

        # Check
        self.assertEqual(computed_limits, expected_limits)

    def test_colnames_to_colindices_should_convert_correctly(self):

        # Given
        origin = pd.DataFrame({'A': [1., 1., 1., 1., 1., 14., 20., -10., 12., 1., 3., -2.],
                               'B': [-5., -3., -2., -1., 1., 1., 0., 10., 1., 1., -3., 0.],
                               'C': [-20., -8., -11., -12., -14., 0., 0., 0., 0., 0., 7., -20.],
                               'D': [-2., 3., 6., 7., 18., 1., 2., 3., 4., 5., -12., -5.],
                               'E': [10., 0., 10., 12., 14., 10., 0., 0., 0., 0., 0., 1.]})

        target_names = ['B', 'C', 'E']
        expected_result = [1, 2, 4]

        # When
        computed_result = colnames_to_colindices(target_names, origin)

        # Check
        self.assertEqual(computed_result, expected_result)

    def test_inputs_targets_split_should_split_correctly(self):

        # Given
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

        # When
        computed_inp, computed_tar = inputs_targets_split(data, input_cols, target_cols,
                                                          samples_length, pred_delay, pred_length)
        expected_inp = data.iloc[:-3]
        expected_tar = data[['B', 'D', 'E']].iloc[7:]

        # Check
        assert_equal(computed_inp.values, expected_inp.values)
        assert_equal(computed_tar.values, expected_tar.values)
        assert_equal(computed_inp[['B', 'D', 'E']].iloc[7].values, computed_tar.iloc[0].values)

    def test_sample_gen_rnn_should_yield_correct_batch_when_limits_are_given(self):

        # Given
        inputs = pd.DataFrame({'A': [1., 1., 1., 1., 1., 14., 20., -10., 12., 1., 3., -2.],
                               'B': [-5., -3., -2., -1., 1., 1., 0., 10., 1., 1., -3., 0.],
                               'C': [-20., -8., -11., -12., -14., 0., 0., 0., 0., 0., 7., -20.],
                               'D': [-2., 3., 6., 7., 18., 1., 2., 3., 4., 5., -12., -5.],
                               'E': [10., 0., 10., 12., 14., 10., 0., 0., 0., 0., 0., 1.]})

        targets = pd.DataFrame({'B': [10., 1., 1., -3., 0., 10., 12., 14],
                                'D': [3., 4., 5., -12., -5., -3., -2., -1.],
                                'E': [0., 0., 0., 0., 1., 1., 1., 1.]})

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

        gen = sample_gen_rnn(inputs, targets, limits=(0, 5), samples_length=5, sampling_step=1, batch_size=2)

        # When
        computed_xbatch, computed_ybatch = next(gen)

        # Check
        assert_equal(computed_xbatch, expected_xbatch)
        assert_equal(computed_ybatch, expected_ybatch)

    def test_sample_gen_rnn_should_yield_all_batches_when_limits_are_not_given(self):
        
        # Given
        inputs = pd.DataFrame({'A': [1., 1., 1., 1., 1., 14., 20., -10., 12., 1., 3., -2.],
                               'B': [-5., -3., -2., -1., 1., 1., 0., 10., 1., 1., -3., 0.],
                               'C': [-20., -8., -11., -12., -14., 0., 0., 0., 0., 0., 7., -20.],
                               'D': [-2., 3., 6., 7., 18., 1., 2., 3., 4., 5., -12., -5.],
                               'E': [10., 0., 10., 12., 14., 10., 0., 0., 0., 0., 0., 1.]})

        targets = pd.DataFrame({'B': [10., 1., 1., -3., 0., 10., 12., 14],
                                'D': [3., 4., 5., -12., -5., -3., -2., -1.],
                                'E': [0., 0., 0., 0., 1., 1., 1., 1.]})

        expected_gen = sample_gen_rnn(inputs, targets, limits=(0, 8), samples_length=5, sampling_step=1, batch_size=3)

        gen = sample_gen_rnn(inputs, targets, samples_length=5, sampling_step=1, batch_size=3)

        for i in range(3):
            # Given
            expected_xbatch, expected_ybatch = next(expected_gen)

            # When
            computed_xbatch, computed_ybatch = next(gen)

            # Check
            assert_equal(computed_xbatch, expected_xbatch)
            assert_equal(computed_ybatch, expected_ybatch)

    def test_sample_gen_rnn_should_reach_dataframe_end(self):

        # Given
        inputs = pd.DataFrame({'A': [1., 1., 1., 1., 1., 14., 20., -10., 12., 1., 3., -2.],
                               'B': [-5., -3., -2., -1., 1., 1., 0., 10., 1., 1., -3., 0.],
                               'C': [-20., -8., -11., -12., -14., 0., 0., 0., 0., 0., 7., -20.],
                               'D': [-2., 3., 6., 7., 18., 1., 2., 3., 4., 5., -12., -5.],
                               'E': [10., 0., 10., 12., 14., 10., 0., 0., 0., 0., 0., 1.]})

        targets = pd.DataFrame({'B': [10., 1., 1., -3., 0., 10., 12., 14],
                                'D': [3., 4., 5., -12., -5., -3., -2., -1.],
                                'E': [0., 0., 0., 0., 1., 1., 1., 1.]})

        gen = sample_gen_rnn(inputs, targets, limits=(0, 8), samples_length=5, sampling_step=1, batch_size=3)

        # When
        last_batch_x, last_batch_y = None, None
        for i in range(3):
            last_batch_x, last_batch_y = next(gen)

        # Check
        self.assertEqual(last_batch_x.shape, (2, 5, 5))
        self.assertEqual(last_batch_y.shape, (2, 3))

    def test_sample_gen_rnn_should_yield_only_xbatch_if_inputs_only_set_true(self):

        # Given
        inputs = pd.DataFrame({'A': [1., 1., 1., 1., 1., 14., 20., -10., 12., 1., 3., -2.],
                               'B': [-5., -3., -2., -1., 1., 1., 0., 10., 1., 1., -3., 0.],
                               'C': [-20., -8., -11., -12., -14., 0., 0., 0., 0., 0., 7., -20.],
                               'D': [-2., 3., 6., 7., 18., 1., 2., 3., 4., 5., -12., -5.],
                               'E': [10., 0., 10., 12., 14., 10., 0., 0., 0., 0., 0., 1.]})

        targets = pd.DataFrame({'B': [10., 1., 1., -3., 0., 10., 12., 14],
                                'D': [3., 4., 5., -12., -5., -3., -2., -1.],
                                'E': [0., 0., 0., 0., 1., 1., 1., 1.]})

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

        gen = sample_gen_rnn(inputs, targets, samples_length=5, sampling_step=1, batch_size=2, inputs_only=True)

        # When
        res = next(gen)

        # Check
        self.assertIsInstance(res, np.ndarray)
        assert_equal(res, expected_xbatch)

    def test_sample_gen_rnn_should_reset_when_reaching_dataframe_end(self):
        # Given
        inputs = pd.DataFrame({'A': [1., 1., 1., 1., 1., 14., 20., -10., 12., 1., 3., -2.],
                               'B': [-5., -3., -2., -1., 1., 1., 0., 10., 1., 1., -3., 0.],
                               'C': [-20., -8., -11., -12., -14., 0., 0., 0., 0., 0., 7., -20.],
                               'D': [-2., 3., 6., 7., 18., 1., 2., 3., 4., 5., -12., -5.],
                               'E': [10., 0., 10., 12., 14., 10., 0., 0., 0., 0., 0., 1.]})

        targets = pd.DataFrame({'B': [10., 1., 1., -3., 0., 10., 12., 14],
                                'D': [3., 4., 5., -12., -5., -3., -2., -1.],
                                'E': [0., 0., 0., 0., 1., 1., 1., 1.]})

        gen = sample_gen_rnn(inputs, targets, limits=(0, 8), samples_length=5, sampling_step=1, batch_size=3)

        # When
        last_batch_x, last_batch_y = None, None
        for i in range(4):
            last_batch_x, last_batch_y = next(gen)

        # Check
        self.assertEqual(last_batch_x.shape, (3, 5, 5))
        self.assertEqual(last_batch_y.shape, (3, 3))
