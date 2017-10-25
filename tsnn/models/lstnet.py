import keras.backend as K
from keras.models import Model
from keras.layers import Input, GRU, Dense, Conv2D, Dropout, concatenate, Add, Lambda


class LSTNet(Model):

    def __init__(self, input_shape, interest_vars,
                 cnn_filters=100, cnn_kernel_height=6, cnn_activation='relu', cnn_use_bias=True,
                 gru_units=100, gru_activation='relu', gru_use_bias=True,
                 gru_skip_units=5, gru_skip_step=24, gru_skip_activation='relu', gru_skip_use_bias=True,
                 ar_window=24, ar_use_bias=True, dropout=0.2):
        """LSTNet network for time series forecasting.

        'Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks', 
        G. Lai, W. Chang, Y. Yang, H. Liu
        Original paper : https://arxiv.org/pdf/1703.07015.pdf

        :param input_shape: tuple of ints - (timesteps, nb_input_features)
        :param interest_vars: list of ints - indices of the features to predict (indices in the input matrix)
            Example : 321 features as inputs, we want to predict the features corresponding to the columns 1, 6 and 315:
            --> interest_vars is then [1, 6, 315]
        :param cnn_filters: int - number of filters in the convolutional layer. Default = 100
        :param cnn_kernel_height: int - height of the convolutional kernel, must be <= timesteps. Default = 6
        :param cnn_activation: Keras activation function - Default = 'relu'
        :param cnn_use_bias: boolean - whether to use bias for the convolutional layer. Default = True
        :param gru_units: int - number of units in the recurrent layer. Default = 100
        :param gru_activation: Keras activation function - Default = 'relu'
        :param gru_use_bias: boolean - whether to use bias for the recurrent layer. Default = True
        :param gru_skip_units: int - number of units in the recurrent-skip layer. Default = 5
        :param gru_skip_step: int - skipped timesteps in the recurrent-skip layer. Default = 24
        :param gru_skip_activation: Keras activation function - Default = 'relu'
        :param gru_skip_use_bias: boolean - whether to use bias for the recurrent-skip layer. Default = True
        :param ar_window: int - number of past values to use as predictors for autoregression - Default = 24
        :param ar_use_bias: boolean - whether to use bias for the autoregression. Default = True
        :param dropout: float in ]0, 1[ - dropout rate for all layers
        """

        self.main_input = Input(shape=input_shape, name="Main_input")
        timesteps, nb_input_features = input_shape

        # Check parameters

        # Convolutional layer
        conv = Lambda(lambda x: K.expand_dims(x, -1), name="Conv_in_expand")(self.main_input)
        conv = Conv2D(filters=cnn_filters,
                      kernel_size=(cnn_kernel_height, nb_input_features),
                      activation=cnn_activation,
                      use_bias=cnn_use_bias,
                      name="Conv")(conv)
        conv = Dropout(dropout, name="Conv_dropout")(conv)
        conv = Lambda(lambda x: K.squeeze(x, axis=2), name="Conv_out_squeeze")(conv)

        # Recurrent layer
        rec = GRU(units=gru_units,
                  use_bias=gru_use_bias,
                  activation=gru_activation,
                  dropout=dropout,
                  name="GRU")(conv)

        # Recurrent-skip layer
        skip_rec = Lambda(lambda x: gru_skip_prep(x, gru_skip_step),
                          name="GRU_skip_inp_prep")(conv)
        skip_rec = GRU(units=gru_skip_units,
                       activation=gru_skip_activation,
                       use_bias=gru_skip_use_bias,
                       dropout=dropout,
                       name="GRU_skip")(skip_rec)
        skip_rec = Lambda(lambda x: K.reshape(x, (-1, gru_skip_step * gru_skip_units)),
                          name="GRU_skip_out_reshape")(skip_rec)

        # Combination of recurrent outputs
        rec = concatenate([rec, skip_rec], axis=1, name="Recurrent_concat")
        res = Dense(len(interest_vars), activation='linear', name="NN_dim_reduce")(rec)

        # Autoregressive component
        ar = Lambda(lambda x: autoreg_prep(x, ar_window, interest_vars), name="Autoreg_prep")(self.main_input)
        ar = Dense(units=1,
                   activation="linear",
                   use_bias=ar_use_bias,
                   name="Autoreg")(ar)
        ar = Lambda(lambda x: K.reshape(x, (-1, len(interest_vars))), name="Autoreg_out_reshape")(ar)

        # Summing NN and Autoregressive branches
        res = Add(name="Sum_NN_Autoreg")([res, ar])
        self.main_output = res

        super(LSTNet, self).__init__(inputs=[self.main_input], outputs=[self.main_output])


def autoreg_prep(x, ar_window, interest_vars):
        """Batch transformations in order to perform autoregression.
        
        :param x: keras.backend.Tensor (3D) - batch of inputs (batch_size, timesteps, nb_features)
        :param ar_window: int - number of past values to use as predictors for autoregression
        :param interest_vars: interest_vars: list of ints - indices of the features to predict (in the input matrix)
        :return: keras.backend.Tensor (2D)  - data formatted for autoregression
        """
        predictors_window = x[:, -ar_window:, :]
        perm_1 = K.permute_dimensions(predictors_window, [2, 1, 0])
        interest_vars_only = K.gather(perm_1, interest_vars)
        perm_2 = K.permute_dimensions(interest_vars_only, [2, 0, 1])
        reshaped = K.reshape(perm_2, (-1, ar_window))
        return reshaped


def gru_skip_prep(conv_x, gru_skip_step):
        """Batch transformations for the recurrent-skip layer.
        
        :param conv_x: keras.backend.Tensor - output of the convolutional layer
        :param gru_skip_step: int - skipped timesteps in the recurrent-skip layer.
        :return: keras.backend.Tensor (3D) - data formatted for recurrent-skip layer.
        """
        cnn_filters = int(conv_x.shape[2])
        possible_jumps = int(conv_x.shape[1]) // gru_skip_step
        jumps_window = conv_x[:, -possible_jumps * gru_skip_step:, :]
        reshaped_window = K.reshape(jumps_window, (-1, possible_jumps, gru_skip_step, cnn_filters))
        permuted_columns = K.permute_dimensions(reshaped_window, [0, 2, 1, 3])
        gru_skip_input = K.reshape(permuted_columns, (-1, possible_jumps, cnn_filters))
        return gru_skip_input
