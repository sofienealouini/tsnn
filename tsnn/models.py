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

        'Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks', G. Lai, W. Chang, Y. Yang, H. Liu
        Original paper : https://arxiv.org/pdf/1703.07015.pdf

        :param input_shape: tuple of ints - (timesteps, nb_input_features)
        :param interest_vars: list of ints - indices of the features to predict (indices in the input matrix)
            Example : 321 features as inputs, we want to predict the features corresponding to the columns 1, 6 and 315:
            interest_vars is then [1, 6, 315]
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

        self.main_input = Input(shape=input_shape)
        timesteps, nb_input_features = input_shape
        possible_jumps = (timesteps - cnn_kernel_height) // gru_skip_step

        # Convolutional layer
        conv = Lambda(lambda x: K.expand_dims(x, -1))(self.main_input)
        conv = Conv2D(filters=cnn_filters,
                      kernel_size=(cnn_kernel_height, nb_input_features),
                      activation=cnn_activation,
                      use_bias=cnn_use_bias)(conv)
        conv = Dropout(dropout)(conv)
        conv = Lambda(lambda x: K.squeeze(x, axis=2))(conv)

        # Recurrent layer
        rec = GRU(units=gru_units,
                  use_bias=gru_use_bias,
                  activation=gru_activation,
                  dropout=dropout)(conv)

        # Recurrent-skip layer
        skip_rec = Lambda(lambda x: x[:, -possible_jumps * gru_skip_step:, :])(conv)
        skip_rec = Lambda(lambda x: K.reshape(x, (-1, possible_jumps, gru_skip_step, cnn_filters)))(skip_rec)
        skip_rec = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1, 3]))(skip_rec)
        skip_rec = Lambda(lambda x: K.reshape(x, (-1, possible_jumps, cnn_filters)))(skip_rec)
        skip_rec = GRU(units=gru_skip_units,
                       activation=gru_skip_activation,
                       use_bias=gru_skip_use_bias,
                       dropout=dropout)(skip_rec)
        skip_rec = Lambda(lambda x: K.reshape(x, (-1, gru_skip_step * gru_skip_units)))(skip_rec)

        # Combination of recurrent outputs
        rec = concatenate([rec, skip_rec], axis=1)
        res = Dense(len(interest_vars), activation='linear')(rec)

        # Autoregressive component
        ar = Lambda(lambda x: x[:, -ar_window:, :])(self.main_input)
        ar = Lambda(lambda x: K.permute_dimensions(x, [2, 1, 0]))(ar)
        ar = Lambda(lambda x: K.gather(x, interest_vars))(ar)
        ar = Lambda(lambda x: K.permute_dimensions(x, [2, 0, 1]))(ar)
        ar = Lambda(lambda x: K.reshape(x, (-1, ar_window)))(ar)
        ar = Dense(units=1,
                   activation="linear",
                   use_bias=ar_use_bias)(ar)
        ar = Lambda(lambda x: K.reshape(x, (-1, len(interest_vars))))(ar)

        # Summing NN and AR branches
        res = Add()([res, ar])
        self.main_output = res

        super(LSTNet, self).__init__(inputs=[self.main_input], outputs=[self.main_output])