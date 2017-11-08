from keras.models import Model
from keras.layers import Input, LSTM, GRU, SimpleRNN, Dense, merge, Permute, Multiply, TimeDistributed, Lambda


class DeepRecurrent(Model):
    def __init__(self, input_shape,
                 rec_layers_dims, rec_layer_type="lstm", rec_use_bias=True, rec_activation='tanh',
                 output_dim=1, dense_use_bias=True, dropout=0.2,
                 attention_dim=None):
        """Deep recurrent network for time series forecasting.
        
        :param input_shape: input_shape: tuple of ints - (timesteps, nb_input_features)
        :param rec_layers_dims: list of ints - dimensions of the recurrent layers given in order. 
            Example : 2 recurrent layers, the first having 128 units and the second 64 
            --> rec_layers_dims is then [128, 64]
        :param rec_layer_type: String - type of recurrent layers to use : one of ['simple_rnn', 'lstm', 'gru']
        :param rec_use_bias: boolean - whether to use bias for the recurrent layers. Default = True
        :param rec_activation: Keras activation function to use for all recurrent layers - Default = 'tanh'
        :param output_dim: int - number of predicted features
        :param dense_use_bias: boolean - whether to use bias for the output layer. Default = True
        :param dropout: float in ]0, 1[ - dropout rate for all layers. Default = 0.2
        :param attention_dim: None or one of ["features", "timesteps"] - dimension along which to apply attention
            - if None, no attention mechanism is used
            - if "features", attention is applied feature-wise. For each timestep, the features are weighted 
            - if "timesteps", attention is applied time-wise. For each feature, the timesteps are weighted 
        """

        type_dict = {"simple_rnn": SimpleRNN, "lstm": LSTM, "gru": GRU}
        rec_layer = type_dict[rec_layer_type]
        nb_rec_layers = len(rec_layers_dims)

        self.main_input = Input(shape=input_shape, name="Main_input")

        rec_output = self.main_input
        i = 0

        for dim in rec_layers_dims:
            i += 1
            is_last_rec_layer = (i == nb_rec_layers)
            rec_output = rec_layer(units=dim,
                                   use_bias=rec_use_bias,
                                   activation=rec_activation,
                                   dropout=dropout,
                                   name="Recurrent_" + str(i),
                                   return_sequences=not is_last_rec_layer)(rec_output)

            # Feature-wise attention
            if attention_dim == "features":
                rec_output = self.apply_feature_wise_attention(rec_output=rec_output,
                                                               is_last_rec_layer=is_last_rec_layer,
                                                               name="Attention_" + str(i))

            # Time-wise attention
            if attention_dim == "timesteps":
                rec_output = self.apply_time_wise_attention(timesteps=input_shape[0],
                                                            rec_output=rec_output,
                                                            is_last_rec_layer=is_last_rec_layer,
                                                            name="Attention_" + str(i))

        self.main_output = Dense(units=output_dim,
                                 activation="linear",
                                 use_bias=dense_use_bias)(rec_output)

        super(DeepRecurrent, self).__init__(inputs=[self.main_input], outputs=[self.main_output])

    def apply_feature_wise_attention(self, rec_output, is_last_rec_layer, name):
        """
        
        :param rec_output: 
        :param is_last_rec_layer: 
        :param name: 
        :return: 
        """
        if not is_last_rec_layer:
            attention_layer = TimeDistributed(Dense(int(rec_output.shape[2]), activation="softmax"), name=name)(
                rec_output)
        if is_last_rec_layer:
            attention_layer = Dense(int(rec_output.shape[1]), activation="softmax", name=name)(rec_output)
        rec_output = Multiply()([rec_output, attention_layer])
        return rec_output

    def apply_time_wise_attention(self, timesteps, rec_output, is_last_rec_layer, name):
        """
        
        :param timesteps: 
        :param rec_output: 
        :param is_last_rec_layer: 
        :param name: 
        :return: 
        """
        if not is_last_rec_layer:
            rec_output = Permute((2, 1))(rec_output)
            attention_layer = TimeDistributed(Dense(timesteps, activation="softmax"), name=name)(rec_output)
            rec_output = Multiply()([rec_output, attention_layer])
            rec_output = Permute((2, 1))(rec_output)
        if is_last_rec_layer:
            attention_layer = Dense(int(rec_output.shape[1]), activation="softmax", name=name)(rec_output)
            rec_output = Multiply()([rec_output, attention_layer])
        return rec_output

