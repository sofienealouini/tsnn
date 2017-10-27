from keras.models import Model
from keras.layers import Input, LSTM, GRU, SimpleRNN, Dense


class DeepRecurrent(Model):
    def __init__(self, input_shape,
                 rec_layers_dims, rec_layer_type="lstm", rec_use_bias=True, rec_activation='tanh',
                 output_dim=1, dense_use_bias=True, dropout=0.2):
        """Deep recurrent network for time series forecasting.
        
        :param input_shape: tuple of ints - (timesteps, nb_input_features)
        :param output_dim: int - number of predicted features
        :param rec_layers_dims: list of ints - dimensions of the recurrent layers given in order. 
            Example : 2 recurrent layers, the first having 128 units and the second 64 
            --> rec_layers_dims is then [128, 64]
        :param rec_layer_type: String - type of recurrent layers to use : one of ['simple_rnn', 'lstm', 'gru']
        :param rec_use_bias: boolean - whether to use bias for the recurrent layers. Default = True
        :param rec_activation: Keras activation function to use for all recurrent layers - Default = 'tanh'
        :param dense_use_bias: boolean - whether to use bias for the output layer. Default = True
        :param dropout: float in ]0, 1[ - dropout rate for all layers. Default = 0.2
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
                                   name="Recurrent_"+str(i),
                                   return_sequences=not is_last_rec_layer)(rec_output)

        self.main_output = Dense(units=output_dim,
                                 activation="linear",
                                 use_bias=dense_use_bias)(rec_output)

        super(DeepRecurrent, self).__init__(inputs=[self.main_input], outputs=[self.main_output])
