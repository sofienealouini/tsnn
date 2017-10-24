from keras.models import Model
from keras.layers import Input, LSTM, GRU, SimpleRNN, Dense


class DeepRecurrent(Model):
    def __init__(self, input_shape, output_dim=1,
                 rec_layers_dims=[128, 64], rec_layer_type="lstm", rec_use_bias=True, rec_activation='relu',
                 dense_use_bias=True, dropout=0.2):

        type_dict = {"simple_rnn": SimpleRNN, "lstm": LSTM, "gru": GRU}
        nb_rec_layers = len(rec_layers_dims)

        rec_layer = type_dict[rec_layer_type]

        ### Vérification paramètres

        self.main_input = Input(shape=input_shape, name="Main_input")
        rec_output = self.main_input
        i = 0

        for dim in rec_layers_dims:
            i += 1
            last_rec_layer = (i == nb_rec_layers)
            rec_output = rec_layer(units=dim,
                                   use_bias=rec_use_bias,
                                   activation=rec_activation,
                                   dropout=dropout,
                                   name="Recurrent_"+str(i),
                                   return_sequences=not last_rec_layer)(rec_output)

        self.main_output = Dense(units=output_dim,
                                 activation="linear",
                                 use_bias=dense_use_bias)(rec_output)

        super(DeepRecurrent, self).__init__(inputs=[self.main_input], outputs=[self.main_output])
