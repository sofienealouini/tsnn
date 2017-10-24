import keras.backend as K
from keras.models import Model
from keras.layers import Input, GRU, Dense, Conv2D, Conv1D, concatenate, Flatten, Lambda, TimeDistributed


class DeepSense(Model):

    def __init__(self,
                 sensor_dims_list,
                 sequence_length,
                 time_window_tau,
                 freq_domain=False,
                 cnn_filters=64,
                 cnn1_kernel_height=2,
                 cnn2_kernel_size=3,
                 cnn3_kernel_size=2,
                 cnn4_kernel_height=2,
                 cnn5_kernel_size=3,
                 cnn6_kernel_size=2,
                 cnn_activation='relu',
                 cnn_use_bias=True,
                 gru_units=32,
                 gru_use_bias=True,
                 gru_activation='relu',
                 dropout=0.1,
                 output_dim=1):

        """DeepSense network for multi-sensor time series forecasting.
        
        'DeepSense: A Unified Deep Learning Framework for Time-Series Mobile Sensing Data Processing', 
        S. Yao, S. Hu, Y. Zhao, A. Zhang, T. Abdelzaher
        Original paper : http://papers.www2017.com.au.s3-website-ap-southeast-2.amazonaws.com/proceedings/p351.pdf
        
        :param sensor_dims_list: list of ints - number of features measured by each sensor.
            Example : for a 3-axis accelerometer and a 3-axis gyroscope, sensor_dims_list = [3, 3] 
        :param sequence_length: int - length of the input sequence (the generator yields a batch of those sequences)
        :param time_window_tau: int - time windows Tau. Each input sequence is cut into smaller windows of size Tau.
            2*time_window_tau must divide sequence_length
        :param freq_domain: boolean - whether to use FFT. If True, the window size is doubled (time --> (amp, pha))
        :param cnn_filters: int - number of filters in the convolutional layers. Default = 64
        :param cnn1_kernel_height: int - height of the first convolutional kernel (at individual sensor level)
            Must be < time_window_tau. Default = 2
        :param cnn2_kernel_size: int - size of the second convolutional kernel (at individual sensor level). Default = 3
        :param cnn3_kernel_size: int - size of the third convolutional kernel (at individual sensor level). Default = 2
        :param cnn4_kernel_height: int - height of the first convolutional kernel (after merging sensors). Default = 2
        :param cnn5_kernel_size: int - size of the second convolutional kernel (after merging sensors). Default = 3
        :param cnn6_kernel_size: int - size of the third convolutional kernel (after merging sensors). Default = 2
        :param cnn_activation: Keras activation function - convolutional layers activation. Default = 'relu'
        :param cnn_use_bias: boolean - whether to use bias for the convolutional layers. Default = True
        :param gru_units: int - number of units in the recurrent layers. Default = 32
        :param gru_use_bias: boolean - whether to use bias for the recurrent layers. Default = True
        :param gru_activation: Keras activation function - recurrent layers activation. Default = 'relu'
        :param dropout: float in ]0, 1[ - dropout rate for all layers
        :param output_dim: int - number of predicted features
        """

        self.nb_sensors = len(sensor_dims_list)
        window_size = 2 * time_window_tau if freq_domain else time_window_tau
        cnn1_stride = 2 if freq_domain else 1

        # 2*time_window_tau must divide sequence_length
        # cnn1_kernel_height Must be < time_window_tau
        # various conditions on kernel sizes and matrix heights...

        self.main_input = [Input(shape=(sequence_length, sensor_k_dim)) for sensor_k_dim in sensor_dims_list]

        split_input = [Lambda(
            lambda x: K.reshape(x, (-1, sequence_length//time_window_tau, window_size, int(input_k.shape[2]))))(input_k)
                       for input_k in self.main_input]

        extended_dim = [Lambda(lambda x: K.expand_dims(x, axis=-1))(input_k) for input_k in split_input]

        # First individual convolutional layer
        conv1 = [TimeDistributed(Conv2D(filters=cnn_filters,
                                        kernel_size=(cnn1_kernel_height, ext_input_k.get_shape().as_list()[-2]),
                                        activation=cnn_activation,
                                        strides=cnn1_stride,
                                        use_bias=cnn_use_bias))(ext_input_k) for ext_input_k in extended_dim]

        flat_conv1_out = [Lambda(lambda x: K.squeeze(x, axis=3))(conv1_out_k) for conv1_out_k in conv1]

        # Second individual convolutional layer
        conv2 = [TimeDistributed(Conv1D(filters=cnn_filters,
                                        kernel_size=cnn2_kernel_size,
                                        activation=cnn_activation,
                                        use_bias=cnn_use_bias))(conv1_out_k) for conv1_out_k in flat_conv1_out]

        # Third individual convolutional layer
        conv3 = [TimeDistributed(Conv1D(filters=cnn_filters,
                                        kernel_size=cnn3_kernel_size,
                                        activation=cnn_activation,
                                        use_bias=cnn_use_bias))(conv2_out_k) for conv2_out_k in conv2]

        flat_conv3_out = [TimeDistributed(Flatten())(conv3_out_k) for conv3_out_k in conv3]

        # Merged outputs
        extended_dim2 = [TimeDistributed(Lambda(lambda x: K.expand_dims(x, axis=-1)))(conv3_out_k)
                         for conv3_out_k in flat_conv3_out]

        merged = concatenate(extended_dim2) if len(extended_dim2) > 1 else extended_dim2[0]

        # Add the "channels" dimension to meet Conv2D expected input format
        extended_dim3 = Lambda(lambda x: K.expand_dims(x, axis=-1))(merged)

        # First global convolutional layer
        conv4 = TimeDistributed(Conv2D(filters=cnn_filters,
                                       kernel_size=(cnn4_kernel_height, self.nb_sensors),
                                       activation=cnn_activation,
                                       use_bias=cnn_use_bias))(extended_dim3)
        flat_conv4_out = TimeDistributed(Lambda(lambda x: K.squeeze(x, axis=2)))(conv4)

        # Second global convolutional layer
        conv5 = TimeDistributed(Conv1D(filters=cnn_filters,
                                       kernel_size=cnn5_kernel_size,
                                       activation=cnn_activation,
                                       use_bias=cnn_use_bias))(flat_conv4_out)

        # Third global convolutional layer
        conv6 = TimeDistributed(Conv1D(filters=cnn_filters,
                                       kernel_size=cnn6_kernel_size,
                                       activation=cnn_activation,
                                       use_bias=cnn_use_bias))(conv5)

        # Flattened output
        flat_conv6_out = TimeDistributed(Flatten())(conv6)

        # First GRU layer
        rec1_out = GRU(units=gru_units,
                       use_bias=gru_use_bias,
                       activation=gru_activation,
                       dropout=dropout,
                       return_sequences=True)(flat_conv6_out)

        # Second GRU layer
        rec2_out = GRU(units=gru_units,
                       use_bias=gru_use_bias,
                       activation=gru_activation,
                       dropout=dropout)(rec1_out)

        self.main_output = Dense(units=output_dim,
                                 activation="linear",
                                 use_bias=True)(rec2_out)

        super(DeepSense, self).__init__(inputs=self.main_input, outputs=[self.main_output])
