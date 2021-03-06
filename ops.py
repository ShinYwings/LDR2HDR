import tensorflow as tf

class conv2d(tf.keras.layers.Layer):
    def __init__(self, output_channels="output_channels", pool_method="pool_method", k_h="k_h", k_w= "k_w", padding="SAME"):
        super(conv2d, self).__init__()
        self.output_channels = output_channels
        self.k_w = k_w
        self.k_h = k_h
        st = 2 if pool_method == 'stride' else 1
        self.strides = [1,st, st,1]
        self.padding = padding
    def build(self, input_shape):
        # TODO input.get_shape()??
        w_init = tf.random.truncated_normal(shape=[self.k_h,self.k_w, input_shape[-1], self.output_channels])
        self.w = tf.Variable(initial_value=w_init, trainable=True)
        bias_init = tf.random.truncated_normal(shape=[self.output_channels])
        self.biases = tf.Variable(initial_value=bias_init, trainable=True) 
        
        super(conv2d, self).build(input_shape)

        # make simpler
        # self.w = self.add_weight(shape=(input_shape[-1], self.units),
        #                        initializer='random_normal',
        #                        trainable=True)
        # self.b = self.add_weight(shape=(self.units,),
        #                        initializer='random_normal',
        #                        trainable=True)
    def call(self, input):
        return tf.nn.bias_add(tf.nn.conv2d(input, self.w, strides=self.strides, padding=self.padding), self.biases)

class deconv2d(tf.keras.layers.Layer):
    def __init__(self, output_channels="output_channels", output_imshape=[],k_h="k_h", k_w= "k_w", padding="SAME", method='resize'):
        super(deconv2d, self).__init__()
        self.output_channels = output_channels
        self.k_w = k_w
        self.k_h = k_h
        self.output_imshape = tf.cast(output_imshape, dtype=tf.int32).numpy()
        self.padding = padding
        self.method= method
    def build(self, input_shape):

        if self.method == 'upsample':
            '''deconv method : checkerboard issue'''
            w_init = tf.random.truncated_normal(shape=[self.k_h, self.k_w, self.output_channels, input_shape[-1]])
            self.w = tf.Variable(initial_value=w_init, trainable=True)
            bias_init = tf.random.truncated_normal(shape=[self.output_channels])
            self.biases = tf.Variable(initial_value=bias_init, trainable=True) 
        elif self.method == 'resize':
            '''resize-conv method http://distill.pub/2016/deconv-checkerboard/'''
            w_init = tf.random.truncated_normal(shape=[self.k_h, self.k_w, input_shape[-1], self.output_channels])
            self.w = tf.Variable(initial_value=w_init, trainable=True)
            bias_init = tf.random.truncated_normal(shape=[self.output_channels])
            self.biases = tf.Variable(initial_value=bias_init, trainable=True)
        
        super(deconv2d, self).build(input_shape)

    def call(self, input):
        batch_size, input_h, _, _ = input.get_shape().as_list() #bhwc
    
        if self.method == 'upsample':
            output_shape = [batch_size, self.output_imshape[0], self.output_imshape[1], self.output_channels]
            strides = int(self.output_imshape[0] / input_h)
            deconv = tf.nn.bias_add(tf.nn.conv2d_transpose(input, self.w, output_shape=output_shape, strides=strides, padding=self.padding), self.biases)  # deconv
        
        elif self.method == 'resize':
            im_resized = tf.image.resize(input, (self.output_imshape[0], self.output_imshape[1]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            strides = [1,1,1,1]
            deconv = tf.nn.bias_add(tf.nn.conv2d(im_resized, self.w, strides=strides, padding=self.padding), self.biases)
        
        return deconv

class fc2d(tf.keras.layers.Layer):
    def __init__(self, fc_dim):
        super(fc2d, self).__init__()
        self.fc_dim = fc_dim

    def build(self,input_shape):
        # batch_size = input.get_shape()[0].value
        # in_height = input.get_shape()[1].value
        # in_width = input.get_shape()[2].value
        # in_channels = input.get_shape()[3].value
        in_height = input_shape[1]
        in_width = input_shape[2]
        in_channels = input_shape[3]
        self.in_dim = in_height * in_width * in_channels

        w_init = tf.random.truncated_normal(shape=[self.in_dim, self.fc_dim])
        self.w = tf.Variable(initial_value=w_init, trainable=True)

        bias_init = tf.random.truncated_normal(shape=[self.fc_dim])
        self.biases = tf.Variable(initial_value=bias_init, trainable=True) 
        super(fc2d, self).build(input_shape)

    def call(self, input):
        
        fc = tf.reshape(input, [-1, self.in_dim])
        fc = tf.matmul(fc, self.w)
        fc = tf.add(fc, self.biases)
        fc = tf.reshape(fc, [-1, 1, 1, self.fc_dim])

        return fc

class dfc2d(tf.keras.layers.Layer):
    def __init__(self, out_height="out_height", out_width="out_width", out_channels="out_channels"):
        # de-fully connected
        # input_:  [batch, 1, 1, fc_dim]
        super(dfc2d, self).__init__()
        self.out_height = out_height
        self.out_width = out_width
        self.out_channels = out_channels
        self.out_dim = out_height*out_width*out_channels
    
    def build(self,input_shape):
        # batch_size = input.get_shape()[0].value
        self.fc_dim = input_shape[-1]

        w_init = tf.random.truncated_normal(shape=[self.fc_dim, self.out_dim])
        self.w = tf.Variable(initial_value=w_init, trainable=True)

        bias_init = tf.random.truncated_normal(shape=[self.out_dim])
        self.biases = tf.Variable(initial_value=bias_init, trainable=True)
        super(dfc2d, self).build(input_shape)

    def call(self, input):
        # fc_dim = input.get_shape()[-1].value
        # input = tf.reshape(input, [-1, fc_dim])
        # w_init = tf.random.truncated_normal(shape=[fc_dim, self.out_dim])
        # w = tf.Variable(initial_value=(w_init,), trainable=True)
        # fc = tf.matmul(input, w)
        # bias_init = tf.random.truncated_normal(shape=self.out_dim)
        # biases = tf.Variable(initial_value=(bias_init,), trainable=True)

        input = tf.reshape(input, [-1, self.fc_dim])
        fc = tf.matmul(input, self.w)
        fc = tf.add(fc, self.biases)
        fc = tf.reshape(fc, [-1, self.out_height, self.out_width, self.out_channels])

        return fc

class batch_normalization(tf.keras.layers.Layer):
    def __init__(self, decay=0.9, epsilon=1e-5, **kwargs):
        super(batch_normalization, self).__init__(**kwargs)
        self.decay = decay
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=[input_shape[-1], ],
                                     initializer=tf.initializers.ones,
                                     trainable=True)
        self.beta = self.add_weight(name='beta',
                                    shape=[input_shape[-1], ],
                                    initializer=tf.initializers.zeros,
                                    trainable=True)
        self.moving_mean = self.add_weight(name='moving_mean',
                                           shape=[input_shape[-1], ],
                                           initializer=tf.initializers.zeros,
                                           trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
                                               shape=[input_shape[-1], ],
                                               initializer=tf.initializers.ones,
                                               trainable=False)
        super(batch_normalization, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        """
        variable = variable * decay + value * (1 - decay)
        """
        delta = variable * self.decay + value * (1 - self.decay)
        return variable.assign(delta)

    def call(self, inputs, training):
        if training:
            batch_mean, batch_variance = tf.nn.moments(inputs, list(range(len(inputs.shape) - 1)))
            mean_update = self.assign_moving_average(self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        
        output = tf.nn.batch_normalization(inputs,
                                           mean=mean,
                                           variance=variance,
                                           offset=self.beta,
                                           scale=self.gamma,
                                           variance_epsilon=self.epsilon)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

class maxpool2d(tf.keras.layers.Layer):
    def __init__(self, kernel_size="kernel_size", strides=None, padding="SAME"):
        
        super(maxpool2d, self).__init__()

        if strides == None:
            strides = kernel_size
        
        self.kernel_size = [1, kernel_size, kernel_size, 1]
        self.strides = [1, strides, strides, 1]
        self.padding = padding
    
    def call(self, x):
        return tf.nn.max_pool(x, ksize= self.kernel_size, strides=self.strides, padding= self.padding)

class avgpool2d(tf.keras.layers.Layer):
    def __init__(self, kernel_size="kernel_size", strides=None, padding="SAME"):

        super(avgpool2d, self).__init__()

        if strides == None:
            strides = kernel_size
        
        self.kernel_size = [1, kernel_size, kernel_size, 1]
        self.strides = [1, strides, strides, 1]
        self.padding = padding
    
    def call(self, x):
        return tf.nn.avg_pool(x, ksize= self.kernel_size, strides=self.strides, padding= self.padding)

class elu(tf.keras.layers.Layer):
    def __init__(self):
        super(elu, self).__init__()
    
    def call(self, x):
        return tf.nn.elu(x)

class dropout(tf.keras.layers.Layer):
    def __init__(self, keep_prob=0.5):
        super(dropout, self).__init__()
        self.keep_prob = keep_prob
    
    def call(self, x, isTraining):

        if isTraining:
            return tf.nn.dropout(x, rate=self.keep_prob)
        else:
            return x

