import tensorflow as tf
from tensorflow.keras import Model
import ops

class transportMatrix(tf.keras.layers.Layer):
    def __init__(self, h=32, w=128, channel=3):
        super(transportMatrix, self).__init__()
        initializer = tf.random_normal_initializer()

        # 4096 = image h, w (32 x 128)
        self.half_h = tf.cast(tf.divide(h,2), dtype=tf.int32)
        self.row_size = tf.multiply(self.half_h, w)
        self.w = tf.Variable(initial_value=initializer(shape=[self.row_size, self.row_size, channel], dtype=tf.float32), trainable=True)
    
    def call(self, x):
        input_b, _, _, input_c = x.get_shape().as_list() #bhwc
        half_outImg = x[:, : self.half_h, :, :]
        half_outImg = tf.reshape(half_outImg, [input_b, self.row_size, 1, input_c])
        tm = tf.einsum("jkl,ikml->ijml",self.w, half_outImg)
        output = tf.reshape(tm, [input_b, 64, 64, input_c])

        return output

class ldr2hdr(Model):
    def __init__(self, fc_dim="fc_dim", im_height="im_height", deconv_method="deconv_method"):

        super(ldr2hdr, self).__init__()

        self.fc_dim = fc_dim
        # self.doSigmoidLast = doSigmoidLast
        self.deconv_method = deconv_method
        self.im_height = im_height
        self.im_width = tf.multiply(2,im_height)
        self.latentVector = None

        # Encoder 
        self.conv1 = ops.conv2d(output_channels=64, k_h=7, k_w=7, pool_method="stride")
        self.bn1 = ops.batch_normalization()
        self.elu1 = ops.elu()
        
        self.conv2 = ops.conv2d(output_channels=128, k_h=5, k_w=5, pool_method="stride")
        self.bn2 = ops.batch_normalization()
        self.elu2 = ops.elu()
        
        self.conv3 = ops.conv2d(output_channels=256, k_h=3, k_w=3, pool_method="stride")
        self.bn3 = ops.batch_normalization()
        self.elu3 = ops.elu()
        
        self.conv4 = ops.conv2d(output_channels=256, k_h=3, k_w=3, pool_method="stride")
        self.bn4 = ops.batch_normalization()
        self.elu4 = ops.elu()

        self.fc = ops.fc2d(self.fc_dim)
        self.bn5 = ops.batch_normalization()
        self.elu5 = ops.elu()
        self.dropout = ops.dropout(0.5)

        # Decoder
        # Original
        # self.defc = ops.dfc2d(out_height=self.conv4.get_shape()[1].value,
        #                         out_width=self.conv4.get_shape()[2].value,
        #                             out_channels=self.conv4.get_shape()[3].value)
        self.defc = ops.dfc2d(out_height=tf.cast(tf.divide(self.im_height, 16), dtype=tf.int32),
                                out_width=tf.cast(tf.divide(self.im_width, 16), dtype=tf.int32),
                                    out_channels=256)
        self.bn6 = ops.batch_normalization()
        self.elu6 = ops.elu()

        self.deconv4 = ops.deconv2d(output_channels=256, output_imshape=[tf.divide(self.im_height,8), tf.divide(self.im_width,8)], k_h=3, k_w=3, method=self.deconv_method)
        self.bn7 = ops.batch_normalization()
        self.elu7 = ops.elu()

        self.deconv3 = ops.deconv2d(output_channels=128, output_imshape=[tf.divide(self.im_height,4), tf.divide(self.im_width,4)], k_h=3, k_w=3, method=self.deconv_method)
        self.bn8 = ops.batch_normalization()
        self.elu8 = ops.elu()

        self.deconv2 = ops.deconv2d(output_channels=64, output_imshape=[tf.divide(self.im_height,2), tf.divide(self.im_width,2)], k_h=5, k_w=5, method=self.deconv_method)
        self.bn9 = ops.batch_normalization()
        self.elu9 = ops.elu()

        self.deconv1 = ops.deconv2d(output_channels=64, output_imshape=[self.im_height, self.im_width], k_h=7, k_w=7, method=self.deconv_method)
        self.bn10 = ops.batch_normalization()
        self.elu10 = ops.elu()

        self.out = ops.conv2d(output_channels=3, k_h=1, k_w=1, pool_method=None)

        self.tm = ops.transportMatrix()

    def encoder(self, x, training="training"):

        self.conv1_output = self.conv1(x)
        bn1 = self.bn1(self.conv1_output, training)
        elu1 = self.elu1(bn1)

        self.conv2_output = self.conv2(elu1)
        bn2 = self.bn2(self.conv2_output, training)
        elu2 = self.elu2(bn2)

        self.conv3_output = self.conv3(elu2)
        bn3 = self.bn3(self.conv3_output, training)
        elu3 = self.elu3(bn3)

        self.conv4_output = self.conv4(elu3)
        bn4 = self.bn4(self.conv4_output, training)
        elu4 = self.elu4(bn4)

        fc1 = self.fc(elu4)
        bn5 = self.bn5(fc1, training)
        elu5 = self.elu5(bn5)
        self.latentVector = self.dropout(elu5, training)

        return self.latentVector

    def decoder(self, training="training"):
        
        # No input
        fc = self.latentVector
        conv4 = self.conv4_output
        conv3 = self.conv3_output
        conv2 = self.conv2_output
        conv1 = self.conv1_output

        defc = self.defc(fc)
        defc = tf.add(defc, conv4)
        bn6 = self.bn6(defc, training)
        elu6 = self.elu6(bn6)

        deconv4 = self.deconv4(elu6)
        deconv4 = tf.add(deconv4, conv3)
        bn7 = self.bn7(deconv4, training)
        elu7 = self.elu7(bn7)

        deconv3 = self.deconv3(elu7)
        deconv3 = tf.add(deconv3, conv2)
        bn8 = self.bn8(deconv3, training)
        elu8 = self.elu8(bn8)

        deconv2 = self.deconv2(elu8)
        deconv2 = tf.add(deconv2, conv1)
        bn9 = self.bn9(deconv2, training)
        elu9 = self.elu9(bn9)

        deconv1 = self.deconv1(elu9)
        bn10 = self.bn10(deconv1, training)
        elu10 = self.elu10(bn10)

        out = self.out(elu10)
        out = tf.nn.sigmoid(out, name='OutputImg')

        tm_out = self.tm(out)

        return out, tm_out


# NOT UPDATED
# def sunPredictior(self, isTraining, reuse=False):
#         fc = self.fc
#         with tf.variable_scope('SunPosition', reuse=reuse):
#             if self.doFCNorFC == 'FC':
#                 fc_fcn = lambda fc, dims: fc2d(fc, dims)
#             elif self.doFCNorFC == 'FCN':
#                 fc_fcn = lambda fc, dims: conv2d(fc, output_channels=dims, k_h=1, k_w=1, pool_method=None, padding='VALID')

#             with tf.variable_scope('fc1'):
#                 sunpos_fc1 = fc_fcn(fc, 32)
#                 sunpos_fc1 = tf.nn.elu(batch_norm(sunpos_fc1, isTraining), name='activation')

#             with tf.variable_scope('fc2'):
#                 sunpos_fc2 = fc_fcn(sunpos_fc1, 16)
#                 sunpos_fc2 = tf.nn.elu(batch_norm(sunpos_fc2, isTraining), name='activation')

#             with tf.variable_scope('fc5'):
#                 sunPos = fc_fcn(sunpos_fc2, 1)
#                 sunPos = tf.nn.relu(sunPos, name='activation')
#                 sunPos = tf.squeeze(sunPos, [1, 2], name='output')  # [N,1,1,D]->[N, D] or [N, D]
#         return sunPos

#     def discriminator_da(self, isTraining, lambdar, reuse=False):
#         fc = gradient_reversal(self.fc, lambdar)
#         with tf.variable_scope("Discriminator", reuse=reuse):
#             with tf.variable_scope('fc1'):
#                 domain_fc = fc2d(fc, 32)
#                 domain_fc = tf.nn.elu(domain_fc, name='activation')

#             with tf.variable_scope('fc2'):
#                 domain_logit = fc2d(domain_fc, 2)
#                 domain_fc = tf.nn.softmax(domain_logit, name='activation')
#                 domain_fc = tf.squeeze(domain_fc, [1, 2], name='domain_out')
#         return domain_fc, domain_logit  # (BATCH, 2)

#     def pred(self, inputs, isTraining, reuse=False):
#         fc = self.encoder(inputs=inputs, isTraining=isTraining, reuse=reuse)
#         sunPos = self.sunPredictior(isTraining=isTraining, reuse=reuse)
#         outImg = self.decoder(isTraining=isTraining, reuse=reuse)
#         return outImg, sunPos, fc