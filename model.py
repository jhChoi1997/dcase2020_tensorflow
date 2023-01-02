import math
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model


class ResidualBlock(Model):
    def __init__(self, n_channel, n_mul, kernel_size, padding, dilation_rate, n_groups):
        super(ResidualBlock, self).__init__()

        self.sigmoid_group_norm = layers.LayerNormalization(axis=(1, 2))
        self.sigmoid_conv = layers.Conv1D(n_channel * n_mul, kernel_size, padding=padding, dilation_rate=dilation_rate, groups=n_groups, kernel_regularizer=regularizers.l2(1e-4))
        self.tanh_group_norm = layers.LayerNormalization(axis=(1, 2))
        self.tanh_conv = layers.Conv1D(n_channel * n_mul, kernel_size, padding=padding, dilation_rate=dilation_rate, groups=n_groups, kernel_regularizer=regularizers.l2(1e-4))

        self.skip_group_norm = layers.LayerNormalization(axis=(1, 2))
        self.skip_conv = layers.Conv1D(n_channel, 1, padding=padding, groups=n_groups, kernel_regularizer=regularizers.l2(1e-4))
        self.residual_group_norm = layers.LayerNormalization(axis=(1, 2))
        self.residual_conv = layers.Conv1D(n_channel * n_mul, 1, padding=padding, groups=n_groups, kernel_regularizer=regularizers.l2(1e-4))

    def call(self, inputs, training=None, mask=None):
        x1 = self.sigmoid_group_norm(inputs)
        x1 = self.sigmoid_conv(x1)
        x2 = self.tanh_group_norm(inputs)
        x2 = self.tanh_conv(x2)
        x1 = tf.nn.sigmoid(x1)
        x2 = tf.nn.tanh(x2)
        x = tf.multiply(x1, x2)

        x1 = self.skip_group_norm(x)
        skip = self.skip_conv(x1)
        x2 = self.residual_group_norm(x)
        residual = self.residual_conv(x2)

        return skip, residual + inputs


class FeatureExtractor(Model):
    def __init__(self, n_blocks, n_channel, n_mul, kernel_size, padding, n_groups):
        super(FeatureExtractor, self).__init__()

        self.n_blocks = n_blocks
        self.kernel_size = kernel_size

        self.group_norm1 = layers.LayerNormalization(axis=(1, 2))
        self.conv1 = layers.Conv1D(n_channel * n_mul, 1, groups=n_groups, kernel_regularizer=regularizers.l2(1e-4))

        self.residual_blocks = [ResidualBlock(n_channel, n_mul, kernel_size, padding, 2**i, n_groups) for i in range(n_blocks)]
        self.concatenate = layers.Concatenate()

    def call(self, inputs, training=None, mask=None):
        x = self.group_norm1(inputs)
        x = self.conv1(x)
        skip_connections = []
        for rb in self.residual_blocks:
            skip, x = rb(x)
            skip = tf.expand_dims(skip, axis=-1)
            skip_connections.append(skip)
        output = self.concatenate(skip_connections)
        return output

    def get_receptive_field(self):
        receptive_field = 1
        for _ in range(self.n_blocks):
            receptive_field = receptive_field * 2 + self.kernel_size - 2
        return receptive_field


class PredictionLayer(Model):
    def __init__(self, n_channel, padding, n_groups, receptive_field):
        super(PredictionLayer, self).__init__()

        self.receptive_field = receptive_field

        self.relu1 = layers.Activation('relu')
        self.group_norm1 = layers.LayerNormalization(axis=(1, 2))
        self.conv1 = layers.Conv1D(n_channel, 1, padding=padding, groups=n_groups, activation='relu', kernel_regularizer=regularizers.l2(1e-4))
        self.relu2 = layers.Activation('relu')
        self.group_norm2 = layers.LayerNormalization(axis=(1, 2))
        self.conv2 = layers.Conv1D(n_channel, 1, padding=padding, groups=n_groups, kernel_regularizer=regularizers.l2(1e-4))

    def call(self, inputs, training=None, mask=None):
        x = self.relu1(inputs)
        x = self.group_norm1(x)
        x = self.conv1(x)
        x = self.relu2(x)
        x = self.group_norm2(x)
        x = self.conv2(x)
        output = x[:, self.receptive_field - 1:-1, :]
        return output


class WaveNet(Model):
    def __init__(self, n_blocks, n_channel, n_mul, kernel_size, padding, n_groups):
        super(WaveNet, self).__init__()

        self.feature_extractor = FeatureExtractor(n_blocks, n_channel, n_mul, kernel_size, padding, n_groups)
        self.receptive_field = self.feature_extractor.get_receptive_field()

        self.prediction = PredictionLayer(n_channel, padding, n_groups, self.receptive_field)

    def call(self, inputs, training=None, mask=None):
        x = self.feature_extractor(inputs)
        x = tf.reduce_sum(x, axis=-1)
        output = self.prediction(x)
        return output


class ResnetBlock(Model):
    def __init__(self, n_filter, projection=False):
        super(ResnetBlock, self).__init__()
        self.projection = projection
        if self.projection:
            self.conv1 = layers.Conv2D(n_filter, 3, strides=2, padding='same', kernel_regularizer=regularizers.l2(1e-4))
        else:
            self.conv1 = layers.Conv2D(n_filter, 3, padding='same', kernel_regularizer=regularizers.l2(1e-4))
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation('relu')

        self.conv2 = layers.Conv2D(n_filter, 3, padding='same', kernel_regularizer=regularizers.l2(1e-4))
        self.bn2 = layers.BatchNormalization()
        if self.projection:
            self.conv3 = layers.Conv2D(n_filter, 1, strides=2, padding='same', kernel_regularizer=regularizers.l2(1e-4))
        else:
            self.conv3 = layers.Conv2D(n_filter, 1, padding='same', kernel_regularizer=regularizers.l2(1e-4))
        self.add = layers.Add()
        self.act2 = layers.Activation('relu')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.projection:
            skip = self.conv3(inputs)
        else:
            skip = inputs
        x = self.add([x, skip])
        output = self.act2(x)
        return output


class ClassificationLayer(Model):
    def __init__(self, n_class, filters, arcface=None):
        super(ClassificationLayer, self).__init__()

        self.arcface = arcface

        self.conv1 = layers.Conv2D(filters, 7, strides=2, padding='same', kernel_regularizer=regularizers.l2(1e-4))
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.Activation('relu')
        self.mp1 = layers.MaxPooling2D(3, strides=2, padding='same')

        self.rb1 = ResnetBlock(filters)
        self.rb2 = ResnetBlock(filters)
        self.rb3 = ResnetBlock(filters * 2, True)
        self.rb4 = ResnetBlock(filters * 2)
        self.rb5 = ResnetBlock(filters * 4, True)
        self.rb6 = ResnetBlock(filters * 4)
        self.rb7 = ResnetBlock(filters * 8, True)
        self.rb8 = ResnetBlock(filters * 8)

        self.gap1 = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(n_class, activation='softmax', kernel_regularizer=regularizers.l2(1e-4))

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs[0])
        x = self.bn1(x)
        x = self.act1(x)
        x = self.mp1(x)
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        x = self.rb6(x)
        x = self.rb7(x)
        x = self.rb8(x)
        x = self.gap1(x)
        if self.arcface is not None:
            output = self.arcface(x, inputs[1])
        else:
            output = self.dense1(x)
        return output


class SegmenatationLayer(Model):
    def __init__(self, n_class, receptive_field):
        super(SegmenatationLayer, self).__init__()
        self.receptive_field = receptive_field

        self.act1 = layers.Activation('relu')
        self.conv1 = layers.Conv1D(n_class, 1, kernel_regularizer=regularizers.l2(1e-4))
        self.softmax1 = layers.Softmax(axis=-1)

    def call(self, inputs, training=None, mask=None):
        x = self.act1(inputs)
        x = self.conv1(x)
        x = x[:, self.receptive_field - 1:-1, :]
        output = self.softmax1(x)
        return output


class MTLClass(Model):
    def __init__(self, n_blocks, n_channel, n_mul, kernel_size, padding, n_groups, n_class, t_frame, arcface=None):
        super(MTLClass, self).__init__()
        self.feature_extractor = FeatureExtractor(n_blocks, n_channel, n_mul, kernel_size, padding, n_groups)
        self.receptive_field = self.feature_extractor.get_receptive_field()
        self.prediction = PredictionLayer(n_channel, padding, n_groups, self.receptive_field)
        self.classification = ClassificationLayer(n_class, filters=8, arcface=arcface)
        # self.classification = tf.keras.applications.resnet50.ResNet50(weights=None,
        #                                                               input_shape=(t_frame, n_channel, n_blocks),
        #                                                               classes=n_class)

    def call(self, inputs, training=None, mask=None):
        skip = self.feature_extractor(inputs[0])
        x = tf.reduce_sum(skip, axis=-1)
        output1 = self.prediction(x)
        output2 = self.classification([skip, inputs[1]])
        return output1, output2


class MTLSeg(Model):
    def __init__(self, n_blocks, n_channel, n_mul, kernel_size, padding, n_groups, n_class):
        super(MTLSeg, self).__init__()
        self.feature_extractor = FeatureExtractor(n_blocks, n_channel, n_mul, kernel_size, padding, n_groups)
        self.receptive_field = self.feature_extractor.get_receptive_field()

        self.prediction = PredictionLayer(n_channel, padding, n_groups, self.receptive_field)
        self.segmentation = SegmenatationLayer(n_class, self.receptive_field)

    def call(self, inputs, training=None, mask=None):
        skip = self.feature_extractor(inputs)
        x = tf.reduce_sum(skip, axis=-1)
        output1 = self.prediction(x)
        output2 = self.segmentation(x)
        return output1, output2


class MTLClassSeg(Model):
    def __init__(self, n_blocks, n_channel, n_mul, kernel_size, padding, n_groups, n_class, t_frame, arcface=None):
        super(MTLClassSeg, self).__init__()

        self.feature_extractor = FeatureExtractor(n_blocks, n_channel, n_mul, kernel_size, padding, n_groups)
        self.receptive_field = self.feature_extractor.get_receptive_field()

        self.prediction = PredictionLayer(n_channel, padding, n_groups, self.receptive_field)
        self.classification = ClassificationLayer(n_class, filters=8, arcface=arcface)
        # self.classification = tf.keras.applications.resnet50.ResNet50(weights=None,
        #                                                               input_shape=(t_frame, n_channel, n_blocks),
        #                                                               classes=n_class)
        self.segmentation = SegmenatationLayer(n_class, self.receptive_field)

    def call(self, inputs, training=None, mask=None):
        skip = self.feature_extractor(inputs[0])
        x = tf.reduce_sum(skip, axis=-1)
        output1 = self.prediction(x)
        output2 = self.classification([skip, inputs[1]])
        output3 = self.segmentation(x)
        return output1, output2, output3


class MultiResolutionWaveNet(Model):
    def __init__(self, n_blocks, n_channel, n_mul, kernel_size, padding, n_groups):
        super(MultiResolutionWaveNet, self).__init__()

        self.feature_extractor = FeatureExtractor(n_blocks, n_channel, n_mul, kernel_size, padding, n_groups)
        self.receptive_field = self.feature_extractor.get_receptive_field()

        self.prediction_layers = [PredictionLayer(n_channel, padding, n_groups, self.receptive_field) for _ in range(n_blocks)]
        self.concatenate = layers.Concatenate()

    def call(self, inputs, training=None, mask=None):
        x = self.feature_extractor(inputs)
        prediction = []
        for idx, layer in enumerate(self.prediction_layers):
            tmp = layer(x[:, :, :, idx])
            tmp = tf.expand_dims(tmp, axis=-1)
            prediction.append(tmp)
        output = self.concatenate(prediction)
        return output


class MultiResolutionSumWaveNet(Model):
    def __init__(self, n_blocks, n_channel, n_mul, kernel_size, padding, n_groups):
        super(MultiResolutionSumWaveNet, self).__init__()

        self.feature_extractor = FeatureExtractor(n_blocks, n_channel, n_mul, kernel_size, padding, n_groups)
        self.receptive_field = self.feature_extractor.get_receptive_field()

        self.prediction_layers = [PredictionLayer(n_channel, padding, n_groups, self.receptive_field) for _ in range(n_blocks)]
        self.concatenate = layers.Concatenate()

    def call(self, inputs, training=None, mask=None):
        x = self.feature_extractor(inputs)
        prediction = []
        for idx, layer in enumerate(self.prediction_layers):
            tmp = layer(x[:, :, :, idx])
            tmp = tf.expand_dims(tmp, axis=-1)
            prediction.append(tmp if len(prediction) == 0 else tmp + prediction[-1])
        output = self.concatenate(prediction)
        return output


class ArcMarginProduct(Model):
    def __init__(self, in_features=64, out_features=7, s=30.0, m=0.7, sub=1, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.sub = sub
        self.weight = self.add_weight('kernel',
                                      shape=[in_features, out_features * sub],
                                      initializer=tf.keras.initializers.GlorotUniform())

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def call(self, inputs, training=None, mask=None):
        norm = tf.keras.layers.Normalization()
        cosine = layers.Dense(norm(self.weight))(norm(inputs[0]))

        if self.sub > 1:
            cosine = cosine.view(-1, self.out_features, self.sub)
            cosine = tf.math.reduce_max(cosine, axis=2)
        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = tf.zeros(cosine.shape)

        one_hot.scatter_nd()

        one_hot.scatter_(1, inputs[1].view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output


