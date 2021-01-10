import tensorflow as tf


# Original Src: https://github.com/bfelbo/DeepMoji/blob/master/deepmoji/attlayer.py
# Adopted and Modified: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77269#454482
class AttentionWeightedAverage2D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.init = tf.keras.initializers.get('uniform')
        super(AttentionWeightedAverage2D, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]
        assert len(input_shape) == 4
        self.W = self.add_weight(shape=(input_shape[3], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self._trainable_weights = [self.W]
        super(AttentionWeightedAverage2D, self).build(input_shape)

    def call(self, x):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 2-dimensional weights
        logits  = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits  = K.reshape(logits, (x_shape[0], x_shape[1], x_shape[2]))
        ai      = K.exp(logits - K.max(logits, axis=[1,2], keepdims=True))
        
        att_weights    = ai / (K.sum(ai, axis=[1,2], keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result         = K.sum(weighted_input, axis=[1,2])
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[3]
        return (input_shape[0], output_len)
