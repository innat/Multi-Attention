import cv2
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras import backend as K

from attentionlayer.gwap_cbam import ChannelAttentionModule, SpatialAttentionModule
from attentionlayer.deep_moji2d import AttentionWeightedAverage2D

image_size = 71
def resize(mnist):
     train_data = []
     for img in mnist:
            resized_img = cv2.resize(img, (image_size, image_size))
            train_data.append(resized_img)
     return train_data

(xtrain, train_target), (_, _) = tf.keras.datasets.mnist.load_data()

# resize and prepare the input 
xtrain = resize(xtrain)
xtrain = np.expand_dims(xtrain, axis=-1)
xtrain = np.repeat(xtrain, 3, axis=-1)
xtrain = xtrain.astype('float32') / 255

# train set / target 
ytrain = tf.keras.utils.to_categorical(train_target, num_classes=10)

class Classifier(tf.keras.Model):
    def __init__(self, dim):
        super(Classifier, self).__init__()
        # Defining All Layers in __init__
        # Layer of Block
        self.Base  = tf.keras.applications.Xception(
            input_shape=(image_size, image_size, 3),
            weights=None,
            include_top=False)
        # Keras Built-in
        self.GAP1 = tf.keras.layers.GlobalAveragePooling2D()
        self.GAP2 = tf.keras.layers.GlobalAveragePooling2D()
        self.BAT  = tf.keras.layers.BatchNormalization()
        self.ADD  = tf.keras.layers.Add()
        self.AVG  = tf.keras.layers.Average()
        self.DROP = tf.keras.layers.Dropout(rate=0.5)
        # Customs
        self.CAN  = ChannelAttentionModule()
        self.SPN1 = SpatialAttentionModule()
        self.SPN2 = SpatialAttentionModule()
        self.AWG  = AttentionWeightedAverage2D()
        # Tail
        self.DENS = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.OUT  = tf.keras.layers.Dense(10,  activation='softmax', dtype=tf.float32)
    
    def call(self, input_tensor, training=False):
        # Base Inputs
        x  = self.Base(input_tensor)
        # Attention Modules 1
        # Channel Attention + Spatial Attention 
        canx   = self.CAN(x)*x
        spnx   = self.SPN1(canx)*canx
        # Global Weighted Average Poolin
        gapx   = self.GAP1(spnx)
        wvgx   = self.GAP2(self.SPN2(canx))
        gapavg = self.AVG([gapx, wvgx])
        # Attention Modules 2
        # Attention Weighted Average (AWG)
        awgavg = self.AWG(x)
        # Summation of Attentions
        x = self.ADD([gapavg, awgavg])
        # Tails
        x = self.BAT(x)
        x = self.DENS(x)
        x  = self.DROP(x, training=training)
        return self.OUT(x)
    
    # AFAIK: The most convenient method to print model.summary() in suclassed model
    def build_graph(self):
        x = tf.keras.layers.Input(shape=(image_size, image_size, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
    
# plot 
model = Classifier((image_size, image_size, 3))
tf.keras.utils.plot_model(
    model.build_graph(), show_shapes=True, 
    show_layer_names=True, expand_nested=False                      
)

model.compile(
          metrics=['accuracy'],
          loss = 'categorical_crossentropy',
          optimizer = 'adam'
          )

model.fit(xtrain, ytrain, epochs=5)
