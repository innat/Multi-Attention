![mattention](https://user-images.githubusercontent.com/17668390/104115771-f7795800-533c-11eb-9a57-0f3282604625.png)

## Multi-Attention

A simple lightweight attention mechanism wrapper in 2D convolution. It consists with attentin mdoules of **Convolution Block Attention Module (CBAM)** and **DeepMoji**. 


[**CBAM**](https://arxiv.org/abs/1807.06521): Convolutional Block Attention Module is a `dual attention` mechanism. It learns the informative features by integrating **channel-wise attention** and **spatial attention** together. The module is set in sequential order, begin with channel-wise followed by the spatial module. Additionally we add a weighting term to the Global Average Pooling (**GAP**) at the end of **CBAM**; such as 

![11](https://user-images.githubusercontent.com/17668390/104115831-bc2b5900-533d-11eb-9fa7-8ef09785a2c2.png)


## Usages 

simply run

```
python train.py
```

```
Epoch 1/5
1875/1875 [==============================] - 178s 89ms/step - loss: 0.4090 - accuracy: 0.8990
Epoch 2/5
1875/1875 [==============================] - 166s 89ms/step - loss: 0.0971 - accuracy: 0.9744
Epoch 3/5
1875/1875 [==============================] - 166s 89ms/step - loss: 0.0910 - accuracy: 0.9775
Epoch 4/5
1875/1875 [==============================] - 166s 89ms/step - loss: 0.0797 - accuracy: 0.9810
Epoch 5/5
1875/1875 [==============================] - 166s 89ms/step - loss: 0.0611 - accuracy: 0.9851
```

Model details 

```python
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
```


For live example, check this [notebook](https://www.kaggle.com/ipythonx/tf-keras-ranzcr-multi-attention-efficientnet/notebook). 


