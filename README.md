![mattention](https://user-images.githubusercontent.com/17668390/104115771-f7795800-533c-11eb-9a57-0f3282604625.png)

## Multi-Attention

A simple lightweight attention mechanism wrapper in 2D convolution. It consists with attentin mdoules of **Convolution Block Attention Module (CBAM)** and **DeepMoji**. 


[**CBAM**](https://arxiv.org/abs/1807.06521): Convolutional Block Attention Module is a `dual attention` mechanism. It learns the informative features by integrating **channel-wise attention** and **spatial attention** together. The module is set in sequential order, begin with channel-wise followed by the spatial module. Additionally we add a weighting term to the Global Average Pooling (**GAP**) at the end of **CBAM**; such as 

![11](https://user-images.githubusercontent.com/17668390/104115831-bc2b5900-533d-11eb-9fa7-8ef09785a2c2.png)
