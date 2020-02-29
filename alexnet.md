# ImageNet Classification with Deep Convolutional Neural Networks  

## 42 Matrix

|      | 定义                               | 特点                                                         |
| ---- | ---------------------------------- | ------------------------------------------------------------ |
| 背景 | competition                        | with a big datasets                                          |
| 对象 | Image                              | from ImageNet  in realistic settings million high-resolution |
| 问题 | Classification overfitting         |                                                              |
| 方法 | Deep Convolutional Neural Networks | using non-saturating neurons  using GPU for convolution using dropout purely supervised |

## 逻辑树

![alxnet思维导图](D:\work_DL\论文阅读\alexnet\alxnet思维导图.PNG)

## 算法

```python
inputs = array([128,227,227,3])
net=Conv2d(inputs,11,11,96,stride=4,relu)
#128,57,57,96
net=Maxpool(net)
#128,28,28,96
net=Conv2d(net,5,5,256,relu)
#128,28,28,256
net=Maxpool(net)
#128,13,13,256
net=Conv2d(net,3,3,384,relu)
#128,13,13,384
net=Conv2d(net,3,3,384,relu)
#128,13,13,384
net=Conv2d(net,3,3,256,relu)
#128,13,13,256
net=Maxpool(net) 
#128,6,6,256
net=Flatten(net)
#128,9216
net=FullyConected(net,4096)
#128,4096
net=FullyConected(net,4096)
#128,4096
net=FullyConected(net,1000)
#128,1000
outputs=Softmax(net)

train = Momentum(learning rate = 0.0005,momentum = 0.9)
#stochastic gradient descent

loss=cross_entropy(labels, outputs）
```

## 实验结果

| 数据库名称        | error rates |
| ----------------- | ----------- |
| ILSVRC-2010-Top-1 | 37.5%       |
| ILSVRC-2010-Top-5 | 17.0%       |
| ILSVRC-2012-Top-1 | 36.7%       |
| ILSVRC-2012-Top-5 | 15.3%       |

