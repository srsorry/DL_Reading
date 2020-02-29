# SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image
Segmentation
## 42 Matrix

|      | 定义               | 特点                                                         |
| ---- | ------------------ | ------------------------------------------------------------ |
| 背景 | autonomous driving | safe fast                                                    |
| 对象 | Image              | Road Scenes,pixels belong to large classes                   |
| 问题 | segmentation       | pixel-wise  both memory and computation time co-occurrences and spatial-context |
| 方法 | SegNet             | Deep Convolutional  Encoder-Decoder  using non-linear upsampling        using the max-pooling indices using a new benchmark challenge |

## 逻辑树

![segnet思维导图](D:\work_DL\论文阅读\segnet\segnet思维导图.PNG)

## 算法

```python
inputs = array([N,360,480,3])
net=Conv2d(inputs,3,3,64,relu)
#N,360,480,64
net=Conv2d(inputs,3,3,64,relu)
#N,360,480,64
net=Maxpool(net)
#N,180,240,64
net=Conv2d(inputs,3,3,128,relu)
#N,180,240,128
net=Conv2d(inputs,3,3,128,relu)
#N,180,240,128
net=Maxpool(net)
#N,90,120,128
net=Conv2d(inputs,3,3,256,relu)
#N,90,120,256
net=Conv2d(inputs,3,3,256,relu)
#N,90,120,256
net=Conv2d(inputs,3,3,256,relu)
#N,90,120,256
net=Maxpool(net)
#N,45,60,256
net=Conv2d(inputs,3,3,512,relu)
#N,45,60,512
net=Conv2d(inputs,3,3,512,relu)
#N,45,60,512
net=Conv2d(inputs,3,3,512,relu)
#N,45,60,512
net=Maxpool(net,stride=1)
#N,45,60,512
net=Upsampling(net,stride=1)
#N,45,60,512
net=Conv2d(inputs,3,3,512,relu)
#N,45,60,512
net=Conv2d(inputs,3,3,512,relu)
#N,45,60,512
net=Conv2d(inputs,3,3,512,relu)
#N,45,60,512
net=Upsampling(net)
#N,90,120,512
net=Conv2d(inputs,3,3,256,relu)
#N,90,120,256
net=Conv2d(inputs,3,3,256,relu)
#N,90,120,256
net=Conv2d(inputs,3,3,256,relu)
#N,90,120,256
net=Upsampling(net)
#N,180,240,256
net=Conv2d(inputs,3,3,128,relu)
#N,180,240,128
net=Conv2d(inputs,3,3,128,relu)
#N,180,240,128
net=Upsampling(net)
#N,360,480,256
net=Conv2d(inputs,3,3,64,relu)
#N,360,480,64
net=Conv2d(inputs,3,3,64,relu)
#N,360,480,64
output= Softmax(net)
#N,360,480,K (K is the number of classes)
"""For the road scenes which have 11 classes we used a mini-batch size of 5 and for indoor scenes with 37 classes we used a mini-batch size of 4"""
loss = cross_entropy(outputs, labels)
train = Momentum(learning_rate = 0.1,momentum = 0.9)
```

## 实验结果

| 数据库名称       | global accuracy (G) | class average accuracy (C) | mIoU  | BF    |
| ---------------- | ------------------- | -------------------------- | ----- | ----- |
| CamVid test set  | 90.40               | 71.20                      | 60.10 | 46.84 |
| SUNRGB-D dataset | 72.63               | 72.63                      | 72.63 | 12.66 |

