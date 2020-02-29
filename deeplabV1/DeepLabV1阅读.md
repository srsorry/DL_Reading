# SEMANTIC IMAGE SEGMENTATION WITH DEEP CONVOLUTIONAL NETS AND FULLY CONNECTED CRFS  

## 42 Matrix

|      | 定义             | 特点                                |
| ---- | ---------------- | ----------------------------------- |
| 背景 | General field    |                                     |
| 对象 | image            | color                               |
| 问题 | SEGMENTATION     | pixel-level                         |
| 方法 | “DeepLab” system | using CRF using ‘atrous’  algorithm |

## 逻辑树

![deeplabv1思维导图](D:\work_DL\论文阅读\DeepLabV1\deeplabv1思维导图.PNG)

## 算法

```python
inputs = array([20,224,224,3])
net=Conv2d(inputs,3,3,64,relu)
#20,224,224,64
net=Conv2d(net,3,3,64,relu)
#20,224,224,64
net=Maxpool(net)
#20,112,112,64
net=Conv2d(net,3,3,128,relu)
#20,112,112,128
net=Conv2d(net,3,3,128,relu)
#20,112,112,128
net=Maxpool(net)
#20,56,56,128
net=Conv2d(inputs,3,3,256,relu)
#20,56,56,256
net=Conv2d(net,3,3,256,relu)
#20,56,56,256
net=Conv2d(net,3,3,256,relu)
#20,56,56,256
net=Maxpool(net)
#20,28,28,256
net=Conv2d(net,3,3,512,relu)
#20,28,28,512
net=Conv2d(net,3,3,512,relu)
#20,28,28,512
net=Conv2d(net,3,3,512,relu)
#20,28,28,512
net=Conv2d(net,3,3,512,relu,hole=2)
#20,28,28,512
net=Conv2d(net,3,3,512,relu,hole=2)
#20,28,28,512
net=Conv2d(net,3,3,512,relu,hole=2)
#20,28,28,512
net=Conv2d(net,4,4,1024,relu,hole=8)
#20,28,28,1024
net=Conv2d(net,1,1,1024,relu)
#20,28,28,1024
net=Conv2d(net,1,1,1000,relu)
#20,28,28,1000
net= Interpolation(net)
#20,224,224,1000
net = crf(net)
loss = cross_entropy(outputs, labels)
train = Momentum(learning_rate = 0.001 ,momentum = 0.9)
#initial learning rate of 0.001 (0.01 for the final classifier layer)
#multiplying the learning rate by 0.1 at every 2000 iterations
```

## 实验结果

PASCAL VOC 2012   mean IOU  71.6

