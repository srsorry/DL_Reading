# Rethinking Atrous Convolution for Semantic Image Segmentation  

## 42 Matrix

|      | 定义               | 特点                                                         |
| ---- | ------------------ | ------------------------------------------------------------ |
| 背景 | General field      |                                                              |
| 对象 | image              | color                                                        |
| 问题 | Segmentation       | Semantic  with reduced feature resolution with objects at multiple scales |
| 方法 | “DeepLabv3” system | using Atrous Convolution using ASPP using Upsampling         |

## 逻辑树

![deeplabV3思维导图](D:\work_DL\论文阅读\deeplabV3思维导图.PNG)

## 算法

```python
inputs = array([16,224,224,3])
#16,224,224,3
net=Conv2d(inputs,7,7,64,relu,stride=2)
#16,112,112,64
net=Maxpool(3,3,stride=2)
#16,56,56,64
net = Bottleneck(net,64,64,256)
#16,56,56,256
net = Bottleneck(net,64,64,256)
#16,56,56,256
net = Bottleneck(net,64,64,256)
#16,56,56,256
net = Maxpool(net)
#16,28,28,256
net = Bottleneck(net,128,128,512)
#16,28,28,512
net = Bottleneck(net,128,128,512)
#16,28,28,512
net = Bottleneck(net,128,128,512)
#16,28,28,512
net = Bottleneck(net,128,128,512)
#16,28,28,512
net = Maxpool(net)
#16,14,14,512
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
Block4 = Bottleneck(net,256,256,1024，atrous rate =2)
#16,14,14,1024
net1 = Conv2d (Block4,1,1,1024)
#16,14,14,1024
net2 = Conv2d (Block4,3,3,1024,rate =6)
#16,14,14,1024
net3 = Conv2d (Block4,3,3,1024,rate =12)
#16,14,14,1024
net4 = Conv2d (Block4,3,3,1024,rate =18)
#16,14,14,1024
image_pool = avg(Block4)
#16,14,14,1024
net = concat(net1,net2,net3,net4,image_pool)
#16,14,14,5120
net = Conv2d(net,1,1,1024)
#16,14,14,1024
net = bilinear_interpolation(net,f=16)
#16,224,224,1024
output = Softmax(net)
#16,224,224,1

loss=cross_entropy(labels, outputs）
train = Momentum(learning_rate = 0.007 ,momentum = 0.9)
# using poly learning rate policy,power = 0.9
# using Multigrid= (1,2,4)
```

## 实验结果

PASCAL VOC 2012   mean IOU  85.7