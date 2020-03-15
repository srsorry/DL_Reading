# Encoder-Decoder with Atrous Separable
Convolution for Semantic Image Segmentation    

## 42 Matrix

|      | 定义                | 特点                                                         |
| ---- | ------------------- | ------------------------------------------------------------ |
| 背景 | General field       |                                                              |
| 对象 | image               | color                                                        |
| 问题 | Segmentation        | Semantic  capturing rich contextual information obtaining sharp object boundaries |
| 方法 | “DeepLabv3+” system | with spatial pyramid pooling  with encoder-decoder with depthwise separable convolution |

## 逻辑树

![deeplabV3+思维导图](D:\work_DL\论文阅读\deeplabV3+思维导图.PNG)

## 算法

```python
inputs = array([32,299,299,3])
#32,299,299,3
net=Conv2d(inputs,3,3,32,stride=2)
#32,150,150,32
net0=Conv2d(net,3,3,64)
#32,150,150,64
net = SepConv(net0,3,3,128)
#32,150,150,128
net = SepConv(net,3,3,128)
#32,150,150,128
net = SepConv(net,3,3,128,stride=2)
#32,75,75,128
net1 = Conv2d(net0,1,1,128,stride=2)
#32,75,75,128
net2 = sum(net1,net)
#32,75,75,128，得到1/4的特征图
net = SepConv(net2,3,3,256)
#32,75,75,256
net = SepConv(net,3,3,256)
#32,75,75,256
net = SepConv(net,3,3,256,stride=2)
#32,38,38,256
net3 = Conv2d(net2,1,1,256,stride=2)
#32,38,38,256
net4 = sum(net3,net)
#32,38,38,256
net = SepConv(net4,3,3,728)
#32,38,38,728
net = SepConv(net,3,3,728)
#32,38,38,728
net = SepConv(net,3,3,728,stride=2)
#32,19,19,728
net5 = Conv2d(net4,1,1,728,stride=2)
#32,19,19,728
net6 = sum(net5,net)
#32,19,19,728
for i range (16):
    net = SepConv(netin,3,3,728)
    net = SepConv(net,3,3,728)
    net = SepConv(net,3,3,728)
    netout = sum(netin,net)
    #32,19,19,728
#得到了1/16的特征图
#ASPP模块
neta = Conv2d (net,1,1,1024)
#32,19,19,728
netb = SepConv (net,3,3,1024,rate =6)
#32,19,19,728
netc = SepConv (net,3,3,1024,rate =12)
#32,19,19,728
netd = SepConv (net,3,3,1024,rate =18)
#32,19,19,728
image_pool = avg(net)
#32,19,19,728
net = concat(neta,netb,netc,netd,image_pool)
#32,19,19,728*5
net = Conv2d(net,1,1,728)
#32,19,19,728
net_upsample4 = bilinear_interpolation(net,f=4)
#32,75,75,728
net = Conv2d(net2,1,1,728)
#32,75,75,728
net = concat(net,net_upsample4)
#32,75,75,728*2
net = Conv2d(net,3,3,N)
#32,75,75,N
net = bilinear_interpolation(net,f=4)
#32,299,299,N
output = Softmax(net)
#32,299,299,1
loss=cross_entropy(labels, outputs）
train = Momentum(learning_rate = 0.05 ,momentum = 0.9)
# rate decay = 0.94 every 2 epochs, and weight decay 4e − 5
```

## 实验结果

PASCAL VOC 2012   mean IOU  89.0

Cityscapes val set   mean IOU   82.1