# SSD: Single Shot MultiBox Detector    
## 42 Matrix

|      | 定义              | 特点                                                         |
| ---- | ----------------- | ------------------------------------------------------------ |
| 背景 | General field     |                                                              |
| 对象 | image             | color                                                        |
| 问题 | detecting objects | real-time                                                    |
| 方法 | SSD               | Single Shot  MultiBox   using a small convolutional filter   using separate predictors |

## 逻辑树

![Non-local](https://github.com/srsorry/DL_Reading/blob/master/SSD/SSD.PNG)

## 算法

```python
inputs = array([32,300,300,3])
net=Conv2d(inputs,3,3,64,relu)
#32,300,300,64
net=Conv2d(net,3,3,64,relu)
#32,300,300,64
net=Maxpool(net)
#32,150,150,64
net=Conv2d(net,3,3,128,relu)
#32,150,150,128
net=Conv2d(net,3,3,128,relu)
#32,150,150,128
net=Maxpool(net)
#32,75,75,128
net=Conv2d(net,3,3,256,relu)
#32,75,75,256
net=Conv2d(net,3,3,256,relu)
#32,75,75,256
net=Conv2d(net,3,3,256,relu)
#32,75,75,256
net=Maxpool(net)
#32,38,38,256
net=Conv2d(net,3,3,512,relu)
#32,38,38,512
net=Conv2d(net,3,3,512,relu)
#32,38,38,512
net0=Conv2d(net,3,3,512,relu)
#32,38,38,512
net1=Conv2d(net0,3,3,84,relu)
#32,38,38,84
net=Maxpool(net0)
#32,19,19,512
net=Conv2d(net,3,3,1024,relu)
#32,19,19,1024
net2=Conv2d(net,1,1,1024,relu)
#32,19,19,1024
net3=Conv2d(net2,3,3,126,relu)
#32,19,19,124
net=Conv2d(net2,1,1,256,relu)
#32,19,19,256
net4=Conv2d(net,3,3,512,relu,stride=2)
#32,10,10,512
net5=Conv2d(net2,3,3,126,relu)
#32,10,10,124
net=Conv2d(net4,1,1,128,relu)
#32,10,10,128
net6=Conv2d(net,3,3,256,relu,stride=2)
#32,5,5,256
net7=Conv2d(net2,3,3,126,relu)
#32,5,5,124
net=Conv2d(net6,1,1,128,relu)
#32,5,5,128
net8=Conv2d(net,3,3,256,relu,padding=0)
#32,3,3,256
net9=Conv2d(net8,3,3,126,relu)
#32,3,3,126
net=Conv2d(net8,1,1,128,relu)
#32,3,3,128
net10=Conv2d(net,3,3,256,relu,padding=0)
#32,1,1,256
net11=Conv2d(net10,3,3,84,relu)
#32,1,1,84
net12=Conv2d(net10,1,1,128,relu)
#32,1,1,128
net=Conv2d(net,3,3,256,relu)
#32,1,1,256
net13=Conv2d(net,3,3,84,relu)
#32,1,1,84

output = NonMaximum(net1,net3,net5,net7,net9,net11,net13)

train = Momentum(learning_rate = 0.1 ,momentum = 0.9)
# earning rate of 0.01 and reducing it by a factor of 10 at every 150k iterations
#momentum of 0.9 and a weight decay of 0.0001
```

$$
L(x,c,l,g)=\frac{1}{N}(L_{conf}(x,c)+aL_{loc}(x,l,g))
$$



## 实验结果

| 数据库名称        | 指标         | 最佳性能 |
| ----------------- | ------------ | -------- |
| PASCAL VOC2007    | mAP          | 81.6     |
| PASCAL VOC2012    | mAP          | 80.0     |
| COCO test-dev2015 | IoU:0.5-0.95 | 26.8     |

