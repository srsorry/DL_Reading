# Non-local Neural Networks  
Convolution for Semantic Image Segmentation    

## 42 Matrix

|      | 定义                      | 特点                                                         |
| ---- | ------------------------- | ------------------------------------------------------------ |
| 背景 | General field             |                                                              |
| 对象 | image video               | color                                                        |
| 问题 | Capturing dependencies    | long-range                                                   |
| 方法 | Non-local Neural Networks | between any two positions easily combined  computationally economical |

## 逻辑树

![Non-local](https://github.com/srsorry/DL_Reading/blob/master/Non-local/Non-local.PNG)

## 算法

```python
inputs = array([64,224,224,3])
#64,224,224,3
net=Conv2d(inputs,7,7,64,relu,stride=2)
#64,112,112,64
net=Maxpool(3,3,stride=2)
#64,56,56,64
net = Bottleneck(net,64,64,256)
#64,56,56,256
net = Bottleneck(net,64,64,256)
#64,56,56,256
net = Bottleneck(net,64,64,256)
#64,56,56,256
net = Maxpool(net)
#64,28,28,256
net = Bottleneck(net,128,128,512)
#64,28,28,512
net = Non-local(net)
#64,28,28,512
net = Bottleneck(net,128,128,512)
#64,28,28,512
net = Non-local(net)
#64,28,28,512
net = Bottleneck(net,128,128,512)
#64,28,28,512
net = Non-local(net)
#64,28,28,512
net = Bottleneck(net,128,128,512)
#64,28,28,512
net = Non-local(net)
#64,28,28,512
net = Maxpool(net)
#64,14,14,512
net = Bottleneck(net,256,256,1024)
#64,14,14,1024
net = Non-local(net)
#64,14,14,1024
net = Bottleneck(net,256,256,1024)
#64,14,14,1024
net = Non-local(net)
#64,14,14,1024
net = Bottleneck(net,256,256,1024)
#64,14,14,1024
net = Non-local(net)
#64,14,14,1024
net = Bottleneck(net,256,256,1024)
#64,14,14,1024
net = Non-local(net)
#64,14,14,1024
net = Bottleneck(net,256,256,1024)
#64,14,14,1024
net = Non-local(net)
#64,14,14,1024
net = Bottleneck(net,256,256,1024)
#64,14,14,1024
net = Non-local(net)
#64,14,14,1024
net = Maxpool(net)
#64,7,7,1024
net = Bottleneck(net,512,512,2048)
#64,7,7,2048
net = Bottleneck(net,512,512,2048)
#64,7,7,2048
net = Bottleneck(net,512,512,2048)
#64,7,7,2048
net = Averagepool(net)
#64,7,7,2048
net=Flatten(net)
#64,100352
net=FullyConected(net,1000)
#64,1000
outputs=Softmax(net)
#64,1
loss=cross_entropy(labels, outputs）
train = Momentum(learning_rate = 0.1 ,momentum = 0.9)
# earning rate of 0.01 and reducing it by a factor of 10 at every 150k iterations
#momentum of 0.9 and a weight decay of 0.0001
```

## 实验结果

| 数据库名称     | 指标                    | 最佳性能 |
| -------------- | ----------------------- | -------- |
| Kinetics top-1 | classification accuracy | 77.7     |
| Kinetics top-5 | classification accuracy | 93.3     |
| Charades       | Classification accuracy | 39.5     |
| COCO           | AP                      | 66.5     |
| COCO           | AP50                    | 87.3     |
| COCO           | AP75                    | 72.8     |

