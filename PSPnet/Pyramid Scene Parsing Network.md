# Pyramid Scene Parsing Network  

## 42 Matrix

|      | 定义                             | 特点                                                         |
| ---- | -------------------------------- | ------------------------------------------------------------ |
| 背景 | automatic driving, robot sensing | safe                                                         |
| 对象 | image                            | color                                                        |
| 问题 | Scene parsing                    | with unrestricted open vocabulary and diverse scenes pixel-wise |
| 方法 | Pyramid Scene Parsing Network    | with Pyramid Pooling Module developing a optimization strategy |

## 逻辑树

![pspnet思维导图](D:\work_DL\论文阅读\PSPnet\pspnet思维导图.PNG)

## 算法

![pspnet](D:\work_DL\论文阅读\PSPnet\pspnet.PNG)

```python
inputs = array([16,224,224,3])
#16,224,224,3
net=Conv2d(net,3,3,64,relu,stride=2)
#16,112,112,64
net=Conv2d(net,3,3,64,relu)
#16,112,112,64
net=Conv2d(net,3,3,64,relu)
#16,112,112,64
net=Maxpool(3,3,stride=2)
#16,56,56,64
net = Bottleneck(net,64,64,256)
#16,56,56,256
net = Bottleneck(net,64,64,256)
#16,56,56,256
net = Bottleneck(net,64,64,256)
#16,56,56,256
#原来的maxpool去除
net = Bottleneck(net,128,128,512)
#16,56,56,512
net = Bottleneck(net,128,128,512)
#16,56,56,512
net = Bottleneck(net,128,128,512)
#16,56,56,512
net = Bottleneck(net,128,128,512)
#16,56,56,512
net = Maxpool(net)
#16,28,28,512
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,256,256,1024)
#16,28,28,1024
net = Bottleneck(net,512,512,2048)
#16,28,28,2048
net = Bottleneck(net,512,512,2048)
#16,28,28,2048
res = Bottleneck(net,512,512,2048)
#16,28,28,2048
#下面为金字塔模型部分
interp_block1 = interp_block(res, 1)
#16,28,28,512
interp_block2 = interp_block(res, 2)
#16,28,28,512
interp_block3 = interp_block(res, 3)
#16,28,28,512
interp_block6 = interp_block(res, 6)
#16,28,28,512
net = concat (res,interp_block1,interp_block2,interp_block3,interp_block4)
#16,28,28,4096
net=Conv2d(net,3,3,512,relu)
#16,28,28,512
net=Conv2d(net,1,1,nb_classes,relu)
#16,28,28,nb_classes
net = Interp(net,8)
#16,224,224,nb_classes
output = softmax(net)
#16,224,224,1

loss = cross_entropy(outputs, labels)
#using auxiliary loss in ResNet101
train = Momentum(learning_rate = 0.01,momentum = 0.9)
#use the poly learning rate policy and power is 0.9 
```

## 实验结果

| 数据库名称      | mIoU | IoU cla | iIoU cla | IoU cat. | iIoU cat. |
| --------------- | ---- | ------- | -------- | -------- | --------- |
| PASCAL VOC 2012 | 85.4 | ~       | ~        | ~        | ~         |
| Cityscapes      | 80.2 | 80.2    | 58.1     | 90.6     | 78.6      |

