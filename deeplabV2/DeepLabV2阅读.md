# DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution,and Fully Connected CRFs  

## 42 Matrix

|      | 定义             | 特点                                                 |
| ---- | ---------------- | ---------------------------------------------------- |
| 背景 | General field    |                                                      |
| 对象 | image            | color                                                |
| 问题 | SEGMENTATION     | pixel-level                                          |
| 方法 | “DeepLab” system | using CRF using atrous convolution using ASPP module |

## 逻辑树 

![deeplabv2思维导图](https://github.com/srsorry/DL_Reading/blob/master/deeplabV2/deeplabv2%E6%80%9D%E7%BB%B4%E5%AF%BC%E5%9B%BE.PNG)

## 算法

```python
inputs = array([10,224,224,3])
net=Conv2d(inputs,3,3,64,relu)
#10,224,224,64
net=Conv2d(net,3,3,64,relu)
#10,224,224,64
net=Maxpool(net)
#10,112,112,64
net=Conv2d(net,3,3,128,relu)
#10,112,112,128
net=Conv2d(net,3,3,128,relu)
#10,112,112,128
net=Maxpool(net)
#10,56,56,128
net=Conv2d(net,3,3,256,relu)
#10,56,56,256
net=Conv2d(net,3,3,256,relu)
#10,56,56,256
net=Conv2d(net,3,3,256,relu)
#10,56,56,256
net=Maxpool(net)
#10,28,28,256
net=Conv2d(net,3,3,512,relu)
#10,28,28,512
net=Conv2d(net,3,3,512,relu)
#10,28,28,512
net=Conv2d(net,3,3,512,relu)
#10,28,28,512
net=Conv2d(net,3,3,512,relu,rate =2)
#10,28,28,512
net=Conv2d(net,3,3,512,relu,rate =2)
#10,28,28,512
pool5=Conv2d(net,3,3,512,relu,rate =2)
#10,28,28,512
net1=Conv2d(pool5,3,3,1024,relu,rate =6)
#10,28,28,1024
net1=Conv2d(net1,1,1,1024,relu)
#10,28,28,1024
net1=Conv2d(net1,1,1,1000,relu)
#10,28,28,1000
net2=Conv2d(pool5,3,3,1024,relu,rate =12)
#10,28,28,1024
net2=Conv2d(net2,1,1,1024,relu)
#10,28,28,1024
net2=Conv2d(net2,1,1,1000,relu)
#10,28,28,1000
net3=Conv2d(pool5,3,3,1024,relu,rate =18)
#10,28,28,1024
net3=Conv2d(net3,1,1,1024,relu)
#10,28,28,1024
net3=Conv2d(net3,1,1,1000,relu)
#10,28,28,1000
net4=Conv2d(pool5,3,3,1024,relu,rate =24)
#10,28,28,1024
net4=Conv2d(net4,1,1,1024,relu)
#10,28,28,1024
net4=Conv2d(net4,1,1,1000,relu)
#10,28,28,1000
net = sum_Fusion(net1,net2,net3,net4)
#10,28,28,4000
net= Interpolation(net)
#10,224,224,4000
net = crf(net)
loss = cross_entropy(outputs, labels)
train = Momentum(learning_rate = 0.001 ,momentum = 0.9)
# using poly policy andthe power =0.9
```

## 实验结果

| 数据库               | mIOU  |
| -------------------- | ----- |
| PASCAL VOC 2012 test | 79.7  |
| PASCAL-Context       | 45.7  |
| PASCAL-Person-Part   | 64.94 |
| Cityscapes           | 71.4  |

