# Fully Convolutional Networks for Semantic Segmentation  

## 42 Matrix

|      | 定义                        | 特点                                                         |
| ---- | :-------------------------- | :----------------------------------------------------------- |
| 背景 | General field               |                                                              |
| 对象 | image                       | color                                                        |
| 问题 | Segmentation                | Semantic    pixelwise    inherent    between semantics and location |
| 方法 | Fully Convolutional Network | Fully Convolutional    using upsampling    using pre-trianing    using  "skip" architecture |

## 逻辑树

![FCN思维导图](https://github.com/srsorry/DL_Reading/blob/master/FCN/FCN%E6%80%9D%E7%BB%B4%E5%AF%BC%E5%9B%BE.PNG)

## 算法

```python
inputs = array([N,224,224,3])
net=Conv2d(inputs,3,3,64,relu)
#N,224,224,64
net=Conv2d(net,3,3,64,relu)
#N,224,224,64
pool1=Maxpool(net)
#N,112,112,64
net=Conv2d(pool1,3,3,128,relu)
#N,112,112,128
net=Conv2d(net,3,3,128,relu)
#N,112,112,128
pool2=Maxpool(net)
#N,56,56,128
net=Conv2d(pool2,3,3,256,relu)
#N,56,56,256
net=Conv2d(net,3,3,256,relu)
#N,56,56,256
net=Conv2d(net,3,3,256,relu)
#N,56,56,256
pool3=Maxpool(net)
#N,28,28,256
net=Conv2d(pool3,3,3,512,relu)
#N,28,28,512
net=Conv2d(net,3,3,512,relu)
#N,28,28,512
net=Conv2d(net,3,3,512,relu)
#N,28,28,512
pool4=Maxpool(net)
#N,14,14,512
net=Conv2d(pool4,3,3,512,relu)
#N,14,14,512
net=Conv2d(net,3,3,512,relu)
#N,14,14,512
net=Conv2d(net,3,3,512,relu)
#N,14,14,512
pool5=Maxpool(net)
#N,7,7,512
net = Conv2d(pool5,7,7,4096,relu,dropout_ratio=0.5)
#N,7,7,4096
net = Conv2d(net,1,1,4096,relu,dropout_ratio=0.5)
#N,7,7,4096
score_fr=Conv2d(net,1,1,60)
#N,7,7,60
fcn_32s_pre = DeConv2d(score_fr,n,n,60,stride=32)
#N,224,224,60
fcn_32s_output = Softmax(fcn_32s_pre)
#N,224,224,1
upscore2 = bilinear_interpolation(score_fr,f=2)
#N,14,14,60
score_pool4 = Conv2d(pool4,1,1,60)
#N,14,14,60
fuse_pool4 = concat(score_pool4,upscore2)
#N,14,14,120
net =bilinear_interpolation(net,f=16)
#N,224,224,120
fcn_16s_output = Softmax(net)
#N,224,224,1
upscore_pool4 = bilinear_interpolation(fuse_pool4,f=2)
#N,28,28,60
score_pool3 = Conv2d(pool3,1,1,60)
#N,28,28,60
fuse_pool3 = concat(score_pool3,upscore_pool4)
#N,28,28,120
net = bilinear_interpolation(fuse_pool3,f=8)
#N,224,224,120
fcn_8s_output = Softmax(net)
#N,224,224,1

loss=cross_entropy(labels, outputs）

#train by SGD with momentum 
 train = Momentum(learning_rate = 0.0001 ,momentum = 0.9)
#our full image training effectively batches each image into a regular grid of large, overlapping patche
```



# 实验结果

| 数据库名称      | pixel acc. | mean IU | mean acc. | f.w. IU |
| --------------- | ------- | -------- | --------------- | --------------- |
| PASCAL VOC 2011 | 90.3 | 62.7     | 75.9 | 83.2 |
| PASCAL VOC 2012 | ~ | 62.2     | ~ | ~ |
| NYUDv2          | 65.4 | 34.0     | 46.1 | 49.5 |
|  SIFT Flow      | 85.2 | 39.5 | 51.7 | 76.1 |
| PASCAL-Context 59 class | 65.9 | 35.1 | 46.5 | 51.0 |
| PASCAL-Context 33 class | 71.8 | 53.5 | 68.0 | 57.7 |

$$
pixel\quad accuracy:\sum_{i}{}{n_{ii}}/\sum_{i}{}{t_i}
\\
mean\quad accuracy:(1/n_{cl})\sum_{i}{}{n_{ii}}/t_i
\\
mean\quad IU:(1/n_{cl})\sum_{i}{}{n_{ii}}/(t_i+\sum_{j}{}{n_{ji}-n_{ii}})
\\
frequency\quad weithted\quad IU:
\\
(\sum_{k}{}{t_k})^{-1}\sum_{i}{}{t_in_{ii}}/(t_i+\sum_{j}{}{n_{ji}-n_{ii}})
$$



