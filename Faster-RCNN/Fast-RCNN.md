# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks  

## 42 Matrix

|      | 定义             | 特点                        |
| ---- | ---------------- | --------------------------- |
| 背景 | General field    |                             |
| 对象 | image            | color                       |
| 问题 | Object Detection | Real-Time                   |
| 方法 | Faster R-CNN     | with RPN with anchors boxes |

## 逻辑树

![Non-local](D:\work_DL\论文阅读\Non-local.PNG)

## 算法

```python
inputs = array([N,800,600,3])
#N,800,600,3
Feature_Map = VGG16(inputs)
#N,50,38,512
net0 = Conv2d(Feature_Map,1,1,18)
#N,50,38,18
net1 = Conv2d(Feature_Map,1,1,36)
#N,50,38,36
net = softmax(net0)
#N,50,38,18
net = Proposal(net,net1,im_info)
#N,Mx(x1,y1,x2,y2)
net = ROIPooling(net,Feature_Map,7,7,0.0625)
#N,Mx(7x7x512)
net=FullyConected(net,4096)
#N,Mx4096
net=FullyConected(net,4096)
#N,Mx4096
cls_score = softmax(net)
#N,(K+1)xM
bbox_reg = bbox_pred(net)
#N,(k+1)x4xM

#训练过程
#loss
#结构图
```

$$
L({p_i},{t_i}) = \frac{1}{N_{cls}}\sum_iL_{cls}(p_i,p^*_i)+\lambda\frac{1}{N_{reg}}\sum_ip_i^*L_{reg}(t_i,t_i^*)
$$





## 实验结果

| 数据库名称      | 指标   | 最佳性能 |
| --------------- | ------ | -------- |
| PASCAL VOC 2007 | mAP    | 78.8     |
| PASCAL VOC 2012 | mAP    | 75.9     |
| MS COCO         | mAP@.5 | 42.7     |

