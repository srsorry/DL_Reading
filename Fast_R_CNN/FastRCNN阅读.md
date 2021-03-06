# Fast R-CNN      
## 42 Matrix

|      | 定义             | 特点                                                  |
| ---- | ---------------- | ----------------------------------------------------- |
| 背景 | General field    |                                                       |
| 对象 | image            | color                                                 |
| 问题 | object detection | more complex with numerous proposals rough to precise |
| 方法 | Fast R-CNN       | using VGG  single-stage   using a multi-task loss     |

## 逻辑树

![Non-local](https://github.com/srsorry/DL_Reading/blob/master/Fast_R_CNN/FASTR-CNN.PNG)

## 算法

![s](https://github.com/srsorry/DL_Reading/blob/master/Fast_R_CNN/%E7%BB%93%E6%9E%84.jfif)


$$
L (p,u,t^u,v) = L_{cls}(p,u) + \lambda[u\geq1]L_{loc}(t^u,v)
\\
L_loc(t^u,v) = \sum_{i\epsilon \{ x,y,w,h\}} smooth_{L_1}(t^{u}_i-v_i)
\\
smooth_{L_1}(x) = \begin{cases} 0.5x^2 \quad if|x|<1 \\ |x| - 0.5 \quad otherwise \end{cases}  
$$


```python

train = Momentum(learning_rate = 0.001 ,momentum = 0.9)
#A momentum of 0:9 and parameter decayof 0.0005
```

## 实验结果

| 数据库名称 | 指标 | 最佳性能 |
| ---------- | ---- | -------- |
| VOC 2007   | mAP  | 70.0     |
| VOC 2010   | mAP  | 68.8     |
| VOC 2012   | mAP  | 68.4     |

