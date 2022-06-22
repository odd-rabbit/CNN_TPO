Reference:
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9693504

建议看看这篇文章:
https://mp.weixin.qq.com/s/65GtdBOS5Xx8lK7vMuymhw


###CNN_no_dependency.ipynb
This notebook does not have any dependencies, through which you can understand the general framework of this project

Data generation -> model training -> prediction results

###CNN.ipynb
Most of the functions used in this notebook are written in other .py.

Focus on the comparison of different model parameters

###evaluate.py
All the evaluation functions and visualization functions are written in it, and the processed prediction results can be put in to get the evaluation results.

###preprocess.py
The generation function of x, y, if you need to modify the generated data in CNN, change it here




###CNN_no_dependency.ipynb

这一个notebook没有任何依赖，通过它你可以了解这个项目的大致框架

数据生成->模型训练->预测结果

###CNN.ipynb
这个notebook的大多数用到的函数都写在了其他的.py中。

专注于对不同模型参数的比较

###CNNmodel.py
里面就是个输入为32*32的Lenet-5

###evaluate.py
里面写了所有的评估函数和可视化函数，将处理后的预测结果放进去能得到评估结果

###preprocess.py
x,y的生成函数，CNN中需要修改生成数据的话在这里面更改

###dataHandler.py
这个是close.csv, high.csv, low.csv的专属数据生成器

###Generator.py
这个是close.csv, high.csv, low.csv的专属结果评估器