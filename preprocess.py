import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
预处理CNN的输入数据
用到了market profile市场轮廓指标
CNN的输入为一张张图片，在默认的例子中，图片的大小为32*32
图片的高为32个价格小区间，该值由图片中所含天数决定
图片的宽则分为4个大区间(clip)，4个大区间中有8个像素(time)，每一个代表一天
生成出来后将它转换为灰度从0.2到1的渐变灰度图
t为需要用到未来多少天
label的生成条件可在label_condition中修改


GenerateXY中
df要求输入dataframe的格式为, index可为任何值（整数，字符串，日期）
index   close   high    low
  0     .....   ....    ...
  1     .....   ....    ...
如果不使用默认time,clip或t，则需要在CNN模型处做更改
'''


class GenerateXY(object):

    def __init__(self, df, start=None, end=None, time=8, clip=4, t=1):

        # dataframe parm
        self.time = time
        self.clip = clip
        self.interval = self.time * self.clip

        self.T = t
        # T future and past is how much future/past data used to generate label
        self.T_future = self.interval

        if start is None:
            self.start = 0
        else:
            self.start = start
        if end is None:
            self.end = -self.T
        else:
            self.end = end-self.T

        self.df = df
        self.high = np.array(df['high'].iloc[self.start: self.end])
        self.low = np.array(df['low'].iloc[self.start: self.end])
        # self.volume = np.array(df['volume'].iloc[self.start: self.end])
        self.close = np.array(df['close'])


    def market_profile_many(self, index):

        cut_list = np.linspace(index, index+self.interval, self.clip + 1).astype(int)
        matrix = np.array([[]])
        # for every cut_interval
        # print(cut_list)
        price_list = np.linspace(np.min(self.low[cut_list[0]:cut_list[-1]]),
                                 np.max(self.high[cut_list[0]:cut_list[-1]]),
                                 self.interval)
        for c in range(1, len(cut_list)):

            length = cut_list[c] - cut_list[c - 1]
            high = self.high[cut_list[c - 1]:cut_list[c]]
            low = self.low[cut_list[c - 1]:cut_list[c]]
            out_array = np.full((self.interval, self.time), 0)
            # here using series max and min, not current cut max and min
            # print(self.close[cut_list[c - 1]:cut_list[c]])
            if np.size(low) < self.time or np.size(high) < self.time:
                return None
            # price_list = np.linspace(np.min(low), np.max(high), self.interval)
            time_list = np.linspace(0, length, self.time + 1).astype(int)
            # for every time_interval
            for i in range(1, len(time_list)):
                index = np.where((low[i-1] <= price_list) & (price_list <= high[i-1]))[0]
                # print(index)
                for row in index:
                    for column in range(self.time):
                        if out_array[row][column] == 0:
                            out_array[row][column] = i
                            break

            # merge all the cut in one big matrix
            if c == 1:
                matrix = out_array
            else:
                matrix = np.hstack((matrix, out_array))
        # 显示图片
        # plt.matshow(matrix)
        # plt.show()
        matrix = matrix.astype(np.float32)
        return matrix

    def to_grayscale(self, matrix):
        # 将图片转为0.6到1之间的渐变灰度图
        gray_scale = np.linspace(0.6, 1, self.time*self.clip)
        matrix[matrix > 0] = 1
        for i in range(len(matrix[0])):
            matrix[:, i] = matrix[:, i] * gray_scale[i]
        return matrix

    # def grayscale_volume(self, matrix, index):
    #     volume = self.volume[index:index+self.interval]
    #     std_volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    #     matrix[matrix > 0] = 1
    #     for i in range(len(matrix[0])):
    #         matrix[:, i] = matrix[:, i] * std_volume[i]
    #     # for i in range(self.clip):
    #     #     clip_volume = np.average(std_volume[i*self.time:(i+1)*self.time])
    #     #     matrix[:, i*self.time:(i+1)*self.time] = matrix[:, i*self.time:(i+1)*self.time] * clip_volume
    #     return matrix

    def label_condition(self, index):
        # 设置生成标签的条件
        price = self.close
        c = index + self.interval - 1
        # print('ic', c)
        # print('c', price[c])
        r = 0.02
        std = np.std(price[c - self.T: c])
        # # t等于需要用到未来多少天
        # if (1+2*r)*price[c-2*self.T] < (1+r)*price[c-self.T] < price[c] > (1+r)*price[c+self.T]:
        #     return 0
        # elif (1-2*r)*price[c-2*self.T] > (1-r)*price[c-self.T] > price[c] < (1-r)*price[c+self.T]:
        #     return 2
        # if price[c-2*self.T]+2*std < price[c-self.T]+std < price[c] > price[c+self.T]+std:
        #     return 0
        # elif price[c-2*self.T]-2*std > price[c-self.T]-std > price[c] < price[c+self.T]-std:
        #     return 2
        # if price[c-2*self.T] < price[c-self.T] < price[c] > price[c+self.T]:
        #     return 0
        # elif price[c-2*self.T] > price[c-self.T] > price[c] < price[c+self.T]:
        #     return 2
        # if (1-r)*price[c-self.T] > price[c] > (1+r)*price[c+self.T]:
        #     return 0
        # elif (1+r)*price[c-self.T] < price[c] < (1-r)*price[c+self.T]:
        #     return 2
        # if price[c-self.T] > price[c] > price[c+self.T]:
        #     return 0
        # elif price[c-self.T] < price[c] < price[c+self.T]:
        #     return 2
        if price[c-self.T]-std > price[c] > price[c+self.T]+std:
            return 0
        elif price[c-self.T]+std < price[c] < price[c+self.T]-std:
            return 2
        else:
            return 1

    def generate_xy(self, onehot=True):
        # 决定标签的判别策略，并生成CNN需要的格式
        x = []
        y = []
        if len(self.df) < self.interval:
            print('length is not enough to generate')
            return None, None
        for i in range(len(self.df)-self.interval-self.T+1):
            # print('start', i)
            matrix = self.market_profile_many(i)
            if matrix is not None:
                # matrix = self.grayscale_volume(matrix, i)
                matrix = self.to_grayscale(matrix)
                label = self.label_condition(i)
                x.append(matrix)
                y.append(label)
                # plt.matshow(matrix, cmap=plt.cm.gray_r)
                # plt.axis('off')
                # plt.show()
        x = np.array(x)
        x = x.reshape(x.shape[0], self.interval, self.interval, 1)
        y = np.array(y)
        # print(y)
        if onehot and len(y) > 0:  # one hot 编码
            onehot_y = np.zeros((y.size, 3))
            onehot_y[np.arange(y.size), y] = 1
            return x, onehot_y
        else:
            return x, y

    def generate_x(self):
        # 生成最后一个x用来预测, 没有标签
        self.high = np.array(self.df['high'])
        self.low = np.array(self.df['low'])
        self.close = np.array(self.df['close'])

        x = []
        if len(self.df) < self.interval:
            print('length is not enough to generate')
            return None, None
        index = len(self.df)-self.interval-self.T+1
        # print('start', index)
        matrix = self.market_profile_many(index)
        if matrix is not None:
            matrix = self.to_grayscale(matrix)
            x.append(matrix)

        x = np.array(x)
        x = x.reshape(x.shape[0], self.interval, self.interval, 1)
        # print(y)

        return x


if __name__ == '__main__':
    df = pd.read_csv('BTCUSDT_1h.csv')
    Generator = GenerateXY(df.loc[0:63], t=8)
    # 生成多个x,y用来训练
    train, label = Generator.generate_xy()
    print('x: ', train.shape)
    print('y: ', label.shape)
    # 生成最后一个x用来预测, 没有标签
    x = Generator.generate_x()
    print('x: ', x.shape)
