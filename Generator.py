import numpy as np
import pandas as pd
from dataHandler import Handler
from dataHandler import alpha_func
from CNNmodel import builder
from tensorflow import keras

'''
get_label：
对所有symbol生成对应的标签

get_predict：
对所有标签进行滚动训练以及预测
比如用第三年的做训练，用来预测第四年

get_position:
可以用get_label 或 get_predict的返回值
返回对应的持仓状况
'''


def get_label(close, high, low, t=1):
    data = close.copy()
    symbols = list(data.columns[1:])
    data['date'] = pd.to_datetime(data['time'])
    start_year = data.loc[0, 'date'].year
    end_year = list(data['date'])[-1].year + 1
    years = range(start_year, end_year)
    data = data.set_index('date')

    interval = 32
    label = data.copy()
    label.iloc[:, 1:] = np.nan

    for year in years:
        year = str(year)
        for symbol in symbols:
            x, y = Handler(close, high, low, t=t).generate_by_date_symbol(symbol, f'{year}/01/01', f'{year}/12/31')
            start, end = Handler(close, high, low, t=t).get_date(f'{year}/01/01', f'{year}/12/31')
            start = data['time'].shift(-interval + t).loc[start]
            end = data['time'].shift(t).loc[end]
            print(f'{year} : {symbol}')
            try:
                y = np.argmax(y, axis=1)
                if len(y) < len(label.loc[start:end, symbol]):
                    start = data['time'].shift(len(y) - 1).loc[end]
                label.loc[start:end, symbol] = y
            except:
                print('contiune')
    return label


def get_predict(close, high, low, t=1):
    data = close.copy()
    symbols = list(data.columns[1:])
    data['date'] = pd.to_datetime(data['time'])
    start_year = data.loc[0, 'date'].year
    end_year = list(data['date'])[-1].year
    years = range(start_year, end_year)
    data = data.set_index('date')
    interval = 32

    predict = data.copy()
    predict.iloc[:, 1:] = np.nan

    for year in years:
        # 这是个未解决的bug，date莫名其妙地出现在close中，第一次for循环没出现，往后就会出现
        try:
            del close['date']
        except:
            pass
        # ————————————————————————————————————————————————————————————
        year1 = str(year)
        x_train, y_train = Handler(close, high, low, t=1).generate_by_date(f'{year1}/01/01', f'{year1}/12/31')
        # 初始化新模型
        input_shape = (32, 32, 32, 1)
        model = builder(input_shape)
        # 训练模型
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=100)
        year2 = str(year+1)
        for symbol in symbols:
            print(f'{year2} : {symbol}')
            x_test, y_test = Handler(close, high, low, t=1).generate_by_date_symbol(symbol, f'{year2}/01/01', f'{year2}/12/31')
            start, end = Handler(close, high, low, t=1).get_date(f'{year2}/01/01', f'{year2}/12/31')
            start = data['time'].shift(-interval + t).loc[start]
            end = data['time'].shift(t).loc[end]
            try:
                y_pred = model.predict(x_test)
                y_pred = np.argmax(y_pred, axis=1)
                if len(y_pred) < len(predict.loc[start:end, symbol]):
                    start = data['time'].shift(len(y_pred) - 1).loc[end]
                predict.loc[start:end, symbol] = y_pred
            except:
                print('contiune')
    return predict


def get_position(df, hold=None, update=True):
    symbols = list(df.columns[1:])

    position = df.copy()
    position.iloc[:, 1:] = np.nan

    for symbol in symbols:
        series = df[symbol].dropna()
        # print(series)
        if len(series) > 0:
            position.loc[series.index[0]:series.index[-1], symbol] = alpha_func(list(series), hold=hold, update=update)
    return position


if __name__ == '__main__':
    df_close = pd.read_csv('../corr/close.csv')
    df_high = pd.read_csv('high.csv')
    df_low = pd.read_csv('low.csv')

    t = 16

    label = get_label(df_close, df_high, df_low, t=t)
    label.to_csv('label.csv', index=False)

    predict = get_predict(df_close, df_high, df_low, t=t)
    predict.to_csv('predict.csv', index=False)

    df_predict = pd.read_csv('predict.csv')
    position = get_position(df_predict, t)
    position.to_csv('position.csv', index=False)
