import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def alpha_func(signal, hold=None, update=True):
    '''
    :param signal: array of signal
    :param hold: Hold period once signal occur
    :param update: whether hold will be affected by other signal
    :return: array of position
    '''

    if hold == 0:
        return np.array([0 for _ in range(len(signal))])
    if hold is None:
        hold = 0

    hold -= 1
    position = [0 for _ in range(len(signal))]
    pre_pos = 0
    hold_count = 0
    for i in range(len(signal)):
        position[i] = pre_pos
        # print(signal[i], pre_pos)
        if signal[i] != 0:

            if pre_pos * signal[i] > 0 and hold > 0:
                hold_count -= 1
            else:
                hold_count = hold
            if update:
                pre_pos = signal[i]

        else:
            if hold_count > 0:
                hold_count -= 1
            elif hold_count == 0 and pre_pos != 0:
                pre_pos = 0
    return np.array(position)


def summary(position, price):
    cash = 1
    init_cash = 1
    current_p = 0
    cash_log = []
    enter = 0
    win = 0
    change_log = []
    pre_price = 0
    for i in range(len(position)):

        if current_p != position[i]:
            # if current_p * position[i] <= 0:
            if current_p == 0:
                change = 0
            else:
                enter += 1
                change = (price[i] - pre_price) * current_p
                if change > 0:
                    win += 1

            pre_price = price[i]
            change_log.append(change)

            current_p = position[i]

            cash += (cash / pre_price) * change

        #             cash *= 0.999
        #         else:
        #             change = stop_order(current_p, price[i], pre_price, 0.02, 0.02)
        #             if change:
        #                 current_p = 0
        #                 enter += 1
        #                 cash += (cash/pre_price) * change
        #                 pre_price = 0
        #                 if change > 0:
        #                     win += 1
        cash_log.append(cash)

    cash_log = np.array(cash_log)
    change_log = np.array(change_log)
    ratio = (cash_log[1:] / cash_log[:-1])
    profit_trade = np.mean(change_log[change_log > 0])
    loss_trade = np.abs(np.mean(change_log[change_log < 0]))
    print('Porfit per trade:', (cash_log[-1] - init_cash) / enter)
    print('Enter count     :', enter)
    print('Precision       :', win / enter)
    print('Net Worth       :', cash_log[-1] / init_cash)
    print('P/L             :', profit_trade / loss_trade)
    print('Annual Return   :', ((cash_log[-1] / init_cash - 1) / (init_cash) / 331) * 365)
    cummax = np.maximum.accumulate(cash_log)
    print('Max Drawdown    :', np.max((cummax - cash_log) / cummax))
    print('Turnover        :', np.mean(np.abs(position[1:] - position[:-1])))
    # plt.plot(cash_log)
    return cash_log


def evaluate_hold(position, price):
    cash = 1
    init_cash = 1
    current_p = 0
    cash_log = []
    for i in range(len(price)):
        if current_p == position[i]:
            continue
        else:
            if current_p == 1:
                change = price[i] - pre_price
            elif current_p == -1:
                change = pre_price - price[i]
            else:
                change = 0
            pre_price = price[i]
            current_p = position[i]
            cash += (cash/pre_price) * change
        cash_log.append(cash)
    return cash_log[-1]/init_cash


def plot_res(price, pred, log, name='unnamed'):
    plot_df = pd.DataFrame({'price': price, 'return': log})
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(111)
    plot_df['price'].plot(ax=ax1, grid=True, alpha=0.8, style='b', label='Price')
    plt.xlabel('Hour')
    ax1.set_ylabel('Price')
    plt.plot(np.where(pred > 0)[0], price[np.where(pred > 0)[0]], '^', markersize=10, color='g', alpha=0.4, label='Long')
    plt.plot(np.where(pred < 0)[0], price[np.where(pred < 0)[0]], 'v', markersize=10, color='r', alpha=0.4, label='Short')
    plt.legend(loc=2)

    ax2 = ax1.twinx()
    plot_df['return'].plot(ax=ax2, label='Return', style='y', alpha=0.7)
    ax2.set_ylabel('Profit')
    plt.legend(loc=1)
    plt.title('Predicted returns and changes in Bitcoin price')
    plt.savefig(f'{name}.png', dpi=400, bbox_inches='tight')
    plt.show()


def plot_hold(price, y_pred):
    with_update = [1]
    without_update = [1]
    for i in range(1, 300):
        position = alpha_func(y_pred, hold=i, update=True)
        with_update.append(evaluate_hold(position, price))
        position = alpha_func(y_pred, hold=i, update=False)
        without_update.append(evaluate_hold(position, price))
    plt.plot(with_update, 'o-', label='update')
    plt.plot(without_update, 'd-', label='without update')
    plt.ylabel('Rate of Return')
    plt.xlabel('Holding Time (hour)')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    test = [1,1,1,2,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1]
    print(test)
    res = alpha_func(test)
    print(res)