import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def abnormal_analytics(abnormal_returns, fig_name='returns.pdf', fig_title='Abnormal returns around glitch event'):
    print('*'*100)
    print(fig_title)
    sufficient_idx = []
    sufficient_ar = []
    p_threshold = 0.05
    for i, ar in enumerate(abnormal_returns):
        if ar[-1][3][1] <= p_threshold:
            sufficient_idx.append(i)
            sufficient_ar.append(ar)

    # print('N sufficient P:', len(sufficient_ar))
    returns = np.array([r[0] for r in sufficient_ar])
    mean_return_days = np.mean(returns, axis=0)

    mean_return_days *= 100

    m, lower, upper = mean_confidence_interval(returns)
    n_days = len(mean_return_days)
    event_index = n_days // 2

    time_axis = range(-int((n_days-1)/2), int((n_days-1)/2 + 1))

    car_11 = np.sum(mean_return_days[event_index-5:event_index+6])
    car_2 = np.sum(mean_return_days[event_index-1:event_index+1])

    print('CAR -5/5:', round(car_11, 2))
    print('CAR -1/0:', round(car_2, 2))

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.fill_between(np.arange(len(lower)), lower, upper, facecolor=plt.cm.viridis(0.7), alpha=0.2)
    ax.plot(mean_return_days, color=plt.cm.viridis(0.4), marker='.', mfc=plt.cm.viridis(0.9))
    ax.plot(lower, color=plt.cm.viridis(0.7))
    ax.plot(upper, color=plt.cm.viridis(0.7))
    ax.axvline(x=event_index, color=plt.cm.inferno(0.4), lw=0.5)
    ax.set_xticks(range(0,n_days,5))
    ax.set_xticklabels(time_axis[::5])

    ax.set_xlabel('Days from event [d]')
    ax.set_ylabel('Abnormal return [%]')
    ax.set_title(fig_title)
    fig.savefig(fig_name, dpi=300)


def mean_confidence_interval(data, confidence=0.95):
    a = 100.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n
