import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def abnormal_analytics_pct(package, package_random):
    means = [p[0] for p in package]
    stds = [p[1] for p in package]

    means_norm = np.array([np.array(m) - m[0] for m in means])

    means_avg = np.mean(means_norm, axis=0)
    means_std = np.std(means_norm, axis=0)
    stretched_avg = np.array([means_avg[0]] * 10 + [means_avg[1]] * 5 + [means_avg[2]] * 5)
    stretched_std = np.array([means_std[0]] * 10 + [means_std[1]] * 5 + [means_std[2]] * 5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    """
    for means in means_norm:
        stretched = [means[0]] * 10 + [means[1]] * 5 + [means[2]] * 5
        ax.plot(stretched, color='black', alpha=0.1)
    """
    for pkg in package_random:
        means = [p[0] for p in pkg]
        means_norm = np.array([np.array(m) - m[0] for m in means])
        means_avg = np.mean(means_norm, axis=0)
        stretched_avg_rand = np.array(stretch(means_avg))
        ax.plot(stretched_avg_rand, color='black')
    ax.plot(stretched_avg, color='red')
    # ax.plot(stretched_avg + stretched_std, color='green')
    # ax.plot(stretched_avg - stretched_std, color='green')
    # ax.plot(stretched_avg, color='red')
    fig.savefig('pct_change.png')


def stretch(arr):
    return [arr[0]] * 10 + [arr[1]] * 5 + [arr[2]] * 5


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

    tickers = [r[1] for r in sufficient_ar]
    returns = np.array([r[0] for r in sufficient_ar])
    returns_event = [r[19] for r in returns]
    dates = [r[2] for r in sufficient_ar]
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



def _abnormal_analytics(package):
    tickers = [r[1] for r in package]
    returns = np.array([r[0] for r in package])
    dates = [r[2] for r in package]

    mean_return_days = np.mean(returns, axis=0)
    std_return_days = np.std(returns, axis=0)

    print(std_return_days)
    print(mean_return_days)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(np.log(std_return_days))
    """
    for ret in returns:
        ax.plot(ret, color='black', lw=0.2)

    ax.plot(mean_return_days, color='red')
    ax.plot(mean_return_days + std_return_days, color='blue')
    ax.plot(mean_return_days - std_return_days, color='blue')
    # ax.set_ylim([-0.1, 0.1])
    """
    fig.savefig('returns.png')
