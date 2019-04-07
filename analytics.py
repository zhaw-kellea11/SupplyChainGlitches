import numpy as np

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


def abnormal_analytics(package):
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
