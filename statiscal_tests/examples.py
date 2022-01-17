"""
Examples
"""
from scipy import stats
from scipy.stats import kendalltau, chi2_contingency
from statsmodels.tsa.stattools import adfuller
from scipy.stats import ttest_ind


def kendall():
    data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
    data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]
    stat, p = kendalltau(data1, data2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably independent')
    else:
        print('Probably dependent')


def chi_square():
    table = [[10, 20, 30], [6, 9, 17]]
    stat, p, dof, expected = chi2_contingency(table)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably independent')
    else:
        print('Probably dependent')


def adulfer():
    data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    stat, p, lags, obs, crit, t = adfuller(data)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably not Stationary')
    else:
        print('Probably Stationary')


def t_test():
    data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
    data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
    stat, p = ttest_ind(data1, data2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')


def mannwhitneyu():
    print("-" * 50)

    data1 = [0.359298587, 0.354261277, 0.359123922, 0.373292613, 0.368569638, 0.757998681, 0.701713025, 0.732921256,
             0.730464725, 0.492448899, 0.525252644, 0.526169757, 0.500618901, 0.71132853, 0.722767451, 0.70771055,
             0.741631732, 0.384061526, 0.319184162, 0.333333401, 0.280497525, 0.266208734, 0.289042707, 0.72818534,
             0.784016139, 0.739683701, 0.732383663, 0.702707847, 0.723492219, 0.506346764, 0.524093497, 0.574022525,
             0.516126517, 0.503358857, 0.481918037, 0.743526784, 0.760619827, 0.749603701, 0.763207205, 0.734506166,
             0.747082702]
    data2 = [0.371513731, 0.348409324, 0.353462656, 0.332520897, 0.723750084, 0.724434349, 0.74205812, 0.761257159,
             0.511683614, 0.502332284, 0.522025207, 0.556793131, 0.70632944, 0.710400593, 0.74624594, 0.340667402,
             0.329657916, 0.312983318, 0.266355323, 0.316387569, 0.287358348, 0.752666112, 0.737775196, 0.724327749,
             0.720742186, 0.712986804, 0.724580676, 0.521233406, 0.542183053, 0.505639977, 0.521999514, 0.492066772,
             0.499935326, 0.716132774, 0.772435733, 0.745203825, 0.749075552, 0.739952925, 0.742489835]

    stat, p = stats.mannwhitneyu(data1, data2, alternative='less')
    print('stat=%.3f, p=%f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')


def main():
    kendall()
    chi_square()
    adulfer()
    t_test()
    mannwhitneyu()


if __name__ == '__main__':
    main()
