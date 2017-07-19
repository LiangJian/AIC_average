from functions import *
import numpy as np
import gvar as gv
import lsqfit


def fcn1(x_, p_):
    c0 = p_['C0']
    return c0 + x_*0.0


def fcn2(x_, p_):
    c0 = p_['C0']  
    c1 = p_['C1']  
    m = p_['m']  
    return c0+c1*np.exp(-m*x_)

n_conf = 200
n_r2 = 16**2*3 + 1
r_cut = 1.0
r2_0 = 0.0
r2_table = []

for ir2 in range(0, n_r2):
    if np.sqrt(ir2) - r2_0 >= r_cut:
        r2_0 = np.sqrt(ir2)
        r2_table.append(ir2)

r2_table = np.array(r2_table)
r_table = np.sqrt(r2_table)


fdata = np.load("result_C_corr_3264.m0.032000.m20.150000.sum08.fold.pick_9_4.npy")

ioff = 8

data = np.zeros(shape=(n_conf, r2_table.size))
for i in range(1, r2_table.size):
    data[:, i] = fdata[:, r2_table[i]]

fit_range = [ioff, r2_table.size - 0]
print('fitting range:', end='\t')
print(fit_range)

data = data[:, fit_range[0]:fit_range[1]]
x = r_table[fit_range[0]:fit_range[1]]
data_ave = np.average(data, 0)
data_cov = np.cov(data, rowvar=False)/n_conf
# data_cov = np.diag(np.diagonal(data_cov))
y = gv.gvar(data_ave, data_cov)

np.random.seed(7)

windows = get_random_window(x.size, 5, 3, 20)

prior0=[]
prior0.append({})
prior0[0]['C0'] = gv.gvar(0.14, 0.1)
prior0.append({})
prior0[1]['C0'] = gv.gvar(0.14, 0.1)
prior0[1]['C1'] = gv.gvar(0.1, 0.1)
prior0[1]['m'] = gv.gvar(1.0, 1.0)
res = aic_fit(x, y, np.array([fcn1, fcn2]), prior0, windows, 'C0')
print('AIC mean value and systematic error:', end='\t')
print(res)

nboot = 50
bs_res = []

print('(begin AIC boot ...', end='\t')
st = time.time()
for ib in range(nboot):
    x_ = np.arange(0, data.shape[0], 1)
    xb_ = get_boot_sample(x_)
    data_new_ = data[xb_, ...]
    data_ave_new_ = np.average(data_new_, 0)
    data_cov_new_ = np.cov(data_new_, rowvar=False) / n_conf
    # data_cov = np.diag(np.diagonal(data_cov))
    y = gv.gvar(data_ave_new_, data_cov_new_)

    res = aic_fit(x, y, np.array([fcn1, fcn2]), prior0, windows, 'C0')
    bs_res.append(res)
ed = time.time()
print('done in %4.2f' % (ed -st) + 's.)')

print('AIC statistical error:%\t', np.std(bs_res))
