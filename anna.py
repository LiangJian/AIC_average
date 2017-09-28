from functions import *
from fitting import print_fit
import numpy as np
import gvar as gv
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def fcn0(x_, p_):
    c0 = p_['C0']
    return c0 + x_*0.0


def fcn1(x_, p_):
    c0 = p_['C0']  
    c1 = p_['C1']  
    m = p_['m']  
    return c0+c1*np.exp(-m*x_)


def fcn2(x_, p_):
    c0 = p_['C0']
    m = p_['m']
    return c0 - 4*np.pi*np.sqrt(x_)*np.exp(-m*x_)/m

def print0(args, end='\n'):
    if rank == 0:
        print(args,end=end)

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


data = np.zeros(shape=(n_conf, r2_table.size))
for i in range(1, r2_table.size):
    data[:, i] = fdata[:, r2_table[i]]


ioff = 10
fit_range = [ioff, r2_table.size - 0]
print0('fitting range:', end='\t')
print0(fit_range)
print0('mpi size: %d'%size)

data = data[:, fit_range[0]:fit_range[1]]
x = r_table[fit_range[0]:fit_range[1]]
data_ave = np.average(data, 0)
data_cov = np.cov(data, rowvar=False)/n_conf
# data_cov = np.diag(np.diagonal(data_cov))
y = gv.gvar(data_ave, data_cov)

#np.random.seed(2)

np.random.seed(0)
#windows = get_random_window_range(x.size, 8, 5, 40)
#`windows = get_random_window_range_fix_size(x.size, 7, 8)
#windows = get_random_window_points(x.size, 8, 4, 60)
windows = get_random_window_points_fix_size(x.size, 7, 60)

#aic_fit = aic_fit_window
aic_fit = aic_fit_filter

prior0 =[]
prior0.append({})
prior0[0]['C0'] = gv.gvar(0.14, np.inf)
prior0.append({})
prior0[1]['C0'] = gv.gvar(0.14, np.inf)
prior0[1]['C1'] = gv.gvar(0.1, np.inf)
prior0[1]['m'] = gv.gvar(1.0, np.inf)
prior0.append({})
prior0[2]['C0'] = gv.gvar(0.14, np.inf)
prior0[2]['m'] = gv.gvar(1.0, np.inf)

p0 =[]
p0.append({})
p0[0]['C0'] = 0.14
p0.append({})
p0[1]['C0'] = 0.14
p0[1]['C1'] = 0.1
p0[1]['m'] = 1.0
p0.append({})
p0[2]['C0'] = 0.14
p0[2]['m'] = 1.0

"""
print(x.size)
cut = 4
fit = lsqfit.nonlinear_fit(data=(x[cut:], y[cut:]), p0=p0[2], fcn=fcn2, debug=True, svdcut=-1e-12)
print_fit(fit)    
exit(0)
"""

res = aic_fit(x, y, np.array([fcn0, fcn2]), (p0[0], p0[2]), windows, 'C0')
print0('AIC mean value and systematic error:', end='\t')
print0(res)

nboot = 200
bs_res = []

if int(nboot/size)*size != nboot:
    print0('wrong mpi size vs nboot: %d %d'%(size, nboot))
    exit(0)
size_per_core = int(nboot/size)
np.random.seed(rank)

print0('(begin AIC boot ...', end='\t')
st = time.time()
comm.Barrier()
for ib in range(rank*size_per_core, (rank + 1)*size_per_core):
    xb_ = get_boot_sample(np.arange(0, data.shape[0], 1))
    data_new_ = data[xb_, ...]
    data_ave_new_ = np.average(data_new_, 0)
    data_cov_new_ = np.cov(data_new_, rowvar=False) / n_conf
    # data_cov = np.diag(np.diagonal(data_cov))
    y = gv.gvar(data_ave_new_, data_cov_new_)

    res = aic_fit(x, y, np.array([fcn0, fcn2]), (p0[0], p0[2]), windows, 'C0')
    bs_res.append(res.mean)

comm.Barrier()
ed = time.time()
print0('done in %4.2f' % (ed - st) + 's.)')

"""
sendbuf = np.zeros(100, dtype='i') + rank
recvbuf = None
if rank == 0:
    recvbuf = np.empty([size, 100], dtype='i')
comm.Gather(sendbuf, recvbuf, root=0)
print0(np.array(recvbuf).shape)
exit(0)
"""

bs_res = np.array(bs_res)
bs_res_all = None
if rank == 0:
    bs_res_all = np.empty([size, bs_res.size])
comm.Gather(bs_res, bs_res_all, root=0)
if rank == 0:
    bs_res_all = np.array(bs_res_all).reshape(nboot)
print0('AIC statistical error:',end='\t')
e_s = 0.
if rank == 0:
    e_s = get_p68_mean_and_error(bs_res_all)
print0(e_s)
