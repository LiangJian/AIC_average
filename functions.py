from error_and_resample import *
import time
import gvar as gv
import scipy
import lsqfit


def print_fit(fit_):
    print('---------------------------------------------------')
    print('chi2/dof =', '%4.2f' % (fit_.chi2/fit_.dof), end='\t')
    print('\t dof =', '%d' % fit_.dof, end='\t')
    print('\t Q =', '%4.2f' % (scipy.special.gammaincc((fit_.dof-fit_.p.size)/2, fit_.chi2/2)))
    print('---------------------------------------------------')
    for key in fit_.p:
        print(key+' =', fit_.p[key])
    print('---------------------------------------------------')


def bs_fitting(fit_, nbs_, seed_=0):
    print('\nBootstrap Analysis with n_boot %d ...' % nbs_)
    st = time.time()
    gv.ranseed(seed_)
    n_boot = nbs_
    bs_res = {}
    for key in fit_.p:
        bs_res[key] = []
    for bs_fit in fit_.bootstrap_iter(n_boot):
        p = bs_fit.pmean
        for key in fit_.p:
            bs_res[key].append(p[key])
    for key in fit_.p:
        bs_res[key] = np.array(bs_res[key])
        # print(key+' =', gv.gvar(np.average(bs_res[key]), np.std(bs_res[key])))
        print(key+' =', gv.gvar(get_p68_mean_and_error(bs_res[key], 0)))
    ed = time.time()
    print('Bootstrap Analysis done, %4.2f s used.' % (ed - st))
    return bs_res


def get_random_window(size_, range_of_length_, fitting_length_, n_range_):
    window_ = []
    count = 0
    count2 = 0
    while True:
        pos_ = int(np.random.random(1)[0] * size_)
        length_ = int(np.random.random(1)[0] * range_of_length_) + fitting_length_
        if [pos_, pos_ + length_] not in window_:
            if pos_ + length_ < size_:
                window_.append([pos_, pos_ + length_])
                count += 1
        if count == n_range_:
            break
        count2 += 1
        if count2 > 10000:
            print('no enough points')
            exit(-1)
    window_ = np.array(window_)
    return window_


def aic_fit(x_, y_, fcns_, prior0_, window_, key_):
    size_ = x_.size
    n_range_ = window_.shape[0]
    res_ = []
    aic_ = []
    for i in range(n_range_):
        x_new_ = x_[window_[i][0]:window_[i][1]]
        y_new_ = y_[window_[i][0]:window_[i][1]]
        for j in range(fcns_.size):
            fit = lsqfit.nonlinear_fit(data=(x_new_, y_new_), prior=prior0_[j], fcn=fcns_[j], debug=True, svdcut=-1e-12)
            res_.append(fit.pmean[key_])
            aic_.append(2*len(prior0_[j]) + fit.chi2)
    aic_ = np.array(aic_)
    aic_ = np.exp(-0.5 * aic_)
    aic_ /= np.sum(aic_)

    res_ = np.array(res_)
    return gv.gvar(np.sum(aic_ * res_), np.sum(aic_ * (res_-np.average(res_))**2)**0.5)
