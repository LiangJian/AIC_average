from error_and_resample import *
import gvar as gv
import lsqfit


def get_random_window_range(size_, range_of_length_, fitting_length_, n_range_):
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
            print('no enough windows')
            exit(-1)
    window_ = np.array(window_)
    return window_


def get_random_window_range_fix_size(size_, fitting_length_, n_range_):
    window_ = []
    count = 0
    count2 = 0
    while True:
        pos_ = int(np.random.random(1)[0] * size_)
        length_ = fitting_length_
        if [pos_, pos_ + length_] not in window_:
            if pos_ + length_ < size_:
                window_.append([pos_, pos_ + length_])
                count += 1
        if count == n_range_:
            break
        count2 += 1
        if count2 > 10000:
            print('no enough windows')
            exit(-1)
    window_ = np.array(window_)
    return window_


def get_random_window_points(size_, range_of_length_, fitting_length_, n_range_):
    window_ = []
    count = 0
    count2 = 0
    a_ = np.arange(size_)
    while True:
        length_ = int(np.random.random(1)[0] * range_of_length_) + fitting_length_
        if length_ > size_:
            count2 += 1
            continue
        b = np.random.choice(a_, length_, replace=False)
        b.sort()
        if any(a_ is x for x in window_):
            count2 += 1
            if count2 > 10000:
                print('no enough windows')
                exit(-1)
            continue
        else:
            window_.append(b)
            count += 1
        if count == n_range_:
            break
        count2 += 1
        if count2 > 10000:
            print('no enough windows')
            exit(-1)

    window_ = np.array(window_)
    return window_


def get_random_window_points_fix_size(size_, fitting_length_, n_range_):
    window_ = []
    count = 0
    count2 = 0
    a_ = np.arange(size_)
    while True:
        b = np.random.choice(a_, fitting_length_, replace=False)
        b.sort()
        if any(a_ is x for x in window_):
            count2 += 1
            if count2 > 10000:
                print('no enough windows')
                exit(-1)
            continue
        else:
            window_.append(b)
            count += 1
        if count == n_range_:
            break
        count2 += 1
        if count2 > 10000:
            print('no enough windows')
            exit(-1)

    window_ = np.array(window_)
    return window_


def aic_fit_window(x_, y_, fcns_, p0_, window_, key_):
    size_ = x_.size
    n_range_ = window_.shape[0]
    res_ = []
    aic_ = []
    for i in range(n_range_):
        x_new_ = x_[window_[i][0]:window_[i][1]]
        y_new_ = y_[window_[i][0]:window_[i][1]]
        for j in range(fcns_.size):
            fit = lsqfit.nonlinear_fit(data=(x_new_, y_new_), p0=p0_[j], fcn=fcns_[j], debug=True, svdcut=-1e-12)
            res_.append(fit.pmean[key_])
            aic_.append(2*len(p0_[j]) + fit.chi2)
    aic_ = np.array(aic_)
    aic_ = np.exp(-0.5 * aic_)
    aic_ /= np.sum(aic_)

    res_ = np.array(res_)
    return gv.gvar(np.sum(aic_ * res_), np.sum(aic_ * (res_-np.average(res_))**2)**0.5)


def aic_fit_filter(x_, y_, fcns_, p0_, filter_, key_):
    size_ = x_.size
    n_range_ = filter_.shape[0]
    res_ = []
    aic_ = []
    for i in range(n_range_):
        x_new_ = x_[filter_[i]]
        y_new_ = y_[filter_[i]]
        for j in range(fcns_.size):
            fit = lsqfit.nonlinear_fit(data=(x_new_, y_new_), p0=p0_[j], fcn=fcns_[j], debug=True, svdcut=-1e-12)
            #print(fit.chi2/fit.dof)
            res_.append(fit.pmean[key_])
            aic_.append(2*len(p0_[j]) + fit.chi2)
    aic_ = np.array(aic_)
    aic_ = np.exp(-0.5 * aic_)
    aic_ /= np.sum(aic_)

    res_ = np.array(res_)
    return gv.gvar(np.sum(aic_ * res_), np.sum(aic_ * (res_-np.average(res_))**2)**0.5)
