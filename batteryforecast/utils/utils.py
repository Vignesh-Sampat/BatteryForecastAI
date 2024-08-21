import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.stats import norm
from scipy.integrate import trapz

def _smooth(x, y):
    yf = interp1d(x, y, kind='quadratic')
    xs = np.linspace(x.min(), x.max(), 1000)
    ys = yf(xs)
    return xs, ys

def _peakchar(x, y, promfrac=0.05, areas=True):
    nanv = np.nan*np.ones((2,))
    # find max peak
    res = find_peaks(y)[0]
    if len(res) == 0:
        if not areas:
            return nanv, nanv
        return nanv, nanv, nanv, nanv
    maxpeak = np.max(y[res])

    # compute the prominence threshold
    prominence = promfrac*maxpeak

    # find peaks exceeding prominence
    res = find_peaks(y, prominence=prominence)[0]

    if len(res) == 0:
        if not areas:
            return nanv, nanv
        return nanv, nanv, nanv, nanv
    locs = x[res]
    vals = y[res]

    indx = np.argsort(vals)[::-1]
    locs = locs[indx]
    vals = vals[indx]

    if len(locs) > 2:
        locs = locs[:2]
        vals = vals[:2]

    npeaks = len(locs)

    if not areas:
        locs_ = np.copy(nanv)
        vals_ = np.copy(nanv)

        locs_[:npeaks] = locs
        vals_[:npeaks] = vals

        return locs_, vals_

    scales = np.array([])
    areas = np.array([])
    for loc, val in zip(locs, vals):
        scale = _peakfit(loc, val, x, y)[0]
        scales = np.append(scales, scale)
        area = _peakarea(x, loc, val, scale)
        areas = np.append(areas, area)
        _peakmse(scale, loc, val, x, y)

    locs_ = np.copy(nanv)
    vals_ = np.copy(nanv)
    scales_ = np.copy(nanv)
    areas_ = np.copy(nanv)

    npeaks = len(locs)
    locs_[:npeaks] = locs
    vals_[:npeaks] = vals
    scales_[:npeaks] = scales
    areas_[:npeaks] = areas

    return locs_, vals_, scales_, areas_


def _peakfit(loc, val, x, y):
    def _peakmse_(scale):
        return _peakmse(scale, loc, val, x, y)
    res = minimize(_peakmse_, x0=0.05, bounds=((0.001, 0.5),))
    return res.x

def _peakmse(scale, loc, val, x, y):
    weights = norm.pdf(x, loc=loc, scale=0.01)
    weights /= weights.max()
    trial = norm.pdf(x, loc=loc, scale=scale)
    trial *= val/trial.max()
    return np.sum(weights*(y - trial)**2)

def _peakarea(x, loc, val, scale):
    trial = norm.pdf(x, loc=loc, scale=scale)
    trial *= val/trial.max()
    return trapz(trial, x)

