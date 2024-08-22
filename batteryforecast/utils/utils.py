import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.stats import norm
from scipy.integrate import trapezoid

import numpy as np
from scipy.integrate import cumtrapz


def get_acc_capacity(df):
    """
    Calculate the accumulated capacity and accumulated energy for each cycle
    in a battery dataset.

    This function processes a DataFrame representing battery test data and
    computes the accumulated capacity and energy for each cycle. The results
    are stored as new columns in the original DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing battery test data. It should
        include columns for 'cycle_number', 'step_index', 'state', 'current',
        'voltage', and 'test_time'.

    Returns:
        pandas.DataFrame: The input DataFrame with added columns 'acc_capacity'
        and 'acc_energy', representing the accumulated capacity and energy
        for each cycle, respectively.
    """

    # Initialize arrays to store accumulated capacity and energy for each cycle
    acc_capacity = np.array([])
    acc_energy = np.array([])

    # Group the DataFrame by cycle number
    for cycle_num, cycle_data in df.groupby(['cycle_number']):
        # Initialize arrays for capacity and energy within the current cycle
        capacity = np.array([])
        energy = np.array([])

        # Initial values for the accumulated capacity and energy within the cycle
        p = 0  # Last accumulated capacity
        q = 0  # Last accumulated energy

        # Process each step within the cycle
        for step_idx, step_data in cycle_data.groupby(['step_index']):
            state = step_data['state'].iloc[0]

            # For charging or discharging steps, calculate capacity and energy
            if state in ['charging', 'discharging']:
                s_cap = cumtrapz(
                    step_data['current'], step_data['test_time'], initial=0) / 3600 + p
                s_ene = cumtrapz(
                    step_data['current'] * step_data['voltage'], step_data['test_time'], initial=0) / 3600 + q
                p = s_cap[-1]  # Update last accumulated capacity
                q = s_ene[-1]  # Update last accumulated energy

            # For other steps, the capacity and energy remain constant
            else:
                s_cap = np.ones(len(step_data)) * p
                s_ene = np.ones(len(step_data)) * q

            # Append calculated capacity and energy for the current step
            capacity = np.append(capacity, s_cap)
            energy = np.append(energy, s_ene)

        # Adjust capacity and energy relative to the minimum value in the cycle
        capacity -= capacity.min()
        energy -= energy.min()

        # Append the accumulated capacity and energy for the cycle
        acc_capacity = np.append(acc_capacity, capacity)
        acc_energy = np.append(acc_energy, energy)

    # Add the accumulated capacity and energy to the DataFrame as new columns
    df['acc_capacity'] = acc_capacity
    df['acc_energy'] = acc_energy

    return df


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
    return trapezoid(trial, x)

