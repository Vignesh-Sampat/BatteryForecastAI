from .base_feature import FeatureExtractor
from ..utils.utils import _peakchar,_smooth
from scipy.signal import find_peaks
import pandas as pd
import numpy as np


class DVDQvsQ(FeatureExtractor):
    """
    This class computes the features from the dV/dQ vs. Q curve during charging or discharging phases
    of battery cycles. It calculates the peak heights, locations, and valley features from the curve.

    Parameters:
    ----------
    state : str, optional
        The state of the battery during the feature extraction. It can be either 'charging' or 'discharging'.
        Default is 'charging'.

    Methods:
    -------
    compute_features(data) -> pd.DataFrame:
        Computes and returns the features from the dV/dQ vs. Q curve for each cycle.

    _dVdQcalc(V, Q) -> tuple:
        Computes the differential voltage (dV) over differential capacity (dQ) and the midpoint of the capacity.
    """

    def __init__(self, state='charging'):
        """
        Initializes the DVDQvsQ feature extractor with the specified state.

        Parameters:
        ----------
        state : str, optional
            The state of the battery ('charging' or 'discharging'). Default is 'charging'.
        """
        self.state = state

    def compute_features(self, data) -> pd.DataFrame:
        """
        Computes the features from the dV/dQ vs. Q curve for each cycle in the data.

        Parameters:
        ----------
        data : pd.DataFrame
            A DataFrame containing the battery data with at least the following columns:
            - 'state': The state of the battery ('charging' or 'discharging').
            - 'method': The method of the cycle (should include 'constant_current').
            - 'cycle_number': The identifier for the battery cycle.
            - 'voltage': The voltage readings.
            - 'acc_capacity': The accumulated capacity readings.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the computed features with columns:
            - 'cycle_number': The cycle number.
            - 'dVdQpeakXloc_state': Location of the dV/dQ peak.
            - 'dVdQpeakXmag_state': Magnitude of the dV/dQ peak.
            - 'dVdQvalleyXloc_state': Location of the dV/dQ valley.
            - 'dVdQvalleyXmag_state': Magnitude of the dV/dQ valley.
        """
        cycle_number = []
        peaklocarr, peakmagarr = [], []
        valleylocarr, valleymagarr = [], []

        # Filter data to only include cycles with the specified state and constant current method
        dquer = data.query(
            f"state=='{self.state}'").query(
            "method=='constant_current'").groupby('cycle_number')

        for cyc, cycle_data in dquer:
            cycle_number.append(cyc)

            # Extract voltage (V) and accumulated capacity (Q) from the cycle data
            V = cycle_data['voltage'].values
            Q = cycle_data['acc_capacity'].values

            # Compute the differential voltage/capacity (dV/dQ) and the midpoint capacity (Qmid)
            Qmid, dVdQ = self._dVdQcalc(V, Q)

            if len(Qmid) < 4:
                # If not enough points, append NaNs to maintain structure
                nanarr = [np.nan, np.nan]
                peaklocarr.append(nanarr)
                peakmagarr.append(nanarr)
                valleylocarr.append(nanarr)
                valleymagarr.append(nanarr)
                continue

            # Filter out extreme values from the dV/dQ curve
            res = find_peaks(dVdQ)[0]
            if len(res) == 0:
                maxpeak = np.sort(dVdQ)[int(-0.1 * len(dVdQ))]
            else:
                maxpeak = np.max(dVdQ[res])
            indx = dVdQ < 2 * maxpeak
            Qmid = Qmid[indx]
            dVdQ = dVdQ[indx]

            if len(Qmid) < 4:
                # Handle case where filtered data is too small
                nanarr = [np.nan, np.nan]
                peaklocarr.append(nanarr)
                peakmagarr.append(nanarr)
                valleylocarr.append(nanarr)
                valleymagarr.append(nanarr)
                continue

            # Smooth the dV/dQ curve for better peak/valley detection
            Qsm, dVdQsm = _smooth(Qmid, dVdQ)
            dVdQmax = dVdQsm.max()

            # Identify and store peaks in the smoothed curve
            peaklocs, peakmags = _peakchar(Qsm, dVdQsm, areas=False)
            peaklocarr.append(peaklocs)
            peakmagarr.append(peakmags)

            # Identify and store valleys by analyzing the inverted curve
            valleylocs, valleymags = _peakchar(Qsm, dVdQmax - dVdQsm, areas=False)
            valleymags = -1 * (valleymags - dVdQmax)
            valleylocarr.append(valleylocs)
            valleymagarr.append(valleymags)

        # Determine the appropriate suffix based on the state
        st = 'ch' if self.state == 'charging' else 'di'

        if len(cycle_number) == 0:
            # If no valid cycles, return an empty DataFrame with the correct structure
            df = pd.DataFrame({'cycle_number': cycle_number})
            peakids = np.arange(2)
            for ii in peakids:
                df[f'dVdQpeak{ii + 1}loc_{st}'] = []
                df[f'dVdQpeak{ii + 1}mag_{st}'] = []
                df[f'dVdQvalley{ii + 1}loc_{st}'] = []
                df[f'dVdQvalley{ii + 1}mag_{st}'] = []
            return df

        # Convert lists to arrays for DataFrame construction
        peaklocarr = np.vstack(peaklocarr)
        peakmagarr = np.vstack(peakmagarr)
        valleylocarr = np.vstack(valleylocarr)
        valleymagarr = np.vstack(valleymagarr)

        # Create DataFrame to hold all computed features
        df = pd.DataFrame({'cycle_number': cycle_number})

        # Populate the DataFrame with peak and valley features
        peakids = np.arange(2)
        for ii in peakids:
            df[f'dVdQpeak{ii + 1}loc_{st}'] = peaklocarr[:, ii]
            df[f'dVdQpeak{ii + 1}mag_{st}'] = peakmagarr[:, ii]
            df[f'dVdQvalley{ii + 1}loc_{st}'] = valleylocarr[:, ii]
            df[f'dVdQvalley{ii + 1}mag_{st}'] = valleymagarr[:, ii]

        # Set the cycle_number as the index of the DataFrame
        df.set_index('cycle_number', inplace=True)

        return df

    def _dVdQcalc(self, V, Q):
        """
        Computes the differential voltage (dV) over differential capacity (dQ) and the midpoint of the capacity (Qmid).

        Parameters:
        ----------
        V : np.ndarray
            Array of voltage values for the cycle.
        Q : np.ndarray
            Array of accumulated capacity values for the cycle.

        Returns:
        -------
        Qmid : np.ndarray
            Midpoints of the capacity values.
        dVdQ : np.ndarray
            Differential voltage over differential capacity.
        """
        Qmid = (Q[1:] + Q[:-1]) / 2
        dQ = np.diff(Q)
        dV = np.diff(V)
        # Remove points where dQ is effectively zero to avoid division errors
        indx = np.abs(dQ) > 1e-10
        dQ = dQ[indx]
        dV = dV[indx]
        Qmid = Qmid[indx]
        # Compute dV/dQ
        dVdQ = dV / dQ

        return Qmid, dVdQ
