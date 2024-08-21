import pandas as pd
import numpy as np
from .base_feature import FeatureExtractor
from ..utils.utils import _peakchar,_smooth

class DQDVvsV(FeatureExtractor):
    """
    Calculate peak heights, locations, and areas from the dQ/dV vs. V curve.

    This class computes various features from the derivative of charge with respect to voltage (dQ/dV)
    during either the charging or discharging state of a battery. The features extracted include the
    location and magnitude of peaks and valleys in the dQ/dV curve, as well as the area under these peaks.

    Attributes:
        state (str): The state of the battery process, either 'charging' or 'discharging'.

    Methods:
        compute_features(data: pd.DataFrame) -> pd.DataFrame:
            Computes the dQ/dV features for each cycle in the dataset.

            Args:
                data (pd.DataFrame): The input dataframe containing battery cycle data.
                                     Must include 'state', 'cycle_number', 'voltage', 'acc_capacity',
                                     and 'method' columns.

            Returns:
                pd.DataFrame: A dataframe with cycle numbers as the index and dQ/dV-related features as columns.
    """

    def __init__(self, state='charging'):
        """
        Initializes the DQDVvsV class with the specified state.

        Args:
            state (str): The state of the battery process to analyze ('charging' or 'discharging').
                         Default is 'charging'.
        """
        self.state = state

    def compute_features(self, data) -> pd.DataFrame:
        """
        Compute dQ/dV features for each battery cycle.

        The method calculates the locations, magnitudes, and areas of peaks and valleys in the dQ/dV vs. V curve
        for the specified battery state.

        Args:
            data (pd.DataFrame): The input dataframe containing battery cycle data.
                                 It must include the following columns:
                                 - 'state': The state of the battery process ('charging', 'discharging').
                                 - 'cycle_number': The unique identifier for each cycle.
                                 - 'voltage': The voltage values during the process.
                                 - 'acc_capacity': The accumulated capacity values during the process.
                                 - 'method': The method used in each step ('constant_current').

        Returns:
            pd.DataFrame: A dataframe with cycle numbers as the index and dQ/dV-related features as columns.
        """
        cycle_number = []
        peaklocarr, peakmagarr, peakareaarr = [], [], []
        valleylocarr, valleymagarr = [], []

        # Filter data by the specified state and method (constant current) and group by cycle_number
        dquer = data.query(
            "state == '{}'".format(self.state)).query(
                "method == 'constant_current'").groupby('cycle_number')

        for cyc, cycle_data in dquer:
            cycle_number.append(cyc)

            V = cycle_data['voltage'].values
            Q = cycle_data['acc_capacity'].values

            # Calculate dQ/dV and the midpoint voltage values
            Vmid, dQdV = self._dQdVcalc(V, Q)

            if len(Vmid) < 4:
                # Handle cases with insufficient data points
                nanarr = [np.nan, np.nan]
                peaklocarr.append(nanarr)
                peakmagarr.append(nanarr)
                peakareaarr.append(nanarr)
                valleylocarr.append(nanarr)
                valleymagarr.append(nanarr)
                continue

            # Smooth the dQ/dV curve
            Vsm, dQdVsm = _smooth(Vmid, dQdV)
            dQdVmax = dQdVsm.max()

            # Identify and characterize peaks in the smoothed dQ/dV curve
            res = _peakchar(Vsm, dQdVsm)
            peaklocs, peakmags, scales, peakareas = res
            peaklocarr.append(peaklocs)
            peakmagarr.append(peakmags)
            peakareaarr.append(peakareas)

            # Identify and characterize valleys in the smoothed dQ/dV curve
            res = _peakchar(Vsm, dQdVmax - dQdVsm)
            valleylocs, valleymags, scales, areas = res
            valleymags = -1 * (valleymags - dQdVmax)
            valleylocarr.append(valleylocs)
            valleymagarr.append(valleymags)

        # Determine the suffix based on the state ('ch' for charging, 'di' for discharging)
        st = 'ch' if self.state == 'charging' else 'di'

        # Handle case with no data
        if len(cycle_number) == 0:
            df = pd.DataFrame({'cycle_number': cycle_number})
            peakids = np.arange(2)
            for ii in peakids:
                df['dQdVpeak' + str(ii + 1) + 'loc_' + st] = []
                df['dQdVpeak' + str(ii + 1) + 'mag_' + st] = []
                df['dQdVpeak' + str(ii + 1) + 'area_' + st] = []
                df['dQdVvalley' + str(ii + 1) + 'loc_' + st] = []
                df['dQdVvalley' + str(ii + 1) + 'mag_' + st] = []
            return df

        # Stack the results for peaks and valleys into arrays
        peaklocarr = np.vstack(peaklocarr)
        peakmagarr = np.vstack(peakmagarr)
        peakareaarr = np.vstack(peakareaarr)
        valleylocarr = np.vstack(valleylocarr)
        valleymagarr = np.vstack(valleymagarr)

        # Create a DataFrame with the extracted dQ/dV features
        df = pd.DataFrame({'cycle_number': cycle_number})

        peakids = np.arange(2)
        for ii in peakids:
            df['dQdVpeak' + str(ii + 1) + 'loc_' + st] = peaklocarr[:, ii]
            df['dQdVpeak' + str(ii + 1) + 'mag_' + st] = peakmagarr[:, ii]
            df['dQdVpeak' + str(ii + 1) + 'area_' + st] = peakareaarr[:, ii]
            df['dQdVvalley' + str(ii + 1) + 'loc_' + st] = valleylocarr[:, ii]
            df['dQdVvalley' + str(ii + 1) + 'mag_' + st] = valleymagarr[:, ii]

        # Set cycle_number as the index
        df.set_index('cycle_number', inplace=True)

        return df

    def _dQdVcalc(self, V, Q):
        """
        Calculate the dQ/dV values from voltage and accumulated capacity.

        Args:
            V (np.array): Array of voltage values.
            Q (np.array): Array of accumulated capacity values.

        Returns:
            Vmid (np.array): Midpoint voltage values.
            dQdV (np.array): Calculated dQ/dV values.
        """
        Vmid = (V[1:] + V[:-1]) / 2
        dQ = np.diff(Q)
        dV = np.diff(V)
        # Sort and remove duplicate V values
        Vmid, indx = np.unique(Vmid, return_index=True)
        dQ = dQ[indx]
        dV = dV[indx]
        # Remove points where dV is zero to avoid infinite dQ/dV
        indx = np.abs(dV) > 1e-10
        dQ = dQ[indx]
        dV = dV[indx]
        Vmid = Vmid[indx]
        # Compute dQ/dV
        dQdV = dQ / dV

        return Vmid, dQdV