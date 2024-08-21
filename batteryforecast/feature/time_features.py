import pandas as pd
import numpy as np
from BatteryForecastAI.batteryforecast.feature.base_feature import FeatureExtractor

class ChargingTime(FeatureExtractor):
    """
    Compute various time-related features for charging or discharging cycles of a battery.

    This class calculates the duration of the overall charging/discharging process and
    the specific phases of constant current (CC) and constant voltage (CV) during the cycle.
    It also computes the fraction of time spent in the CC phase relative to the entire cycle duration.

    Attributes:
        state (str): The state of the battery process, either 'charging' or 'discharging'.

    Methods:
        compute_features(data: pd.DataFrame) -> pd.DataFrame:
            Computes time-related features for each cycle in the dataset.

            Args:
                data (pd.DataFrame): The input dataframe containing battery cycle data.
                                     Must include 'state', 'cycle_number', 'test_time', and 'method' columns.

            Returns:
                pd.DataFrame: A dataframe with cycle numbers as the index and time-related features as columns.
    """

    def __init__(self, state='charging'):
        """
        Initializes the ChargingTime class with the specified state.

        Args:
            state (str): The state of the battery process to analyze ('charging' or 'discharging').
                         Default is 'charging'.
        """
        self.state = state

    def compute_features(self, data) -> pd.DataFrame:
        """
        Compute time-related features for each battery cycle.

        The method calculates the total time, constant current (CC) time, constant voltage (CV) time,
        and the fraction of CC time relative to the total cycle time for the specified battery state.

        Args:
            data (pd.DataFrame): The input dataframe containing battery cycle data.
                                 It must include the following columns:
                                 - 'state': The state of the battery process ('charging', 'discharging').
                                 - 'cycle_number': The unique identifier for each cycle.
                                 - 'test_time': The time stamps of the process.
                                 - 'method': The method used in each step ('constant_current', 'constant_voltage').

        Returns:
            pd.DataFrame: A dataframe with cycle numbers as the index and time-related features as columns.
        """
        cycle_number, t_, tCC, tCV, tCCvsCVfrac = [], [], [], [], []

        # Filter data by the specified state (charging or discharging) and group by cycle_number
        quer = data.query("state == '{}'".format(self.state)).groupby('cycle_number')

        for cyc, cycle_data in quer:
            cycle_number.append(cyc)

            # Extract the test_time array for the current cycle
            t = cycle_data['test_time'].values

            # Calculate the total time duration for the cycle
            t_.append(t[-1] - t[0])

            # Identify the time intervals for constant current (CC) and constant voltage (CV) phases
            selCC = cycle_data['method'].values == 'constant_current'
            selCV = cycle_data['method'].values == 'constant_voltage'

            # Calculate the time spent in the CC phase
            tmp = t[selCC]
            tCC.append(tmp[-1] - tmp[0] if len(tmp) >= 1 else 0)

            # Calculate the time spent in the CV phase
            tmp = t[selCV]
            tCV.append(tmp[-1] - tmp[0] if len(tmp) >= 1 else 0)

            # Calculate the fraction of CC time relative to the total cycle time
            tCCvsCVfrac.append(tCC[-1] / t_[-1])

        # Determine the suffix based on the state ('ch' for charging, 'di' for discharging)
        st = 'ch' if self.state == 'charging' else 'di'

        # Create a DataFrame with all the extracted time-related features
        df = pd.DataFrame({
            'cycle_number': cycle_number,
            't_' + st: t_,
            'tCC_' + st: tCC,
            'tCV_' + st: tCV,
            'tCCvsCVfrac_' + st: tCCvsCVfrac
        })

        # Set cycle_number as the index
        df.set_index('cycle_number', inplace=True)

        return df


class TimeIntervalEqualVoltageChange(FeatureExtractor):
    """
    Computes the time interval during equal voltage change for battery data.

    This class calculates the time interval during which the battery voltage remains between two specified points
    (Vlow and Vhigh) during a charging or discharging cycle. It measures how long the battery's voltage stays within
    this range in each cycle.

    Parameters:
    - Vlow (float): Lower voltage point. Default is 3.5.
    - Vhigh (float): Upper voltage point. Default is 4.0.
    - state (str): Battery state, either 'charging' or 'discharging'. Default is 'charging'.

    Returns:
    - pd.DataFrame: DataFrame containing the time interval during equal voltage change.
      The DataFrame has the following columns:
        - TIEVC_state: Time interval during equal voltage change, where 'state' is replaced
          with 'ch' for charging or 'di' for discharging.
    """

    def __init__(self, Vlow=3.5, Vhigh=4.0, state='charging'):
        """
        Initializes the TimeIntervalEqualVoltageChange instance with the given parameters.

        Parameters:
        - Vlow (float): Lower voltage point.
        - Vhigh (float): Upper voltage point.
        - state (str): Battery state ('charging' or 'discharging').
        """
        self.Vlow = Vlow
        self.Vhigh = Vhigh
        self.state = state

    def compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the time interval during equal voltage change for each cycle.

        This method processes the input data to compute the time difference between the specified
        voltage levels (Vlow and Vhigh) within each cycle based on the battery state.

        Parameters:
        - data (pd.DataFrame): DataFrame containing battery cycle data with columns 'voltage',
          'test_time', 'state', and 'cycle_number'.

        Returns:
        - pd.DataFrame: DataFrame with the time interval during equal voltage change for each
          cycle. Columns include:
            - 'TIEVC_ch' for charging state or 'TIEVC_di' for discharging state.
        """
        cycle_number = []
        TIEVC = []

        # Filter data based on the battery state and group by cycle number
        filtered_data = data.query("state == @self.state").groupby('cycle_number')

        for cycle_num, cycle_data in filtered_data:
            cycle_number.append(cycle_num)

            V = cycle_data['voltage'].values
            t = cycle_data['test_time'].values

            # Find the indices of the closest voltages to Vlow and Vhigh
            indxlow = np.argmin(np.abs(V - self.Vlow))
            indxhigh = np.argmin(np.abs(V - self.Vhigh))

            # Calculate time interval based on the state
            if self.state == 'charging':
                TIEVC.append(t[indxhigh] - t[indxlow])
            elif self.state == 'discharging':
                TIEVC.append(t[indxlow] - t[indxhigh])
            else:
                raise ValueError("Invalid state. Must be 'charging' or 'discharging'.")

        # Create DataFrame with the results
        st = 'ch' if self.state == 'charging' else 'di'
        df = pd.DataFrame({
            'cycle_number': cycle_number,
            f'TIEVC_{st}': TIEVC
        })

        df.set_index('cycle_number', inplace=True)
        return df

