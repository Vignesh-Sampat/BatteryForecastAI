import pandas as pd
import numpy as np
from .base_feature import FeatureExtractor
from ..utils.utils import get_acc_capacity

class SOHfraction(FeatureExtractor):
    """
    Compute the State of Health (SOH) as a fraction of the nominal capacity.

    This class calculates the SOH for each charging cycle in the dataset by
    dividing the maximum accumulated capacity during that cycle by the nominal capacity.

    Attributes:
        Qnom (float): The nominal capacity (in Amp-hours) of the battery. Default is 1 Amp-hour.

    Methods:
        compute_features(data: pd.DataFrame) -> pd.DataFrame:
            Computes the SOH for each charging cycle in the input data.

            Args:
                data (pd.DataFrame): The input dataframe containing battery cycle data.
                                     Must include 'state', 'cycle_number', and 'acc_capacity' columns.

            Returns:
                pd.DataFrame: A dataframe with cycle numbers as the index and SOH values as the column 'SOH'.
    """

    def __init__(self, Qnom=1):
        """
        Initializes the SOHfraction class with the given nominal capacity.

        Args:
            Qnom (float): The nominal capacity of the battery. Default is 1 Amp-hour.
        """
        self.Qnom = Qnom

    def compute_features(self, data) -> pd.DataFrame:
        """
        Computes the SOH for each charging cycle in the input data.

        The method filters the data to include only charging cycles, then calculates
        the SOH by dividing the maximum accumulated capacity by the nominal capacity (Qnom).

        Args:
            data (pd.DataFrame): The input dataframe containing battery raw data from battery-data-toolkit.
                                 It must have the following columns:
                                 - 'state': The state of the battery ('charging', 'discharging', etc.).
                                 - 'cycle_number': The unique identifier for each charging cycle.
                                 - 'acc_capacity': The accumulated capacity during the cycle.

        Returns:
            pd.DataFrame: A dataframe with cycle numbers as the index and SOH values in a column named 'SOH'.
        """
        # include the accumulated capacity and accumulated energy
        data = get_acc_capacity(data)

        # Initialize lists to store cycle numbers and corresponding SOH values
        cycle_number = []
        SOH = []

        # Filter data to include only rows where the state is 'charging'
        charging_data = data.query("state == 'charging'")

        # Group data by cycle_number and compute SOH for each cycle
        for cyc, cycle_data in charging_data.groupby('cycle_number'):
            cycle_number.append(cyc)
            SOH_value = np.max(cycle_data['acc_capacity'].values) / self.Qnom
            SOH.append(SOH_value)

        # Create a DataFrame with cycle_number as the index and SOH as the column
        df = pd.DataFrame({
            'cycle_number': cycle_number,
            'SOH': SOH
        })

        # Set cycle_number as the index
        df.set_index('cycle_number', inplace=True)

        return df
