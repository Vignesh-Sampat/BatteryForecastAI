import pandas as pd
import numpy as np
from .base_feature import FeatureExtractor


class BasicFeatures(FeatureExtractor):
    """
    Extracts key features for both charging (ch) and discharging (di) states from battery cycle data.

    Features extracted:
        1. Imed_ch/Imed_di: Median current during charging/discharging.
        2. Vavg_ch/Vavg_di: Average voltage during charging/discharging.
        3. Q_ch/Q_di: Charge capacity during charging/discharging.
        4. E_ch/E_di: Energy during charging/discharging.
        5. Qeff/Eeff: Efficiency of charge/energy, comparing discharging to charging.
        6. Inorm_ch/Inorm_di: Normalized median current during charging/discharging.

    Methods:
        compute_features(data: pd.DataFrame) -> pd.DataFrame:
            Extracts and returns the above features for each cycle in the input dataset.
    """

    def compute_features(self, data) -> pd.DataFrame:
        """
        Compute key features for each battery cycle in both charging and discharging states.

        Args:
            data (pd.DataFrame): Input dataframe containing battery cycle data.
                                 Must include 'state', 'cycle_number', 'current',
                                 'test_time', 'acc_capacity', and 'acc_energy' columns.

        Returns:
            pd.DataFrame: A dataframe with cycle numbers as the index and extracted features as columns.
        """
        # Lists to store the features for each cycle
        cycle_number = []
        Imed_ch, Imed_di = [], []
        Vavg_ch, Vavg_di = [], []
        Q_ch, Q_di, E_ch, E_di = [], [], [], []
        Qeff, Eeff = [], []
        Inorm_ch, Inorm_di = [], []

        # Calculate the global median current for charging and discharging states
        Iallmed_ch = np.median(data.query("state == 'charging'")['current'].values)
        Iallmed_di = -1 * np.median(data.query("state == 'discharging'")['current'].values)

        # Group the data by cycle_number and compute features for each cycle
        for cyc, cycle_data in data.groupby('cycle_number'):
            cycle_number.append(cyc)

            # Separate the data into charging and discharging states
            cycle_data_ch = cycle_data.query("state == 'charging'")
            cycle_data_di = cycle_data.query("state == 'discharging'")

            # Compute features for the charging state
            if len(cycle_data_ch['test_time'].values) > 3:
                Imed_ch.append(np.median(cycle_data_ch['current'].values))
                acc_cap_ch = cycle_data_ch['acc_capacity'].values
                acc_ene_ch = cycle_data_ch['acc_energy'].values
                Q_ch.append(acc_cap_ch.max() - acc_cap_ch.min())  # Total charge during charging
                E_ch.append(acc_ene_ch.max() - acc_ene_ch.min())  # Total energy during charging
                # The average voltage during charging is calculated as E_ch[-1] / Q_ch[-1].
                # Energy(E)=∫V⋅Idt
                # Charge(Q)=∫Idt
                # Vavg = Qch[−1]/Ech[−1]
                Vavg_ch.append(E_ch[-1] / Q_ch[-1])  # Average voltage during charging
                Inorm_ch.append(Imed_ch[-1] / Iallmed_ch)  # Normalized median current during charging
            else:
                # Append NaN if data is insufficient for feature computation
                Imed_ch.append(np.nan)
                Q_ch.append(np.nan)
                E_ch.append(np.nan)
                Vavg_ch.append(np.nan)
                Inorm_ch.append(np.nan)

            # Compute features for the discharging state
            if len(cycle_data_di['test_time'].values) > 3:
                Imed_di.append(-1 * np.median(cycle_data_di['current'].values))
                acc_cap_di = cycle_data_di['acc_capacity'].values
                acc_ene_di = cycle_data_di['acc_energy'].values
                Q_di.append(acc_cap_di.max() - acc_cap_di.min())  # Total charge during discharging
                E_di.append(acc_ene_di.max() - acc_ene_di.min())  # Total energy during discharging
                # The average voltage during charging is calculated as E_ch[-1] / Q_ch[-1].
                # Energy(E)=∫V⋅Idt
                # Charge(Q)=∫Idt
                # Vavg = Qch[−1]/Ech[−1]
                Vavg_di.append(E_di[-1] / Q_di[-1])  # Average voltage during discharging
                Inorm_di.append(Imed_di[-1] / Iallmed_di)  # Normalized median current during discharging
            else:
                # Append NaN if data is insufficient for feature computation
                Imed_di.append(np.nan)
                Q_di.append(np.nan)
                E_di.append(np.nan)
                Vavg_di.append(np.nan)
                Inorm_di.append(np.nan)

            # Compute efficiency features if both charging and discharging data are available
            if len(cycle_data_ch['test_time'].values) > 3 and len(cycle_data_di['test_time'].values) > 3:
                Qeff.append(100 * Q_di[-1] / Q_ch[-1])  # Charge efficiency (discharging vs charging)
                Eeff.append(100 * E_di[-1] / E_ch[-1])  # Energy efficiency (discharging vs charging)
            else:
                Qeff.append(np.nan)
                Eeff.append(np.nan)

        # Create a DataFrame with all the extracted features
        df = pd.DataFrame({
            'cycle_number': cycle_number,
            'Imed_ch': Imed_ch,
            'Imed_di': Imed_di,
            'Vavg_ch': Vavg_ch,
            'Vavg_di': Vavg_di,
            'Q_ch': Q_ch,
            'Q_di': Q_di,
            'E_ch': E_ch,
            'E_di': E_di,
            'Qeff': Qeff,
            'Eeff': Eeff,
            'Inorm_ch': Inorm_ch,
            'Inorm_di': Inorm_di
        })

        # Set cycle_number as the index
        df.set_index('cycle_number', inplace=True)

        return df
