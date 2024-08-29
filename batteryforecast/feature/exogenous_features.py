import pandas as pd
import numpy as np
from .base_feature import FeatureExtractor

class Exogenous(FeatureExtractor):

    def compute_features(self, data) -> pd.DataFrame:
        # Computation of C rate charge/Discharge
        # Temperature
        pass