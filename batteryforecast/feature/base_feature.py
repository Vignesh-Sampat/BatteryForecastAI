from abc import ABC, abstractmethod
import pandas as pd

class FeatureExtractor(ABC):
    @abstractmethod
    def compute_features(self, data) -> pd.DataFrame:
        pass
