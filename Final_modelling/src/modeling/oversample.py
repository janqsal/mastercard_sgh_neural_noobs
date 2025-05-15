from imblearn.over_sampling import RandomOverSampler
import pandas as pd

def oversample(X: pd.DataFrame, y: pd.Series, *, random_state: int = 42):
    ros = RandomOverSampler(random_state=random_state)
    return ros.fit_resample(X, y)
