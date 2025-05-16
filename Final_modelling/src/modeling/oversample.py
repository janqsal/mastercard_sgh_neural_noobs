def oversample(X: pd.DataFrame, y: pd.Series, *, random_state: int = 42):
    """
    Randomly oversamples the minority class to balance the dataset.
    """
    ros = RandomOverSampler(random_state=random_state)
    return ros.fit_resample(X, y)
