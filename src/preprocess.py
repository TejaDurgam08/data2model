import pandas as pd

def handle_missing_values(df, method, fill_value=None):
    if df.isnull().sum().sum() == 0:
        return df  # No missing values to handle

    elif method == "Drop rows":
        return df.dropna()
    
    elif method == "Drop columns":
        return df.dropna(axis=1)
    
    elif method == "Fill with mean":
        return df.fillna(df.mean(numeric_only=True))
    
    elif method == "Fill with median":
        return df.fillna(df.median(numeric_only=True))
    
    elif method == "Fill with mode":
        return df.fillna(df.mode().iloc[0])
    
    elif method == "Fill with constant" and fill_value is not None:
        return df.fillna(fill_value)
    
    elif method == "Interpolate":
        return df.interpolate()
    
    else:
        return df  # No change
