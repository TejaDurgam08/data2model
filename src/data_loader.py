import pandas as pd
import io
import json

def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    
    elif file.name.endswith('.json'):
        content = file.read()
        return pd.json_normalize(json.loads(content))
    
    else:
        raise ValueError("Unsupported file type")
