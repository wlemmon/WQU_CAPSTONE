from datetime import datetime
import os
start = datetime(2011, 1, 1)
end = datetime(2012, 12, 31)
project_root = '/Users/lemmonw/Desktop/WQU_CAPSTONE'
def rel(dir):
    return os.path.join(project_root, dir)
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 50000)

def clean_adj_close(df):
    df = df['Adj Close']
    # print(np.where(cols.max() > 1000000000))
    todrop=df.columns[np.where(df.max() > 1000000000)]
    return df.drop(columns=todrop, axis=1