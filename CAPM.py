"""This code is pretty straight forward, just basic capm formula applied to each row"""

import pandas as pd
import numpy as np
from datetime import datetime


df = pd.read_csv("merge_data_w_beta_alpha.csv")
capm_data = pd.read_csv("Econ Data\\capm_data.csv")
for i in range(len(capm_data)):
    date = str(capm_data.at[i, "Date"])  
    year = date[:4]  
    month = date[4:6]
    day = date[6:]  
    date = f"{year}-{month}-{day}"
    capm_data.at[i, "Date"] = date
capm_data.set_index("Date", inplace=True)

df["CAPM"] = [0]*len(df)
for i in range(len(df)):
    try:
        date = df.at[i,"Date"]
        date = str(datetime.strptime(date, "%m/%d/%Y").strftime("%Y-%m-%d"))
        beta = df.at[i,"beta"]
        rp = capm_data.at[date,"RP"]
        rf = capm_data.at[date,"RF"]
        capm = rf + beta*rp
        df.at[i,"CAPM"] = capm
    except:
        df.at[i,"CAPM"] = np.NaN
df.to_csv("merged_data_capm.csv",index=False)
