import pandas as pd
from sqlalchemy import create_engine, inspect
from .config import Categorical_cols

#convert data from phishing.db to pandas dataframe
def load_data():
    engine = create_engine("sqlite:///data/phishing.db")
    df = pd.read_sql_table("phishing_data", engine)
    return df

#Clean data by dropping unnecessary columns , zeroing inapproprate negative values and cleaning string values
def load_cleaned_data():
    df = load_data()
    df = df.drop(columns=["Unnamed: 0"])
    df.loc[df["NoOfImage"] < 0, "NoOfImage"] = 0
    for cate in Categorical_cols:
        df[cate] = df[cate].astype(str).str.strip().str.lower()
    df["LineOfCode_missing"] = df["LineOfCode"].isna().astype(int)
    df.columns = df.columns.map(str).str.strip()
    return df