from pymongo import MongoClient
from dagster import op, Out, In, DagsterType
from dagster_pandas import PandasColumn, create_dagster_pandas_dataframe_type
from datetime import datetime
import pandas as pd

mongo_connection_string = "mongodb://dap:dap@127.0.0.1"

DrugsDataFrame = create_dagster_pandas_dataframe_type(
    name="DrugsDataFrame",
    columns=[
        PandasColumn.string_column("conditions"),
        PandasColumn.string_column("drug"),
        PandasColumn.string_column("indication"),
        PandasColumn.string_column("type"),
        PandasColumn.float_column("effective"),
        PandasColumn.float_column("easeofuse"),
        PandasColumn.float_column("satisfaction"),
    ],
)

drug_columns = {
    "conditions": "conditions",
    "drug": "drug",
    "Indication": "Indication",
    "type": "type",
    "effective": "effective",
    "easeofuse": "easeofuse",
    "satisfaction": "satisfaction",


}

@op(ins={'start': In(bool)}, out=Out(DrugsDataFrame))
def extract_drug(start=True) -> DrugsDataFrame:
    conn = MongoClient(mongo_connection_string)
    db = conn["drugcsv"]
    drug = pd.DataFrame(db.drugrating.find({}))
    drug.drop(
        columns=["_id"],
        axis=1,
        inplace=True
    )
    
    conn.close()
    return drug

@op(ins={'drug': In(DrugsDataFrame)}, out=Out(None))
def stage_extracted_drug(drug):
    drug.to_csv("staging/drug.csv",index=False,sep="\t")
