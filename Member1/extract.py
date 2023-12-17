from pymongo import MongoClient
from dagster import op, Out, In, DagsterType
from dagster_pandas import PandasColumn, create_dagster_pandas_dataframe_type
from datetime import datetime
import pandas as pd
mongo_connection_string = "mongodb://dap:dap@127.0.0.1"

DrugDataFrame = create_dagster_pandas_dataframe_type(
    name="DrugDataFrame",
    columns=[
        PandasColumn.integer_column("patient_id",non_nullable=True, unique=True),
        PandasColumn.string_column("drugName"),
        PandasColumn.string_column("condition"),
        PandasColumn.float_column("rating"),
        PandasColumn.string_column("date"),
        PandasColumn.integer_column("usefulcount"),
        PandasColumn.integer_column("review_length")
        
    ],
)

drug_columns = {
    "patient_id": "patient_id",
    "drugName": "drugName",
    "condition": "condition",
    "rating": "rating",
    "date":"date",
    "usefulCount": "usefulcount",
    "review_length": "review_length"
   
}

@op(ins={'start': In(bool)}, out=Out(DrugDataFrame))
def extract_drug(start=True) -> DrugDataFrame:
    conn = MongoClient(mongo_connection_string)
    db = conn["drugdb"]
    drug = pd.DataFrame(db.drug_rating.find({}))
    drug.drop(
        columns=["_id", "review"],
        axis=1,
        inplace=True
    )
    drug.rename(
        columns=drug_columns,
        inplace=True
    )
   
    conn.close()
    return drug

@op(ins={'drug': In(DrugDataFrame)}, out=Out(None))
def stage_extracted_drug(drug):
    drug.to_csv("staging/drug.csv",index=False,sep="\t")