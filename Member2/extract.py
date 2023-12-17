from pymongo import MongoClient
from dagster import op, Out, In, DagsterType
from dagster_pandas import PandasColumn, create_dagster_pandas_dataframe_type
from datetime import datetime
import pandas as pd
mongo_connection_string = "mongodb://dap:dap@127.0.0.1"

DrugDataFrame2 = create_dagster_pandas_dataframe_type(
    name="DrugDataFrame2",
    columns=[
        PandasColumn.string_column("rxnorm_id",non_nullable=True),
        PandasColumn.string_column("drug_name"),
        PandasColumn.string_column("drug_tier"),
        PandasColumn.string_column("prior_authorization"),
        
    ],
)
drug_columns = {
    "drug_name": "drug_name",
    "rxnorm_id": "rxnorm_id",
    "drug_tier": "drug_tier",
    "prior_authorization": "prior_authorization",
   
}

@op(ins={'start': In(bool)}, out=Out(DrugDataFrame2))
def extract_drug(start=True) -> DrugDataFrame2:
    conn = MongoClient(mongo_connection_string)
    db = conn["drugdb2"]
    drug = pd.DataFrame(db.drugdb2.find({}))
    drug.drop(
        columns=["_id", "_index_url","_formulary_url","plan_id_type","plan_id","quantity_limit","step_therapy"],
        axis=1,
        inplace=True
    )
    
   
    conn.close()
    return drug

@op(ins={'drug': In(DrugDataFrame2)}, out=Out(None))
def stage_extracted_drug(drug):
    drug.to_csv("staging/drug.csv",index=False,sep="\t")