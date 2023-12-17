from dagster import op, Out, In
from dagster_pandas import PandasColumn, create_dagster_pandas_dataframe_type
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import pandas as pd

TransformDrugData = create_dagster_pandas_dataframe_type(
    name="TransformDrugData",
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

@op(ins={'start':In(None)},out=Out(TransformDrugData))
def transform_extracted_drug(start=True) -> TransformDrugData:
    drug = pd.read_csv("staging/drug.csv", sep="\t")
    drug["conditions"] = drug["conditions"] .astype(str)
    drug["drug"] = drug["drug"].astype(str)
    drug["indication"] = drug["indication"].astype(str)
    drug["type"] = drug["type"].astype(str)
    drug["effective"] = drug["effective"]
    drug["easeofuse"] = drug["easeofuse"]
    drug["satisfaction"] = drug["satisfaction"]
    
    
    return drug
@op(ins={'drug': In(TransformDrugData)}, out=Out(None))
def stage_transformed_drug(drug):
    drug.to_csv(
        "staging/transformed_drug.csv",
        sep="\t",
        index=False
    )