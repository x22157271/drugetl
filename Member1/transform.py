from dagster import op, Out, In
from dagster_pandas import PandasColumn, create_dagster_pandas_dataframe_type
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import pandas as pd

TransformedDrugDataFrame = create_dagster_pandas_dataframe_type(
    name="TransformedDrugDataFrame",
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

@op(ins={'start':In(None)},out=Out(TransformedDrugDataFrame))
def transform_extracted_drug(start=True) -> TransformedDrugDataFrame:
    drug = pd.read_csv("staging/drug.csv", sep="\t")
    drug["patient_id"] = drug["patient_id"] 
    drug["drugName"] = drug["drugName"]
    drug["condition"] = drug["condition"] 
    drug["rating"] = drug["rating"]
    drug["date"] = pd.to_datetime(drug["date"], errors='coerce').dt.strftime('%Y-%m-%d')
    drug["usefulcount"] = drug["usefulcount"]
    drug["review_length"] = drug["review_length"]
    
   
    return drug

@op(ins={'drug': In(TransformedDrugDataFrame)}, out=Out(None))
def stage_transformed_drug(drug):
    drug.to_csv(
        "staging/transformed_drug.csv",
        sep="\t",
        index=False
    )