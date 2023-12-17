from dagster import op, Out, In
from dagster_pandas import PandasColumn, create_dagster_pandas_dataframe_type
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import pandas as pd

TransformedDrugDataFrame2 = create_dagster_pandas_dataframe_type(
    name="TransformedDrugDataFrame2",
    columns=[
        PandasColumn.string_column("drug_name",non_nullable=True),
        PandasColumn.string_column("rxnorm_id"),
        PandasColumn.string_column("drug_tier"),
        PandasColumn.string_column("prior_authorization"),
    ],
)

@op(ins={'start':In(None)},out=Out(TransformedDrugDataFrame2))
def transform_extracted_drug2(start=True) -> TransformedDrugDataFrame2:
    drug = pd.read_csv("staging/drug.csv", sep="\t")
    
    
    drug["drug_name"] = drug["drug_name"].astype(str)
    drug["rxnorm_id"] = drug["rxnorm_id"].astype(str)
    drug["drug_tier"] = drug["drug_tier"].astype(str)
    drug["prior_authorization"] = drug["prior_authorization"].astype(str)

    
   
    return drug

@op(ins={'drug': In(TransformedDrugDataFrame2)}, out=Out(None))
def stage_transformed_drug(drug):
    drug.to_csv(
        "staging/transformed_drug.csv",
        sep="\t",
        index=False
    )