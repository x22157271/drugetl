import webbrowser
from dagster import job
from extract import *
from transform import *
from load import *
@op(out=Out(bool))
def load_dimensions(drugdim):
    return drugdim 

@job
def etl():
 drugdim=load_drug_dimension(stage_transformed_drug(
                                    transform_extracted_drug(
                                        stage_extracted_drug(
                                            extract_drug()
                                        )
                                    )
                                )
                            ),
                           