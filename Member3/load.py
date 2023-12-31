from dagster import op, Out, In, get_dagster_logger
from sqlalchemy import create_engine, exc
from sqlalchemy.pool import NullPool
import pandas as pd

postgres_connection_string = "postgresql://dap:dap@127.0.0.1:5432/drug_data"

@op(ins={'start': In(None)},out=Out(bool))
def load_drug_dimension(start=True):
    logger = get_dagster_logger()
    drug = pd.read_csv("staging/drug.csv", sep="\t")
    try:
        engine = create_engine(postgres_connection_string,poolclass=NullPool)
        engine.execute("TRUNCATE public.drug_data1;")
        rowcount = drug.to_sql(
            name="drug_data1",
            schema="public",
            con=engine,
            index=False,
            if_exists="append"
        )
        logger.info("%i records loaded" % rowcount)
        engine.dispose(close=True)
        return rowcount > 0
    except exc.SQLAlchemyError as error:
        logger.error("Error: %s" % error)
        return False