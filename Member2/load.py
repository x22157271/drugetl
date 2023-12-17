
from dagster import op, Out, In, get_dagster_logger
from sqlalchemy import create_engine, exc
from sqlalchemy.pool import NullPool
import pandas as pd
postgres_connection_string = "postgresql://dap:dap@127.0.0.1:5432/drugs"

@op(ins={'empty': In(None)}, out=Out(bool))
def load_drug_dimension(empty):
    logger = get_dagster_logger()
    drug = pd.read_csv("staging/transformed_drug.csv", sep="\t")
    try:
        engine = create_engine(postgres_connection_string,poolclass=NullPool)
        engine.execute("TRUNCATE public.drug1;")
        rowcount = drug.to_sql(
            name="drug1",
            schema="public",
            con=engine,
            index=False,
            if_exists="append",
            method="multi"
        )
        logger.info("%i drug records loaded" % rowcount)
        engine.dispose(close=True)
        return rowcount > 0
    except exc.SQLAlchemyError as error:
        logger.error("Error: %s" % error)
        return False