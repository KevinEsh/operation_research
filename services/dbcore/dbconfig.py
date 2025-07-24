from os import environ
from typing import Annotated, Any, Generator

from dotenv import load_dotenv
from fastapi import Depends
from sqlmodel import Session, SQLModel, create_engine

load_dotenv("../../.env")
# check if the database file exists
# If not, it will be created when the first table is created
DATABASE_SERVER = environ.get("DATABASE_SERVER", "duckdb")
DATABASE_NAME = "dbcore"

if DATABASE_SERVER == "duckdb":
    DATABASE_URL = "duckdb:///./data/core.db"
elif DATABASE_SERVER == "postgresql":
    POSTGRES_USER = environ.get("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = environ.get("POSTGRES_PASSWORD", "postgres")
    POSTGRES_HOST = environ.get("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = 5432
    DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{DATABASE_NAME}"
else:
    raise ValueError(f"Unsupported DATABASE_SERVER: {DATABASE_SERVER}")

print(f"Connecting to database at {DATABASE_URL}")

engine = create_engine(DATABASE_URL, echo=False)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, Any, None]:
    with Session(engine) as session:
        yield session


SessionType = Annotated[Session, Depends(get_session)]
