from __future__ import annotations

import os
import sqlite3
from collections.abc import Mapping
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker


DATABASE_URL_ENV = "LLM_BENCHMARK_DATABASE_URL"
DEFAULT_DATABASE_URL = "sqlite:///runtime/llm_benchmark.db"


def get_database_url(environment: Mapping[str, str] | None = None) -> str:
    values = os.environ if environment is None else environment
    return values.get(DATABASE_URL_ENV, DEFAULT_DATABASE_URL)


def create_db_engine(database_url: str | None = None, *, echo: bool = False) -> Engine:
    url = database_url or get_database_url()
    parsed = make_url(url)
    if parsed.drivername.startswith("sqlite") and parsed.database not in (None, "", ":memory:"):
        Path(parsed.database).expanduser().parent.mkdir(parents=True, exist_ok=True)

    connect_args = {"check_same_thread": False} if parsed.drivername.startswith("sqlite") else {}
    engine = create_engine(url, echo=echo, connect_args=connect_args)
    if parsed.drivername.startswith("sqlite"):
        _enable_sqlite_foreign_keys(engine)
    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


def _enable_sqlite_foreign_keys(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection: object, connection_record: object) -> None:
        del connection_record
        if not isinstance(dbapi_connection, sqlite3.Connection):
            return
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()
