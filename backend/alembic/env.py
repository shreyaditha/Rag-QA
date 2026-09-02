import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Load alembic config
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import Base from app so autogenerate can see models
from app.models import Base  # noqa: E402  (models registers with Base)
target_metadata = Base.metadata


def get_url() -> str:
    """Read DATABASE_URL from env and convert to sync psycopg2 URL for Alembic."""
    url = os.environ.get("DATABASE_URL", "")
    # asyncpg driver is not supported by Alembic directly; use psycopg2
    url = url.replace("postgresql+asyncpg", "postgresql+psycopg2")
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    cfg_section = config.get_section(config.config_ini_section, {})
    cfg_section["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        cfg_section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
