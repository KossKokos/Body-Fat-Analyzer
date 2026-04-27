import ssl
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

from config.settings import settings
from database.models import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
# target_metadata = None

# other values from the config, defined by the needs of env.py,
# can be acquired:
target_metadata = Base.metadata
config.set_main_option("sqlalchemy.url", settings.SQLALCHEMY_DATABASE_URL)# ... etc.

def get_connect_args() -> dict:
    db_url = settings.SQLALCHEMY_DATABASE_URL
    db_ssl_mode = str(settings.DB_SSL_MODE).strip().lower()

    is_pg8000 = db_url.startswith("postgresql+pg8000://")
    ssl_required = db_ssl_mode == "require"

    print(f"ALEMBIC DB_SSL_MODE={db_ssl_mode}", flush=True)
    print(f"ALEMBIC DB DRIVER={db_url.split('://', 1)[0]}", flush=True)
    print(f"ALEMBIC SSL REQUIRED={ssl_required}", flush=True)

    if is_pg8000 and ssl_required:
        return {
            "ssl_context": ssl.create_default_context(),
        }

    return {}

def run_migrations_offline() -> None:
    context.configure(
        url=settings.SQLALCHEMY_DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()



def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = settings.SQLALCHEMY_DATABASE_URL

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args=get_connect_args(),
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

# def run_migrations_offline() -> None:
#     """Run migrations in 'offline' mode.

#     This configures the context with just a URL
#     and not an Engine, though an Engine is acceptable
#     here as well.  By skipping the Engine creation
#     we don't even need a DBAPI to be available.

#     Calls to context.execute() here emit the given string to the
#     script output.

#     """
#     url = config.get_main_option("sqlalchemy.url")
#     context.configure(
#         url=url,
#         target_metadata=target_metadata,
#         literal_binds=True,
#         dialect_opts={"paramstyle": "named"},
#     )

#     with context.begin_transaction():
#         context.run_migrations()


# def run_migrations_online() -> None:
#     """Run migrations in 'online' mode.

#     In this scenario we need to create an Engine
#     and associate a connection with the context.

#     """
#     connectable = engine_from_config(
#         config.get_section(config.config_ini_section, {}),
#         prefix="sqlalchemy.",
#         poolclass=pool.NullPool,
#     )

#     with connectable.connect() as connection:
#         context.configure(
#             connection=connection, target_metadata=target_metadata
#         )

#         with context.begin_transaction():
#             context.run_migrations()


# if context.is_offline_mode():
#     run_migrations_offline()
# else:
#     run_migrations_online()
