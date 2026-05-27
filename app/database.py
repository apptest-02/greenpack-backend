"""
Greenpack Pro — Database Configuration
Supports SQLite (Mode A: offline) and PostgreSQL (Mode B: cloud)
via the same SQLAlchemy models
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import event
from app.config import get_settings
import logging

log = logging.getLogger(__name__)
settings = get_settings()


class Base(DeclarativeBase):
    pass


def create_engine():
    db_url = settings.db_url

    if "sqlite" in db_url:
        engine = create_async_engine(
            db_url,
            echo=settings.log_level == "DEBUG",
            connect_args={"check_same_thread": False},
        )
        # Enable WAL mode for SQLite (better concurrent read performance)
        @event.listens_for(engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.close()
    else:
        engine = create_async_engine(
            db_url,
            echo=settings.log_level == "DEBUG",
            pool_size=10,
            max_overflow=20,
        )

    return engine


engine = create_engine()
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    """FastAPI dependency — provides database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Create all tables on startup"""
    from app.models import base  # noqa - imports all models
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database initialized")


async def check_db_integrity() -> bool:
    """Check SQLite database integrity on startup"""
    if "sqlite" not in settings.db_url:
        return True
    async with AsyncSessionLocal() as session:
        try:
            result = await session.execute(
                __import__("sqlalchemy").text("PRAGMA integrity_check")
            )
            check = result.scalar()
            if check != "ok":
                log.error(f"Database integrity check failed: {check}")
                return False
            return True
        except Exception as e:
            log.error(f"Database integrity check error: {e}")
            return False
