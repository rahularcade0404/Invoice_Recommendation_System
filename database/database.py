import datetime
import os

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Float, Integer, String, URL, create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

required_settings = ("DB_USER", "DB_PASSWORD", "DB_NAME")
missing_settings = [name for name in required_settings if not os.getenv(name)]
if missing_settings:
    raise RuntimeError("Missing MySQL settings in the project .env file: "
        + ", ".join(missing_settings)
    )

DATABASE_URL = URL.create(
    drivername="mysql+pymysql",
    username=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "3306")),
    database=os.environ["DB_NAME"],
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# SQL Table Model to store generated recommendations
class SavedRecommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)
    contact_name = Column(String(255), index=True)
    engine_type = Column(String(64))  # 'Collaborative', 'Majority_Reorder', or 'State_Majority'
    recommended_part_group = Column(String(255))
    recommended_qty = Column(Integer, nullable=True)
    recommendation_score = Column(String(64))
    state = Column(String(255), nullable=True)
    


class MajorityReorderRecommendation(Base):
    __tablename__ = "majority_reorders"

    id = Column(Integer, primary_key=True, index=True)
    contact_name = Column(String(255), index=True)
    recommended_part_group = Column(String(255))
    material = Column(String(255))
    material_desc = Column(String(512))
    previously_bought_qty = Column(Integer, nullable=True)
    recommended_qty = Column(Integer, nullable=True)
    recommendation_score = Column(String(64))
    state = Column(String(255), nullable=True)
    


class StateRecommendation(Base):
    __tablename__ = "same_state_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    contact_name = Column(String(255), index=True)
    recommended_part_group = Column(String(255))
    previously_bought_qty = Column(Integer, nullable=True)
    recommended_qty = Column(Integer, nullable=True)
    recommendation_score = Column(String(64))
    state = Column(String(255), nullable=True)

# Initialize DB Tables
def init_db():
    inspector = inspect(engine)
    if inspector.has_table("majority_reorders"):
        column_names = {
            column["name"]
            for column in inspector.get_columns("majority_reorders")
        }
        if "previously_bought_qty" not in column_names:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE majority_reorders "
                        "ADD COLUMN previously_bought_qty INTEGER NULL"
                    )
                )

    if inspector.has_table("same_state_recommendations"):
        column_names = {
            column["name"]
            for column in inspector.get_columns("same_state_recommendations")
        }
        if "previously_bought_qty" not in column_names:
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE same_state_recommendations "
                        "ADD COLUMN previously_bought_qty INTEGER NULL"
                    )
                )

    if inspector.has_table("recommendations"):
        column_names = {
            column["name"] for column in inspector.get_columns("recommendations")
        }
        if ("item_or_part_group" in column_names and "recommended_part_group" not in column_names):
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE recommendations "
                        "CHANGE COLUMN item_or_part_group "
                        "recommended_part_group VARCHAR(255)"
                    )
                )
    Base.metadata.create_all(bind=engine)


# DB Dependency for API Endpoints
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()