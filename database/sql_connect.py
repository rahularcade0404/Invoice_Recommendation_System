from sqlalchemy import text

from database.database import engine

with engine.connect() as connection:
    connection.execute(text("SELECT 1"))

print("Database connection established successfully.")