from database.database import engine, get_db
from sqlalchemy import text

def test_connection():
    try:
        # Obtain a connection from the engine
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("Successfully connected to MySQL database!")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_connection()