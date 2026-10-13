# Local MySQL setup

## 1. Confirm the MySQL database exists

Start the MySQL Server Windows service and confirm the `Rahul` database exists in MySQL Workbench. If it does not exist, run:

```sql
CREATE DATABASE Rahul
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;
```

The app will use the MySQL `root` account configured in `.env`. For shared or deployed environments, create a dedicated database user with access only to this database.

## 2. Configure the project

From the project folder in PowerShell, copy the example configuration:

```powershell
Copy-Item .env.example .env
notepad .env
```

Set `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, and `DB_NAME` in `.env` to match your MySQL server. `.env` is excluded by `.gitignore`; do not commit it.

## 3. Install dependencies and test MySQL

Activate the virtual environment, then run:

```powershell
python -m pip install -r requirements.txt
python main.py
```

The connection test should report success. If it fails, check that MySQL Server is running, port `3306` is reachable, the database exists, and the `.env` settings match your MySQL credentials.

## 4. Start the API and persist recommendations

```powershell
python -m uvicorn api:app --reload
```

API startup creates the `recommendations` table if needed. The app then generates recommendations from the Excel workbook and replaces the saved recommendation rows in MySQL. The workbook itself remains the input source; this pipeline does not import invoice rows into MySQL.

To generate and save recommendations without starting the API, run:

```powershell
python batch_db_pipeline.py
```

This batch command reads the workbook in the project folder and refreshes the same MySQL recommendations table.