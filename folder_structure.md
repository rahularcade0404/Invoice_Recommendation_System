invoice_recommendation_project/
├── app/
│   ├── __init__.py
│   ├── main.py             # App initialization & entry point
│   ├── config.py           # Settings and Pydantic BaseSettings
│   ├── database.py         # DB session and connection setup
|   |-- data_loader.py      # read multiple excel files from files folder
│   ├── api/                # API Endpoints (Routers)
│   │   ├── __init__.py
│   │   ├── api.py    # recommendation engine api covering two APIs
│   │-- files/              # only excel/csv files         
│   │   │   └── 
│   ├── models/             # Database/ORM models (e.g., SQLAlchemy)
│   │   ├── __init__.py
│   │   └── user.py
│   ├── schemas/            # Data validation models (Pydantic)
│   │   ├── __init__.py
│   │   └── user.py
│   └── services/           # Business logic layer
│       ├── __init__.py
│       └── user_service.py
|       |__recommendation_service.py     #  DB CRUD operations for recommendation table
├── tests/                   # Pytest test suite mirroring app/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_auth.py
├── .env                    # Local environment variables
├── .gitignore
├── requirements.txt        # Package dependencies
└── README.md
