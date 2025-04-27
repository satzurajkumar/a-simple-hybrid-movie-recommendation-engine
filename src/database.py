# src/database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)

# --- Explicitly load .env from the project root ---
# Construct the path to the .env file relative to this script's directory
# Assumes database.py is in a 'src' folder one level below the project root where .env is
script_dir = os.path.dirname(os.path.abspath(__file__)) # Gets the directory of database.py (e.g., .../src)
project_root = os.path.dirname(script_dir) # Go one level up to the project root (e.g., ...)
dotenv_path = os.path.join(project_root, '.env') # Construct the full path to .env

logging.info(f"Looking for .env file at: {dotenv_path}")

# Load the .env file using the explicit path
# override=True ensures that the .env file takes precedence over existing system vars if found
loaded = load_dotenv(dotenv_path=dotenv_path, override=True)

if not loaded:
    logging.warning(f".env file not found or not loaded from path: {dotenv_path}")
# --- End explicit load ---


DATABASE_URL = os.getenv("DATABASE_URL")

# --- Debugging Print Statement ---
logging.info(f"Attempting to connect with DATABASE_URL: {DATABASE_URL}")
# --- End Debugging Print Statement ---

if not DATABASE_URL:
    logging.error("DATABASE_URL environment variable not set or empty after attempting load.")
    raise ValueError("DATABASE_URL environment variable not set. Please check your .env file and its location.")

# Check if the URL contains the expected prefix (optional check)
if DATABASE_URL and "pymysql" not in DATABASE_URL: # Added check if DATABASE_URL is not None
    logging.warning(f"DATABASE_URL does not contain 'pymysql'. Current URL: {DATABASE_URL}")


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
