from typing import Optional, List, Dict
import pandas as pd
import psycopg2
import os
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__)

class DatabaseService:
    """
    Class for database operations and data extraction.
    """
    def __init__(self,
                 host: str = "localhost",
                 port: int = 5432,
                 dbname: str = "your_database",
                 user: str = "your_username",
                 password: str = "your_password"):
        """
        Initialize the database service.

        Args:
            host: Database host
            port: Database port
            dbname: Database name
            user: Database username
            password: Database password
        """
        self.host = os.getenv("LEMURS_POSTGRES_HOST", host)
        self.port = int(os.getenv("LEMURS_POSTGRES_PORT", port))
        self.dbname = os.getenv("LEMURS_POSTGRES_DB", dbname)
        self.user = os.getenv("LEMURS_POSTGRES_USER", user)
        self.password = os.getenv("LEMURS_POSTGRES_PASSWORD", password)
        self.connection = None

    def connect(self) -> bool:
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.user,
                password=self.password
            )
            return True
        except psycopg2.Error as e:
            logger.error(f"Error connecting to PostgreSQL database: {e}")
            return False

    def disconnect(self):
        """Disconnect from database"""
        if self.connection:
            self.connection.close()

    def extract_from_database(self, table_name: str) -> pd.DataFrame:
        """
        Extract data from a database table into a pandas DataFrame.

        Args:
            table_name: Name of the database table to extract

        Returns:
            DataFrame containing all records from the table

        Raises:
            Exception: If database connection fails or query execution fails
        """
        # Connect if not already connected
        if not self.connection or self.connection.closed:
            if not self.connect():
                raise Exception("Failed to connect to database")
        
        try:
            cursor = self.connection.cursor()

            # First check if 'id' column exists
            check_query = f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}' AND column_name = 'id'
            """
            cursor.execute(check_query)
            has_id = cursor.fetchone() is not None
            cursor.close()

            # Build query with ORDER BY only if id column exists
            if has_id:
                query = f"SELECT * FROM {table_name} ORDER BY id"
            else:
                query = f"SELECT * FROM {table_name}"

            df = pd.read_sql(query, self.connection)
            return df

        except Exception as e:
            logger.error(f"Error extracting data from {table_name}: {e}")
            raise
