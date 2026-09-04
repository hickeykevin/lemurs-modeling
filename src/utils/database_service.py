from typing import Optional, List, Dict
import os
import re
import io
import pandas as pd
import psycopg2
import paramiko

# Compatibility shim: sshtunnel 0.4.0 references paramiko.DSSKey which was removed in newer Paramiko
if not hasattr(paramiko, "DSSKey"):
    class _DummyDSSKey:
        pass
    paramiko.DSSKey = _DummyDSSKey

from sshtunnel import SSHTunnelForwarder

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__)

class DatabaseService:
    """
    Class for database operations and data extraction via SSH tunneling.
    """
    def __init__(self,
                 host: Optional[str] = None,
                 port: Optional[int] = None,
                 dbname: Optional[str] = None,
                 user: Optional[str] = None,
                 password: Optional[str] = None,
                 ssh_host: Optional[str] = None,
                 ssh_port: Optional[int] = None,
                 ssh_user: Optional[str] = None,
                 ssh_pkey: Optional[str] = None,
                 ssh_key: Optional[str] = None,
                 ssh_password: Optional[str] = None):
        """
        Initialize the database service.

        Args:
            host: Database host (from the perspective of the database/SSH server, default 'localhost')
            port: Database port (default 5432)
            dbname: Database name
            user: Database username
            password: Database password
            ssh_host: SSH bastion / server host
            ssh_port: SSH port (default 22)
            ssh_user: SSH username
            ssh_pkey: Path to SSH private key file (e.g. ~/.ssh/id_rsa)
            ssh_key: Raw SSH private key string or path
            ssh_password: SSH password (if not using key-based authentication)
        """
        self.host = host if host is not None else os.getenv("LEMURS_POSTGRES_HOST", "localhost")
        self.port = int(port if port is not None else os.getenv("LEMURS_POSTGRES_PORT", 5432))
        self.dbname = dbname if dbname is not None else os.getenv("LEMURS_POSTGRES_DB", "your_database")
        self.user = user if user is not None else os.getenv("LEMURS_POSTGRES_USER", "your_username")
        self.password = password if password is not None else os.getenv("LEMURS_POSTGRES_PASSWORD", "your_password")

        # SSH Tunnel configuration
        self.ssh_host = ssh_host if ssh_host is not None else os.getenv("LEMURS_SSH_HOST")
        self.ssh_port = int(ssh_port if ssh_port is not None else os.getenv("LEMURS_SSH_PORT", 22))
        self.ssh_user = ssh_user if ssh_user is not None else os.getenv("LEMURS_SSH_USER")
        self.ssh_pkey = ssh_pkey if ssh_pkey is not None else os.getenv("LEMURS_SSH_KEY_PATH")
        self.ssh_key = ssh_key if ssh_key is not None else os.getenv("LEMURS_SSH_KEY")
        self.ssh_password = ssh_password if ssh_password is not None else os.getenv("LEMURS_SSH_PASSWORD")

        self.connection = None
        self.tunnel: Optional[SSHTunnelForwarder] = None

    def _resolve_ssh_pkey(self):
        """
        Resolve the private key from raw key string, key path, or existing PKey object.
        Returns a paramiko.PKey instance, a path string to the key file, or None.
        """
        key_source = self.ssh_key or self.ssh_pkey
        if not key_source:
            return None

        # If already a paramiko PKey object
        if hasattr(key_source, "get_name") and hasattr(key_source, "asbytes"):
            return key_source

        if isinstance(key_source, str):
            key_source_str = key_source.strip().strip('"').strip("'").strip()

            # Check if it's an existing file path first
            expanded = os.path.expanduser(key_source_str)
            if os.path.exists(expanded):
                return expanded

            # Handle escaped literal \n commonly present when storing multi-line secrets on one line in .env
            if "\\n" in key_source_str:
                key_source_str = key_source_str.replace("\\n", "\n")

            candidates = []

            # Check if headers are present
            match = re.search(r"-----BEGIN ([A-Z ]+)-----(.+?)-----END \1-----", key_source_str, re.DOTALL)
            if match:
                header_type = match.group(1).strip()
                body = re.sub(r"\s+", "", match.group(2).strip())
                body_lines = [body[i:i+64] for i in range(0, len(body), 64)]
                candidates.append(f"-----BEGIN {header_type}-----\n" + "\n".join(body_lines) + f"\n-----END {header_type}-----")
                candidates.append(key_source_str)
            else:
                # Bare base64 payload: clean all whitespace and wrap in standard headers
                body = re.sub(r"\s+", "", key_source_str)
                body_lines = [body[i:i+64] for i in range(0, len(body), 64)]
                formatted_body = "\n".join(body_lines)
                candidates.extend([
                    f"-----BEGIN OPENSSH PRIVATE KEY-----\n{formatted_body}\n-----END OPENSSH PRIVATE KEY-----",
                    f"-----BEGIN RSA PRIVATE KEY-----\n{formatted_body}\n-----END RSA PRIVATE KEY-----",
                    f"-----BEGIN EC PRIVATE KEY-----\n{formatted_body}\n-----END EC PRIVATE KEY-----",
                    f"-----BEGIN PRIVATE KEY-----\n{formatted_body}\n-----END PRIVATE KEY-----",
                ])

            for candidate in candidates:
                for key_cls in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
                    try:
                        f = io.StringIO(candidate)
                        return key_cls.from_private_key(f, password=self.ssh_password)
                    except Exception:
                        continue

            logger.warning("Failed to parse raw SSH key string into a paramiko PKey.")

        return key_source

    def connect(self) -> bool:
        """Connect to PostgreSQL database through an encrypted SSH tunnel."""
        try:
            if not self.ssh_host:
                raise ValueError("SSH host is required for database connection (LEMURS_SSH_HOST).")

            tunnel_kwargs = {
                "ssh_address_or_host": (self.ssh_host, self.ssh_port),
                "remote_bind_address": (self.host, self.port),
            }
            if self.ssh_user:
                tunnel_kwargs["ssh_username"] = self.ssh_user

            resolved_pkey = self._resolve_ssh_pkey()
            if resolved_pkey:
                tunnel_kwargs["ssh_pkey"] = resolved_pkey

            if self.ssh_password:
                tunnel_kwargs["ssh_password"] = self.ssh_password

            logger.info(f"Opening SSH tunnel to {self.ssh_host}:{self.ssh_port} -> {self.host}:{self.port}")
            self.tunnel = SSHTunnelForwarder(**tunnel_kwargs)
            self.tunnel.start()

            self.connection = psycopg2.connect(
                host="127.0.0.1",
                port=self.tunnel.local_bind_port,
                dbname=self.dbname,
                user=self.user,
                password=self.password
            )
            return True
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL database: {e}")
            self.disconnect()
            return False

    def disconnect(self):
        """Disconnect from database and close any active SSH tunnel."""
        if self.connection:
            try:
                self.connection.close()
            except Exception as e:
                logger.warning(f"Error closing PostgreSQL connection: {e}")
            finally:
                self.connection = None

        if self.tunnel:
            try:
                self.tunnel.stop()
            except Exception as e:
                logger.warning(f"Error stopping SSH tunnel: {e}")
            finally:
                self.tunnel = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

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
