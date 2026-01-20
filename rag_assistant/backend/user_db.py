"""
User database management with SQLite.
Handles user registration, authentication, and API key management.
"""
import sqlite3
import secrets
import hashlib
import bcrypt
from typing import Optional, Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a bcrypt hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


def hash_api_key(api_key: str) -> str:
    """Hash an API key using SHA-256."""
    return hashlib.sha256(api_key.encode('utf-8')).hexdigest()


def generate_api_key() -> str:
    """Generate a secure random API key."""
    return secrets.token_urlsafe(32)


class UserDatabase:
    """Manages user authentication and API keys."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.USERS_DB_PATH
        self._init_db()
    
    def _get_connection(self):
        """Get a database connection."""
        return sqlite3.connect(self.db_path)
    
    def _init_db(self):
        """Initialize the database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # API keys table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                key_hash TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def register_user(self, username: str, password: str) -> Optional[str]:
        """
        Register a new user and return an API key.
        Returns None if username already exists.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Hash password
            password_hash = hash_password(password)
            
            # Insert user
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash)
            )
            user_id = cursor.lastrowid
            
            # Generate and store API key
            api_key = generate_api_key()
            key_hash = hash_api_key(api_key)
            
            cursor.execute(
                "INSERT INTO api_keys (user_id, key_hash) VALUES (?, ?)",
                (user_id, key_hash)
            )
            
            conn.commit()
            return api_key
            
        except sqlite3.IntegrityError:
            # Username already exists
            return None
        finally:
            conn.close()
    
    def login_user(self, username: str, password: str) -> Optional[str]:
        """
        Authenticate a user and return their API key.
        Returns None if authentication fails.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Get user
            cursor.execute(
                "SELECT id, password_hash FROM users WHERE username = ?",
                (username,)
            )
            result = cursor.fetchone()
            
            if not result:
                return None
            
            user_id, password_hash = result
            
            # Verify password
            if not verify_password(password, password_hash):
                return None
            
            # Get or create API key
            cursor.execute(
                "SELECT key_hash FROM api_keys WHERE user_id = ? LIMIT 1",
                (user_id,)
            )
            key_result = cursor.fetchone()
            
            if key_result:
                # Return existing key (we can't reverse the hash, so generate new one)
                # Actually, we should store the original key or generate a new one
                # For simplicity, generate a new key on each login
                api_key = generate_api_key()
                key_hash = hash_api_key(api_key)
                
                # Update existing key
                cursor.execute(
                    "UPDATE api_keys SET key_hash = ?, created_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                    (key_hash, user_id)
                )
            else:
                # Create new key
                api_key = generate_api_key()
                key_hash = hash_api_key(api_key)
                
                cursor.execute(
                    "INSERT INTO api_keys (user_id, key_hash) VALUES (?, ?)",
                    (user_id, key_hash)
                )
            
            conn.commit()
            return api_key
            
        finally:
            conn.close()
    
    def verify_api_key(self, api_key: str) -> Optional[Dict]:
        """
        Verify an API key and return user info.
        Returns None if key is invalid.
        """
        key_hash = hash_api_key(api_key)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT u.id, u.username
                FROM users u
                JOIN api_keys k ON u.id = k.user_id
                WHERE k.key_hash = ?
            """, (key_hash,))
            
            result = cursor.fetchone()
            
            if result:
                return {
                    "user_id": result[0],
                    "username": result[1]
                }
            return None
            
        finally:
            conn.close()


# Global instance
db = UserDatabase()
