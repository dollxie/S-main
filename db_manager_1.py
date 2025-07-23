import hashlib
import os
from datetime import datetime, timedelta
from pymongo import MongoClient
from bson.objectid import ObjectId

class DatabaseManager:
    def __init__(self, db_name=os.getenv('MONGODB_DATABASE_NAME', 'user_auth_db')):
        # IMPORTANT: Store your MongoDB connection string in an environment variable.
        # Example: MONGODB_CONNECTION_STRING = "mongodb://user:pass@host..."
        # Ensure this string is not hardcoded in your repository.
        connection_string = os.environ.get("MONGODB_CONNECTION_STRING")
        if not connection_string:
            # Handle missing connection string
            print("ERROR: MONGODB_CONNECTION_STRING environment variable not set.")
            # In a real app, you might raise an error or have a default local fallback (use with caution)
            raise ValueError("MongoDB connection string not configured")

        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        self.users = self.db.users
        self.sessions = self.db.sessions
        
        self.users.create_index('username', unique=True)
        self.users.create_index('email', unique=True)
        self.sessions.create_index('session_token', unique=True)
    
    def _generate_salt(self):
        return os.urandom(32).hex()
    
    def _hash_password(self, password, salt):
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return password_hash
    
    def register_user(self, username, email, password):
        try:
            # Check if username or email already exists
            if self.users.find_one({'$or': [{'username': username}, {'email': email}]}):
                return False, "Username or email already exists."
            
            # Generate salt and hash password
            salt = self._generate_salt()
            password_hash = self._hash_password(password, salt)
            
            # Insert new user
            user_data = {
                'username': username,
                'email': email,
                'password_hash': password_hash,
                'salt': salt,
                'created_at': datetime.now(),
                'last_login': None
            }
            
            self.users.insert_one(user_data)
            return True, "User registered successfully."
        except Exception as e:
            return False, f"Error registering user: {str(e)}"
    
    def authenticate_user(self, username, password):
        try:
            # Get user data
            user = self.users.find_one({'username': username})
            
            if not user:
                return False, None, "Invalid username or password."
            
            # Check password
            input_hash = self._hash_password(password, user['salt'])
            if input_hash != user['password_hash']:
                return False, None, "Invalid username or password."
            
            # Update last login time
            self.users.update_one(
                {'_id': user['_id']},
                {'$set': {'last_login': datetime.now()}}
            )
            
            return True, str(user['_id']), "Authentication successful."
        except Exception as e:
            return False, None, f"Authentication error: {str(e)}"
    
    def create_session(self, user_id, expiry_hours=24):
        """Create a new session for the user."""
        try:
            # Generate session token
            session_token = os.urandom(32).hex()
            
            # Calculate expiry time
            expires_at = datetime.now() + timedelta(hours=expiry_hours)
            
            # Insert session
            session_data = {
                'user_id': user_id,
                'session_token': session_token,
                'created_at': datetime.now(),
                'expires_at': expires_at
            }
            
            self.sessions.insert_one(session_data)
            return session_token
        except Exception as e:
            print(f"Error creating session: {str(e)}")
            return None
    
    def validate_session(self, session_token):
        """Validate a session token and return user_id if valid."""
        try:
            # Get session data
            session = self.sessions.find_one({'session_token': session_token})
            
            if not session:
                return None
            
            # Check if session has expired
            if datetime.now() > session['expires_at']:
                # Delete expired session
                self.sessions.delete_one({'session_token': session_token})
                return None
            
            return session['user_id']
        except Exception as e:
            print(f"Error validating session: {str(e)}")
            return None
    
    def delete_session(self, session_token):
        try:
            self.sessions.delete_one({'session_token': session_token})
            return True
        except Exception as e:
            print(f"Error deleting session: {str(e)}")
            return False
    
    def get_user_info(self, user_id):
        try:
            # Convert string ID to ObjectId if necessary
            if isinstance(user_id, str):
                user_id = ObjectId(user_id)
                
            user = self.users.find_one({'_id': user_id})
            
            if not user:
                return None
            
            user_info = {
                "id": str(user['_id']),
                "username": user['username'],
                "email": user['email'],
                "created_at": user['created_at'],
                "last_login": user['last_login']
            }
            
            return user_info
        except Exception as e:
            print(f"Error getting user info: {str(e)}")
            return None