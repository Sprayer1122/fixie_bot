"""
FixieBot Configuration
Central configuration file for all application settings.
"""

import os

# Flask Configuration
class Config:
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'fixiebot-secret-key-2024'
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000
    
    # ML Model Configuration
    ML_CONFIG = {
        'max_features': 1000,  # TF-IDF max features
        'n_estimators': 100,   # Random Forest estimators
        'random_state': 42,    # Random seed for reproducibility
        'test_size': 0.2,      # Test split ratio
    }
    
    # Data Configuration
    DATA_CONFIG = {
        'historical_data_path': 'dataset/historical_tickets.csv',
        'new_data_path': 'dataset/new_tickets.csv',
        'model_save_path': 'models/',  # Directory to save trained models
    }
    
    # Chatbot Configuration
    CHAT_CONFIG = {
        'max_similar_tickets': 3,      # Number of similar tickets to show
        'min_confidence_threshold': 0.3,  # Minimum confidence to show prediction
        'typing_delay_ms': 1000,       # Delay before showing typing indicator
    }
    
    # UI Configuration
    UI_CONFIG = {
        'app_name': 'FixieBot',
        'app_description': 'AI-Powered Ticket Fix Predictor',
        'welcome_message': 'Welcome to FixieBot! I can help you predict the most likely fix for customer tickets.',
        'example_queries': [
            'Customer reported slow loading times in the dashboard',
            'User interface buttons not responding',
            'Database connection errors occurring frequently',
            'Request for new feature in payment module',
            'Application crashes when uploading large files',
            'Login authentication issues for multiple users'
        ]
    }
    
    # Logging Configuration
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': 'fixiebot.log'
    }

# Development Configuration
class DevelopmentConfig(Config):
    DEBUG = True
    ML_CONFIG = {
        **Config.ML_CONFIG,
        'n_estimators': 50,  # Faster training for development
    }

# Production Configuration
class ProductionConfig(Config):
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    ML_CONFIG = {
        **Config.ML_CONFIG,
        'n_estimators': 200,  # More estimators for better accuracy
    }

# Testing Configuration
class TestingConfig(Config):
    TESTING = True
    ML_CONFIG = {
        **Config.ML_CONFIG,
        'n_estimators': 10,  # Minimal for testing
    }

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'default')
    return config.get(env, config['default']) 