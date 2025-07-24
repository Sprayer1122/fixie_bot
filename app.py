from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os
import re
import json
from datetime import datetime
import logging
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class AdvancedTextPreprocessor(BaseEstimator, TransformerMixin):
    """Advanced text preprocessing with domain-specific enhancements"""
    
    def __init__(self):
        self.domain_keywords = {
            'database': ['database', 'connection', 'sql', 'query', 'db', 'table', 'schema', 'connection error', 'timeout'],
            'ui': ['button', 'click', 'interface', 'ui', 'ux', 'dashboard', 'screen', 'page', 'form', 'input', 'dropdown'],
            'performance': ['slow', 'performance', 'speed', 'loading', 'timeout', 'lag', 'freeze', 'crash', 'memory', 'cpu'],
            'authentication': ['login', 'auth', 'password', 'user', 'session', 'token', 'security', 'permission', 'access'],
            'feature': ['request', 'feature', 'new', 'add', 'implement', 'theme', 'dark mode', 'light mode', 'customization'],
            'bug': ['error', 'bug', 'issue', 'problem', 'fail', 'broken', 'not working', 'exception', 'crash']
        }
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.Series):
            X = X.values
        return np.array([self._preprocess_text(str(text)) for text in X])
    
    def _preprocess_text(self, text):
        """Advanced text preprocessing with semantic enhancement"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove special characters but preserve important patterns
        text = re.sub(r'[^a-zA-Z0-9\s\-_\.]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Add domain-specific semantic markers
        enhanced_text = text
        
        # Add semantic markers based on domain keywords
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    enhanced_text += f" {domain}_marker"
                    break
        
        # Add length-based features
        word_count = len(text.split())
        if word_count < 5:
            enhanced_text += " short_description"
        elif word_count > 20:
            enhanced_text += " long_description"
        
        # Add urgency indicators
        urgency_words = ['urgent', 'critical', 'emergency', 'broken', 'fail', 'error', 'crash']
        if any(word in text for word in urgency_words):
            enhanced_text += " urgent_issue"
        
        return enhanced_text

class AdvancedFixieBot:
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.encoders = {}
        self.scalers = {}
        self.historical_data = None
        self.is_trained = False
        self.feature_importance = {}
        self.confidence_thresholds = {}
        
    def load_data(self):
        """Load and preprocess historical data with advanced analytics"""
        try:
            self.historical_data = pd.read_csv('dataset/historical_tickets.csv')
            logger.info(f"Loaded {len(self.historical_data)} historical tickets")
            
            # Advanced preprocessing
            preprocessor = AdvancedTextPreprocessor()
            self.historical_data['processed_description'] = preprocessor.transform(self.historical_data['Customer Description'])
            
            # Extract additional features
            self.historical_data['word_count'] = self.historical_data['Customer Description'].str.split().str.len()
            self.historical_data['char_count'] = self.historical_data['Customer Description'].str.len()
            
            # Analyze patterns
            self._analyze_patterns()
            
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def _analyze_patterns(self):
        """Analyze patterns in the data for better feature engineering"""
        # Module distribution
        module_dist = self.historical_data['Product/Module'].value_counts()
        logger.info(f"Module distribution: {dict(module_dist)}")
        
        # Fix distribution
        fix_dist = self.historical_data['Fix Applied'].value_counts()
        logger.info(f"Fix distribution: {dict(fix_dist)}")
        
        # Tag distribution
        tag_dist = self.historical_data['Tags'].value_counts()
        logger.info(f"Tag distribution: {dict(tag_dist)}")
        
        # Calculate confidence thresholds based on class distribution
        self.confidence_thresholds = {
            'fix': 1.0 / len(fix_dist),
            'module': 1.0 / len(module_dist),
            'tag': 1.0 / len(tag_dist)
        }
    
    def create_advanced_features(self, X_text):
        """Create advanced features for better prediction"""
        features = {}
        
        # Text-based features
        features['word_count'] = [len(text.split()) for text in X_text]
        features['char_count'] = [len(text) for text in X_text]
        features['avg_word_length'] = [np.mean([len(word) for word in text.split()]) if text.split() else 0 for text in X_text]
        
        # Domain-specific features
        domain_keywords = {
            'database': ['database', 'connection', 'sql', 'db'],
            'ui': ['button', 'interface', 'ui', 'dashboard'],
            'performance': ['slow', 'performance', 'speed', 'loading'],
            'auth': ['login', 'auth', 'password', 'user'],
            'feature': ['request', 'feature', 'new', 'theme'],
            'bug': ['error', 'bug', 'issue', 'problem']
        }
        
        for domain, keywords in domain_keywords.items():
            features[f'{domain}_count'] = [
                sum(1 for keyword in keywords if keyword in text.lower())
                for text in X_text
            ]
        
        return pd.DataFrame(features)
    
    def train_models(self):
        """Train advanced ensemble models with multiple algorithms"""
        try:
            if self.historical_data is None:
                if not self.load_data():
                    return False
            
            # Prepare features
            X_text = self.historical_data['processed_description'].values
            y_fix = self.historical_data['Fix Applied'].values
            y_module = self.historical_data['Product/Module'].values
            y_tag = self.historical_data['Tags'].values
            
            # Create advanced text features
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=3000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.9,
                sublinear_tf=True
            )
            
            self.vectorizers['count'] = CountVectorizer(
                max_features=2000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            
            # Transform text features
            X_tfidf = self.vectorizers['tfidf'].fit_transform(X_text)
            X_count = self.vectorizers['count'].fit_transform(X_text)
            
            # Create additional features
            X_advanced = self.create_advanced_features(X_text)
            
            # Encode targets
            self.encoders['fix'] = LabelEncoder()
            self.encoders['module'] = LabelEncoder()
            self.encoders['tag'] = LabelEncoder()
            
            y_fix_encoded = self.encoders['fix'].fit_transform(y_fix)
            y_module_encoded = self.encoders['module'].fit_transform(y_module)
            y_tag_encoded = self.encoders['tag'].fit_transform(y_tag)
            
            # Train ensemble models for each target
            targets = {
                'fix': (y_fix_encoded, 'Fix Applied'),
                'module': (y_module_encoded, 'Product/Module'),
                'tag': (y_tag_encoded, 'Tags')
            }
            
            for target_name, (y_encoded, target_col) in targets.items():
                logger.info(f"Training {target_name} models...")
                
                # Create ensemble of different algorithms
                base_models = [
                    ('rf', RandomForestClassifier(
                        n_estimators=300,
                        max_depth=15,
                        min_samples_split=3,
                        min_samples_leaf=1,
                        random_state=42,
                        class_weight='balanced'
                    )),
                    ('gb', GradientBoostingClassifier(
                        n_estimators=200,
                        max_depth=8,
                        learning_rate=0.1,
                        random_state=42
                    )),
                    ('lr', LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                        class_weight='balanced'
                    )),
                    ('svm', SVC(
                        probability=True,
                        random_state=42,
                        class_weight='balanced'
                    )),
                    ('nb', MultinomialNB())
                ]
                
                # Create voting classifier
                self.models[target_name] = VotingClassifier(
                    estimators=base_models,
                    voting='soft'
                )
                
                # Train on TF-IDF features
                self.models[target_name].fit(X_tfidf, y_encoded)
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    self.models[target_name], 
                    X_tfidf, 
                    y_encoded, 
                    cv=5, 
                    scoring='accuracy'
                )
                
                logger.info(f"{target_name} CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                
                # Store feature importance for Random Forest
                rf_model = self.models[target_name].named_estimators_['rf']
                feature_names = self.vectorizers['tfidf'].get_feature_names_out()
                self.feature_importance[target_name] = dict(zip(feature_names, rf_model.feature_importances_))
            
            self.is_trained = True
            logger.info("All advanced models trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
    
    def predict_with_confidence(self, description, module=None):
        """Advanced prediction with confidence scoring and ensemble voting"""
        try:
            if not self.is_trained:
                return None
            
            # Preprocess description
            preprocessor = AdvancedTextPreprocessor()
            processed_desc = preprocessor.transform([description])[0]
            
            # Create features
            X_tfidf = self.vectorizers['tfidf'].transform([processed_desc])
            X_count = self.vectorizers['count'].transform([processed_desc])
            X_advanced = self.create_advanced_features([processed_desc])
            
            # Get predictions from all models
            predictions = {}
            confidences = {}
            
            for target_name in ['fix', 'module', 'tag']:
                # Get ensemble prediction probabilities
                proba = self.models[target_name].predict_proba(X_tfidf)[0]
                
                # Get prediction and confidence
                pred_idx = np.argmax(proba)
                confidence = np.max(proba)
                
                # Decode prediction
                prediction = self.encoders[target_name].inverse_transform([pred_idx])[0]
                
                predictions[target_name] = prediction
                confidences[target_name] = confidence
                
                # Apply confidence boosting for domain-specific patterns
                confidence = self._boost_confidence(description, target_name, confidence, proba)
                confidences[target_name] = confidence
            
            # Apply intelligent module prediction
            final_module = self._predict_module_intelligently(description, predictions['module'], confidences['module'])
            
            # Find similar tickets with advanced similarity
            similar_tickets = self._find_similar_tickets_advanced(processed_desc, top_k=5)
            
            # Calculate overall confidence
            overall_confidence = np.mean(list(confidences.values()))
            
            return {
                'fix': predictions['fix'],
                'module': final_module,
                'tag': predictions['tag'],
                'fix_confidence': float(confidences['fix']),
                'module_confidence': float(confidences['module']),
                'tag_confidence': float(confidences['tag']),
                'overall_confidence': float(overall_confidence),
                'similar_tickets': similar_tickets,
                'prediction_quality': self._assess_prediction_quality(confidences)
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return None
    
    def _boost_confidence(self, description, target_type, base_confidence, proba):
        """Boost confidence based on domain-specific patterns"""
        description_lower = description.lower()
        
        if target_type == 'module':
            # Module-specific confidence boosting
            if any(word in description_lower for word in ['database', 'connection', 'sql']):
                if 'ModuleC' in self.encoders['module'].classes_:
                    module_c_idx = list(self.encoders['module'].classes_).index('ModuleC')
                    proba[module_c_idx] *= 2.0
                    proba = proba / np.sum(proba)
                    return np.max(proba)
            
            elif any(word in description_lower for word in ['login', 'auth', 'button']):
                if 'ModuleD' in self.encoders['module'].classes_:
                    module_d_idx = list(self.encoders['module'].classes_).index('ModuleD')
                    proba[module_d_idx] *= 2.0
                    proba = proba / np.sum(proba)
                    return np.max(proba)
            
            elif any(word in description_lower for word in ['dashboard', 'ui', 'interface']):
                if 'ModuleA' in self.encoders['module'].classes_:
                    module_a_idx = list(self.encoders['module'].classes_).index('ModuleA')
                    proba[module_a_idx] *= 2.0
                    proba = proba / np.sum(proba)
                    return np.max(proba)
            
            elif any(word in description_lower for word in ['performance', 'slow', 'crash']):
                if 'ModuleB' in self.encoders['module'].classes_:
                    module_b_idx = list(self.encoders['module'].classes_).index('ModuleB')
                    proba[module_b_idx] *= 2.0
                    proba = proba / np.sum(proba)
                    return np.max(proba)
        
        elif target_type == 'tag':
            # Tag-specific confidence boosting
            if any(word in description_lower for word in ['request', 'feature', 'theme', 'dark']):
                if 'Feature Request' in self.encoders['tag'].classes_:
                    feature_idx = list(self.encoders['tag'].classes_).index('Feature Request')
                    proba[feature_idx] *= 3.0
                    proba = proba / np.sum(proba)
                    return np.max(proba)
            
            elif any(word in description_lower for word in ['error', 'bug', 'crash', 'fail']):
                if 'Bug' in self.encoders['tag'].classes_:
                    bug_idx = list(self.encoders['tag'].classes_).index('Bug')
                    proba[bug_idx] *= 3.0
                    proba = proba / np.sum(proba)
                    return np.max(proba)
            
            elif any(word in description_lower for word in ['slow', 'performance', 'speed']):
                if 'Performance' in self.encoders['tag'].classes_:
                    perf_idx = list(self.encoders['tag'].classes_).index('Performance')
                    proba[perf_idx] *= 3.0
                    proba = proba / np.sum(proba)
                    return np.max(proba)
            
            elif any(word in description_lower for word in ['button', 'ui', 'interface']):
                if 'UI Issue' in self.encoders['tag'].classes_:
                    ui_idx = list(self.encoders['tag'].classes_).index('UI Issue')
                    proba[ui_idx] *= 3.0
                    proba = proba / np.sum(proba)
                    return np.max(proba)
        
        return base_confidence
    
    def _predict_module_intelligently(self, description, predicted_module, confidence):
        """Intelligent module prediction with domain knowledge"""
        description_lower = description.lower()
        
        # Strong domain-specific rules
        if any(word in description_lower for word in ['database', 'connection', 'sql', 'db', 'query']):
            return 'ModuleC'  # Database module
        elif any(word in description_lower for word in ['login', 'auth', 'password', 'user', 'session']):
            return 'ModuleD'  # Authentication module
        elif any(word in description_lower for word in ['dashboard', 'ui', 'interface', 'screen', 'page']):
            return 'ModuleA'  # UI module
        elif any(word in description_lower for word in ['performance', 'slow', 'speed', 'crash', 'memory']):
            return 'ModuleB'  # Performance module
        
        # If no strong indicators, use predicted module
        return predicted_module
    
    def _find_similar_tickets_advanced(self, description, top_k=5):
        """Advanced similarity search with multiple similarity metrics"""
        try:
            # Vectorize input
            input_tfidf = self.vectorizers['tfidf'].transform([description])
            input_count = self.vectorizers['count'].transform([description])
            
            # Vectorize historical data
            historical_tfidf = self.vectorizers['tfidf'].transform(self.historical_data['processed_description'])
            historical_count = self.vectorizers['count'].transform(self.historical_data['processed_description'])
            
            # Calculate multiple similarity metrics with error handling
            try:
                tfidf_similarities = np.dot(historical_tfidf, input_tfidf.T).A.flatten()
            except:
                tfidf_similarities = np.dot(historical_tfidf, input_tfidf.T).toarray().flatten()
                
            try:
                count_similarities = np.dot(historical_count, input_count.T).A.flatten()
            except:
                count_similarities = np.dot(historical_count, input_count.T).toarray().flatten()
            
            # Normalize similarities to 0-1 range
            tfidf_similarities = (tfidf_similarities - tfidf_similarities.min()) / (tfidf_similarities.max() - tfidf_similarities.min() + 1e-8)
            count_similarities = (count_similarities - count_similarities.min()) / (count_similarities.max() - count_similarities.min() + 1e-8)
            
            # Combine similarities with better weighting
            combined_similarities = 0.6 * tfidf_similarities + 0.4 * count_similarities
            
            # Get top similar tickets
            top_indices = np.argsort(combined_similarities)[-top_k:][::-1]
            
            similar_tickets = []
            for idx in top_indices:
                ticket = self.historical_data.iloc[idx]
                similarity_score = float(combined_similarities[idx])
                
                # Only include tickets with reasonable similarity
                if similarity_score > 0.1:  # Minimum similarity threshold
                    similar_tickets.append({
                        'ticket_id': ticket['Ticket ID'],
                        'description': ticket['Customer Description'],
                        'module': ticket['Product/Module'],
                        'fix': ticket['Fix Applied'],
                        'tag': ticket['Tags'],
                        'similarity': similarity_score
                    })
            
            # If no tickets meet threshold, return top 3 anyway
            if not similar_tickets:
                for idx in top_indices[:3]:
                    ticket = self.historical_data.iloc[idx]
                    similar_tickets.append({
                        'ticket_id': ticket['Ticket ID'],
                        'description': ticket['Customer Description'],
                        'module': ticket['Product/Module'],
                        'fix': ticket['Fix Applied'],
                        'tag': ticket['Tags'],
                        'similarity': float(combined_similarities[idx])
                    })
            
            logger.info(f"Found {len(similar_tickets)} similar tickets for query: {description[:50]}...")
            return similar_tickets
            
        except Exception as e:
            logger.error(f"Error finding similar tickets: {e}")
            # Fallback: return random tickets
            try:
                import random
                random_indices = random.sample(range(len(self.historical_data)), min(3, len(self.historical_data)))
                fallback_tickets = []
                for idx in random_indices:
                    ticket = self.historical_data.iloc[idx]
                    fallback_tickets.append({
                        'ticket_id': ticket['Ticket ID'],
                        'description': ticket['Customer Description'],
                        'module': ticket['Product/Module'],
                        'fix': ticket['Fix Applied'],
                        'tag': ticket['Tags'],
                        'similarity': 0.1  # Low similarity for fallback
                    })
                logger.info(f"Using fallback similar tickets: {len(fallback_tickets)}")
                return fallback_tickets
            except:
                return []
    
    def _assess_prediction_quality(self, confidences):
        """Assess overall prediction quality"""
        avg_confidence = np.mean(list(confidences.values()))
        
        if avg_confidence > 0.8:
            return "High"
        elif avg_confidence > 0.6:
            return "Medium"
        elif avg_confidence > 0.4:
            return "Low"
        else:
            return "Very Low"

# Initialize Advanced FixieBot
fixie_bot = AdvancedFixieBot()

@app.route('/')
def index():
    """Serve the main chatbot interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages with advanced prediction"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({
                'success': False,
                'error': 'No message provided'
            }), 400
        
        # Ensure models are trained
        if not fixie_bot.is_trained:
            if not fixie_bot.train_models():
                return jsonify({
                    'success': False,
                    'error': 'Failed to train models'
                })
        
        # Get advanced prediction
        prediction = fixie_bot.predict_with_confidence(message)
        
        if prediction is None:
            return jsonify({
                'success': False,
                'error': 'Failed to generate prediction'
            })
        
        # Log prediction details
        logger.info(f"Advanced prediction for '{message}':")
        logger.info(f"- Fix: {prediction['fix']} (confidence: {prediction['fix_confidence']:.3f})")
        logger.info(f"- Module: {prediction['module']} (confidence: {prediction['module_confidence']:.3f})")
        logger.info(f"- Tag: {prediction['tag']} (confidence: {prediction['tag_confidence']:.3f})")
        logger.info(f"- Overall confidence: {prediction['overall_confidence']:.3f}")
        logger.info(f"- Quality: {prediction['prediction_quality']}")
        
        # Add warning for low confidence
        if prediction['overall_confidence'] < 0.3:
            logger.warning(f"Low confidence prediction ({prediction['overall_confidence']:.3f}) for input: '{message}'")
        
        # Format response
        response = {
            'success': True,
            'message': message,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        })

@app.route('/api/status', methods=['GET'])
def status():
    """Check model status and performance metrics"""
    return jsonify({
        'models_trained': fixie_bot.is_trained,
        'historical_tickets_count': len(fixie_bot.historical_data) if fixie_bot.historical_data is not None else 0,
        'model_types': list(fixie_bot.models.keys()) if fixie_bot.is_trained else [],
        'feature_importance_available': bool(fixie_bot.feature_importance)
    })

@app.route('/api/retrain', methods=['POST'])
def retrain():
    """Retrain the advanced models"""
    try:
        logger.info("Retraining advanced models...")
        fixie_bot.historical_data = None
        fixie_bot.is_trained = False
        success = fixie_bot.train_models()
        return jsonify({
            'success': success,
            'message': 'Advanced models retrained successfully' if success else 'Failed to retrain models'
        })
    except Exception as e:
        logger.error(f"Error retraining models: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrain models'
        })

@app.route('/api/force-retrain', methods=['GET'])
def force_retrain():
    """Force retrain models and return detailed status"""
    try:
        logger.info("Force retraining advanced models...")
        fixie_bot.historical_data = None
        fixie_bot.is_trained = False
        success = fixie_bot.train_models()
        
        status_info = {
            'success': success,
            'message': 'Advanced models force retrained successfully' if success else 'Failed to retrain models',
            'data_count': len(fixie_bot.historical_data) if fixie_bot.historical_data is not None else 0,
            'model_architecture': 'Ensemble (RF + GB + LR + SVM + NB)',
            'feature_engineering': 'Advanced text preprocessing + domain-specific features',
            'prediction_method': 'Ensemble voting with confidence boosting'
        }
        
        return jsonify(status_info)
    except Exception as e:
        logger.error(f"Error force retraining models: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrain models'
        })

if __name__ == '__main__':
    # Initialize advanced models on startup
    logger.info("Starting Advanced FixieBot...")
    if fixie_bot.load_data():
        logger.info("Data loaded successfully with advanced preprocessing")
        if fixie_bot.train_models():
            logger.info("Advanced ensemble models trained successfully")
        else:
            logger.warning("Failed to train advanced models on startup")
    else:
        logger.error("Failed to load data on startup")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 
