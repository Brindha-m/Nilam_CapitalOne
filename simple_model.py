import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

class SimpleCropRecommendationModel:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.feature_columns = None
        self.target_encoder = LabelEncoder()

    def preprocess_data(self, df):
        """Preprocess the dataset for Random Forest model"""
        df = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['State', 'District', 'Soil_Type', 'pH_Classification',
                              'Soil_Health', 'Salinity_Level', 'Cropping_Season',
                              'Crop_Category', 'Irrigation_Type']
        
        for col in categorical_columns:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    try:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        # Handle unseen labels
                        unseen_mask = ~df[col].astype(str).isin(self.label_encoders[col].classes_)
                        if unseen_mask.any():
                            most_common_class = self.label_encoders[col].classes_[0]
                            df.loc[unseen_mask, col] = most_common_class
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))

        # Handle Suitable_Months
        if 'Suitable_Months' in df.columns:
            df['Suitable_Months_Count'] = df['Suitable_Months'].str.count(',') + 1
            df = df.drop('Suitable_Months', axis=1)

        # Remove non-numeric and target columns
        exclude_columns = ['Crop', 'Crop_Variety', 'Data_Collection_Date']
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in exclude_columns:
            if col in numeric_columns:
                numeric_columns.remove(col)

        self.feature_columns = numeric_columns
        return df[numeric_columns]

    def train(self, df):
        """Train the model"""
        print("🔄 Training Simple Crop Recommendation Model...")
        
        # Encode target variable
        y = self.target_encoder.fit_transform(df['Crop'])
        print(f"📊 Number of crop classes: {len(self.target_encoder.classes_)}")
        
        # Preprocess features
        X = self.preprocess_data(df)
        print(f"🔧 Number of features: {len(self.feature_columns)}")
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ Model Accuracy: {accuracy:.4f}")

        # Print classification report
        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.target_encoder.classes_))

        return accuracy

    def predict(self, input_data):
        """Make predictions"""
        try:
            # Preprocess input
            X_processed = self.preprocess_data(input_data)
            
            # Ensure feature order matches training and add missing features
            if hasattr(self, 'feature_columns') and self.feature_columns is not None:
                # Add missing columns with default values
                for col in self.feature_columns:
                    if col not in X_processed.columns:
                        X_processed[col] = 0.0  # Default value for missing features
                
                # Reorder columns to match training order
                X_processed = X_processed[self.feature_columns]
            
            X_scaled = self.scaler.transform(X_processed)

            # Predict probabilities
            predictions = self.model.predict_proba(X_scaled)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions)
            
            crop_name = self.target_encoder.inverse_transform([predicted_class])[0]

            # Get all predictions with crop names
            all_predictions_with_names = []
            for i, prob in enumerate(predictions[0]):
                crop = self.target_encoder.inverse_transform([i])[0]
                all_predictions_with_names.append((crop, prob))
            
            # Sort by probability
            all_predictions_with_names.sort(key=lambda x: x[1], reverse=True)

            return crop_name, confidence, all_predictions_with_names

        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None, None, None

    def save_model(self, model_path="models/simple_crop_model.pkl", encoders_path="models/simple_encoders.pkl", scaler_path="models/simple_scaler.pkl"):
        """Save model and preprocessors"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoders, encoders_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.target_encoder, "models/simple_target_encoder.pkl")
        joblib.dump(self.feature_columns, "models/simple_feature_columns.pkl")

    def load_model(self, model_path="models/simple_crop_model.pkl", encoders_path="models/simple_encoders.pkl", scaler_path="models/simple_scaler.pkl"):
        """Load model and preprocessors"""
        self.model = joblib.load(model_path)
        self.label_encoders = joblib.load(encoders_path)
        self.scaler = joblib.load(scaler_path)
        self.target_encoder = joblib.load("models/simple_target_encoder.pkl")
        self.feature_columns = joblib.load("models/simple_feature_columns.pkl")

def train_simple_model():
    """Main training function"""
    # Load data
    df = pd.read_csv('data/nilamdata.csv')
    print(f"📊 Dataset loaded with {len(df)} samples")
    
    # Initialize and train model
    model = SimpleCropRecommendationModel()
    accuracy = model.train(df)
    
    # Save model
    model.save_model()
    print("💾 Model training completed and saved!")
    
    return model

if __name__ == "__main__":
    train_simple_model()
