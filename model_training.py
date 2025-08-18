import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import warnings
warnings.filterwarnings('ignore')

class CropRecommendationModel:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = None
        self.target_encoder = LabelEncoder()

    def preprocess_data(self, df):
        """Preprocess the dataset for LSTM model"""
        # Handle missing values
        df = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['State', 'District', 'Soil_Type', 'pH_Classification',
                              'Soil_Health', 'Salinity_Level', 'Cropping_Season',
                              'Crop_Category', 'Irrigation_Type']
        
        # Note: Pest_Disease_Incidence is numerical in the dataset, not categorical
        
        for col in categorical_columns:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    try:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        # Handle unseen labels by mapping to the most common class
                        unseen_mask = ~df[col].astype(str).isin(self.label_encoders[col].classes_)
                        if unseen_mask.any():
                            # Replace unseen values with the most common class
                            most_common_class = self.label_encoders[col].classes_[0]
                            df.loc[unseen_mask, col] = most_common_class
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))

        # Handle Suitable_Months (convert string to numeric representation)
        if 'Suitable_Months' in df.columns:
            df['Suitable_Months_Count'] = df['Suitable_Months'].str.count(',') + 1
            df = df.drop('Suitable_Months', axis=1)

        # Remove non-numeric and target columns
        exclude_columns = ['Crop', 'Crop_Variety', 'Data_Collection_Date']
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove excluded columns from numeric_columns
        for col in exclude_columns:
            if col in numeric_columns:
                numeric_columns.remove(col)

        self.feature_columns = numeric_columns
        return df[numeric_columns]

    def create_sequences(self, X, y, sequence_length=10):
        """Create sequences for LSTM input"""
        X_seq, y_seq = [], []
        
        # If we don't have enough data for sequences, pad with repeats
        if len(X) < sequence_length:
            # Repeat the data to create minimum sequence length
            repeat_factor = (sequence_length // len(X)) + 1
            X_repeated = pd.concat([X] * repeat_factor, ignore_index=True)[:sequence_length]
            y_repeated = pd.concat([y] * repeat_factor, ignore_index=True)[:sequence_length]
            X_seq.append(X_repeated.values)
            y_seq.append(y_repeated.iloc[-1])
        else:
            # Normal sequence creation
            for i in range(len(X) - sequence_length + 1):
                X_seq.append(X.iloc[i:(i + sequence_length)].values)
                y_seq.append(y.iloc[i + sequence_length - 1])

        return np.array(X_seq), np.array(y_seq)

    def build_model(self, input_shape, num_classes):
        """Build bidirectional stacked LSTM model"""
        model = Sequential([
            # First bidirectional LSTM layer
            Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
                         input_shape=input_shape),
            BatchNormalization(),
            
            # Second bidirectional LSTM layer
            Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
            BatchNormalization(),
            
            # Third bidirectional LSTM layer
            Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2)),
            BatchNormalization(),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, df, sequence_length=10):
        """Train the model"""
        # Encode target variable
        y = self.target_encoder.fit_transform(df['Crop'])
        
        # Preprocess features
        X = self.preprocess_data(df)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, pd.Series(y), sequence_length)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
        )

        # Build model
        self.model = self.build_model(
            (sequence_length, len(self.feature_columns)),
            len(self.target_encoder.classes_)
        )

        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
        ]

        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=10,  # More epochs for better learning
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate model
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred_classes)
        print(f"Model Accuracy: {accuracy:.4f}")

        return history

    def predict(self, input_data, sequence_length=10):
        """Make predictions"""
        try:
            # Preprocess input
            X_processed = self.preprocess_data(input_data)
            
            # Ensure feature order matches training
            if hasattr(self, 'feature_columns') and self.feature_columns is not None:
                X_processed = X_processed[self.feature_columns]
            
            X_scaled = self.scaler.transform(X_processed)

            # Create sequence (repeat data if less than sequence_length)
            if len(X_scaled) < sequence_length:
                X_scaled = np.tile(X_scaled, (sequence_length // len(X_scaled) + 1, 1))[:sequence_length]
            
            X_seq = X_scaled[-sequence_length:].reshape(1, sequence_length, -1)

            # Predict
            predictions = self.model.predict(X_seq, verbose=0)
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

    def save_model(self, model_path="models/crop_model.h5", encoders_path="models/encoders.pkl", scaler_path="models/scaler.pkl"):
        """Save model and preprocessors"""
        self.model.save(model_path)
        joblib.dump(self.label_encoders, encoders_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.target_encoder, "models/target_encoder.pkl")
        joblib.dump(self.feature_columns, "models/feature_columns.pkl")

    def load_model(self, model_path="models/crop_model.h5", encoders_path="models/encoders.pkl", scaler_path="models/scaler.pkl"):
        """Load model and preprocessors"""
        self.model = tf.keras.models.load_model(model_path)
        self.label_encoders = joblib.load(encoders_path)
        self.scaler = joblib.load(scaler_path)
        self.target_encoder = joblib.load("models/target_encoder.pkl")
        self.feature_columns = joblib.load("models/feature_columns.pkl")

def train_model():
    """Main training function"""
    # Load data
    df = pd.read_csv('data/nilamdata.csv')
    
    # Initialize and train model
    model = CropRecommendationModel()
    history = model.train(df)
    
    # Save model
    model.save_model()
    print("Model training completed and saved!")
    
    return model

if __name__ == "__main__":
    train_model()
