import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib


def create_synthetic_dataset():
    data = {
        'Fan-In': np.random.randint(1, 20, 1000),
        'Fan-Out': np.random.randint(1, 10, 1000),
        'Signal Type': np.random.choice(['Control', 'Data'], 1000),
        'Gate Count': np.random.randint(5, 50, 1000),
        'Clock Frequency (MHz)': np.random.randint(50, 500, 1000),
        'Combinational Depth': np.random.randint(1, 20, 1000)
    }
    return pd.DataFrame(data)

# Step 2: Feature Engineering
def preprocess_data(df):
    # Encode categorical variables
    label_encoder = LabelEncoder()
    df['Signal Type'] = label_encoder.fit_transform(df['Signal Type'])
    df['Signal Type'] = df['Signal Type'].astype('category')  # Convert to categorical type

    # Normalize numerical features
    scaler = StandardScaler()
    numerical_features = ['Fan-In', 'Fan-Out', 'Gate Count', 'Clock Frequency (MHz)']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Separate features and target variable
    X = df.drop('Combinational Depth', axis=1)
    y = df['Combinational Depth']
    return X, y, label_encoder, scaler

# Step 3: Train-Test Split
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the XGBoost model
def train_model(X_train, y_train):
    # Enable categorical feature support
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        enable_categorical=True  # Enable categorical feature support
    )
    model.fit(X_train, y_train)
    return model

# Step 5: Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    return y_pred

# Step 6: Save the model and preprocessing objects
def save_model(model, label_encoder, scaler, filename):
    joblib.dump({'model': model, 'label_encoder': label_encoder, 'scaler': scaler}, filename)
    print(f"Model and preprocessing objects saved to {filename}")

# Step 7: Predict combinational depth for a new signal
def predict_depth(model, label_encoder, scaler, new_signal):
    new_signal_df = pd.DataFrame([new_signal])
    try:
        new_signal_df['Signal Type'] = label_encoder.transform(new_signal_df['Signal Type'])
        new_signal_df['Signal Type'] = new_signal_df['Signal Type'].astype('category')  # Convert to categorical type
    except ValueError as e:
        print(f"Error: {e}. Ensure the 'Signal Type' is one of {label_encoder.classes_}")
        return None

    numerical_features = ['Fan-In', 'Fan-Out', 'Gate Count', 'Clock Frequency (MHz)']
    new_signal_df[numerical_features] = scaler.transform(new_signal_df[numerical_features])
    predicted_depth = model.predict(new_signal_df)
    print(f"Predicted Combinational Depth: {predicted_depth[0]}")
    return predicted_depth[0]

# Main function to run the entire pipeline
def main():
    # Step 1: Create synthetic dataset
    df = create_synthetic_dataset()
    print("Synthetic dataset created.")

    # Step 2: Preprocess data
    X, y, label_encoder, scaler = preprocess_data(df)
    print("Data preprocessing completed.")

    # Step 3: Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("Data split into training and testing sets.")

    # Step 4: Train the model
    model = train_model(X_train, y_train)
    print("Model training completed.")

    # Step 5: Evaluate the model
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    save_model(model, label_encoder, scaler, 'combinational_depth_predictor.pkl')

    new_signal = {
        'Fan-In': 10,
        'Fan-Out': 4,
        'Signal Type': 'Control',  # This will be encoded
        'Gate Count': 20,
        'Clock Frequency (MHz)': 150
    }
    print("Predicting combinational depth for a new signal...")
    predict_depth(model, label_encoder, scaler, new_signal)

if __name__ == "__main__":
    main()