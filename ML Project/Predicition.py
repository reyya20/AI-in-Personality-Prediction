import pandas as pd
import joblib
import re

# Define file paths
model_path = r'C:\Users\User\Desktop\Paper work\handwriting_model.joblib'
scaler_path = r'C:\Users\User\Desktop\Paper work\scaler.joblib'
file_path_new_data = r'C:\Users\User\Desktop\Paper work\new_dataset.txt'

# Load the saved model and scaler
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    raise

# Load and preprocess new data
try:
    # Try loading with different delimiters
    new_data = pd.read_csv(file_path_new_data, delimiter=',')  # Change delimiter if needed
    print("New data loaded successfully.")
except Exception as e:
    print(f"Error loading new data: {e}")
    raise

# Print column names for debugging
print("Columns in new data:", new_data.columns)

# Strip extra spaces from column names if needed
new_data.columns = new_data.columns.str.strip()

# Verify data after loading
print(new_data.head())

# Check for missing values
if new_data.isnull().sum().any():
    print("Warning: Missing values found in the new data.")
    new_data = new_data.dropna()  # Drop rows with missing values

# Update column names as needed based on the output
# For example, if column names are different, replace them accordingly
try:
    X_new = new_data[['Size', 'Slant', 'Avg Distance Between Characters']].copy()
except KeyError as e:
    print(f"Column error: {e}")
    raise

# Convert 'Slant' column from string representation to numerical values
def convert_slant(slant_str):
    try:
        match = re.search(r'\d+\.?\d*', str(slant_str))
        if match:
            return float(match.group(0))
        else:
            return None
    except Exception as e:
        print(f"Error converting slant value: {e}")
        return None

X_new['Slant'] = X_new['Slant'].apply(lambda x: convert_slant(x))

# Drop rows with NaN values after conversion
X_new = X_new.dropna()

# Normalize the new data using the loaded scaler
try:
    X_new_scaled = scaler.transform(X_new)
    print("Data normalized successfully.")
except Exception as e:
    print(f"Error normalizing data: {e}")
    raise

# Predict using the loaded model
try:
    predictions = model.predict(X_new_scaled)
    print("Predictions made successfully.")
    print(predictions)
except Exception as e:
    print(f"Error making predictions: {e}")
    raise
