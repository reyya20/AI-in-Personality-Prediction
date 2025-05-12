import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Define the conversion function if required
def convert_slant(slant_value):
    try:
        return float(slant_value)
    except ValueError:
        return 0.0

# Load the dataset
file_path = r'C:\Users\User\Desktop\Paper work\handwriting_features_all_traits.txt'
data = pd.read_csv(file_path)

# Select and preprocess the data
X = data[['Size', 'Slant', 'Avg Distance Between Characters']].copy()

# Convert 'Slant' column to float
X['Slant'] = X['Slant'].apply(lambda x: convert_slant(x))

# Drop rows with NaN values
X = X.dropna()

# Initialize the scaler and fit on the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for later use
scaler_path = r'C:\Users\User\Desktop\scaler.joblib'
joblib.dump(scaler, scaler_path)

# You can now use X_scaled for training or other purposes
print(X_scaled)
