import re
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Define convert_slant function
def convert_slant(slant_str):
    match = re.search(r'\d+\.?\d*', slant_str)
    if match:
        return float(match.group(0))
    else:
        return None

# Load the saved scaler
scaler_path = r'C:\Users\User\Desktop\Paper work\scaler.joblib'
scaler = joblib.load(scaler_path)

# Load your model
model_path = r'C:\Users\User\Desktop\Paper work\handwriting_model.joblib'
model = joblib.load(model_path)

# Load and preprocess new data
file_path_new_data = r'C:\Users\User\Desktop\Paper work\new_dataset.txt'
new_data = pd.read_csv(file_path_new_data)

# Preprocess the new data
X_new = new_data[['Size', 'Slant', 'Avg Distance Between Characters']].copy()
X_new['Slant'] = X_new['Slant'].apply(lambda x: convert_slant(str(x)))
X_new = X_new.dropna()

# Normalize the new data using the loaded scaler
X_new_scaled = scaler.transform(X_new)

# Get predicted probabilities
predicted_probs = model.predict_proba(X_new_scaled)

# Get the class names (traits) from the model
class_names = model.classes_

# Create a DataFrame to display the probabilities
prob_df = pd.DataFrame(predicted_probs, columns=class_names)
prob_df['Image Name'] = new_data['Image Name']

# Merge the probabilities with the original data
result = pd.merge(new_data[['Image Name', 'Trait']], prob_df, on='Image Name')

# Display the available columns for debugging
print(result.columns)

# Display the results
print(result[['Image Name', 'Trait'] + list(class_names)])

# Plotting the predicted probabilities
for trait in class_names:
    plt.figure(figsize=(10, 6))
    plt.bar(result['Image Name'], result[trait], color='skyblue')
    plt.xlabel('Image Name')
    plt.ylabel('Probability')
    plt.title(f'Predicted Probability of {trait}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
