import pandas as pd
import numpy as np
import pickle

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Data Encoding and Scaling
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer


# Pipeline
from sklearn.pipeline import Pipeline

# Clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Flask
from flask import Flask, request, jsonify, render_template


''' This Function is to fill all missing values '''
def fill_missing_values(df):
    df = df.copy()
    fill_values = {
        'is_premium': 0,
        'subscription_status': 1,
        'company_size': 'More than 1000',
        'total_sub': 0,
        'is_unlimited': 1,
        'subscription_amount_in_dollars': 0,
        'number_of_subscriptions': 0,
        'number_of_invitations': 0,
        'job_posted': 0,
        'number_of_kits': 0,
        'number_of_recorded_interviews': 0,
        'number_of_live_interviews': 0,
        'days_since_last_login': 'More than 1 Year'
    }
    return df.fillna(fill_values)

def ordinal_encoder(df):
    # Define the ordinal column and the ordering
    ordinal_col = ['company_size', 'days_since_last_login']
    company_size_order = ['1-25', '26-100', '101-500', '500-1000', 'More than 1000']
    login_days_order = ['Less than 1 Week', '1-4 Weeks', '1-3 Months', '3-6 Months', '6-12 Months', 'More than 1 Year']

    total_order = [company_size_order, login_days_order]
    # Initialize OrdinalEncoder with specified categories
    ordinal = OrdinalEncoder(categories=total_order)
    
    # Fit and transform the data (make sure input is 2D for encoding)
    encoded = ordinal.fit_transform(df[ordinal_col].astype(str))

    # Shifting encoding to start from 1
    encoded += 1
    
    # Convert the encoded result to a DataFrame with the appropriate column name
    encoded_df = pd.DataFrame(encoded, columns=[f'{col}_ord' for col in ordinal_col], index=df.index)

    # Drop the original column
    df.drop(columns=ordinal_col, inplace=True)

    # Concatenate the encoded column to the original dataframe
    df = pd.concat([df, encoded_df], axis=1)
    
    return df

log_cols = [
    'total_sub',
    'subscription_amount_in_dollars',
    'number_of_subscriptions',
    'number_of_invitations',
    'job_posted',
    'number_of_kits',
    'number_of_recorded_interviews',
    'number_of_live_interviews',
    'days_since_last_login'
]

def log_transform(df):
    df = df.copy()
    for col in log_cols:
        if col in df.columns:
            # fill NaNs
            df[col] = df[col].fillna(0)
            # if a number is less than zero, turn it into zero;
            df[col] = df[col].clip(lower=0)
            # safe log1p
            df[col] = np.log1p(df[col])
            
    return df

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f) 

MODEL_PATH = "ClientLoyaltyRecommender_model.pth"
X_TENSOR_PATH = "X_tensor.pkl"

X_tensor = load_pickle(X_TENSOR_PATH)

df = pd.read_csv('experiment_data1.csv')
df.drop('jobma_catcher_id', axis=1, inplace=True)
compare_df = pd.read_csv('experiment_data1.csv')

class AutoEncoder(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # bottleneck
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_shape)
        )

    def forward(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
input_shape = df.shape[1]    
model = AutoEncoder(input_shape)

# Load model
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

with torch.no_grad():
    encoder, _ = model(X_tensor)

latent_np = encoder.cpu().numpy()

# Load Pipeline
with open("pipeline.pkl", "rb") as pipeline_file:
    pipeline = pickle.load(pipeline_file)

def recommend(user_input, model, latent_embeddings, compare_df, pipeline, top_k=5):
    # Transform the user input using the pipeline
    user_df = pd.DataFrame([user_input])
    user_input_transformed = pipeline.transform(user_df)
    user_input_tensor = torch.tensor(user_input_transformed, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        # Get the user embedding from the model
        user_embedding, _ = model(user_input_tensor)

        # Normalize the user embedding
        user_embedding = F.normalize(user_embedding, p=2, dim=1)

        # Convert latent embeddings to tensor and normalize
        latent_embeddings_tensor = torch.tensor(latent_embeddings, dtype=torch.float32)
        latent_embeddings_tensor = F.normalize(latent_embeddings_tensor, p=2, dim=1)

        # Calculate cosine similarity
        similarities = F.cosine_similarity(user_embedding, latent_embeddings_tensor, dim=1)

        # Get top K most similar indices
        top_indices = similarities.topk(top_k).indices.cpu().numpy()

        # Get the top K recommendations from the compare_df
        recommended = compare_df.iloc[top_indices].copy()
        recommended['similarity'] = similarities[top_indices].cpu().numpy()

    return recommended


# Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('intro.html')

@app.route('/form')
def form_page():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON. Set 'Content-Type: application/json'"}), 415

    data = request.get_json()
    try:
        recommended_df = recommend(data, model, latent_embeddings=latent_np, compare_df=compare_df, pipeline=pipeline, top_k=5)
        recommendations = recommended_df.to_dict(orient='records')

        return jsonify({"recommendations": recommendations})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)