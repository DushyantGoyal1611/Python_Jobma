import os
import pandas as pd
import numpy as np
import warnings
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Data Encoding and Scaling
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Pipeline
from sklearn.pipeline import Pipeline
import pickle


# Code Starts from here ------------------------------------------------------------------------------------------------------------------------------------

# Ignores the Warnings
warnings.filterwarnings('ignore')

# Loading .env file into my python code
load_dotenv()

login_order = {
        'Less than 1 Week':0,
        '1-4 Weeks':1,
        '1-3 Months':2,
        '3-6 Months':3,
        '6-12 Months':4,
        'More than 1 Year':5
    }

def create_connection():
    print('creating connection with DB')
    user = os.getenv("DB_USER")
    raw_password = os.getenv("DB_PASSWORD")
    password = quote_plus(raw_password)
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    db = os.getenv("DB_NAME")

    # Credentials of mySQL connection
    connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"
    engine = create_engine(connection_string)
    print('connection created successfully')
    return engine

def create_df(engine,catcher_only=False):
    print('creating DFs=============')
    if catcher_only:
        print('Catcher DF is requested')
        catcher_df = pd.read_sql('Select jobma_catcher_id, is_premium, jobma_catcher_parent, jobma_verified, subscription_status, company_size FROM jobma_catcher', con=engine) 
        return catcher_df
    print("Wallet DF")
    wallet_df = pd.read_sql('Select catcher_id AS jobma_catcher_id, is_unlimited FROM wallet', con=engine)
    print("Subscription DF")
    subscription_df = pd.read_sql('Select catcher_id AS jobma_catcher_id, currency, subscription_amount FROM subscription_history', con=engine)
    print("Invitation DF")
    invitation_df = pd.read_sql('Select jobma_catcher_id, jobma_interview_mode, jobma_interview_status FROM jobma_pitcher_invitations', con=engine)
    print("Job posting DF")
    job_posting_df = pd.read_sql('Select jobma_catcher_id FROM jobma_employer_job_posting', con=engine)
    print("kit DF")
    kit_df = pd.read_sql('Select catcher_id AS jobma_catcher_id FROM job_assessment_kit', con=engine)
    print('Login DF')
    login_df = pd.read_sql('Select jobma_role_id, jobma_user_id AS jobma_catcher_id, jobma_last_login FROM jobma_login',con=engine)
    # Closing the Connection
    engine.dispose()
    return wallet_df,subscription_df,invitation_df,job_posting_df,kit_df,login_df

# catcher_df
def fetching_catcher_df(catcher_df):
    print("Processing catcher DF")
    catcher_df['jobma_verified'] = catcher_df['jobma_verified'].replace({'0':0, '1':1})
    catcher_df.drop(catcher_df[catcher_df['is_premium'] == ''].index, inplace=True)
    catcher_df['is_premium'] = catcher_df['is_premium'].replace({'0':0, '1':1})
    catcher_df['subscription_status'] = catcher_df['subscription_status'].replace({'0':0, '1':1, '2':0})

    return catcher_df

# wallet_df
def fetching_wallet_df(wallet_df):
    print("Processing wallet DF")
    wallet_df.rename(columns={'catcher_id': 'jobma_catcher_id'}, inplace=True)
    wallet_df['is_unlimited'] = wallet_df['is_unlimited'].replace({'0':0, '1':1})
    wallet_df = wallet_df[wallet_df['is_unlimited'] != '']
    wallet_df.drop_duplicates(inplace=True)

    return wallet_df

# subscription_df
def fetching_subscription_df(subscription_df):
    print("Processing subscription DF")
    subscription_df.rename(columns={'catcher_id': 'jobma_catcher_id'}, inplace=True)
    subscription_df.loc[subscription_df['currency'] == '1', 'subscription_amount'] /= 85.23
    subscription_df = subscription_df.groupby('jobma_catcher_id').agg(
        subscription_amount_in_dollars = ('subscription_amount', 'sum'),
        number_of_subscriptions = ('subscription_amount', 'count'),
    ).reset_index()
    subscription_df.drop_duplicates(inplace=True)
    return subscription_df

# login_df
def fetching_login_df(login_df):
    print("Processing login DF")
    login_df = login_df[login_df['jobma_role_id'] == 3].copy()
    login_df.rename(columns={'jobma_user_id': 'jobma_catcher_id'}, inplace=True)

    # Calculating Number of Gaps between last login and today
    login_df['jobma_last_login'] = pd.to_datetime(login_df['jobma_last_login'], errors='coerce')
    login_df['activity_duration'] = (pd.Timestamp('today') - login_df['jobma_last_login']).dt.days
    login_df['activity_duration'].fillna(5, inplace=True)
    login_df['activity_duration'] = login_df['activity_duration'].astype(int)

    # Binning
    bins = [-1,7,30,90,180,365,float('inf')]
    labels = ['Less than 1 Week', '1-4 Weeks', '1-3 Months', '3-6 Months', '6-12 Months', 'More than 1 Year']
    login_df['activity_duration'] = pd.cut(login_df['activity_duration'], bins=bins, labels=labels)
    login_df = login_df[['jobma_catcher_id', 'activity_duration']]

    return login_df

def fetching_features(invitation_df, job_posting_df, kit_df):
    print("Fetching features")
    for df in [invitation_df, job_posting_df, kit_df]:
        if 'catcher_id' in df.columns:
            df.rename(columns={'catcher_id': 'jobma_catcher_id'}, inplace=True)

    job_posting_df['job_posted'] = job_posting_df['jobma_catcher_id'].map(job_posting_df['jobma_catcher_id'].value_counts())
    kit_df['number_of_kits'] = kit_df['jobma_catcher_id'].map(kit_df['jobma_catcher_id'].value_counts())

    invitation_df = invitation_df[invitation_df['jobma_interview_status'] == '2']
    invitation_df['number_of_invitations'] = invitation_df['jobma_catcher_id'].map(invitation_df['jobma_catcher_id'].value_counts())
    invitation_df.drop('jobma_interview_status', axis=1, inplace=True)
    invitation_df['interview_completed'] = invitation_df['jobma_catcher_id'].map(invitation_df['jobma_catcher_id'].value_counts())
    invitation_df = invitation_df[invitation_df['jobma_interview_mode'].isin(['1', '2'])].copy()
    interview_counts = invitation_df.groupby(['jobma_catcher_id', 'jobma_interview_mode']).size().unstack(fill_value=0)
    interview_counts = interview_counts.rename(columns={'1': 'number_of_recorded_interviews', '2': 'number_of_live_interviews'})
    invitation_df = invitation_df.merge(interview_counts, on='jobma_catcher_id', how='left')
    
    invitation_df.drop('jobma_interview_mode', axis=1, inplace=True)
    invitation_df = invitation_df.drop_duplicates()

    job_posting_df = job_posting_df[['jobma_catcher_id', 'job_posted']].drop_duplicates()
    kit_df = kit_df[['jobma_catcher_id', 'number_of_kits']].drop_duplicates()

    return invitation_df, job_posting_df, kit_df

# Merging all the DataFrames
def merging_df(catcher_df, wallet_df, subscription_df, invitation_df, job_posting_df, kit_df, login_df):
    print("Merging DFs")
    final_df = catcher_df.copy()

    # Left join each table one by one
    final_df = final_df.merge(wallet_df, on='jobma_catcher_id', how='left')
    final_df = final_df.merge(subscription_df, on='jobma_catcher_id', how='left')
    final_df = final_df.merge(invitation_df, on='jobma_catcher_id', how='left')
    final_df = final_df.merge(job_posting_df, on='jobma_catcher_id', how='left')
    final_df = final_df.merge(kit_df, on='jobma_catcher_id', how='left')
    final_df = final_df.merge(login_df, on='jobma_catcher_id', how='left')
    final_df.drop_duplicates(inplace=True)

    # For Total Sub
    sub_counts = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent').size()
    final_df['total_sub'] = final_df['jobma_catcher_id'].map(sub_counts).fillna(0).astype(int)

    # For Kits
    sub_kits_sum = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')['number_of_kits'].sum()
    kits_mapped = final_df['jobma_catcher_id'].map(sub_kits_sum).fillna(0)
    final_df['number_of_kits'] = final_df['number_of_kits'].fillna(0) + kits_mapped
    final_df['number_of_kits'] = final_df['number_of_kits'].astype(int)
    
    # For Invitations
    sub_invitations_sum = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')['number_of_invitations'].sum()
    invitations_mapped = final_df['jobma_catcher_id'].map(sub_invitations_sum).fillna(0)
    final_df['number_of_invitations'] = final_df['number_of_invitations'].fillna(0) + invitations_mapped
    final_df['number_of_invitations'] = final_df['number_of_invitations'].astype(int)
    
    # For Job Posted
    sub_job_posted_sum = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')['job_posted'].sum()
    job_posted_mapped = final_df['jobma_catcher_id'].map(sub_job_posted_sum).fillna(0)
    final_df['job_posted'] = final_df['job_posted'].fillna(0) + job_posted_mapped
    final_df['job_posted'] = final_df['job_posted'].astype(int)

    # For Recorded Interviews
    sub_recorded_sum = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')['number_of_recorded_interviews'].sum()
    recorded_mapped = final_df['jobma_catcher_id'].map(sub_recorded_sum).fillna(0)
    final_df['number_of_recorded_interviews'] = final_df['number_of_recorded_interviews'].fillna(0) + recorded_mapped
    final_df['number_of_recorded_interviews'] = final_df['number_of_recorded_interviews'].astype(int)

    # For Live Interviews
    sub_live_sum = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')['number_of_live_interviews'].sum()
    live_mapped = final_df['jobma_catcher_id'].map(sub_live_sum).fillna(0)
    final_df['number_of_live_interviews'] = final_df['number_of_live_interviews'].fillna(0) + live_mapped
    final_df['number_of_live_interviews'] = final_df['number_of_live_interviews'].astype(int)

    # For Interview Completed
    sub_to_parent_sum = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')['interview_completed'].sum()
    final_df.loc[final_df['jobma_catcher_id'].isin(sub_to_parent_sum.index), 'interview_completed'] += final_df['jobma_catcher_id'].map(sub_to_parent_sum).fillna(0).astype(int)

    # For Minimum Login Days
    # login_order = {
    #     'Less than 1 Week':0,
    #     '1-4 Weeks':1,
    #     '1-3 Months':2,
    #     '3-6 Months':3,
    #     '6-12 Months':4,
    #     'More than 1 Year':5
    # }

    final_df['activity_duration'] = final_df['activity_duration'].map(login_order).fillna(5).astype(int)
    sub_min_login = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')['activity_duration'].min()
    final_df.loc[final_df['jobma_catcher_id'].isin(sub_min_login.index), 'activity_duration'] = final_df.loc[final_df['jobma_catcher_id'].isin(sub_min_login.index), 'jobma_catcher_id'].map(sub_min_login)

    verified_df = final_df[final_df['jobma_verified'] == 1].copy()
    df = verified_df[verified_df['jobma_catcher_parent'] == 0].copy()
    df.drop(['jobma_catcher_parent', 'jobma_verified'], axis=1, inplace=True)
    
    compare_df = df.copy()
    df.drop('jobma_catcher_id', axis=1, inplace=True)

    print(f"Final merged df shape is {df.shape}")

    return df, compare_df

# This Function is to fill all missing values
def fill_missing_values(final_df):
    final_df = final_df.copy()

    # Step 1: Replace inf with NaN first
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Step 2: Fill NaNs
    fill_values = {
        'is_premium': 0,
        'subscription_status': 0,
        'company_size': '1-25',
        'is_unlimited': 0,
        'subscription_amount_in_dollars': 0,
        'number_of_subscriptions': 0,
        'number_of_invitations': 0,
        'interview_completed': 0,
        'number_of_recorded_interviews': 0,
        'number_of_live_interviews': 0,
        'job_posted': 0,
        'number_of_kits': 0,
        'activity_duration': 5,
        'total_sub': 0,
    }
    final_df.fillna(fill_values, inplace=True)

    # Step 3: Explicitly cast to int for the appropriate columns
    int_columns = [
        'is_premium',
        'subscription_status',
        'is_unlimited',
        'number_of_subscriptions',
        'jobma_interview_status',
        'number_of_invitations',
        'interview_completed',
        'number_of_recorded_interviews',
        'number_of_live_interviews',
        'job_posted',
        'number_of_kits',
        'activity_duration',
        'total_sub',
    ]
    for col in int_columns:
        if col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0).astype(int)

    return final_df


# Data Encoding 
def ordinal_encoder(df):
    ordinal_col = ['company_size']
    company_size_order = ['1-25', '26-100', '101-500', '500-1000', 'More than 1000']

    ordinal = OrdinalEncoder(categories=[company_size_order])

    encoded = ordinal.fit_transform(df[ordinal_col].astype(str))

    encoded_df = pd.DataFrame(encoded, columns=[f' {col}_ord' for col in ordinal_col], index=df.index)

    df.drop(columns=ordinal_col, inplace=True)

    df = pd.concat([df, encoded_df], axis=1)

    return df

# Log Transformation 
def log_transform(df):
    log_cols = [
        'subscription_amount_in_dollars',
        'number_of_subscriptions',
        'interview_completed',
        'number_of_invitations',
        'number_of_recorded_interviews',
        'number_of_live_interviews',
        'job_posted',
        'number_of_kits',
        'activity_duration'
        'total_sub'
    ]

    df = df.copy()
    for col in log_cols:
        if col in df.columns:
            # fill NaNs
            df[col] = df[col].fillna(0)
            # if a number is less than zero, turn it into zero
            df[col] = df[col].clip(lower=0)
            # safe log1p
            df[col] = np.log1p(df[col])

    return df

# Tensor Conversion
def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

class MergingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        engine = create_connection()
        wallet_df,subscription_df,invitation_df,job_posting_df,kit_df, login_df = create_df(engine)
        self.wallet_df = wallet_df
        self.subscription_df = subscription_df
        self.invitation_df = invitation_df
        self.job_posting_df = job_posting_df
        self.kit_df = kit_df
        self.login_df = login_df

    def fit(self, X, y=None):
        return self

    def transform(self, catcher_df):
        catcher = fetching_catcher_df(catcher_df)
        wallet = fetching_wallet_df(self.wallet_df)
        subscription = fetching_subscription_df(self.subscription_df)
        login = fetching_login_df(self.login_df)
        invitation, job_posting, kit = fetching_features(
            self.invitation_df,
            self.job_posting_df,
            self.kit_df,
        )
        final_df, compare_df = merging_df(catcher, wallet, subscription, invitation, job_posting, kit, login)
        self.compare_df_ = fill_missing_values(compare_df)
        self.compare_df_.to_csv("compare_df.csv", index=False)
        return final_df
    
class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = fill_missing_values(X)
        X = ordinal_encoder(X)
        X = log_transform(X)
        return X
    
# Model (AutoEncoder in this case)
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
    
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def train_and_save_model():
    print("Starting with model training")
    # Pipelines
    merge_pipeline = Pipeline([
        ('merge', MergingTransformer())
    ])

    # Step 2: Preprocess merged data
    preprocess_pipeline = Pipeline([
        ('preprocessing', PreprocessingTransformer()),
        ('scaler', StandardScaler()),
        ('to_tensor', FunctionTransformer(to_tensor))
    ])

    full_pipeline = Pipeline([
        ('merge_pipeline', merge_pipeline),
        ('preprocess_pipeline', preprocess_pipeline)
    ])

    X_tensor = full_pipeline.fit_transform(create_df(create_connection(),catcher_only=True))
    
    with open('full_pipeline.pkl', 'wb') as f:
        pickle.dump(full_pipeline, f)
    
    X_df = pd.DataFrame(X_tensor)
    X_data = CustomDataset(X_tensor)

    BATCH_SIZE = 16
    dataloader = DataLoader(X_data, batch_size=BATCH_SIZE, shuffle=True)

    # Initializing the Model 
    input_shape = X_df.shape[1]
    model_1 = AutoEncoder(input_shape)

    # Important Parameters 
    learning_rate = 0.001
    epochs = 50
    patience = 5
    delta = 1e-4
    best_loss = float('inf')
    epochs_no_improve = 0
    training_losses = []

    # Loss Function, Optimizers and LR Scheduler 
    loss_function = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model_1.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    for epoch in range(epochs):
        model_1.train()
        epoch_loss = 0

        for batch in dataloader:
            encoded, decoded = model_1(batch)
            loss = loss_function(decoded, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        training_losses.append(avg_loss)
        scheduler.step(avg_loss)

        print(f"Epoch: {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

        # Early Stopping
        if avg_loss < best_loss - delta:
            best_loss = avg_loss
            epochs_no_improve = 0

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Print current learning rate
        for param_group in optimizer.param_groups:
            print(f"Learning Rate: {param_group['lr']:.6f}")
    print("Saving Model and Embeddings")
    torch.save(model_1.state_dict(),'model.pth')
    encoder, _ = model_1(X_tensor)
    latent_np = encoder.cpu().detach().numpy()

    with open('latent_np.pkl', 'wb') as embeddings_file:
        pickle.dump(latent_np, embeddings_file)

    print("Model and Embeddings Saved")

def predict(user_input, n=5):
    with open('full_pipeline.pkl', 'rb') as f:
        full_pipeline = pickle.load(f)

    # Extract pipelines
    preprocess_pipeline = full_pipeline.named_steps['preprocess_pipeline']

    # access compare_df using .csv file instead of fetching it from database
    compare_df = pd.read_csv('compare_df.csv')

    # Transform user input only with preprocessing pipeline
    user_df = pd.DataFrame([user_input])
    transformed_input = preprocess_pipeline.transform(user_df)
    print(f'Transformed User Input Type is: {type(transformed_input)}')

    # Load trained model
    input_shape = transformed_input.shape[1]
    model_1 = AutoEncoder(input_shape)
    model_1.load_state_dict(torch.load('model.pth'))
    model_1.eval()

    # Load latent embeddings
    with open('latent_np.pkl', 'rb') as embeddings_file:
        embeddings = pickle.load(embeddings_file)
    print(f'Embedding Type is {type(embeddings)}')
    
    # Generate user embedding
    with torch.no_grad():
        user_embedding, _ = model_1(transformed_input)
        user_embedding = F.normalize(user_embedding, p=2, dim=1)
        print(f'User Embedding Type is: {type(user_embedding)}')

        latent_embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        latent_embeddings = F.normalize(latent_embeddings_tensor, p=2, dim=1)
        print(f'Latent Embedding Type is {type(latent_embeddings)}')

        # Compute cosine similarity and get top-N
        similarities = F.cosine_similarity(user_embedding, latent_embeddings, dim=1)
        top_indices = similarities.topk(n).indices.cpu().numpy()

    # Final Recommendation DataFrame
    recommended = compare_df.iloc[top_indices].copy()
    
    # To show the actual activity duration
    recommended['activity_duration'] = recommended['activity_duration'].replace({v:k for k,v in login_order.items()})

    # To show the Similarity Score
    recommended['similarity'] = similarities[top_indices].cpu().numpy()

    print(recommended)

# Expecting catcher_id to fetch all the details of user

def predict_using_catcher_id(catcher_id, n=5):
    with open('full_pipeline.pkl', 'rb') as f:
        full_pipeline = pickle.load(f)

    # Extract Pipelines
    preprocess_pipeline = full_pipeline.named_steps['preprocess_pipeline']
    compare_df = pd.read_csv('compare_df.csv')

    user_row = compare_df[compare_df['jobma_catcher_id'] == catcher_id]

    if user_row.empty:
        raise ValueError(f"No data found for catcher_id: {catcher_id}")

    user_input = user_row.drop(columns=['jobma_catcher_id']).iloc[0].to_dict()
    user_df = pd.DataFrame([user_input])
    transformed_input = preprocess_pipeline.transform(user_df)
    print(f'Transformed User Input Type is: {type(transformed_input)}')

    # Load trained model
    input_shape = transformed_input.shape[1]
    model_1 = AutoEncoder(input_shape)
    model_1.load_state_dict(torch.load('model.pth'))
    model_1.eval()

     # Load latent embeddings
    with open('latent_np.pkl', 'rb') as embeddings_file:
        embeddings = pickle.load(embeddings_file)
    print(f'Embedding Type is {type(embeddings)}')
    
    # Generate user embedding
    with torch.no_grad():
        user_embedding, _ = model_1(transformed_input)
        user_embedding = F.normalize(user_embedding, p=2, dim=1)
        print(f'User Embedding Type is: {type(user_embedding)}')

        latent_embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        latent_embeddings = F.normalize(latent_embeddings_tensor, p=2, dim=1)
        print(f'Latent Embedding Type is {type(latent_embeddings)}')

        # Compute cosine similarity and get top-N
        similarities = F.cosine_similarity(user_embedding, latent_embeddings, dim=1)
        top_indices = similarities.topk(n).indices.cpu().numpy()

    # Final Recommendation DataFrame
    recommended = compare_df.iloc[top_indices].copy()

    # To show the actual activity duration
    recommended['activity_duration'] = recommended['activity_duration'].replace({v:k for k,v in login_order.items()})

    # To show the Similarity Score
    recommended['similarity'] = similarities[top_indices].cpu().numpy()

    print(recommended)


user_pref_good = {'is_premium':0,
                 'subscription_status':0,
                 'company_size':'101-500', 
                 'is_unlimited':0,
                 'subscription_amount_in_dollars': 100.00,
                 'number_of_subscriptions':1,
                 'number_of_invitations':25,
                'interview_completed':10,
                'number_of_recorded_interviews':8,
                'number_of_live_interviews':5,
                'job_posted':4,
                'number_of_kits':7,
                'activity_duration':4,
                'total_sub':1,
            }

user_pref_test = {'is_premium':1,
                 'subscription_status':1,
                 'company_size':'1-25',
                 'is_unlimited':1,
                 'subscription_amount_in_dollars': 125.0,
                 'number_of_subscriptions':1,
                 'number_of_invitations':18,
                  'interview_completed':10,
                  'number_of_recorded_interviews':3,
                'number_of_live_interviews':1,
                 'job_posted':1,
                 'number_of_kits':1,
                'activity_duration':5,
                'total_sub':1
            }

# Starting training the model
# train_and_save_model()

# Making Predictions (Recommendations)

# Using full dictionary
predict(user_pref_good, 5)
predict(user_pref_test, 5)

# Using only catcher id
predict_using_catcher_id(6025, 5)
predict_using_catcher_id(6189, 5)