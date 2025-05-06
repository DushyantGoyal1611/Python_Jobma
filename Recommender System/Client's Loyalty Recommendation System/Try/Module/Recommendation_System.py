import os
import pandas as pd
import numpy as np
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

# Loading .env file into my python code
load_dotenv()

user = os.getenv("DB_USER")
raw_password = os.getenv("DB_PASSWORD")
password = quote_plus(raw_password)
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
db = os.getenv("DB_NAME")

# Credentials of mySQL connection
connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"
engine = create_engine(connection_string)

catcher_df = pd.read_sql('Select * FROM jobma_catcher', con=engine)  # Done  (about catcher's account)
wallet_df = pd.read_sql('Select * FROM wallet', con=engine)  # Done  ('subscription type, plan type')
subscription_df = pd.read_sql('Select * FROM subscription_history', con=engine)  # Done  (number and amount of subscription bought by the catcher)
invitation_df = pd.read_sql('Select * FROM jobma_pitcher_invitations', con=engine)  # Done   (number of invitations sent by the catcher)
job_posting_df = pd.read_sql('Select * FROM jobma_employer_job_posting', con=engine) # Done    (number of jobs posted by the catcher)
kit_df = pd.read_sql('Select * FROM job_assessment_kit', con=engine) # Done   (number of kits created by the catcher)
recorded_interview_df = pd.read_sql('Select * FROM jobma_interviews', con=engine) # Done (number of recorded interviews created by the catcher)
live_interview_df = pd.read_sql('Select * FROM jobma_interviews_online', con=engine) # Done (number of live interviews created by the catcher)
login_df = pd.read_sql('Select * FROM jobma_login',con=engine)  # Done (To Find number of days catcher didn't logged in)

# catcher_df
def fetching_catcher_df(catcher_df):
    catcher_df = catcher_df[['jobma_catcher_id', 'is_premium', 'jobma_catcher_parent', 'jobma_verified', 'subscription_status', 'company_size']]
    catcher_df['jobma_verified'] = catcher_df['jobma_verified'].replace({'0':0, '1':1})
    catcher_df.drop(catcher_df[catcher_df['is_premium'] == ''].index, inplace=True)
    catcher_df['is_premium'] = catcher_df['is_premium'].replace({'0':0, '1':1})
    catcher_df['subscription_status'] = catcher_df['subscription_status'].replace({'0':0, '1':1, '2':0})

    return catcher_df

# wallet_df
def fetching_wallet_df(wallet_df):
    wallet_df.rename(columns={'catcher_id': 'jobma_catcher_id'}, inplace=True)
    wallet_df = wallet_df[['jobma_catcher_id', 'is_unlimited']]
    wallet_df['is_unlimited'] = wallet_df['is_unlimited'].replace({'0':0, '1':1})
    wallet_df.drop(wallet_df[wallet_df['is_unlimited'] == ''].index, inplace=True)
    wallet_df.drop_duplicates(inplace=True)

    return wallet_df

# subscription_df
def fetching_subscription_df(subscription_df):
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
    login_df = login_df[login_df['jobma_role_id'] == 3].copy()
    login_df.rename(columns={'jobma_user_id': 'jobma_catcher_id'}, inplace=True)

    # Calculating Number of Gaps between last login and today
    login_df['jobma_last_login'] = pd.to_datetime(login_df['jobma_last_login'], errors='coerce')
    login_df['days_since_last_login'] = (pd.Timestamp('today') - login_df['jobma_last_login']).dt.days
    login_df['days_since_last_login'].fillna(9999, inplace=True)
    login_df['days_since_last_login'] = login_df['days_since_last_login'].astype(int)

    # Binning
    bins = [-1,7,30,90,180,365,float('inf')]
    labels = ['Less than 1 Week', '1-4 Weeks', '1-3 Months', '3-6 Months', '6-12 Months', 'More than 1 Year']
    login_df['days_since_last_login'] = pd.cut(login_df['days_since_last_login'], bins=bins, labels=labels)
    login_df = login_df[['jobma_catcher_id', 'days_since_last_login']]

    return login_df

# Fetching Features
def fetching_features(invitation_df, job_posting_df, kit_df, recorded_interview_df, live_interview_df):
    for df in [invitation_df, job_posting_df, kit_df, recorded_interview_df, live_interview_df]:
        if 'catcher_id' in df.columns:
            df.rename(columns={'catcher_id': 'jobma_catcher_id'}, inplace=True)

    invitation_df['number_of_invitations'] = invitation_df['jobma_catcher_id'].map(invitation_df['jobma_catcher_id'].value_counts()).astype(int)
    job_posting_df['job_posted'] = job_posting_df['jobma_catcher_id'].map(job_posting_df['jobma_catcher_id'].value_counts()).astype(int)
    kit_df['number_of_kits'] = kit_df['jobma_catcher_id'].map(kit_df['jobma_catcher_id'].value_counts()).astype(int)
    recorded_interview_df['number_of_recorded_interviews'] = recorded_interview_df['jobma_catcher_id'].map(recorded_interview_df['jobma_catcher_id'].value_counts()).astype(int)
    live_interview_df['number_of_live_interviews'] = live_interview_df['jobma_catcher_id'].map(live_interview_df['jobma_catcher_id'].value_counts()).astype(int)    

    invitation_df = invitation_df[['jobma_catcher_id', 'number_of_invitations']].drop_duplicates()
    job_posting_df = job_posting_df[['jobma_catcher_id', 'job_posted']].drop_duplicates()
    kit_df = kit_df[['jobma_catcher_id', 'number_of_kits']].drop_duplicates()
    recorded_interview_df = recorded_interview_df[['jobma_catcher_id', 'number_of_recorded_interviews']].drop_duplicates()
    live_interview_df = live_interview_df[['jobma_catcher_id', 'number_of_live_interviews']].drop_duplicates()

    return invitation_df, job_posting_df, kit_df, recorded_interview_df, live_interview_df

# Merging DataFrames
def merging_df(catcher_df, wallet_df, subscription_df, invitation_df, job_posting_df, kit_df, recorded_interview_df, live_interview_df, login_df):
    final_df = catcher_df.copy()

    # Left join each table one by one
    final_df = final_df.merge(wallet_df, on='jobma_catcher_id', how='left')
    final_df = final_df.merge(subscription_df, on='jobma_catcher_id', how='left')
    final_df = final_df.merge(invitation_df, on='jobma_catcher_id', how='left')
    final_df = final_df.merge(job_posting_df, on='jobma_catcher_id', how='left')
    final_df = final_df.merge(kit_df, on='jobma_catcher_id', how='left')
    final_df = final_df.merge(recorded_interview_df, on='jobma_catcher_id', how='left')
    final_df = final_df.merge(live_interview_df, on='jobma_catcher_id', how='left')
    final_df = final_df.merge(login_df, on='jobma_catcher_id', how='left')
    final_df.drop_duplicates(inplace=True)

    # For Total Sub
    sub_counts = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent').size()
    final_df['total_sub'] = final_df['jobma_catcher_id'].map(sub_counts).fillna(0).astype(int)

    # For Kits
    sub_kits_sum = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')['number_of_kits'].sum()
    final_df.loc[final_df['jobma_catcher_id'].isin(sub_kits_sum.index), 'number_of_kits'] += final_df['jobma_catcher_id'].map(sub_kits_sum)

    # For Invitations
    sub_invitations_sum = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')['number_of_invitations'].sum()
    final_df.loc[final_df['jobma_catcher_id'].isin(sub_kits_sum.index), 'number_of_invitations'] += final_df['jobma_catcher_id'].map(sub_invitations_sum)
    
    # For Job Posted
    sub_job_posted_sum = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')['job_posted'].sum()
    final_df.loc[final_df['jobma_catcher_id'].isin(sub_job_posted_sum.index), 'job_posted'] += final_df['jobma_catcher_id'].map(sub_job_posted_sum)

    # For Recorded Interviews
    sub_recorded_sum = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')['number_of_recorded_interviews'].sum()
    final_df.loc[final_df['jobma_catcher_id'].isin(sub_recorded_sum.index), 'number_of_recorded_interviews'] += final_df['jobma_catcher_id'].map(sub_recorded_sum)

    # For Live Interviews
    sub_live_sum = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')['number_of_live_interviews'].sum()
    final_df.loc[final_df['jobma_catcher_id'].isin(sub_live_sum.index), 'number_of_live_interviews'] += final_df['jobma_catcher_id'].map(sub_live_sum)

    # For Minimum Login Days
    login_order = {
        'Less than 1 Week':0,
        '1-4 Weeks':1,
        '1-3 Months':2,
        '3-6 Months':3,
        '6-12 Months':4,
        'More than 1 Year':5
    }

    final_df['days_since_last_login'] = final_df['days_since_last_login'].map(login_order).fillna(5).astype(int)
    sub_min_login = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')['days_since_last_login'].min()
    final_df.loc[final_df['jobma_catcher_id'].isin(sub_min_login.index), 'days_since_last_login'] = final_df.loc[final_df['jobma_catcher_id'].isin(sub_min_login.index), 'jobma_catcher_id'].map(sub_min_login)
    

    verified_df = final_df[final_df['jobma_verified'] == 1].copy()
    df = verified_df[verified_df['jobma_catcher_parent'] == 0].copy()
    df.drop('jobma_catcher_parent', axis=1, inplace=True)
    
    compare_df = df.copy()
    df.drop('jobma_catcher_id', axis=1, inplace=True)

    print(f"Final merged df shape is {df.shape}")

    return df, compare_df

# This Function is to fill all missing values
def fill_missing_values(final_df):
    final_df = final_df.copy()
    fill_values = {
        'is_premium': 0,
        'jobma_verified': 0,
        'subscription_status': 0,
        'company_size': 'More than 1000',
        'is_unlimited': 1,
        'subscription_amount_in_dollars': 0,
        'number_of_subscriptions': 0,
        'number_of_invitations': 0,
        'job_posted': 0,
        'number_of_kits': 0,
        'number_of_recorded_interviews': 0,
        'number_of_live_interviews': 0,
        'days_since_last_login': 'More than 1 Year',
        'total_sub': 0,
    }
    return final_df.fillna(fill_values)

# Data Encoding 
def ordinal_encoder(df):
    # ordinal_col = ['company_size', 'days_since_last_login']
    ordinal_col = ['company_size']
    company_size_order = ['1-25', '26-100', '101-500', '500-1000', 'More than 1000']
    # login_days_order = ['Less than 1 Week', '1-4 Weeks', '1-3 Months', '3-6 Months', '6-12 Months', 'More than 1 Year']

    # total_order = [company_size_order, login_days_order]
    ordinal = OrdinalEncoder(categories=[company_size_order])

    encoded = ordinal.fit_transform(df[ordinal_col].astype(str))
    # encoded += 1

    encoded_df = pd.DataFrame(encoded, columns=[f' {col}_ord' for col in ordinal_col], index=df.index)

    df.drop(columns=ordinal_col, inplace=True)

    df = pd.concat([df, encoded_df], axis=1)

    return df

# Log Transformation 
def log_transform(df):
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

class MergingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, wallet_df, subscription_df, login_df,
                 invitation_df, job_posting_df, kit_df, recorded_interview_df, live_interview_df):
        self.wallet_df = wallet_df
        self.subscription_df = subscription_df
        self.login_df = login_df
        self.invitation_df = invitation_df
        self.job_posting_df = job_posting_df
        self.kit_df = kit_df
        self.recorded_interview_df = recorded_interview_df
        self.live_interview_df = live_interview_df

    def fit(self, X, y=None):
        return self

    def transform(self, catcher_df):
        parents_df = fetching_catcher_df(catcher_df)
        wallet = fetching_wallet_df(self.wallet_df)
        subscription = fetching_subscription_df(self.subscription_df)
        login = fetching_login_df(self.login_df)
        invitation, job_posting, kit, recorded, live = fetching_features(
            self.invitation_df,
            self.job_posting_df,
            self.kit_df,
            self.recorded_interview_df,
            self.live_interview_df
        )
        final_df, compare_df = merging_df(parents_df, wallet, subscription, invitation, job_posting, kit, recorded, live, login)
        self.compare_df_ = fill_missing_values(compare_df)
        return final_df
    
class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = fill_missing_values(X)
        X = ordinal_encoder(X)
        X = log_transform(X)
        return X

# Pipelines
merge_pipeline = Pipeline([
    ('merge', MergingTransformer(wallet_df=wallet_df,
                                 subscription_df=subscription_df,
                                 login_df=login_df,
                                 invitation_df=invitation_df,
                                 job_posting_df=job_posting_df,
                                 kit_df=kit_df,
                                 recorded_interview_df=recorded_interview_df,
                                 live_interview_df=live_interview_df))
])

# Step 2: Preprocess merged data
preprocess_pipeline = Pipeline([
    ('preprocessing', PreprocessingTransformer()),
    ('scaler', StandardScaler()),
    ('to_tensor', FunctionTransformer(lambda x: torch.tensor(x, dtype=torch.float32)))
])

full_pipeline = Pipeline([
    ('merge_pipeline', merge_pipeline),
    ('preprocess_pipeline', preprocess_pipeline)
])

X_tensor = full_pipeline.fit_transform(catcher_df)
X_df = pd.DataFrame(X_tensor)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
X_data = CustomDataset(X_tensor)

BATCH_SIZE = 16
dataloader = DataLoader(X_data, batch_size=BATCH_SIZE, shuffle=True)

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
    
# Initializing the Model 
input_shape = X_df.shape[1]
model_1 = AutoEncoder(input_shape)
model_1

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

model_1.eval()

with torch.no_grad():
    encoder, _ = model_1(X_tensor)

latent_np = encoder.cpu().numpy()

compare_df = merge_pipeline.named_steps['merge'].compare_df_

user_pref_good = {'is_premium':0,
                  'jobma_verified': 1,
                 'subscription_status':0,
                 'company_size':'101-500', 
                 'is_unlimited':0,
                 'subscription_amount_in_dollars': 100.00,
                 'number_of_subscriptions':1,
                 'number_of_invitations':25,
                 'job_posted':4,
                 'number_of_kits':7,
                 'number_of_recorded_interviews':8,
                'number_of_live_interviews':5,
                'days_since_last_login':4,
                'total_sub':1,
            }

def recommend(user_input, model, latent_embeddings, compare_df, pipeline, top_k=5):
    # Transform the user input using the pipeline
    user_df = pd.DataFrame([user_input])
    user_input_tensor = pipeline.transform(user_df)
    print(type(user_input_tensor))
    # user_input_tensor = torch.tensor(user_input_transformed, dtype=torch.float32)

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

result = recommend(user_pref_good, model_1, latent_np, compare_df, preprocess_pipeline, top_k=5)
print(result)

user_pref_test = {'is_premium':1,
                  'jobma_verified': 1,
                 'subscription_status':1,
                 'company_size':'1-25',
                 'is_unlimited':1,
                 'subscription_amount_in_dollars': 125.0,
                 'number_of_subscriptions':1,
                 'number_of_invitations':18,
                 'job_posted':3,
                 'number_of_kits':3,
                 'number_of_recorded_interviews':3,
                'number_of_live_interviews':1,
                'days_since_last_login':4,
                'total_sub':2
            }

result1 = recommend(user_pref_test, model_1, latent_np, compare_df, preprocess_pipeline, top_k=10)
result1