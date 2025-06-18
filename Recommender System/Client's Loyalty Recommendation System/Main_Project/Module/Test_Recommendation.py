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

warnings.filterwarnings('ignore')
# Loading .env file into my python code
load_dotenv()

def create_connection():
    print('creating connection with DB')
    try:
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
    except Exception as e:
        print(f'Error creating connection with DB: {e}')
        return None
    
def create_df(engine,catcher_only=False):
    print('creating DFs=============')
    try:
        if engine is not None:
            if catcher_only:
                print('Catcher DF is requested')
                catcher_df = pd.read_sql("Select jobma_catcher_id, jobma_catcher_parent, subscription_status, company_size FROM jobma_catcher where jobma_verified = '1' ", con=engine) 
                return catcher_df
            print("Wallet DF")
            wallet_df = pd.read_sql("Select catcher_id AS jobma_catcher_id, is_unlimited FROM wallet where is_unlimited <> '' ", con=engine)
            print("Subscription DF")
            subscription_df = pd.read_sql("Select catcher_id AS jobma_catcher_id, currency, subscription_amount FROM subscription_history", con=engine)
            print("Invitation DF")
            invitation_df = pd.read_sql("Select jobma_catcher_id, jobma_interview_mode, jobma_interview_status FROM jobma_pitcher_invitations", con=engine)
            print("Job posting DF")
            job_posting_df = pd.read_sql("Select jobma_catcher_id FROM jobma_employer_job_posting", con=engine)
            print("kit DF")
            kit_df = pd.read_sql("Select catcher_id AS jobma_catcher_id FROM job_assessment_kit", con=engine)
            print('Login DF')
            login_df = pd.read_sql("Select jobma_user_id AS jobma_catcher_id, jobma_last_login FROM jobma_login where jobma_role_id = 3",con=engine)
            
            # Closing the Connection
            engine.dispose()
            return wallet_df, subscription_df,invitation_df,job_posting_df,kit_df,login_df
        else:
            print('Engine is not Supporting (Problem in create_df)')
    except Exception as e:
        print(f'Error in create_df: {e}')
        return None
    
# catcher_df
def fetching_catcher_df(catcher_df):
    print("Processing catcher DF")
    try:
        catcher_df['subscription_status'] = catcher_df['subscription_status'].replace({'0':0, '1':1, '2':0})
        return catcher_df
    except KeyError as e:
        print(f'Key Not Found: {e}')
        return pd.DataFrame()
    
# wallet_df
def fetching_wallet_df(wallet_df):
    print("Processing wallet DF")
    try:
        wallet_df['is_unlimited'] = wallet_df['is_unlimited'].replace({'0':0, '1':1})
        wallet_df.drop_duplicates(inplace=True)
        return wallet_df
    except KeyError as e:
        print(f'Key Not Found: {e}')
        return pd.DataFrame()
    
# subscription_df
def fetching_subscription_df(subscription_df):
    print("Processing subscription DF")
    try:
        subscription_df.loc[subscription_df['currency'] == '1', 'subscription_amount'] /= 85.23
        subscription_df = subscription_df.groupby('jobma_catcher_id').agg(
            subscription_amount_in_dollars = ('subscription_amount', 'sum'),
            number_of_subscriptions = ('subscription_amount', 'count'),
        ).reset_index()
        subscription_df['subscription_amount_in_dollars'] = subscription_df['subscription_amount_in_dollars'].round(3)
        subscription_df.drop_duplicates(inplace=True)
        return subscription_df
    except KeyError as e:
        print(f'Key Not Found: {e}')
        return pd.DataFrame()
    
# login_df
def fetching_login_df(login_df):
    print("Processing login DF")

    # Calculating Number of Gaps between last login and today
    try:
        login_df['jobma_last_login'] = pd.to_datetime(login_df['jobma_last_login'], errors='coerce')
        login_df['activity_duration'] = (pd.Timestamp('today') - login_df['jobma_last_login']).dt.days
        login_df['activity_duration'].fillna(370, inplace=True)
        login_df['activity_duration'] = login_df['activity_duration'].astype(int)
    
        # Binning
        bins = [-1,7,30,90,180,365,float('inf')]
        labels = ['Less than 1 Week', '1-4 Weeks', '1-3 Months', '3-6 Months', '6-12 Months', 'More than 1 Year']
        login_df['activity_duration'] = pd.cut(login_df['activity_duration'], bins=bins, labels=labels)
        login_df = login_df[['jobma_catcher_id', 'activity_duration']]
        return login_df
    except KeyError as e:
        print(f'Key Not Found: {e}')
        return pd.DataFrame()
    
def fetching_features(invitation_df, job_posting_df, kit_df):
    print("Fetching features")

    try:
        job_posting_df['job_posted'] = job_posting_df['jobma_catcher_id'].map(job_posting_df['jobma_catcher_id'].value_counts())
        kit_df['number_of_kits'] = kit_df['jobma_catcher_id'].map(kit_df['jobma_catcher_id'].value_counts())
    
        invitation_df['number_of_invitations'] = invitation_df['jobma_catcher_id'].map(invitation_df['jobma_catcher_id'].value_counts())
        invitation_df = invitation_df[invitation_df['jobma_interview_mode'].isin(['1', '2'])].copy()
        interview_counts = invitation_df.groupby(['jobma_catcher_id', 'jobma_interview_mode']).size().unstack(fill_value=0)
        interview_counts = interview_counts.rename(columns={'1': 'number_of_recorded_interviews', '2': 'number_of_live_interviews'})
        invitation_df = invitation_df.merge(interview_counts, on='jobma_catcher_id', how='left')
    
        #------
        invitation_df = invitation_df[invitation_df['jobma_interview_status'] != '0']
        invitation_df['interview_completed'] = invitation_df['jobma_catcher_id'].map(invitation_df['jobma_catcher_id'].value_counts())
        invitation_df.drop(['jobma_interview_mode', 'jobma_interview_status'], axis=1, inplace=True)
        #------
        invitation_df = invitation_df.drop_duplicates()
    
        job_posting_df = job_posting_df[['jobma_catcher_id', 'job_posted']].drop_duplicates()
        kit_df = kit_df[['jobma_catcher_id', 'number_of_kits']].drop_duplicates()
    
        return invitation_df, job_posting_df, kit_df
    except Exception as e:
        print(f'Error in fetching_features: {e}')
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    
def merging_df(catcher_df, wallet_df, subscription_df, invitation_df, job_posting_df, kit_df, login_df):
    print("Merging DFs")
    try:
        final_df = catcher_df.copy()
    
        # Left join each table one by one
        try:
            if isinstance(final_df, pd.DataFrame):
                final_df = final_df.merge(login_df, on='jobma_catcher_id', how='left')
                final_df = final_df.merge(wallet_df, on='jobma_catcher_id', how='left')
                final_df = final_df.merge(subscription_df, on='jobma_catcher_id', how='left')
                final_df = final_df.merge(invitation_df, on='jobma_catcher_id', how='left')
                final_df = final_df.merge(job_posting_df, on='jobma_catcher_id', how='left')
                final_df = final_df.merge(kit_df, on='jobma_catcher_id', how='left')
                final_df.drop_duplicates(inplace=True)
            else:
                raise TypeError('Final DF is not a DataFrame')
        except TypeError as e:
            print(f'Error in Merging DFs: {e}')
            return None
        except Exception as e:
            print(f'Error Merging DataFrames: {e}')
            return None
    
        # For Total Sub
        try:
            sub_counts = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent').size()
            final_df['total_sub'] = final_df['jobma_catcher_id'].map(sub_counts).fillna(0).astype(int)
        except KeyError as e:
            print(f'Error in Total Sub in merging_df: {e}')
        except TypeError as e:
            print(f'Error of Int Conversion in Total Sub in merging_df: {e}')        
    
        # For Kits
        try:
            sub_kits_sum = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')['number_of_kits'].sum()
            kits_mapped = final_df['jobma_catcher_id'].map(sub_kits_sum).fillna(0)
            final_df['number_of_kits'] = final_df['number_of_kits'].fillna(0) + kits_mapped
            final_df['number_of_kits'] = final_df['number_of_kits'].astype(int)
        except KeyError as e:
            print(f'Error in Kits in merging_df: {e}')
        except TypeError as e:
            print(f'Error of Int Conversion in Kits in merging_df: {e}')
        
        # For Invitations
        try:
            sub_invitations_sum = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')['number_of_invitations'].sum()
            invitations_mapped = final_df['jobma_catcher_id'].map(sub_invitations_sum).fillna(0)
            final_df['number_of_invitations'] = final_df['number_of_invitations'].fillna(0) + invitations_mapped
            final_df['number_of_invitations'] = final_df['number_of_invitations'].astype(int)
        except KeyError as e:
            print(f'Error in Invitations in merging_df: {e}')
        except TypeError as e:
            print(f'Error of Int Conversion in Invitations in merging_df: {e}')
        
        # For Job Posted
        try:
            sub_job_posted_sum = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')['job_posted'].sum()
            job_posted_mapped = final_df['jobma_catcher_id'].map(sub_job_posted_sum).fillna(0)
            final_df['job_posted'] = final_df['job_posted'].fillna(0) + job_posted_mapped
            final_df['job_posted'] = final_df['job_posted'].astype(int)
        except KeyError as e:
            print(f'Error in Job Posted in merging_df: {e}')
        except TypeError as e:
            print(f'Error of Int Conversion in Job Posted in merging_df: {e}')
    
        # For Recorded Interviews
        try:
            sub_recorded_sum = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')['number_of_recorded_interviews'].sum()
            recorded_mapped = final_df['jobma_catcher_id'].map(sub_recorded_sum).fillna(0)
            final_df['number_of_recorded_interviews'] = final_df['number_of_recorded_interviews'].fillna(0) + recorded_mapped
            final_df['number_of_recorded_interviews'] = final_df['number_of_recorded_interviews'].astype(int)
        except KeyError as e:
            print(f'Error in Recorded Interviews in merging_df: {e}')
        except TypeError as e:
            print(f'Error of Int Conversion in Recorded Interviews in merging_df: {e}')
    
        # For Live Interviews
        try:
            sub_live_sum = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')['number_of_live_interviews'].sum()
            live_mapped = final_df['jobma_catcher_id'].map(sub_live_sum).fillna(0)
            final_df['number_of_live_interviews'] = final_df['number_of_live_interviews'].fillna(0) + live_mapped
            final_df['number_of_live_interviews'] = final_df['number_of_live_interviews'].astype(int)
        except KeyError as e:
            print(f'Error in Live Interviews in merging_df: {e}')
        except TypeError as e:
            print(f'Error of Int Conversion in Live Interviews in merging_df: {e}')
    
        # For Interview Completed
        try:
            sub_to_parent_sum = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')['interview_completed'].sum()
            final_df.loc[final_df['jobma_catcher_id'].isin(sub_to_parent_sum.index), 'interview_completed'] += final_df['jobma_catcher_id'].map(sub_to_parent_sum).fillna(0).astype(int)
        except KeyError as e:
            print(f'Error in Live Interviews in merging_df: {e}')
        except TypeError as e:
            print(f'Error of Int Conversion in Live Interviews in merging_df: {e}')
        
        # For Minimum Login Days
        try:
            login_order = {
                'Less than 1 Week':0,
                '1-4 Weeks':1,
                '1-3 Months':2,
                '3-6 Months':3,
                '6-12 Months':4,
                'More than 1 Year':5
            }
        
            # For Login
            final_df['activity_duration'] = final_df['activity_duration'].map(login_order).fillna(5).astype(int)
            # It will calculate the minimum activity of subcatcher
            sub_min_login = final_df[final_df['jobma_catcher_parent'] != 0].groupby('jobma_catcher_parent')['activity_duration'].min()
            catcher_mask = final_df['jobma_catcher_id'].isin(sub_min_login.index)
            final_df.loc[catcher_mask, 'activity_duration'] = np.minimum(
                final_df.loc[catcher_mask, 'activity_duration'],
                final_df.loc[catcher_mask, 'jobma_catcher_id'].map(sub_min_login)
            )
        except Exception as e:
            print(f'Error in Login in merging_df: {e}')
    
        df = final_df[final_df['jobma_catcher_parent'] == 0].copy()
        df.drop(['jobma_catcher_parent'], axis=1, inplace=True)
        
        compare_df = df.copy()
        df.drop('jobma_catcher_id', axis=1, inplace=True)
    
        print(f"Final merged df shape is {df.shape}")
    
        return df, compare_df
    except Exception as e:
        print(f'Key Not Found: {e}')

    return None, None

def fill_missing_values(final_df):
    final_df = final_df.copy()

    # Step 1: Replace inf with NaN first
    try: 
        final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
        # Step 2: Fill NaNs
        fill_values = {
            'subscription_status': 0,
            'company_size': '1-25',
            'activity_duration': 5,
            'is_unlimited': 0,
            'subscription_amount_in_dollars': 0,
            'number_of_subscriptions': 0,
            'number_of_invitations': 0,
            'number_of_recorded_interviews': 0,
            'number_of_live_interviews': 0,
            'interview_completed': 0,
            'job_posted': 0,
            'number_of_kits': 0,
            'total_sub': 0,
        }
        final_df.fillna(fill_values, inplace=True)
        
    except KeyError as e:
        print(f'Key Not Found: {e}')
    except Exception as e:
        print(f'Error during fillna step: {e}')

    # Step 3: Explicitly cast to int for the appropriate columns
    try:
        int_columns = [
            'subscription_status',
            'activity_duration',
            'is_unlimited',
            'number_of_subscriptions',
            'number_of_invitations',
            'number_of_recorded_interviews',
            'number_of_live_interviews',
            'interview_completed',
            'job_posted',
            'number_of_kits',
            'total_sub',
        ]
        
        for col in int_columns:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0).astype(int)
            else:
                print(f'Column {col} not found in DataFrame')
    except Exception as e:
        print(f'Error during Int Conversion: {e}')

    return final_df

# Data Encoding 
def ordinal_encoder(df):
    try:
        ordinal_col = ['company_size']
        company_size_order = ['1-25', '26-100', '101-500', '500-1000', 'More than 1000']
    
        ordinal = OrdinalEncoder(categories=[company_size_order])

        encoded = ordinal.fit_transform(df[ordinal_col])
    
        encoded_df = pd.DataFrame(encoded, columns=[f' {col}_ord' for col in ordinal_col], index=df.index)
    
        df.drop(columns=ordinal_col, inplace=True)
    
        df = pd.concat([df, encoded_df], axis=1)
        return df
        
    except KeyError as e:
        print(f"Missing column in the input DataFrame: {e}")
        return df
    except Exception as e:
        print(f'Error in ordinal_encoder: {e}')
        return pd.DataFrame()
    
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
            'activity_duration',
            'total_sub'
        ]

    if not isinstance(df, pd.DataFrame):
        print("Error: Input received in Log Transformation is not a DataFrame.")
        return df

    if not log_cols:
        print("Log columns list is empty.")
        return df

    df = df.copy()

    for col in log_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)        # Handle NaNs
            df[col] = df[col].clip(lower=0)    # if a number is less than zero, turn it into zero
            df[col] = np.log1p(df[col])        # Apply log1p safely

    return df

# Will only work if the input this function is receiving is "Numpy Array".
def to_tensor(x):
    try:
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float32)
        else:
            raise TypeError('Input is Not Numpy Array')
    except TypeError as e:
        print(f"Error Converting Values to Tensors: {e}")
        return None
    
class MergingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        engine = create_connection()
        wallet_df, subscription_df, invitation_df, job_posting_df, kit_df, login_df = create_df(engine)
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
    
class AutoEncoder(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_shape)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def train_and_save_model():
    print("Starting with model training")

    try:
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
    
        BATCH_SIZE = 32
        dataloader = DataLoader(X_data, batch_size=BATCH_SIZE, shuffle=True)
    
        # Initializing the Model 
        input_shape = X_df.shape[1]
        model_1 = AutoEncoder(input_shape)
    
        # Important Parameters 
        learning_rate = 0.001
        epochs = 100
        # epochs = 50
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
    except Exception as e:
        print(f'Error in Training: {e}')

# Expecting catcher_id to fetch all the details of user

def predict_using_catcher_id(catcher_id, n=5):

    #For Minimum Login Days
    login_order = {
        'Less than 1 Week':0,
        '1-4 Weeks':1,
        '1-3 Months':2,
        '3-6 Months':3,
        '6-12 Months':4,
        'More than 1 Year':5
    }
    
    # Load the full pipeline and model
    with open('full_pipeline.pkl', 'rb') as f:
        full_pipeline = pickle.load(f)

    # Extract Pipelines
    preprocess_pipeline = full_pipeline.named_steps['preprocess_pipeline']
    compare_df = pd.read_csv('compare_df.csv')

    # Get the row corresponding to the input catcher_id
    if isinstance(compare_df, pd.DataFrame):
        user_row = compare_df[compare_df['jobma_catcher_id'] == catcher_id]
        print(user_row)
    else:
        print('Compare df is not a DataFrame')

    # Exception Handling
    try:
        if user_row.empty:
            raise ValueError(f"No data found for catcher_id: {catcher_id}")
    except ValueError as e:
        print(e)
        return pd.DataFrame()

    user_input = user_row.drop(columns=['jobma_catcher_id']).iloc[0].to_dict()
    user_df = pd.DataFrame([user_input])
    transformed_input = preprocess_pipeline.transform(user_df)

    # Load trained autoencoder model
    input_shape = transformed_input.shape[1]
    model_1 = AutoEncoder(input_shape)
    model_1.load_state_dict(torch.load('model.pth'))
    model_1.eval()

    # Load latent embeddings
    with open('latent_np.pkl', 'rb') as embeddings_file:
        embeddings = pickle.load(embeddings_file)

    # Generate user embedding
    try:
        with torch.no_grad():
            user_embedding, _ = model_1(torch.tensor(transformed_input, dtype=torch.float32))
            user_embedding = F.normalize(user_embedding, p=2, dim=1)
    
            latent_embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
            latent_embeddings = F.normalize(latent_embeddings_tensor, p=2, dim=1)
    
            # Compute cosine similarity and get top n+1 (to account for the input catcher_id)
            if isinstance(user_embedding, torch.Tensor) and isinstance(latent_embeddings, torch.Tensor):
                similarities = F.cosine_similarity(user_embedding, latent_embeddings, dim=1)
                top_indices = similarities.topk(n + 1).indices.cpu().numpy()
            else:
                raise TypeError('Embeddings Error')
    except TypeError as e:
        print(f'Embeddings are not in Tensor: {e}')
        return None
    except Exception as e:
        print(f'Error in predict: {e}')
    
    # Final Recommendation DataFrame
    recommended = compare_df.iloc[top_indices].copy()

    # Drop the row corresponding to the input catcher_id
    recommended = recommended[recommended['jobma_catcher_id'] != catcher_id]

    # Ensure exactly 'n' rows are returned
    recommended = recommended.head(n)

    # Add similarity score to the recommendations
    recommended['similarity'] = similarities[top_indices[:len(recommended)]].cpu().numpy()[:len(recommended)]

    # Replace encoded activity_duration with actual values
    try:
        if 'activity_duration' in recommended.columns:
            recommended['activity_duration'] = recommended['activity_duration'].replace({v:k for k,v in login_order.items()})
        else:
            raise  KeyError("'activity_duration' column not found in DataFrame.")
    except Exception as e:
        print(f"An error occurred: {e}")

    print(recommended)

# Expecting all details from user_input
def predict(user_input, n=5):

    #For Minimum Login Days
    login_order = {
        'Less than 1 Week':0,
        '1-4 Weeks':1,
        '1-3 Months':2,
        '3-6 Months':3,
        '6-12 Months':4,
        'More than 1 Year':5
    }
    
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
    try:
        with torch.no_grad():
            user_embedding, _ = model_1(transformed_input)
            user_embedding = F.normalize(user_embedding, p=2, dim=1)
            print(f'User Embedding Type is: {type(user_embedding)}')
    
            latent_embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
            latent_embeddings = F.normalize(latent_embeddings_tensor, p=2, dim=1)
            print(f'Latent Embedding Type is {type(latent_embeddings)}')
    
            # Compute cosine similarity and get top-N
            if isinstance(user_embedding, torch.Tensor) and isinstance(latent_embeddings, torch.Tensor):
                similarities = F.cosine_similarity(user_embedding, latent_embeddings, dim=1)
                top_indices = similarities.topk(n).indices.cpu().numpy()
            else:
                raise TypeError('Embeddings Error')
    except TypeError as e:
        print(f'Embeddings are not in Tensor: {e}')
        return None
    except Exception as e:
        print(f'Error in predict: {e}')
    
    # Final Recommendation DataFrame
    recommended = compare_df.iloc[top_indices].copy()
    
    # To show the actual activity duration
    try:
        if 'activity_duration' in recommended.columns:
            recommended['activity_duration'] = recommended['activity_duration'].replace({v:k for k,v in login_order.items()})
        else:
            raise KeyError("'activity_duration' column not found in DataFrame.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # To show the Similarity Score
    recommended['similarity'] = similarities[top_indices].cpu().numpy()
    print(user_input)
    
    print(recommended)

user_pref_good = {
                 'subscription_status':0,
                 'company_size':'101-500',
                'activity_duration':4,
                 'is_unlimited':0,
                 'subscription_amount_in_dollars': 100.00,
                 'number_of_subscriptions':1,
                 'number_of_invitations':25,
                'number_of_recorded_interviews':8,
                'number_of_live_interviews':5,
                'interview_completed':10,
                'job_posted':4,
                'number_of_kits':7,
                'total_sub':1,
            }

user_pref_test = {
                  'subscription_status':1,
                'company_size':'1-25',
                  'activity_duration':3,
                 'is_unlimited':1,
                 'subscription_amount_in_dollars': 125.0,
                 'number_of_subscriptions':1,
                 'number_of_invitations':18,
                  'number_of_recorded_interviews':523,
                'number_of_live_interviews':1,
                'interview_completed':10,
                 'job_posted':1,
                 'number_of_kits':1,
                'total_sub':1
            }

train_and_save_model()

predict(user_pref_good, 5)
predict(user_pref_test, 5)

predict_using_catcher_id(6025, 5)
predict_using_catcher_id(6189, 5)