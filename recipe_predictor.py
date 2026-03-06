import pandas as pd
import numpy as np
import ast
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from surprise import SVD, Reader, Dataset

def load_and_preprocess(data_path):
    # Load raw and processed data
    interactions = pd.read_csv(f'{data_path}RAW_interactions.csv')
    recipes = pd.read_csv(f'{data_path}RAW_recipes.csv')
    
    # Merge interaction data with recipe metadata
    df = pd.merge(interactions, recipes, left_on='recipe_id', right_on='id')
    df.drop(columns=['date', 'id', 'review', 'contributor_id', 'submitted', 'description', 'steps'], inplace=True)
    
    # Handle missing values
    df['name'] = df['name'].fillna('Unknown')
    return df

def generate_embeddings(df):
    print("Generating latent factor embeddings via SVD...")
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'recipe_id', 'rating']], reader)
    trainset = data.build_full_trainset()

    # Train SVD to get latent factors for users and items
    model = SVD(n_factors=1, lr_all=0.01, reg_all=0.3, n_epochs=20)
    model.fit(trainset)

    user_embeddings = {trainset.to_raw_uid(i): model.pu[i] for i in range(trainset.n_users)}
    recipe_embeddings = {trainset.to_raw_iid(i): model.qi[i] for i in range(trainset.n_items)}

    df['user_factor'] = df['user_id'].map(lambda x: user_embeddings[x][0])
    df['recipe_factor'] = df['recipe_id'].map(lambda x: recipe_embeddings[x][0])
    
    return df

def feature_engineering(df):
    print("Engineering content-based features...")
    # Calculate global user bias
    df['avg_user_rating'] = df.groupby('user_id')['rating'].transform('mean')
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    df[['minutes', 'n_steps', 'n_ingredients']] = scaler.fit_transform(df[['minutes', 'n_steps', 'n_ingredients']])
    
    return df

def main():
    path = "data"
    df = load_and_preprocess(path)
    df = generate_embeddings(df)
    df = feature_engineering(df)

    # Define features for the Hybrid Linear Model
    features = ['user_factor', 'recipe_factor', 'avg_user_rating', 'minutes', 'n_steps', 'n_ingredients']
    X = df[features]
    y = df['rating']

    # Simple train/test split for validation
    split = int(0.8 * len(df))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train model
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)

    # Get predictions and calculate MSE
    predictions = reg_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Validation MSE: {mse:.4f}")
    
    # Print feature importance (coefficients)
    for feature, coef in zip(features, reg_model.coef_):
        print(f"Feature: {feature:15} | Coefficient: {coef:.4f}")

if __name__ == "__main__":
    main()
