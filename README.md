# Recipe Rating Predictor
This project develops a Hybrid Recommender System to predict star ratings for a massive dataset of recipes and user interactions.

## Project Overview
Predicting users' culinary preferences is challenging due to high variability in taste and recipe complexity. This system employs a hybrid approach that combines Collaborative Filtering (latent factors) with Content-Based Filtering (recipe metadata) to improve prediction accuracy relative to standard baseline models

## Methodology
1. Latent Factor Discovery (Collaborative Filtering)
  - SVD (Singular Value Decomposition): Used the scikit-surprise library to decompose the user-item interaction matrix
  - Embeddings: Extracted 1-dimensional latent factors for both users and recipes to capture "hidden" preferences not explicitly stated in the metadata

2. Content-Based Features
  - Recipe Complexity: Incorporates the number of steps (n_steps), number of ingredients (n_ingredients), and total preparation time (minutes)
  - User Bias: Tracks the average rating given by each user to account for "harsh" vs. "lenient" critics

3. Hybrid Modeling
  - The final prediction is generated via a Linear Regression model that uses the SVD factors and recipe metadata as joint input features. This allows the system to remain robust even when user interaction data is sparse.

## Tech Stack
  - Data Analysis: pandas, numpy
  - Visualization: matplotlib, seaborn
  - Recommendation Engine: scikit-surprise (SVD)
  - Machine Learning: scikit-learn (Linear Regression, MinMaxScaler)

## Results & Insights
The hybrid model achieved a Validation MSE of approximately 0.55.
By analyzing the model coefficients, we found that:
  - User Latent Factors and Average User Rating were the strongest predictors of the final score
  - Preparation Time had a slight negative correlation with ratings, suggesting users may penalize overly long recipes
