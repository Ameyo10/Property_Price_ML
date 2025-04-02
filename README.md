# Kaggle House Prices Prediction

## Overview
This repository contains my work on the **House Prices - Advanced Regression Techniques** competition on **Kaggle**. The goal of this project is to predict the sales price of residential homes in Ames, Iowa, using a dataset with **79 explanatory variables** that describe different aspects of the houses.

This competition is ideal for those with basic knowledge of machine learning, looking to practice **feature engineering** and apply **advanced regression techniques** like **Random Forest** and **Gradient Boosting**.

## Dataset
The dataset consists of:
- **Train.csv**: Contains the training data with labeled sale prices.
- **Test.csv**: Contains the test data without sale prices (to be predicted).
- **Sample_submission.csv**: Example submission format.

The dataset was compiled by **Dean De Cock** for educational purposes and is an extended alternative to the well-known **Boston Housing dataset**.

## Objective
- Predict the final sale price of each house in the **test set**.
- The evaluation metric is **Root-Mean-Squared-Error (RMSE)** between the logarithm of the predicted and actual sale price.

## Project Workflow
1. **Data Understanding & Preprocessing**
   - Handling missing values
   - Feature selection & transformation
   - Encoding categorical variables
   
2. **Exploratory Data Analysis (EDA)**
   - Distribution analysis of sale price
   - Relationship between variables
   - Detecting outliers
   
3. **Feature Engineering**
   - Creating new features
   - Handling skewed data
   - Scaling numerical features
   
4. **Model Selection & Training**
   - Baseline models (Linear Regression, Decision Trees)
   - Advanced models (Random Forest, XGBoost, LightGBM)
   - Hyperparameter tuning
   
5. **Evaluation & Submission**
   - Calculating RMSE
   - Generating submission file

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Kaggle-House-Prices.git
   cd Kaggle-House-Prices
   ```
## Contributing
If you'd like to improve this project, feel free to fork the repository, make changes, and submit a pull request.

## Acknowledgments
- Kaggle for hosting the competition
- Dean De Cock for compiling the dataset
- The Kaggle community for valuable discussions

**Happy Coding! 🚀**

