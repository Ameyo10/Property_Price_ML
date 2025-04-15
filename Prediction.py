import pandas as pd
import joblib
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer

def predict_new_data(new_data_path):
    # Load model and feature information
    model = joblib.load('property_price_model.pkl')
    important_features = joblib.load('important_features.pkl')
    
    # Load and prepare new data
    new_data = pd.read_csv(new_data_path)
    new_data_encoded = pd.get_dummies(new_data, drop_first=True)
    
    # Align columns with training data
    train_columns = important_features
    new_data_processed = new_data_encoded.reindex(columns=train_columns, fill_value=0)
    
    # Make predictions
    predictions = model.predict(new_data_processed)
    
    # Create output dataframe
    result = pd.DataFrame({
        'Id': new_data['Id'],
        'Predicted_SalePrice': predictions
    })
    
    return result

# Making Predictions
if __name__ == "__main__":
    
    # Reading the test data
    test_data = pd.read_csv('D:/Python Projects/Python-Projects/Machine_Learning_Projects_Discord/Property_Price/test.csv')
    
    # Doing predictions (returns a DataFrame with 'Id' and 'Predicted_SalePrice')
    predictions_df = predict_new_data(
        'D:/Python Projects/Python-Projects/Machine_Learning_Projects_Discord/Property_Price/test.csv'
    )
    
    # Merge predictions with original test data
    final_data = test_data.merge(predictions_df, on='Id', how='left')
    
    # Creating a new csv file with the predictions
    final_data.to_csv(
        'D:/Python Projects/Python-Projects/Machine_Learning_Projects_Discord/Property_Price/test_with_predictions.csv', 
        index=False
    )
    print("Predictions saved to test_with_predictions.csv")