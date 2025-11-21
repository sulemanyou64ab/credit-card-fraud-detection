"""
Data preprocessing and splitting module
"""

from sklearn.model_selection import train_test_split
from config import VALID_SIZE, TEST_SIZE, RANDOM_STATE, TARGET, PREDICTORS


def prepare_data(data_df):
    """
    Prepare data by splitting into train, validation, and test sets
    
    Args:
        data_df (pd.DataFrame): Input dataset
        
    Returns:
        tuple: (train_df, valid_df, test_df)
    """
    print("\nSplitting data into train, validation, and test sets...")
    
    # Split into train and test
    train_df, test_df = train_test_split(
        data_df, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        shuffle=True
    )
    
    # Split train into train and validation
    train_df, valid_df = train_test_split(
        train_df, 
        test_size=VALID_SIZE, 
        random_state=RANDOM_STATE, 
        shuffle=True
    )
    
    print(f"Train set: {train_df.shape[0]} samples")
    print(f"Validation set: {valid_df.shape[0]} samples")
    print(f"Test set: {test_df.shape[0]} samples")
    
    return train_df, valid_df, test_df


def get_X_y(df, target=TARGET, predictors=PREDICTORS):
    """
    Extract features and target from dataframe
    
    Args:
        df (pd.DataFrame): Input dataframe
        target (str): Target column name
        predictors (list): List of predictor column names
        
    Returns:
        tuple: (X, y)
    """
    X = df[predictors]
    y = df[target].values
    return X, y
