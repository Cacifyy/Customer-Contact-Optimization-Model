import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

def train_model(data: pd.DataFrame) -> HistGradientBoostingClassifier:
    """
    Function to train the model using GridSearchCV to find the best parameters.
    :param data: DataFrame containing feature matrix and target column ('success')
    :return: Trained Gradient Boosting Classifier
    """
    data = data[['balance', 'previous', 'campaign',
                 'poutcome=success', 'poutcome=failure',
                 'poutcome=other', 'success']]

    # Separate features and target
    X = data.drop('success', axis=1)
    y = data['success']
    # Create the campaign_previous_interaction feature
    # data['campaign_previous_interaction'] = data['campaign'] * data['previous']

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5)

    # Initialize and train the Gradient Boosting Classifier
    gbc = HistGradientBoostingClassifier()
    
    # Define a parameter grid for hyperparameter tuning
    param_grid = {
        'max_iter': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'max_leaf_nodes': [15, 25, 31, 50]
    }
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(gbc, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
    
    # Fit model to the training data
    grid_search.fit(X_train, y_train)
    
    # Get best estimator
    best_gb = grid_search.best_estimator_

    # Evaluate model on validation set
    y_pred = best_gb.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Model accuracy on validation data: {accuracy * 100:.2f}%")

    # Save the best model
    with open('best_gb_model.pkl', 'wb') as model_file:
        pickle.dump(best_gb, model_file)

    return best_gb

def calculate_metrics(y_true: pd.Series, contact_customers: np.ndarray, data: pd.DataFrame) -> None:
    """
    Helper function to calculate and print accuracy, sensitivity, specificity, and profit.
    :param y_true: True labels (success column) as a Series
    :param contact_customers: Binary array indicating which customers were selected to contact
    """
    # Calculate and print accuracy
    accuracy = accuracy_score(y_true, contact_customers)
    print(f"Model accuracy on provided data: {accuracy * 100:.2f}%")

    # Calculate and print sensitivity
    sensitivity = recall_score(y_true, contact_customers)
    print(f"Model sensitivity (recall) on provided data: {sensitivity * 100:.2f}%")

    # Calculate and print specificity
    tn, fp, fn, tp = confusion_matrix(y_true, contact_customers).ravel()
    specificity = tn / (tn + fp)
    print(f"Model specificity on provided data: {specificity * 100:.2f}%")

    # Calculate and print total profit
    successful_contacts = (contact_customers == 1) & (y_true == 1)
    customer_profit = (0.04 * data['balance'] - 10) * successful_contacts
    total_profit = customer_profit.sum()
    print(f'Total profit: ${total_profit:.2f}')

def identify_customers(data: pd.DataFrame) -> np.ndarray:
    """
    Function that accepts a dataset and returns a binary array showing which customers to contact.
    This function will be called to evaluate your program - DO NOT CHANGE ITS SIGNATURE
    :param data: DataFrame with the same format as 3625_assign2_data_train, BUT without the target column (the "success" column")
    :return: a binary array of length "n", where n is the number of instances in data. Indices set to 1 indicate the customers that should be contacted
    """
    # Load the trained model
    with open('best_gb_model.pkl', 'rb') as model_file:
        best_gb = pickle.load(model_file)

    # Check if 'success' column is present for calculating metrics 
    if 'success' in data.columns:
        y_true = data['success']
        data = data.drop('success', axis=1)
    else:
        y_true = None

    training_columns = best_gb.feature_names_in_

    # Ensure the input data has the same columns as the training data
    data = data[training_columns]

    # Predict probabilities of success for the new customers
    prob_success = best_gb.predict_proba(data)[:, 1]

    #Calculate expected profit for each customer
    expected_profit = (prob_success * 0.04 * data['balance']) - 10

    # Identify customers to contact (expected profit > 0)
    contact_customers = (expected_profit > -9.9).astype(int)

    # Calculate metrics on data where the outcome is known
    if y_true is not None:
        calculate_metrics(y_true, contact_customers, data)

    return contact_customers.values

if __name__ == '__main__':
    # Example of how to load the training data, splitting into X and y
    dataset = pd.read_csv('./data/3625_assign2_data_train.csv', index_col=None)

    dataset_total = (0.04 * dataset['balance'] - 10) * dataset['success']
    total_profit2 = dataset_total.sum()
    print(f'Total profit (training): ${total_profit2:.2f}')
    X = dataset.drop('success', axis=1)
    y = dataset['success']

    print(f'Number of instances: {X.shape[0]}, Number of attributes: {X.shape[1]}')

    # Train the model
    train_model(dataset)

    # Identify customers to contact
    contact_list = identify_customers(X)

    customer_profit = (0.04 * X['balance'] - 10) * contact_list
    total_profit = customer_profit.sum()
    print(f'Total profit (training): ${total_profit:.2f}')
