import pandas as pd
import numpy as np
import streamlit as st
from fuzzywuzzy import process
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
sns.set(style="white",font_scale=1.5)
sns.set(rc={"axes.facecolor":"#FFFAF0","figure.facecolor":"#FFFAF0"})
sns.set_context("poster",font_scale = .7)

# Function to load data
@st.cache_data
def load_data(file):
    if file is not None:
        if file.name.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format")
        return data
    else:
        return None

# Function to rename columns automatically
@st.cache_data
def rename_columns_auto(df_current, df_leavers):
    column_mappings = {}
    used_names = set()

    for current_column in df_current.columns:
        best_match, score = process.extractOne(current_column, df_leavers.columns)
        
        if score >= 50:  
            if best_match not in df_current.columns and best_match not in used_names:
                column_mappings[current_column] = best_match
                used_names.add(best_match)
            else:
                alt_best_match, alt_score = process.extractOne(current_column, df_leavers.columns)
                if alt_best_match not in df_current.columns and alt_best_match not in used_names:
                    column_mappings[current_column] = alt_best_match
                    used_names.add(alt_best_match)
    df_current = df_current.rename(columns=column_mappings)
    return df_current

# Function to drop columns not common in both datasets
@st.cache_data
def drop_columns_not_common(df_current, df_leavers):
    common_columns = list(set(df_current.columns) & set(df_leavers.columns))
    df_current = df_current[common_columns]
    df_leavers = df_leavers[common_columns]
    return df_current, df_leavers

# Function to preprocess data and encode it
@st.cache_data
def preprocess_data(cleaned_data):
    encoded_cleaned_data = pd.DataFrame()
    all_columns_dict = {}

    if not cleaned_data.empty:
        # Storing Values of data to encode
        encoded_cleaned_data = cleaned_data.copy()
        # Label Encoding
        for col in sorted(list(encoded_cleaned_data.select_dtypes(include='object').columns)):
            label = LabelEncoder()
            encoded_cleaned_data[col] = label.fit_transform(encoded_cleaned_data[col].astype(str))
            all_columns_dict[col] = dict(zip(label.classes_, label.transform(label.classes_)))
    return encoded_cleaned_data, all_columns_dict

# Function to train the model
@st.cache_data
def train_model(X_train_resampled, y_train_resampled):
    model_params = {
    'learning_rate': 0.01,
    'max_depth': 5,
    'reg_alpha': 0.4,
    'reg_lambda': 0.4,
    'gamma': 0.1,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 1,
    'n_estimators': 1000,
    'random_state': 2,
    'eval_metric': 'rmse',
    'objective': 'reg:tweedie',  # Tweedie loss function
    'tweedie_variance_power': 1  # Tweedie variance power parameter
}

    num_folds = 3
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=2)
    param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'n_estimators': [500, 1000, 1500]
}
    model = XGBRegressor(**model_params)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               scoring='neg_mean_squared_error', cv=kf)
    grid_search.fit(X_train_resampled, y_train_resampled)

    # Using the best model for prediction
    _best_model = grid_search.best_estimator_

    # Print best hyperparameters
    st.write("Best hyperparameters:")
    st.write(grid_search.best_params_)

    # Model training
    _best_model.fit(X_train_resampled, y_train_resampled, eval_set=[(X_train_resampled, y_train_resampled), (X_test, y_test)])#, eval_metric='rmse')
    return _best_model

# Function to plot metrics
@st.cache_data
def plot_metrics(_best_model):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(_best_model.evals_result_['validation_0']['rmse'], label='Training RMSE')
    ax.plot(_best_model.evals_result_['validation_1']['rmse'], label='Validation RMSE')
    ax.set_xlabel('Number of Rounds')
    ax.set_ylabel('RMSE')
    ax.set_title('Training and Validation RMSE over Boosting Rounds')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    pass

# Function to reverse encoded columns
@st.cache_data
def reverse_encode_columns(df, columns_dict):
    flipped_columns = {}
    for col, col_dict in columns_dict.items():
        flipped_columns[col] = {v: k for k, v in col_dict.items()}
        df[col] = df[col].replace(flipped_columns[col])
    return df

# Function to handle missing values
def handle_missing_values(df):
    if df.isnull().sum().sum() > 0:
        st.error("Missing values detected")
        option = st.radio("Choose an option to handle missing values:", ('Fill data', 'Drop Null Values'))
        if option == 'Fill data':
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Fill categorical columns with modal value
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    # Fill numerical columns with mean
                    df[col] = df[col].fillna(df[col].mean())
        elif option == 'Drop Null Values':
            # Drop rows with null values
            df = df.dropna()
            df.reset_index(drop=True, inplace=True)
        
    return df

# Streamlit UI
st.title("HR Analytics App")

current_data = st.file_uploader("Upload Current Staff Dataset: ", type=['csv', 'xlsx'])
leavers_data = st.file_uploader("Upload Leavers Dataset: ", type=['csv', 'xlsx'])

if current_data and leavers_data:
    df_current = load_data(current_data)
    df_leavers = load_data(leavers_data)

    # Rename columns in df_current
    df_current = rename_columns_auto(df_current, df_leavers)
    
    df_current, df_leavers = drop_columns_not_common(df_current, df_leavers)


    # Ensure the 'Leaving Date' and 'Start Date' are in the format dd/mm/yyyy
    df_leavers['Leaving Date'] = pd.to_datetime(df_leavers['Leaving Date'], format='%d/%m/%Y')
    df_leavers['Start Date'] = pd.to_datetime(df_leavers['Start Date'], format='%d/%m/%Y')

    # Calculate Length of Service and Attrition
    df_leavers['Length of Service'] = abs(((df_leavers['Leaving Date'] - df_leavers['Start Date']) / pd.Timedelta(days=30.44)).round(0))
    df_leavers['Attrition'] = 1
    # Calculate Length of Service and Attrition for df_leavers
   # df_leavers['Length of Service'] = abs(((pd.to_datetime(df_leavers['Leaving Date']) - pd.to_datetime(df_leavers['Start Date'])) / pd.Timedelta(days=30.44)).round(0))
    #df_leavers['Attrition'] = 1
    
    # Calculate Length of Service and Attrition for df_current
    today_date = datetime.today().strftime('%Y-%m-%d')
    df_current['Length of Service'] = abs(((pd.to_datetime(today_date) - pd.to_datetime(df_current['Start Date'])) / pd.Timedelta(days=30.44)).round(0))
    df_current['Attrition'] = 0

    combined_data = pd.concat([df_leavers, df_current], ignore_index=True)

    # Sidebar
    st.sidebar.info('Data Cleaning Options')
    st.sidebar.title("Drop Unwanted Values")
    drop_values = {}
    for column in combined_data.columns:
        values_to_drop = st.sidebar.multiselect(f"Select values to drop for {column}:", combined_data[column].unique())
        drop_values[column] = values_to_drop

    cleaned_data = combined_data.copy()
    for column, values_to_drop in drop_values.items():
        cleaned_data = cleaned_data[~cleaned_data[column].isin(values_to_drop)]

    cleaned_data.reset_index(drop=True, inplace=True)

    st.sidebar.title("Drop Unwanted Columns")
    columns_to_drop = st.sidebar.multiselect("Select columns to drop:", combined_data.columns)
    missing_values = cleaned_data.isnull().sum().sum() > 0 
    if columns_to_drop:
        cleaned_data = cleaned_data.drop(columns=columns_to_drop, axis=1)
        # Handle missing values in the cleaned data
        cleaned_data = handle_missing_values(cleaned_data)
        st.write("Dataset with Selected Columns Dropped and Missing Values Handled")
        st.write(cleaned_data)

    elif not columns_to_drop and missing_values:# and columns_to_drop is None:
    # Handle missing values in the cleaned data
        cleaned_data = handle_missing_values(cleaned_data)
        st.write("Cleaned Dataset with Missing Values Handled")
        st.write(cleaned_data)
    else:
        st.write("Combined Dataset with no Columns Dropped")
        st.write(cleaned_data)
else:
    st.error("Upload Dataset to Get Started")  


# Preprocess data and encode it
encoded_cleaned_data, all_columns_dict = preprocess_data(cleaned_data)
X = encoded_cleaned_data.drop(['Attrition'], axis=1)  # Features (input variables)
y = encoded_cleaned_data['Attrition']  # Target variable

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Check for null values in X and y
nulls_X = X.isnull().sum().sum()
nulls_y = y.isnull().sum().sum()

if nulls_X or nulls_y > 0:
    st.error("Fill or Remove Null Values")
else:
    # Data Augmentation using SMOTE
    smote = SMOTE(random_state=2)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train the model
    _best_model = train_model(X_train_resampled, y_train_resampled)

    # Plot training and evaluation metrics
    plot_metrics(_best_model)

    # Storing best prediction values
    y_pred = _best_model.predict(X_test)
    train_y_pred = _best_model.predict(X_train_resampled)

    # Storing Training Performance Metrics
    train_rmse_XGB = mean_squared_error(y_train_resampled, train_y_pred, squared=False)
    train_r2_XGB = r2_score(y_train_resampled, train_y_pred)
    train_mae_XGB = mean_absolute_error(y_train_resampled, train_y_pred)

    # Storing Evaluation Performance Metrics
    eval_rmse_XGB = mean_squared_error(y_test, y_pred, squared=False)
    eval_r2_XGB = r2_score(y_test, y_pred)
    eval_mae_XGB = mean_absolute_error(y_test, y_pred)

    # Printing training and Evaluation Metrics
    st.write(f"Training and Test RMSE are : {train_rmse_XGB:.4f} and {eval_rmse_XGB:.4f} respectively")
    st.write(f"Training and Test r2_score are: {train_r2_XGB:.4f} and {eval_r2_XGB:.4f} respectively")
    st.write(f"Training and Test MAE are: {train_mae_XGB:.4f} and {eval_mae_XGB:.4f} respectively")

total_current_staffs_features = X[y == 0]
# Making predictions on the entire dataset (current staffs)
total_current_staffs_predictions = _best_model.predict(total_current_staffs_features)

total_current_staffs_features['Prediction'] = total_current_staffs_predictions
sorted_total_current_staffs = total_current_staffs_features.sort_values(by='Prediction', ascending=False)
sorted_total_current_staffs.reset_index(drop=True, inplace=True)

no_attrition = cleaned_data.drop('Attrition', axis=1)
columns_to_reverse_encode = no_attrition.columns
sorted_total_current_staffs_decoded = reverse_encode_columns(sorted_total_current_staffs, {col: all_columns_dict.get(col, {}) for col in columns_to_reverse_encode})

# Calculate the quantiles based on the specified percentages
quantiles = pd.qcut(sorted_total_current_staffs_decoded['Prediction'], [0, 0.7, 0.92, 1], labels=['Low Risk', 'Mid Risk', 'High Risk'])

# Add the "Risk Level" column to the DataFrame
sorted_total_current_staffs_decoded['Risk Level'] = quantiles

st.write(sorted_total_current_staffs_decoded)



# Allow user to select the feature they want to visualize
feature_to_visualize = st.selectbox("Select feature to visualize:", sorted_total_current_staffs_decoded.columns)

# Adding a "Visualize" button
if st.button("Visualize"):
    # Selecting the top 8% of rows of the dataframe
    subset_df = sorted_total_current_staffs_decoded.head(int(0.08 * len(sorted_total_current_staffs_decoded)))

    # Counting the occurrences of each feature in the subset_df
    feature_counts_subset = subset_df[feature_to_visualize].value_counts()

    # If there are more than 12 unique values, select only the top 12
    if len(feature_counts_subset) > 12:
        feature_counts_subset = feature_counts_subset.head(12)

    # Calculating the percentage of each feature in the subset_df
    feature_percentages_subset = (feature_counts_subset / len(subset_df)) * 100

    # Counting the occurrences of each feature in sorted_total_current_staffs_decoded
    feature_counts_total = sorted_total_current_staffs_decoded[feature_to_visualize].value_counts()

    # Calculating the percentage of each feature in sorted_total_current_staffs_decoded
    feature_percentages_total = (feature_counts_subset/feature_counts_total) * 100

    feature_percentages_total.dropna(inplace=True)

    colors = sns.color_palette('Set2')

    # Creating the first plot for counts and percentages of subset_df
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    bars = plt.bar(feature_counts_subset.index, feature_counts_subset.values, color=colors)
    plt.title(f'Counts of {feature_to_visualize} in Subset')
    plt.xlabel(feature_to_visualize)
    plt.ylabel('Count')
    plt.xticks(rotation=25, ha='right')

    # Display of percentage on top of each bar
    for bar, percentage in zip(bars, feature_percentages_subset):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{percentage:.1f}%', ha='center', va='bottom', color='black', fontsize=10)

    # Creating the second plot for counts of feature_to_visualize in sorted_total_current_staffs_decoded
    plt.subplot(1, 2, 2)
    bars = plt.bar(feature_percentages_total.index, feature_percentages_total.values, color=colors)
    plt.title(f'Percentage of Predicted Leavers per {feature_to_visualize}')
    plt.xlabel(feature_to_visualize)
    plt.ylabel('Percentage')
    plt.xticks(rotation=25, ha='right')

    # Display of percentage on top of each bar
    for bar, percentage in zip(bars, feature_percentages_total):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{percentage:.1f}%', ha='center', va='bottom', color='black', fontsize=10)

    plt.tight_layout()
    st.pyplot(plt)


# Streamlit UI for scenario testing

# Function for training the model
@st.cache_data
def train_new_model(X_train_new, y_train_new):
    model_params = {
        'boosting_type': 'gbdt',
        'num_leaves': 50,
        'learning_rate': 0.01,
        'n_estimators': 500,
        'min_child_weight': 1,
        'random_state': 2,
        'objective': 'tweedie',
        'tweedie_variance_power': 1
    }

    num_folds = 5
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=2)
    param_grid = {
        'num_leaves': [15, 31, 50],
        'learning_rate': [0.001, 0.01, 0.1],
        'n_estimators': [500, 1000, 1500]
    }
    model = lgb.LGBMRegressor(**model_params)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               scoring='neg_mean_squared_error', cv=kf)
    grid_search.fit(X_train_new, y_train_new)

    # Using the best model for prediction
    _new_model = grid_search.best_estimator_

    return _new_model

# Function for scenario testing
@st.cache_data
def run_scenario_test(_new_model, input_values, feature_to_predict, desired_attrition_prediction):
   # Convert user input values into a DataFrame
    input_df = pd.DataFrame([input_values])

    # Encode categorical columns using corresponding dictionaries in all_columns_dict
    for column, col_dict in all_columns_dict.items():
        if column in input_df.columns:
            input_df[column] = input_df[column].map(col_dict).fillna(input_df[column])

    # Predict the value of the selected feature based on the input values and desired prediction value for Attrition
    feature_prediction = _new_model.predict(input_df)

    return feature_prediction

# Streamlit UI for scenario testing
st.title("Attrition Prediction - Scenario Testing")

# Allow user to select a feature as the new target variable
feature_to_predict = st.selectbox('Select Feature to Predict:', X.columns)

# Data Splitting
X_new = total_current_staffs_features.drop([feature_to_predict], axis=1)
y_new = total_current_staffs_features[feature_to_predict]
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=2)


# Allow user to input values for all features except the feature to be predicted (feature_to_predict)
input_values = {}
for column_name in no_attrition.columns:
    if column_name != feature_to_predict:
        if no_attrition[column_name].dtype == 'object':
            # If the column is categorical (object type), use dropdown
            input_values[column_name] = st.selectbox(f"Select value for {column_name}", no_attrition[column_name].unique())
        else:
            # If the column is numerical, allow user to input values
            input_values[column_name] = st.number_input(f"Enter value for {column_name}", step=0.01)
       
desired_attrition_prediction = st.number_input('Desired Prediction for Attrition', value=0.035, step=0.001)
input_values['Prediction'] = desired_attrition_prediction

# Train the model
_new_model = train_new_model(X_train_new, y_train_new)

# Create a button for scenario testing
if st.button('Run Scenario Test'):
    # Call the function for scenario testing
    feature_prediction = run_scenario_test(_new_model, input_values, feature_to_predict, desired_attrition_prediction)

    # Display the result
    st.write(f"Predicted value of {feature_to_predict}: {feature_prediction}")

# SSR Prediction
st.title("Student-Staff-Ratio Determination")
SSR_data = st.file_uploader('Upload Staff-Student-Data', type=['csv', 'xlsx'])

if SSR_data is not None:
    SSR_data = pd.read_excel(SSR_data)  # Read Excel file
    
    # Selecting the column that covers the period of data (e.g., 'Year')
    selected_feature = st.selectbox("Select the column that covers the period of data, e.g., 'Year', 'Day'", options=SSR_data.columns)
    
    # Selecting a unique value from the selected feature to replace '2024'
    unique_values = SSR_data[selected_feature].unique()
    unique_value_selected = st.selectbox(f"Select the '{selected_feature}' you want to predict", options=unique_values)
    
    # Selecting the target feature/column to predict
    target_feature = st.selectbox("Select the target feature/column you want to predict", options=SSR_data.columns)

    # Filtering the historical data based on the selected feature and the unique value
    historical_data = SSR_data[SSR_data[selected_feature] != unique_value_selected]
    data_2024 = SSR_data[SSR_data[selected_feature] == unique_value_selected]
    st.write(historical_data)
else:
    st.write("No file uploaded. Please upload a file.")


# Define the features (independent variables) and target (dependent variables)
input_features = SSR_data.drop([target_feature], axis=1)



# Create a list to store the predicted values
professional_predictions = []

# Loop through each 'Subject_Group' and predict Total_FTE for 2024
for group, group_data in historical_data.groupby('Subject_Group'):
    # Extract the features and target data for the group
    X = group_data[input_features]
    y = group_data[target_feature]

    # Check if there are matching records for this 'Subject_Group' in 2024
    # Create a list to store the predicted values
professional_predictions = []

# Loop through each 'Subject_Group' and predict Total_FTE for 2024
for group, group_data in historical_data.groupby('Subject_Group'):
    # Extract the features and target data for the group
    X = group_data[input_features]
    y = group_data[target_feature]

    # Check if there are matching records for this 'Subject_Group' in 2024
    if group in data_2024['Subject_Group'].values:
        # Extract the features for predicting 'Total_FTE' in 2024
        X_2024 = data_2024[data_2024['Subject_Group'] == group][input_features]

        # Get common columns between training and prediction datasets
        common_columns = X.columns.intersection(X_2024.columns)

        # Ensure both datasets have the same columns
        X = X[common_columns]
        X_2024 = X_2024[common_columns]

        # Creating model and fitting it to the professional_df
        model = XGBRegressor(learning_rate=0.1, max_depth=2, reg_alpha=0.4, reg_lambda=0.4, random_state=2)
        model.fit(X, y)

        # Using the model to predict 'Total_FTE' for 2024
        predicted_values = model.predict(X_2024)

        # Append the predicted values to professional_predictions list
        professional_predictions.append(predicted_values)

        # Append the prediction to the list
        #professional_predictions.append(prediction)

# Convert the list of predictions to a DataFrame
predicted_fte_df = pd.DataFrame(professional_predictions)


#Model Evaluation

# Storing best prediction values
y_pred = model.predict(X_2024)
train_y_pred = model.predict(X)

#Storing Training Performance Metrics
train_rmse = mean_squared_error(y, train_y_pred, squared=False)
train_r2 = r2_score(y, train_y_pred)
train_mae = mean_absolute_error(y, train_y_pred)

#Printing training and Evaluation Metrics
print(f"Training RMSE: {train_rmse:.4f}")
#print(f"Test RMSE with best model: {eval_rmse_XGB:.4f}")
print(f"Training r2_score: {train_r2:.4f}")
#print(f"Test r2_score with best model: {eval_r2_XGB:.4f}")
print(f"Training MAE: {train_mae:.4f}")
#print(f"Test MAE with best model: {eval_mae_XGB:.4f}") 

# Update the original 'student_staff_data' with the predicted values for 2024
predicted_student_staff_data = pd.concat([historical_data, predicted_fte_df], ignore_index=True)



# Define the 'Subject_Group' and year for which you want to create the plot
subject_group = "Music"
years = range(2019, 2025)  # Assuming you want to plot from 2019 to 2024

# Filter the data for the specified 'Subject_Group'
filtered_data = predicted_student_staff_data[predicted_student_staff_data['Subject_Group'] == subject_group]

# Extract 'Total_Staff' and 'Total_Students' values for the specified 'Subject_Group'
staff_values = filtered_data['Total_FTE']
student_values = filtered_data['Total_Students']

# Create a line plot to visualize changes over the years
plt.figure(figsize=(10, 6))
plt.plot(years, staff_values, marker='o', label='Total_FTE')
plt.plot(years, student_values, marker='o', label='Total_Students')
plt.title(f'{subject_group} - FTE and Students Over the Years')
plt.xlabel('Year')
plt.ylabel('Values')
plt.legend()
plt.grid(True)

# Add text annotations for each data point
for year, staff, students in zip(years, staff_values, student_values):
    plt.annotate(f'{staff:.2f}', (year, staff), textcoords="offset points", xytext=(0,10), ha='center')
    plt.annotate(f'{students:.0f}', (year, students), textcoords="offset points", xytext=(0,-20), ha='center')

# Show the plot
plt.show()