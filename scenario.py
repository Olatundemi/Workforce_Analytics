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