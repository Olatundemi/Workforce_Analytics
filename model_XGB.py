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
