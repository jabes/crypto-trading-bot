import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense

# Load the data
df = pd.read_csv('bitcoin_data.csv')

# Prepare the data
X = df.drop(['price'], axis=1)
y = df['price']

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train the LightGBM model
lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
gbm = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=[lgb_train, lgb_test], early_stopping_rounds=50, verbose_eval=100)

# Make predictions with LightGBM model
y_pred_lgb = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# Train a Keras neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Make predictions with Keras neural network model
y_pred_nn = model.predict(X_test)

# Evaluate the models
print('LightGBM RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred_lgb)))
print('Keras Neural Network RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred_nn)))
