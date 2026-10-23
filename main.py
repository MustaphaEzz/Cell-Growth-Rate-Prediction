# =====================================================================
# MAIN PROGRAM
# Your code starts here. This is the file that will be run.
# =====================================================================


import os
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Load the dataset (Unit 3: Pandas)
df = pd.read_csv(os.path.join('datasets', 'dataset_cell_growth_4factor.csv'))
features = df.columns[1:]
target = df.columns[0]
x = df.loc[:, features].to_numpy()
y = df.loc[:, target].to_numpy()

# Split the data (Unit 4: Linear Regression)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

# Polynomial regression and hyperparameter optimization (Unit 4: Example 08)
model_poly = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(include_bias=False),
    LinearRegression()
)
param_grid_poly = dict(polynomialfeatures__degree=[i for i in range(1, 11)])
model_gscv_poly = GridSearchCV(
    model_poly, param_grid=param_grid_poly,
    cv=4, scoring='neg_root_mean_squared_error', return_train_score=True
)
model_gscv_poly.fit(x_train, y_train)

# KNN regression and hyperparameter optimization (Unit 4: Example 09)
model_knn = make_pipeline(StandardScaler(), KNeighborsRegressor())
param_grid_knn = dict(
    kneighborsregressor__n_neighbors=[i for i in range(1, 81)],
    kneighborsregressor__weights=['uniform', 'distance']
)
model_gscv_knn = GridSearchCV(
    model_knn, param_grid=param_grid_knn,
    cv=4, scoring='neg_root_mean_squared_error', return_train_score=True
)
model_gscv_knn.fit(x_train, y_train)

# Compare final models on the held-out test data (Unit 4: Example 10)
results = []
for name, model in [('Polynomial', model_gscv_poly), ('KNN', model_gscv_knn)]:
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print('\n', name)
    print('Best parameters:', model.best_params_)
    print('CV RMSE:', -model.best_score_)
    print('Training RMSE:', root_mean_squared_error(y_train, y_train_pred))
    print('Testing RMSE:', root_mean_squared_error(y_test, y_test_pred))
    print('Testing R2:', r2_score(y_test, y_test_pred))
    results.append({
        'Model': name,
        'CV RMSE': -model.best_score_,
        'Test RMSE': root_mean_squared_error(y_test, y_test_pred),
        'Test R2': r2_score(y_test, y_test_pred)
    })

# Export comparison tables (Unit 3: Pandas; Unit 4: Examples 08 and 09)
os.makedirs('tables', exist_ok=True)
os.makedirs('figures', exist_ok=True)
pd.DataFrame(results).to_csv(os.path.join('tables', 'model_comparison.csv'), index=False)

df_poly = pd.DataFrame(model_gscv_poly.cv_results_)
df_poly.loc[:, [
    'param_polynomialfeatures__degree', 'mean_train_score', 'mean_test_score',
    'std_test_score'
]].to_csv(os.path.join('tables', 'polynomial_comparison.csv'), index=False)

df_knn = pd.DataFrame(model_gscv_knn.cv_results_)
df_knn.loc[:, [
    'param_kneighborsregressor__n_neighbors',
    'param_kneighborsregressor__weights',
    'mean_train_score', 'mean_test_score', 'std_test_score'
]].to_csv(os.path.join('tables', 'knn_comparison.csv'), index=False)

# Polynomial validation curve (Unit 4: Example 08)
degrees = df_poly.loc[:, 'param_polynomialfeatures__degree'].to_numpy()
rmse_train = -df_poly.loc[:, 'mean_train_score'].to_numpy()
rmse_valid = -df_poly.loc[:, 'mean_test_score'].to_numpy()
fig, ax = plt.subplots()
ax.plot(degrees[:4], rmse_train[:4], label='Train (Poly)')
ax.plot(degrees[:4], rmse_valid[:4], label='Valid (Poly)')
ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('Growth Rate RMSE [d^-1]')
ax.legend()
fig.savefig(os.path.join('figures', 'polynomial_comparison.png'))
plt.close(fig)

# KNN validation curve (Unit 4: Example 09)
fig, ax = plt.subplots()
for weight in param_grid_knn['kneighborsregressor__weights']:
    df_sub = df_knn.loc[df_knn['param_kneighborsregressor__weights'] == weight, :]
    df_sub = df_sub.sort_values(by=['param_kneighborsregressor__n_neighbors'])
    neighbors = df_sub.loc[:, 'param_kneighborsregressor__n_neighbors'].to_numpy()
    rmse_train = -df_sub.loc[:, 'mean_train_score'].to_numpy()
    rmse_valid = -df_sub.loc[:, 'mean_test_score'].to_numpy()
    ax.plot(neighbors, rmse_train, label='Train (KNN: ' + weight + ')')
    ax.plot(neighbors, rmse_valid, label='Valid (KNN: ' + weight + ')')
ax.set_xlabel('Number of Neighbors (k)')
ax.set_ylabel('Growth Rate RMSE [d^-1]')
ax.legend()
fig.savefig(os.path.join('figures', 'knn_comparison.png'))
plt.close(fig)

# Actual versus predicted values (Unit 3: Matplotlib scatter plots)
y_pred = model_gscv_knn.predict(x_test)
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred,alpha=0.6)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
ax.set_xlabel('Actual Growth Rate [d^-1]')
ax.set_ylabel('Predicted Growth Rate [d^-1]')
fig.savefig(os.path.join('figures', 'knn_predictions.png'))
plt.close(fig)

# High and low observed growth conditions (Unit 3: Pandas)
df_sorted = df.sort_values(by=[target], ascending=False)
df_sorted.head(10).to_csv(os.path.join('tables', 'high_growth_conditions.csv'), index=False)

