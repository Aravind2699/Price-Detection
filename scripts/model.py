from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor

def build_pipeline():
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        # ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ('model', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42))
    ])
    return pipeline