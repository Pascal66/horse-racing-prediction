from category_encoders import CatBoostEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', CatBoostEncoder(cols=self.categorical_features), self.categorical_features),
        ('num', 'passthrough', self.numerical_features)
    ]
)

model = XGBClassifier(
    n_estimators=3000,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method='hist',
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

model.fit(X_train_enc, y_train, eval_set=[(X_val_enc, y_val)], early_stopping_rounds=100, verbose=False)

calibrated = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
calibrated.fit(X_val_enc, y_val)