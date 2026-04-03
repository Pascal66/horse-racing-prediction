import optuna
from xgboost import XGBClassifier


def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 1000, 4000),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
        "tree_method": "gpu_hist"
    }

    model = XGBClassifier(**params)

    model.fit(X_train_enc, y_train,
              eval_set=[(X_val_enc, y_val)],
              early_stopping_rounds=50,
              verbose=False)

    preds = model.predict_proba(X_val_enc)[:, 1]
    return log_loss(y_val, preds)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

study.best_params
