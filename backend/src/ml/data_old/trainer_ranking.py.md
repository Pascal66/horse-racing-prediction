from xgboost import XGBRanker

ranker = XGBRanker(
    objective='rank:pairwise',
    n_estimators=2000,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    random_state=42
)

# group = nombre de chevaux par course
group_train = train_df.groupby("race_id").size().to_list()

ranker.fit(
    X_train_enc,
    y_train,
    group=group_train,
    eval_set=[(X_val_enc, y_val)],
    eval_group=[val_df.groupby("race_id").size().to_list()],
    verbose=False
)
scores = ranker.predict(X_test_enc)

# Normalisation intra-course
df['score'] = scores
df['proba'] = df.groupby('race_id')['score'].transform(lambda x: x / x.sum())