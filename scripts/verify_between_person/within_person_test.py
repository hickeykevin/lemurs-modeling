import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = "logs/verify_between_person/runs/"
rng = np.random.default_rng(0)
for run in ["social_connection_xgboost", "social_connection_z_xgboost", "user_cv_social_connection_xgboost",
            "user_cv_social_connection_z_xgboost", "social_stress_z_xgboost", "user_cv_social_stress_z_xgboost",
            "centered_xgboost", "abl_z_sleep"]:
    df = pd.read_csv(R + run + "/pooled_predictions.csv"); df = df[df.stage == "test"]
    g = [(d.y_true.to_numpy(), d.prob_class_1.to_numpy()) for _, d in df.groupby("app_user_id") if d.y_true.nunique() == 2]
    auc = np.array([roc_auc_score(y, p) for y, p in g])
    boot = [rng.choice(auc, len(auc)).mean() for _ in range(5000)]
    # null: shuffle labels within each person, 2000 times
    null = np.array([np.mean([roc_auc_score(rng.permutation(y), p) for y, p in g]) for _ in range(2000)])
    print(f"{run:40s} within-person {auc.mean():.3f} [{np.percentile(boot,2.5):.2f}, {np.percentile(boot,97.5):.2f}] "
          f"people>0.5: {(auc>0.5).sum()}/{len(auc)}  perm p={(np.sum(null>=auc.mean())+1)/2001:.3f}")
