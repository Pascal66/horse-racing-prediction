C:\Users\hmaro\AppData\Roaming\Python\Scripts\uv.exe run F:/git/horse-racing-prediction/.venv/Scripts/python.exe F:\git\horse-racing-prediction\backend\src\ml\trainer_tabnet_gpu.py 
2026-03-29 13:13:20,645 - ML.HyperStackTrainer - INFO - --- STARTING TABNET TOURNAMENT ---
2026-03-29 13:13:20,675 - ML.Loader - INFO - Loading SQL data...
2026-03-29 13:13:23,922 - ML.Loader - INFO - Calculating historical statistics...
2026-03-29 13:13:25,042 - ML.Loader - INFO - Merging datasets...
2026-03-29 13:13:25,342 - ML.HyperStackTrainer - INFO - Data Loaded: (537733, 32) rows
2026-03-29 13:13:25,375 - ML.HyperStackTrainer - INFO - --- Training Target: STEEPLECHASE ---
2026-03-29 13:15:00,364 - ML.HyperStackTrainer - INFO - Training Fold 5/5 for steeplechase...


2026-03-29 13:15:44,695 - ML.HyperStackTrainer - INFO - Target steeplechase Metrics: {'logloss': 0.36967475639312486, 'auc': 0.713632669646485, 'roi': np.float64(-2.6562345262059817e-14), 'win_rate': np.float64(0.9999999999999998), 'avg_odds': np.float64(16.079760319573904), 'count': 107}
2026-03-31 13:40:07,312 - ML.TabnetTrainer - INFO - Target steeplechase Metrics: {'logloss': 0.375828082409402, 'auc': 0.6996581196581196, 'roi': np.float64(-15.462962962962978), 'win_rate': np.float64(0.8453703703703702), 'avg_odds': np.float64(16.098416886543536), 'count': 108}
--- TABNET FEATURE IMPORTANCES (26 features) ---
                       feature  importance
         num__market_sentiment    0.192209
           num__reference_odds    0.096820
           cat__racetrack_code    0.089253
   num__declared_runners_count    0.079529
             cat__trainer_name    0.058907
  num__reference_odds_rel_race    0.057906
    num__winnings_rank_in_race    0.050266
            num__hist_earnings    0.050073
           num__hist_avg_speed    0.038575
                 num__odds_log    0.027077
              num__is_debutant    0.026749
    num__hist_avg_speed_z_race    0.025421
        num__odds_rank_in_race    0.025127
              cat__jockey_name    0.021933
num__career_winnings_rank_race    0.020208
               num__distance_m    0.019565
                      cat__sex    0.019445
 num__reference_odds_rank_race    0.015514
          num__career_winnings    0.014812
               cat__discipline    0.014300
            cat__terrain_label    0.012176
        num__horse_age_at_race    0.010852
           cat__shoeing_status    0.009901
        num__relative_winnings    0.008374
               num__race_month    0.008236
               cat__track_type    0.006774
--------------------------------------------------

2026-03-29 13:15:46,629 - ML.HyperStackTrainer - INFO - --- Training Target: HAIE ---
2026-03-29 13:19:32,654 - ML.HyperStackTrainer - INFO - Target haie Metrics: {'logloss': 0.3258196666776442, 'auc': 0.7400857346987377, 'roi': np.float64(15.882352941176453), 'win_rate': np.float64(1.1588235294117646), 'avg_odds': np.float64(19.583345195729535), 'count': 170}
--- TABNET FEATURE IMPORTANCES (26 features) ---
                       feature  importance
         num__market_sentiment    0.166323
                 num__odds_log    0.084581
             cat__trainer_name    0.073777
            num__hist_earnings    0.066279
        num__odds_rank_in_race    0.065463
           cat__racetrack_code    0.059526
  num__reference_odds_rel_race    0.052674
          num__career_winnings    0.038663
              num__is_debutant    0.034183
           num__reference_odds    0.034021
num__career_winnings_rank_race    0.032560
    num__winnings_rank_in_race    0.031640
                      cat__sex    0.029587
        num__horse_age_at_race    0.028730
   num__declared_runners_count    0.028687
               num__distance_m    0.027203
               cat__track_type    0.022851
              cat__jockey_name    0.021271
           num__hist_avg_speed    0.019095
            cat__terrain_label    0.017812
               cat__discipline    0.013857
           cat__shoeing_status    0.012736
 num__reference_odds_rank_race    0.012539
        num__relative_winnings    0.010398
               num__race_month    0.009894
    num__hist_avg_speed_z_race    0.005650
--------------------------------------------------

2026-03-29 13:19:36,135 - ML.HyperStackTrainer - INFO - --- Training Target: MONTE ---
2026-03-29 13:26:11,316 - ML.HyperStackTrainer - INFO - Target monte Metrics: {'logloss': 0.30654894981210956, 'auc': 0.8109439657270825, 'roi': np.float64(36.18367346938777), 'win_rate': np.float64(1.3618367346938778), 'avg_odds': np.float64(27.13864421910727), 'count': 245}

--- TABNET FEATURE IMPORTANCES (26 features) ---
                       feature  importance
         num__market_sentiment    0.206012
    num__hist_avg_speed_z_race    0.126822
           cat__racetrack_code    0.083659
           num__reference_odds    0.079697
        num__odds_rank_in_race    0.047025
            num__hist_earnings    0.045266
                 num__odds_log    0.034889
  num__reference_odds_rel_race    0.031896
              cat__jockey_name    0.031733
           cat__shoeing_status    0.029453
          num__career_winnings    0.028574
 num__reference_odds_rank_race    0.028031
        num__horse_age_at_race    0.028010
    num__winnings_rank_in_race    0.025268
           num__hist_avg_speed    0.023638
        num__relative_winnings    0.017261
   num__declared_runners_count    0.017046
              num__is_debutant    0.016843
             cat__trainer_name    0.016430
               num__distance_m    0.015629
num__career_winnings_rank_race    0.015568
               cat__discipline    0.013660
               num__race_month    0.012205
                      cat__sex    0.010685
               cat__track_type    0.008022
            cat__terrain_label    0.006678
--------------------------------------------------

2026-03-29 13:26:16,330 - ML.HyperStackTrainer - INFO - --- Training Target: ATTELE ---
2026-03-29 14:09:26,120 - ML.HyperStackTrainer - INFO - Target attele Metrics: {'logloss': 0.2622677201254551, 'auc': 0.8254936099585063, 'roi': np.float64(58.31258644536654), 'win_rate': np.float64(1.5831258644536654), 'avg_odds': np.float64(37.56768627433849), 'count': 1446}
--- TABNET FEATURE IMPORTANCES (26 features) ---
                       feature  importance
         num__market_sentiment    0.237165
    num__hist_avg_speed_z_race    0.123309
           cat__shoeing_status    0.058297
            num__hist_earnings    0.056457
              cat__jockey_name    0.055886
             cat__trainer_name    0.051199
           num__hist_avg_speed    0.050074
          num__career_winnings    0.046081
                 num__odds_log    0.043000
        num__odds_rank_in_race    0.041204
           cat__racetrack_code    0.038959
        num__relative_winnings    0.038794
           num__reference_odds    0.036623
        num__horse_age_at_race    0.023540
  num__reference_odds_rel_race    0.021244
 num__reference_odds_rank_race    0.014894
               cat__track_type    0.013834
   num__declared_runners_count    0.011520
            cat__terrain_label    0.010648
              num__is_debutant    0.008318
num__career_winnings_rank_race    0.007635
               num__distance_m    0.004064
    num__winnings_rank_in_race    0.002901
               cat__discipline    0.002164
                      cat__sex    0.001629
               num__race_month    0.000561
--------------------------------------------------

2026-03-29 14:09:59,980 - ML.HyperStackTrainer - INFO - --- Training Target: PLAT ---
2026-03-29 14:43:05,160 - ML.HyperStackTrainer - INFO - Target plat Metrics: {'logloss': 0.2638083698610925, 'auc': 0.7398846747813838, 'roi': np.float64(-8.958837772397121), 'win_rate': np.float64(0.9104116222760288), 'avg_odds': np.float64(22.917967522262966), 'count': 826}
--- TABNET FEATURE IMPORTANCES (26 features) ---
                       feature  importance
         num__market_sentiment    0.280017
            num__hist_earnings    0.190387
                 num__odds_log    0.088147
           num__reference_odds    0.076545
          num__career_winnings    0.071180
        num__odds_rank_in_race    0.042753
             cat__trainer_name    0.041765
           cat__racetrack_code    0.041764
  num__reference_odds_rel_race    0.024556
              cat__jockey_name    0.021989
               cat__discipline    0.016928
   num__declared_runners_count    0.015787
        num__horse_age_at_race    0.013045
        num__relative_winnings    0.008621
num__career_winnings_rank_race    0.008080
                      cat__sex    0.007917
            cat__terrain_label    0.007640
               num__distance_m    0.007425
           num__hist_avg_speed    0.007311
              num__is_debutant    0.006627
               num__race_month    0.005154
           cat__shoeing_status    0.004053
    num__winnings_rank_in_race    0.004005
    num__hist_avg_speed_z_race    0.003427
 num__reference_odds_rank_race    0.003047
               cat__track_type    0.001831
--------------------------------------------------

2026-03-29 14:43:29,044 - ML.HyperStackTrainer - INFO - --- Training Target: CROSS ---
2026-03-29 14:43:47,331 - ML.HyperStackTrainer - INFO - Target cross Metrics: {'logloss': 0.35125109651635616, 'auc': 0.8256880733944955, 'roi': np.float64(-1.818181818181828), 'win_rate': np.float64(0.9818181818181817), 'avg_odds': np.float64(14.633587786259541), 'count': 22}

--- TABNET FEATURE IMPORTANCES (26 features) ---
                       feature  importance
         num__market_sentiment    0.099431
  num__reference_odds_rel_race    0.092730
 num__reference_odds_rank_race    0.059665
           num__reference_odds    0.057236
        num__odds_rank_in_race    0.050781
    num__winnings_rank_in_race    0.050715
           cat__racetrack_code    0.048908
num__career_winnings_rank_race    0.036940
                 num__odds_log    0.036157
        num__horse_age_at_race    0.035928
            num__hist_earnings    0.035769
   num__declared_runners_count    0.035513
               num__distance_m    0.031840
        num__relative_winnings    0.030610
             cat__trainer_name    0.030240
           num__hist_avg_speed    0.027785
          num__career_winnings    0.027587
                      cat__sex    0.027048
            cat__terrain_label    0.026624
               cat__track_type    0.026456
               num__race_month    0.025893
    num__hist_avg_speed_z_race    0.024897
           cat__shoeing_status    0.024826
              cat__jockey_name    0.021124
               cat__discipline    0.018993
              num__is_debutant    0.016303
--------------------------------------------------

