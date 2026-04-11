# backend/src/api/backtest_service.py
import logging

import pandas as pd
from typing import  Dict, Any
from .repositories import RaceRepository
from .kelly_multi_races import analyze_multiple_races

logger = logging.getLogger(__name__)
MIN_BETS_FOR_RELIABLE_TRAINER = 50  # Ignore les vieux modèles polluants

class BacktestService:
    def __init__(self, repository: RaceRepository):
        self.repository = repository

    def run_backtest(self) -> Dict[str, Any]:
        raw_data = []
        try:
            raw_data = self.repository.get_backtest_data()
        except Exception as e:
            logger.error(f"Backtest repository call failed: {e}", exc_info=True)

        if not raw_data:
            return {"error": "No data available for backtesting"}

        df = pd.DataFrame(raw_data)

        # Suppression des lignes sans prédiction
        df = df.dropna(subset=['proba_winner'])
        if df.empty:
            return {"error": "No predictions found in data"}

        # Ensure correct types
        df['program_date'] = pd.to_datetime(df['program_date'])
        df['win_probability'] = pd.to_numeric(df['proba_winner'], errors='coerce').fillna(0.0)
        
        if 'proba_top3_place' in df.columns:
            df['place_probability'] = pd.to_numeric(df['proba_top3_place'], errors='coerce').fillna(0.0)
        else:
            df['place_probability'] = 0.0

        df['dividend_per_1e'] = pd.to_numeric(df['dividend_per_1e'], errors='coerce').fillna(0.0)
        df['finish_rank'] = pd.to_numeric(df['finish_rank'], errors='coerce').fillna(0).astype(int)
        df['effective_odds'] = df['live_odds'].fillna(df['reference_odds']).fillna(1.0).clip(lower=1.05)

        model_results = {}
        for model in df['model_version'].unique():
            if not model: continue
            model_df = df[df['model_version'] == model].copy()

            bets = []
            for race_id, group in model_df.groupby('race_id'):
                if group['win_probability'].isna().all():
                    continue
                    
                best_idx = group['win_probability'].idxmax()
                best_horse = group.loc[best_idx]

                is_win = int(best_horse['finish_rank']) == 1
                is_placed = int(best_horse['finish_rank']) <= 3

                # Dividende SG
                win_div = group[(group['bet_type'] == 'E_SIMPLE_GAGNANT') & 
                                (group['combination'] == str(best_horse['program_number']))]['dividend_per_1e']
                payout_sg = float(win_div.iloc[0]) if not win_div.empty else (float(best_horse['effective_odds']) if is_win else 0.0)

                # Dividende SP
                place_div = group[(group['bet_type'] == 'E_SIMPLE_PLACE') & 
                                  (group['combination'] == str(best_horse['program_number']))]['dividend_per_1e']
                payout_sp = float(place_div.iloc[0]) if not place_div.empty else (1.1 if is_placed else 0.0)
                # --- Multi-bets (Couple, Trio) ---
                payout_cg, payout_cp, payout_trio = 0.0, 0.0, 0.0

                # CG: Top 2 by win_probability
                top2_win = group.nlargest(2, 'win_probability')
                if len(top2_win) == 2:
                    nums = set(map(int, top2_win['program_number']))
                    if (top2_win['finish_rank'].between(1, 2)).all():
                        cg_rows = group[group['bet_type'].str.contains('COUPLE_GAGNANT', na=False)]
                        for _, row in cg_rows.iterrows():
                            try:
                                comb_nums = set(map(int, str(row['combination']).replace('-', ' ').split()))
                                if comb_nums == nums:
                                    payout_cg = float(row['dividend_per_1e'])
                                    break
                            except:
                                continue

                # CP: Top 2 by place_probability
                top2_place = group.nlargest(2, 'place_probability')
                if len(top2_place) == 2:
                    nums = set(map(int, top2_place['program_number']))
                    if (top2_place['finish_rank'].between(1, 3)).all():
                        cp_rows = group[group['bet_type'].str.contains('COUPLE_PLACE', na=False)]
                        for _, row in cp_rows.iterrows():
                            try:
                                comb_nums = set(map(int, str(row['combination']).replace('-', ' ').split()))
                                if nums.issubset(comb_nums):
                                    payout_cp = float(row['dividend_per_1e'])
                                    break
                            except:
                                continue

                # Trio: Top 3 by win_probability
                top3_win = group.nlargest(3, 'win_probability')
                if len(top3_win) == 3:
                    nums = set(map(int, top3_win['program_number']))
                    if (top3_win['finish_rank'].between(1, 3)).all():
                        trio_rows = group[
                            group['bet_type'].str.contains('TRIO', na=False) & ~group['bet_type'].str.contains('ORDRE',
                                                                                                               na=False)]
                        for _, row in trio_rows.iterrows():
                            try:
                                comb_nums = set(map(int, str(row['combination']).replace('-', ' ').split()))
                                if comb_nums == nums:
                                    payout_trio = float(row['dividend_per_1e'])
                                    break
                            except:
                                continue

                bets.append({
                    "race_id": race_id,
                    "discipline": str(best_horse['discipline']),
                    "month": int(best_horse['program_date'].month),
                    "payout_sg": payout_sg,
                    "payout_sp": payout_sp,
                    "payout_cg": payout_cg,
                    "payout_cp": payout_cp,
                    "payout_trio": payout_trio,
                    "win": is_win,
                    "placed": is_placed,
                    "odds": float(best_horse['effective_odds'])
                })

            bets_df = pd.DataFrame(bets)
            roi_sg, roi_sp, roi_cg, roi_cp, roi_trio = 0.0, 0.0, 0.0, 0.0, 0.0
            win_rate, place_rate = 0.0, 0.0
            seasonal = {}

            if not bets_df.empty and len(bets_df) >= MIN_BETS_FOR_RELIABLE_TRAINER:
                roi_sg = (bets_df['payout_sg'].sum() - len(bets_df)) / len(bets_df) * 100
                roi_sp = (bets_df['payout_sp'].sum() - len(bets_df)) / len(bets_df) * 100
                roi_cg = (bets_df['payout_cg'].sum() - len(bets_df)) / len(bets_df) * 100
                roi_cp = (bets_df['payout_cp'].sum() - len(bets_df)) / len(bets_df) * 100
                roi_trio = (bets_df['payout_trio'].sum() - len(bets_df)) / len(bets_df) * 100
                win_rate = bets_df['win'].mean() * 100
                place_rate = bets_df['placed'].mean() * 100

                # GROUPBY MULTI-INDEX FIX: Re-structuration manuelle pour JSON
                seasonal_raw = bets_df.groupby(['discipline', 'month']).apply(lambda x: {
                    "roi": float(((x['payout_sg'].sum() - len(x)) / len(x) * 100) if len(x) > 0 else 0),
                    "roi_place": float(((x['payout_sp'].sum() - len(x)) / len(x) * 100) if len(x) > 0 else 0),
                    "roi_cg": float(((x['payout_cg'].sum() - len(x)) / len(x) * 100) if len(x) > 0 else 0),
                    "roi_cp": float(((x['payout_cp'].sum() - len(x)) / len(x) * 100) if len(x) > 0 else 0),
                    "roi_trio": float(((x['payout_trio'].sum() - len(x)) / len(x) * 100) if len(x) > 0 else 0),
                    "win_rate": float(x['win'].mean() * 100) if len(x) > 0 else 0,
                    "count": int(len(x)),
                    "avg_odds": float(x['odds'].mean())
                }, include_groups=False)

                for (disc, mon), metrics in seasonal_raw.items():
                    d_key = str(disc)
                    if d_key not in seasonal: seasonal[d_key] = {}
                    seasonal[d_key][int(mon)] = metrics

                model_results[str(model)] = {
                    "roi": float(roi_sg),
                    "roi_place": float(roi_sp),
                    "roi_cg": float(roi_cg),
                    "roi_cp": float(roi_cp),
                    "roi_trio": float(roi_trio),
                    "win_rate": float(win_rate),
                    "place_rate": float(place_rate),
                    "total_bets": int(len(bets_df)),
                    "avg_odds": float(bets_df['odds'].mean()),
                    "seasonal_analysis": seasonal
                }

        # --- Sniper Strategy Evaluation ---
        df['implied_prob'] = 1 / df['effective_odds'].replace(0, 100)
        df['edge'] = df['win_probability'] - df['implied_prob']

        sniper_mask = (df['edge'] >= 0.05) & (df['effective_odds'] >= 2.0) & (df['effective_odds'] <= 25.0)
        sniper_bets_df = df[sniper_mask].copy()

        sniper_final_bets = []
        for race_id, group in sniper_bets_df.groupby('race_id'):
            best = group.sort_values('win_probability', ascending=False).iloc[0]
            is_win = int(best['finish_rank']) == 1
            win_div = group[(group['bet_type'] == 'E_SIMPLE_GAGNANT') & (group['combination'] == str(best['program_number']))]['dividend_per_1e']
            payout = float(win_div.iloc[0]) if not win_div.empty else (float(best['effective_odds']) if is_win else 0.0)
            sniper_final_bets.append({"payout": payout, "win": is_win})

        sniper_results = {}
        if sniper_final_bets:
            s_df = pd.DataFrame(sniper_final_bets)
            sniper_results = {
                "roi": float((s_df['payout'].sum() - len(s_df)) / len(s_df) * 100),
                "win_rate": float(s_df['win'].mean() * 100),
                "total_bets": int(len(s_df))
            }

        # --- Kelly Strategy Evaluation ---
        kelly_results = {}
        available_models = [m for m in df['model_version'].unique() if m]
        if available_models:
            sim_model = 'global' if 'global' in available_models else available_models[0]
            sim_df = df[df['model_version'] == sim_model].copy()
            sim_df['live_odds'] = sim_df['effective_odds']

            current_bankroll = 1000.0
            total_profit = 0.0
            total_staked = 0.0
            horses_per_race = []
            kelly_count = 0

            for _, day_df in sim_df.groupby(sim_df['program_date'].dt.date):
                for race_id, race_df in day_df.groupby('race_id'):
                    res = analyze_multiple_races(race_df, bankroll=current_bankroll, kelly_fraction=0.5)
                    if not res.get('fractions'): continue

                    horses_per_race.append(len(res['fractions']))
                    for prog_num, fraction in res['fractions'].items():
                        if fraction <= 0: continue
                        stake = current_bankroll * fraction
                        horse = race_df[race_df['program_number'] == int(prog_num)].iloc[0]
                        is_win = int(horse['finish_rank']) == 1
                        win_div = race_df[(race_df['bet_type'] == 'E_SIMPLE_GAGNANT') & (race_df['combination'] == str(prog_num))]['dividend_per_1e']
                        odds_payout = float(win_div.iloc[0]) if not win_div.empty else (float(horse['effective_odds']) if is_win else 0.0)
                        
                        profit = (stake * odds_payout) - stake
                        current_bankroll += profit
                        total_profit += profit
                        total_staked += stake
                        kelly_count += 1

            if kelly_count > 0:
                kelly_results = {
                    "roi": float((total_profit / total_staked) * 100) if total_staked > 0 else 0.0,
                    "total_profit": float(total_profit),
                    "total_bets": int(kelly_count),
                    "final_bankroll": float(current_bankroll),
                    "avg_horses_per_race": float(sum(horses_per_race) / len(horses_per_race)) if horses_per_race else 0.0
                }

        # --- Recommended Composite Strategy Evaluation ---
        composite_results = {}
        composite_bets = []

        # Mapping best algorithms per context
        contexts = df[['discipline', 'program_date']].copy()
        contexts['month'] = contexts['program_date'].dt.month
        unique_contexts = contexts[['discipline', 'month']].drop_duplicates()
        best_algos = {}
        for _, row in unique_contexts.iterrows():
            best_algos[(row['discipline'], row['month'])] = self.repository.get_best_model_for_context(
                row['discipline'], row['month'])

        for race_id, race_group in df.groupby('race_id'):
            disc = race_group['discipline'].iloc[0]
            mon = race_group['program_date'].iloc[0].month
            target_algo = best_algos.get((disc, mon))

            # On cherche le modèle qui correspond au meilleur algo, ou le premier disponible
            if target_algo:
                model_group = race_group[race_group['model_version'].str.contains(target_algo, case=False, na=False)]
            else:
                model_group = pd.DataFrame()

            if model_group.empty:
                # Fallback: on prend le modèle avec la plus haute proba moyenne (arbitraire mais robuste)
                model_group = race_group[race_group['model_version'] == race_group['model_version'].iloc[0]]

            best_idx = model_group['win_probability'].idxmax()
            best_horse = model_group.loc[best_idx]
            is_win = int(best_horse['finish_rank']) == 1
            win_div = model_group[(model_group['bet_type'] == 'E_SIMPLE_GAGNANT') & (
                        model_group['combination'] == str(best_horse['program_number']))]['dividend_per_1e']
            payout_sg = float(win_div.iloc[0]) if not win_div.empty else (
                float(best_horse['effective_odds']) if is_win else 0.0)

            # Reprendre les logiques multi-bets pour le composite
            payout_cg, payout_cp, payout_trio = 0.0, 0.0, 0.0

            top2_win = model_group.nlargest(2, 'win_probability')
            if len(top2_win) == 2:
                nums = set(map(int, top2_win['program_number']))
                if (top2_win['finish_rank'].between(1, 2)).all():
                    cg_rows = model_group[model_group['bet_type'].str.contains('COUPLE_GAGNANT', na=False)]
                    for _, row in cg_rows.iterrows():
                        try:
                            if set(map(int, str(row['combination']).replace('-', ' ').split())) == nums:
                                payout_cg = float(row['dividend_per_1e'])
                                break
                        except:
                            continue

            top2_place = model_group.nlargest(2, 'place_probability')
            if len(top2_place) == 2:
                nums = set(map(int, top2_place['program_number']))
                if (top2_place['finish_rank'].between(1, 3)).all():
                    cp_rows = model_group[model_group['bet_type'].str.contains('COUPLE_PLACE', na=False)]
                    for _, row in cp_rows.iterrows():
                        try:
                            if nums.issubset(set(map(int, str(row['combination']).replace('-', ' ').split()))):
                                payout_cp = float(row['dividend_per_1e'])
                                break
                        except:
                            continue

            top3_win = model_group.nlargest(3, 'win_probability')
            if len(top3_win) == 3:
                nums = set(map(int, top3_win['program_number']))
                if (top3_win['finish_rank'].between(1, 3)).all():
                    trio_rows = model_group[
                        model_group['bet_type'].str.contains('TRIO', na=False) & ~model_group['bet_type'].str.contains(
                            'ORDRE', na=False)]
                    for _, row in trio_rows.iterrows():
                        try:
                            if set(map(int, str(row['combination']).replace('-', ' ').split())) == nums:
                                payout_trio = float(row['dividend_per_1e'])
                                break
                        except:
                            continue

            composite_bets.append({
                "payout_sg": payout_sg,
                "payout_cg": payout_cg,
                "payout_cp": payout_cp,
                "payout_trio": payout_trio,
                "win": is_win
            })

        if composite_bets:
            c_df = pd.DataFrame(composite_bets)
            composite_results = {
                "roi": float((c_df['payout_sg'].sum() - len(c_df)) / len(c_df) * 100),
                "roi_cg": float((c_df['payout_cg'].sum() - len(c_df)) / len(c_df) * 100),
                "roi_cp": float((c_df['payout_cp'].sum() - len(c_df)) / len(c_df) * 100),
                "roi_trio": float((c_df['payout_trio'].sum() - len(c_df)) / len(c_df) * 100),
                "win_rate": float(c_df['win'].mean() * 100),
                "total_bets": int(len(c_df))
            }

        return {
            "trainers": model_results,
            "strategies": {
                "sniper": sniper_results,
                "kelly": kelly_results,
                "composite": composite_results
            }
        }
