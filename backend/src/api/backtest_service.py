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

                bets.append({
                    "race_id": race_id,
                    "discipline": str(best_horse['discipline']),
                    "month": int(best_horse['program_date'].month),
                    "payout_sg": payout_sg,
                    "payout_sp": payout_sp,
                    "win": is_win,
                    "placed": is_placed,
                    "odds": float(best_horse['effective_odds'])
                })

            bets_df = pd.DataFrame(bets)
            roi_sg, roi_sp, win_rate, place_rate = 0.0, 0.0, 0.0, 0.0
            seasonal = {}

            if not bets_df.empty and len(bets_df) >= MIN_BETS_FOR_RELIABLE_TRAINER:
                roi_sg = (bets_df['payout_sg'].sum() - len(bets_df)) / len(bets_df) * 100
                roi_sp = (bets_df['payout_sp'].sum() - len(bets_df)) / len(bets_df) * 100
                win_rate = bets_df['win'].mean() * 100
                place_rate = bets_df['placed'].mean() * 100

                # GROUPBY MULTI-INDEX FIX: Re-structuration manuelle pour JSON
                seasonal_raw = bets_df.groupby(['discipline', 'month']).apply(lambda x: {
                    "roi": float(((x['payout_sg'].sum() - len(x)) / len(x) * 100) if len(x) > 0 else 0),
                    "roi_place": float(((x['payout_sp'].sum() - len(x)) / len(x) * 100) if len(x) > 0 else 0),
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

        return {
            "trainers": model_results,
            "strategies": {
                "sniper": sniper_results,
                "kelly": kelly_results
            }
        }
