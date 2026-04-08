import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .repositories import RaceRepository
from .kelly_multi_races import analyze_multiple_races

class BacktestService:
    def __init__(self, repository: RaceRepository):
        self.repository = repository

    def run_backtest(self) -> Dict[str, Any]:
        raw_data = self.repository.get_backtest_data()
        if not raw_data:
            return {"error": "No data available for backtesting"}

        df = pd.DataFrame(raw_data)

        # Ensure correct types
        df['program_date'] = pd.to_datetime(df['program_date'])
        df['proba_winner'] = df['proba_winner'].astype(float)
        df['dividend_per_1e'] = df['dividend_per_1e'].astype(float)
        df['finish_rank'] = df['finish_rank'].astype(int)

        # Determine effective odds for ROI calculation if dividends are missing
        # (Fall back to pmu reference_odds if no dividend)
        df['effective_odds'] = df['live_odds'].fillna(df['reference_odds']).fillna(1.0).clip(lower=1.05)

        # --- Trainer Performance ---
        model_results = {}
        for model in df['model_version'].unique():
            if not model: continue
            model_df = df[df['model_version'] == model].copy()

            # Simple Gagnant ROI
            # Pick best horse per race according to model
            bets = []
            for race_id, group in model_df.groupby('race_id'):
                best_horse = group.loc[group['proba_winner'].idxmax()]

                # Check if it won
                is_win = best_horse['finish_rank'] == 1

                # Get dividend if available for SG
                # combination for SG is the program_number
                win_div = group[(group['bet_type'] == 'SG') & (group['combination'] == str(best_horse['program_number']))]['dividend_per_1e']

                payout = win_div.iloc[0] if not win_div.empty else (best_horse['effective_odds'] if is_win else 0.0)
                if win_div.empty and not is_win: payout = 0.0

                bets.append({
                    "race_id": race_id,
                    "discipline": best_horse['discipline'],
                    "month": best_horse['program_date'].month,
                    "payout": payout,
                    "win": is_win,
                    "odds": best_horse['effective_odds']
                })

            bets_df = pd.DataFrame(bets)
            if not bets_df.empty:
                roi = (bets_df['payout'].sum() - len(bets_df)) / len(bets_df) * 100
                win_rate = bets_df['win'].mean() * 100
                model_results[model] = {
                    "roi": roi,
                    "win_rate": win_rate,
                    "total_bets": len(bets_df),
                    "avg_odds": bets_df['odds'].mean(),
                    "by_discipline": bets_df.groupby('discipline').apply(lambda x: {
                        "roi": (x['payout'].sum() - len(x)) / len(x) * 100,
                        "count": len(x)
                    }).to_dict()
                }

        # --- Sniper Strategy Evaluation ---
        # Strategy: Edge >= 0.05, Odds [2, 25]
        df['implied_prob'] = 1 / df['effective_odds']
        df['edge'] = df['proba_winner'] - df['implied_prob']

        sniper_mask = (df['edge'] >= 0.05) & (df['effective_odds'] >= 2.0) & (df['effective_odds'] <= 25.0)
        sniper_bets_df = df[sniper_mask].copy()

        # Deduplicate to pick one best per race if multiple qualify
        sniper_final_bets = []
        for race_id, group in sniper_bets_df.groupby('race_id'):
            # Pick the highest probability among those that meet the criteria
            best = group.sort_values('proba_winner', ascending=False).iloc[0]
            is_win = best['finish_rank'] == 1
            win_div = group[(group['bet_type'] == 'SG') & (group['combination'] == str(best['program_number']))]['dividend_per_1e']
            payout = win_div.iloc[0] if not win_div.empty else (best['effective_odds'] if is_win else 0.0)
            if win_div.empty and not is_win: payout = 0.0

            sniper_final_bets.append({"payout": payout, "win": is_win})

        sniper_results = {}
        if sniper_final_bets:
            s_df = pd.DataFrame(sniper_final_bets)
            sniper_results = {
                "roi": (s_df['payout'].sum() - len(s_df)) / len(s_df) * 100,
                "win_rate": s_df['win'].mean() * 100,
                "total_bets": len(s_df)
            }

        # --- Kelly Strategy Evaluation ---
        # For Kelly, we simulate race by race
        kelly_results = {}
        # Use the best model (e.g., hyperstack or global) for Kelly simulation
        # If we have multiple models, we pick one or aggregate. Let's use 'global' if exists or first available.
        available_models = [m for m in df['model_version'].unique() if m]
        if available_models:
            sim_model = 'global' if 'global' in available_models else available_models[0]
            sim_df = df[df['model_version'] == sim_model].copy()
            sim_df['live_odds'] = sim_df['effective_odds'] # For analyze_multiple_races compatibility

            # Group by day to apply bankroll management
            daily_bankroll = 1000.0
            total_profit = 0.0
            total_bets_count = 0

            for date, day_df in sim_df.groupby(sim_df['program_date'].dt.date):
                kelly_report = analyze_multiple_races(day_df, bankroll=daily_bankroll, kelly_fraction=0.5)

                # Calculate actual payout for each bet
                for race_id, allocation in kelly_report.get('bankroll_allocation', {}).items():
                    race_info = day_df[day_df['race_id'] == race_id]
                    fractions = kelly_report['courses'][race_id]['fractions']

                    for prog_num, frac in fractions.items():
                        stake = frac * daily_bankroll
                        total_bets_count += 1

                        horse = race_info[race_info['program_number'] == int(prog_num)]
                        if not horse.empty:
                            horse = horse.iloc[0]
                            is_win = horse['finish_rank'] == 1

                            # Get actual dividend
                            win_div = day_df[(day_df['race_id'] == race_id) & (day_df['bet_type'] == 'SG') & (day_df['combination'] == str(prog_num))]['dividend_per_1e']
                            payout_per_1e = win_div.iloc[0] if not win_div.empty else (horse['effective_odds'] if is_win else 0.0)
                            if win_div.empty and not is_win: payout_per_1e = 0.0

                            profit = (stake * payout_per_1e) - stake
                            total_profit += profit

            if total_bets_count > 0:
                kelly_results = {
                    "roi": (total_profit / (total_bets_count * 1.0)) * 100, # This is simplified ROI
                    "total_profit": total_profit,
                    "total_bets": total_bets_count
                }

        return {
            "trainers": model_results,
            "strategies": {
                "sniper": sniper_results,
                "kelly": kelly_results
            }
        }
