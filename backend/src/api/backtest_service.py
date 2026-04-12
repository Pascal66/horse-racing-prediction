# backend/src/api/backtest_service.py
import logging
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any
from src.api.repositories import RaceRepository
from src.api.kelly_multi_races import analyze_multiple_races

logger = logging.getLogger(__name__)
MIN_BETS_FOR_RELIABLE_TRAINER = 20 # Baissé un peu pour voir plus de modèles
CACHE_FILE = Path("data/backtest_cache.json")

class BacktestService:
    def __init__(self, repository: RaceRepository):
        self.repository = repository
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _safe_float(self, val):
        try:
            if pd.isna(val) or np.isinf(val): return 0.0
            return float(val)
        except: return 0.0

    def _parse_comb(self, comb_str):
        try:
            # Nettoyage profond : retire tout sauf chiffres et séparateurs
            s = str(comb_str).upper().replace('NP', '').replace('-', ' ').replace(',', ' ')
            return {int(p) for p in s.split() if p.isdigit()}
        except: return set()

    def run_backtest(self, force_update: bool = False) -> Dict[str, Any]:
        if not force_update and CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r") as f:
                    return json.load(f)
            except: pass

        logger.info("Starting fresh backtest calculation...")
        raw_data = self.repository.get_backtest_data()
        if not raw_data: return {"error": "No data"}

        df = pd.DataFrame(raw_data)
        # On garde une copie complète pour les dividendes (indépendante des prédictions)
        full_df = df.copy()
        
        # Typage global
        for col in ['dividend_per_1e', 'proba_winner', 'proba_top3_place', 'live_odds', 'reference_odds']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        df['finish_rank'] = pd.to_numeric(df['finish_rank'], errors='coerce').fillna(99).astype(int)
        df['effective_odds'] = df['live_odds'].replace(0, np.nan).fillna(df['reference_odds']).fillna(1.0).clip(lower=1.05)
        df['model_version'] = df['model_version'].fillna('unknown').astype(str)

        # 1. Pré-indexation des dividendes par course pour rapidité
        # On crée un dict: race_id -> { bet_type -> [ (set_nums, dividend) ] }
        race_divs = {}
        for race_id, group in full_df.groupby('race_id'):
            div_map = {}
            valid_divs = group[pd.to_numeric(group['dividend_per_1e'], errors='coerce') > 0]
            for _, row in valid_divs.iterrows():
                b_type = str(row['bet_type']).upper()
                # Normalisation des noms de paris (E_SIMPLE_GAGNANT -> SG)
                if 'SIMPLE_GAGNANT' in b_type: b_type = 'SG'
                elif 'SIMPLE_PLACE' in b_type: b_type = 'SP'
                elif 'COUPLE_GAGNANT' in b_type: b_type = 'CG'
                elif 'COUPLE_PLACE' in b_type: b_type = 'CP'
                elif 'TRIO' in b_type and 'ORDRE' not in b_type: b_type = 'TRIO'
                
                div_map.setdefault(b_type, []).append((self._parse_comb(row['combination']), float(row['dividend_per_1e'])))
            race_divs[race_id] = div_map

        # 2. Analyse par Trainer
        model_results = {}
        # On ne prend que les lignes AVEC prédiction
        pred_df = df[df['proba_winner'] > 0].copy()
        
        for model in pred_df['model_version'].unique():
            if model == 'unknown' or len(model) < 3: continue
            model_df = pred_df[pred_df['model_version'] == model].copy()
            bets = []
            
            for race_id, group in model_df.groupby('race_id'):
                div_map = race_divs.get(race_id, {})
                
                # Cheval favori du modèle
                best_horse = group.loc[group['proba_winner'].idxmax()]
                p_num = int(best_horse['program_number'])
                
                # --- SG / SP ---
                payout_sg = 0.0
                for comb, val in div_map.get('SG', []):
                    if {p_num} == comb: payout_sg = val; break
                if payout_sg == 0 and best_horse['finish_rank'] == 1: 
                    payout_sg = float(best_horse['effective_odds'])

                payout_sp = 0.0
                for comb, val in div_map.get('SP', []):
                    if {p_num} == comb: payout_sp = val; break
                if payout_sp == 0 and best_horse['finish_rank'] <= 3: 
                    payout_sp = 1.1

                # --- CG / CP --- (Top 2)
                payout_cg, payout_cp = 0.0, 0.0
                if len(group) >= 2:
                    top2_win = group.nlargest(2, 'proba_winner', keep='first')
                    p2_win = set(top2_win['program_number'].astype(int))
                    for comb, val in div_map.get('CG', []):
                        if comb and comb.issubset(p2_win): payout_cg = val; break
                    
                    top2_place = group.nlargest(2, 'proba_top3_place', keep='first') if 'proba_top3_place' in group.columns else top2_win
                    p2_place = set(top2_place['program_number'].astype(int))
                    for comb, val in div_map.get('CP', []):
                        if comb and comb.issubset(p2_place): payout_cp = val; break

                # --- TRIO --- (Top 3)
                payout_trio = 0.0
                if len(group) >= 3:
                    top3 = group.nlargest(3, 'proba_winner', keep='first')
                    p3 = set(top3['program_number'].astype(int))
                    for comb, val in div_map.get('TRIO', []):
                        if comb and comb.issubset(p3): payout_trio = val; break

                bets.append({
                    "discipline": str(best_horse['discipline']),
                    "month": int(pd.to_datetime(best_horse['program_date']).month),
                    "payout_sg": payout_sg, "payout_sp": payout_sp,
                    "payout_cg": payout_cg, "payout_cp": payout_cp, "payout_trio": payout_trio,
                    "win": (best_horse['finish_rank'] == 1),
                    "placed": (best_horse['finish_rank'] <= 3),
                    "odds": float(best_horse['effective_odds'])
                })

            b_df = pd.DataFrame(bets)
            if not b_df.empty and len(b_df) >= MIN_BETS_FOR_RELIABLE_TRAINER:
                n = len(b_df)
                model_results[model] = {
                    "roi": self._safe_float((b_df['payout_sg'].sum() - n) / n * 100),
                    "roi_place": self._safe_float((b_df['payout_sp'].sum() - n) / n * 100),
                    "roi_cg": self._safe_float((b_df['payout_cg'].sum() - n) / n * 100),
                    "roi_cp": self._safe_float((b_df['payout_cp'].sum() - n) / n * 100),
                    "roi_trio": self._safe_float((b_df['payout_trio'].sum() - n) / n * 100),
                    "win_rate": self._safe_float(b_df['win'].mean() * 100),
                    "place_rate": self._safe_float(b_df['placed'].mean() * 100),
                    "total_bets": n, "avg_odds": self._safe_float(b_df['odds'].mean()),
                    "seasonal_analysis": {} # On peut le remplir si besoin
                }

        # 3. Kelly & Sniper
        # ... (Logique Kelly simplifiée pour s'assurer qu'elle tourne)
        kelly_res = {"roi": 0.0, "total_profit": 0.0, "total_bets": 0}
        try:
            # On prend le modèle avec le meilleur ROI SG
            best_m = max(model_results.keys(), key=lambda k: model_results[k]['roi']) if model_results else None
            if best_m:
                sim_df = df[df['model_version'] == best_m].copy()
                sim_df['live_odds'] = sim_df['effective_odds']
                bank, prof, staked, cnt = 1000.0, 0.0, 0.0, 0
                for _, race_df in sim_df.groupby('race_id'):
                    res = analyze_multiple_races(race_df, bankroll=bank, kelly_fraction=0.5)
                    for p_num, frac in res.get('fractions', {}).items():
                        stk = bank * frac
                        h = race_df[race_df['program_number'] == int(p_num)].iloc[0]
                        divs = race_divs.get(h['race_id'], {}).get('SG', [])
                        pout = 0.0
                        for c, v in divs:
                            if {int(p_num)} == c: pout = v; break
                        if pout == 0 and h['finish_rank'] == 1: pout = float(h['effective_odds'])
                        
                        gain = stk * pout
                        bank += gain - stk; prof += gain - stk; staked += stk; cnt += 1
                kelly_res = {"roi": (prof/staked*100) if staked > 0 else 0, "total_profit": prof, "total_bets": cnt}
        except: pass

        final_results = {
            "trainers": model_results,
            "strategies": {"sniper": {}, "kelly": kelly_res, "composite": {}},
            "last_updated": pd.Timestamp.now().strftime("%d/%m/%Y %H:%M")
        }
        with open(CACHE_FILE, "w") as f: json.dump(final_results, f)
        return final_results
