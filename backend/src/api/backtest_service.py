# backend/src/api/backtest_service.py
import logging
import json
import os
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.api.repositories import RaceRepository
from src.api.kelly_multi_races import analyze_multiple_races

logger = logging.getLogger(__name__)
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
            s = str(comb_str).upper().replace('NP', '0').replace('-', ' ').replace(',', ' ')
            return {int(p) for p in s.split() if p.isdigit()}
        except: return set()

    def _normalize_model_name(self, name: str) -> str:
        if not name or name == 'unknown': return 'unknown'
        n = name.lower()
        algo = 'tabnet' if 'tabnet' in n else 'ltr' if 'ltr' in n else 'hyperstack'
        disc = 'attele' if 'attele' in n else 'monte' if 'monte' in n else 'plat' if 'plat' in n else \
               'haie' if 'haie' in n else 'steeplechase' if 'steeplechase' in n else 'cross' if 'cross' in n else 'global'
        return f"{disc}_{algo}"

    def calculate_roi_for_df(self, df: pd.DataFrame, race_divs: dict, min_bets: int = 1) -> Dict[str, Any]:
        if df.empty: return {}
        df = df.copy()
        df['model_version_norm'] = df['model_version'].apply(self._normalize_model_name)
        df = df[df['finish_rank'] > 0]
        if df.empty: return {}

        results = {}
        pred_df = df[df['win_probability'] > 0].drop_duplicates(subset=['participant_id', 'model_version']).copy()

        for (model, discipline), m_disc_df in pred_df.groupby(['model_version_norm', 'discipline']):
            model_entry = results.setdefault(model, {"disciplines": {}})
            # stats = {bet_type: {"return": 0.0, "staked": 0, "wins": 0, "sum_odds": 0.0}}
            stats = {
                'SG': {"return": 0.0, "staked": 0, "wins": 0, "sum_odds": 0.0},
                'SP': {"return": 0.0, "staked": 0, "wins": 0, "sum_odds": 0.0},
                'CG': {"return": 0.0, "staked": 0, "wins": 0, "sum_odds": 0.0},
                'CP': {"return": 0.0, "staked": 0, "wins": 0, "sum_odds": 0.0},
                'TRIO': {"return": 0.0, "staked": 0, "wins": 0, "sum_odds": 0.0}
            }
            
            for race_id, group in m_disc_df.groupby('race_id'):
                div_map = race_divs.get(race_id, {})
                top3 = group.nlargest(3, 'win_probability')
                p_nums = list(top3['program_number'].astype(int))
                
                # 1. Simple (3 mises par course)
                stats['SG']["staked"] += 3; stats['SP']["staked"] += 3
                race_won_sg, race_won_sp = False, False
                
                for p in p_nums:
                    h_row = group[group['program_number'] == p].iloc[0]
                    eff_odds = float(h_row['effective_odds'])
                    stats['SG']["sum_odds"] += eff_odds
                    stats['SP']["sum_odds"] += eff_odds
                    
                    g_sg, g_sp = 0.0, 0.0
                    for c, v in div_map.get('SG', []):
                        if {p} == c or (p != 0 and c == {0}): g_sg = v; break
                    if g_sg == 0 and h_row['finish_rank'] == 1: g_sg = eff_odds
                    if g_sg > 0: race_won_sg = True
                    stats['SG']["return"] += g_sg

                    for c, v in div_map.get('SP', []):
                        if {p} == c or (p != 0 and c == {0}): g_sp = v; break
                    if g_sp == 0 and 1 <= h_row['finish_rank'] <= 3: g_sp = 1.1
                    if g_sp > 0: race_won_sp = True
                    stats['SP']["return"] += g_sp

                if race_won_sg: stats['SG']["wins"] += 1
                if race_won_sp: stats['SP']["wins"] += 1

                # 2. Couple (3 combinaisons)
                if len(p_nums) >= 2:
                    stats['CG']["staked"] += 3; stats['CP']["staked"] += 3
                    combos = [{p_nums[0], p_nums[1]}]
                    if len(p_nums) == 3: combos.extend([{p_nums[0], p_nums[2]}, {p_nums[1], p_nums[2]}])
                    
                    race_won_cg, race_won_cp = False, False
                    for combo in combos:
                        for c, v in div_map.get('CG', []):
                            if c and c.issubset(combo): stats['CG']["return"] += v; race_won_cg = True; break
                        for c, v in div_map.get('CP', []):
                            if c and c.issubset(combo): stats['CP']["return"] += v; race_won_cp = True; break
                    if race_won_cg: stats['CG']["wins"] += 1
                    if race_won_cp: stats['CP']["wins"] += 1

                # 3. Trio (1 mise)
                if len(p_nums) == 3:
                    stats['TRIO']["staked"] += 1
                    p3_set = set(p_nums)
                    for c, v in div_map.get('TRIO', []):
                        if c and c.issubset(p3_set): stats['TRIO']["return"] += v; stats['TRIO']["wins"] += 1; break

            model_entry["disciplines"][discipline] = {
                bt: {
                    "return": s["return"],
                    "nb_bets": int(s["staked"]),
                    "nb_wins": int(s["wins"]),
                    "roi": self._safe_float((s["return"] - s["staked"]) / s["staked"] * 100) if s["staked"] > 0 else 0.0,
                    "avg_odds": self._safe_float(s["sum_odds"] / s["staked"]) if s["staked"] > 0 else 0.0
                }
                for bt, s in stats.items() if s["staked"] > 0
            }

        # Agrégation globale par modèle
        final_dict = {}
        for model, data in results.items():
            max_profit = -9999.0
            for bt in ['SG', 'SP', 'CG', 'CP', 'TRIO']:
                tr, tm, tw = 0.0, 0, 0
                for d_stats in data["disciplines"].values():
                    if bt in d_stats:
                        s = d_stats[bt]
                        tm += s['nb_bets']; tw += s['nb_wins']; tr += s['return']
                
                profit = self._safe_float(tr - tm)
                max_profit = max(max_profit, profit)
                suffix = f"_{bt.lower()}" if bt != 'SG' else ""
                data[f"profit{suffix}"] = profit
                data[f"roi{suffix}"] = self._safe_float((tr - tm) / tm * 100) if tm > 0 else -100.0
                data[f"nb_bets{suffix if suffix else '_sg'}"] = int(tm)
                data[f"nb_wins{suffix if suffix else '_sg'}"] = int(tw)
            
            data["max_daily_profit"] = max_profit
            # Win Rate global basé sur le nombre de courses gagnées (SG)
            data["win_rate"] = self._safe_float(data["nb_wins_sg"] / (data["nb_bets_sg"]/3) * 100) if data["nb_bets_sg"] > 0 else 0
            data["count"] = int(data["nb_bets_sg"])
            if (data["count"]/3) >= min_bets: final_dict[model] = data
        return final_dict

    def get_daily_stats_from_db(self, date: datetime.date) -> Dict[str, Any]:
        rows = self.repository.get_daily_performance(date)
        if not rows: return {}
        results = {}
        for r in rows:
            model = self._normalize_model_name(r['model_version'])
            m = results.setdefault(model, {"max_daily_profit": -9999.0})
            bt = r['bet_type'].upper()
            pk, rk, nk, wk = (f"profit_{bt.lower()}" if bt!='SG' else "profit"), (f"roi_{bt.lower()}" if bt!='SG' else "roi"), (f"nb_bets_{bt.lower()}" if bt!='SG' else "count"), (f"nb_wins_{bt.lower()}" if bt!='SG' else "nb_wins")
            
            nb = int(r['nb_bets'])
            wins = int(r['nb_wins'])
            roi = float(r['roi'])
            
            m[nk] = m.get(nk, 0) + nb
            m[wk] = m.get(wk, 0) + wins
            # On stocke le profit net
            profit = (nb * roi / 100.0)
            m[pk] = round(m.get(pk, 0.0) + profit, 2)
            m[rk] = round((m[pk] / m[nk] * 100.0), 1) if m[nk] > 0 else 0.0
            m["max_daily_profit"] = max(m["max_daily_profit"], m[pk])
            
            if bt == 'SG': m['win_rate'] = round((m['nb_wins'] / (m['count']/3) * 100), 1) if m['count'] > 0 else 0
        return results

    def get_period_stats(self, date_start, date_end) -> Dict[str, Any]:
        raw_data = self.repository.get_backtest_data(date_start, date_end)
        if not raw_data: return {"trainers": {}, "audit": {"missing_races": []}}
        df = self._prepare_df(pd.DataFrame(raw_data))
        race_divs = self._index_divs(df)
        return {"trainers": self.calculate_roi_for_df(df, race_divs, min_bets=1), "audit": {"missing_races": []}}

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        ref = pd.to_numeric(df.get('reference_odds', 0), errors='coerce').fillna(10.0)
        l30 = pd.to_numeric(df.get('live_odds_30mn', 0), errors='coerce').replace(0, np.nan)
        df['effective_odds'] = l30.fillna(ref).clip(lower=1.01)
        df['win_probability'] = pd.to_numeric(df.get('proba_winner', 0), errors='coerce').fillna(0.0)
        df['place_probability'] = pd.to_numeric(df.get('proba_top3_place', 0), errors='coerce').fillna(df['win_probability'] * 2.0)
        df['finish_rank'] = pd.to_numeric(df.get('finish_rank', 0), errors='coerce').fillna(0).astype(int)
        df['model_version'] = df.get('model_version', 'unknown').fillna('unknown')
        return df

    def _index_divs(self, df: pd.DataFrame) -> dict:
        race_divs = {}
        for rid, group in df.groupby('race_id'):
            d_map = {}
            for _, row in group[pd.to_numeric(group.get('dividend_per_1e', 0), errors='coerce') > 0].iterrows():
                bt = str(row['bet_type']).upper()
                k = 'SG' if 'SIMPLE_GAGNANT' in bt else 'SP' if 'SIMPLE_PLACE' in bt else 'CG' if 'COUPLE_GAGNANT' in bt else 'CP' if 'COUPLE_PLACE' in bt else 'TRIO' if ('TRIO' in bt and 'ORDRE' not in bt) else None
                if k: d_map.setdefault(k, []).append((self._parse_comb(row['combination']), float(row['dividend_per_1e'])))
            race_divs[rid] = d_map
        return race_divs

    def run_backtest(self, force_update: bool = False) -> Dict[str, Any]:
        if not force_update and CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r") as f: return json.load(f)
            except: pass
        raw_data = self.repository.get_backtest_data()
        if not raw_data: return {"error": "No data"}
        df = self._prepare_df(pd.DataFrame(raw_data))
        race_divs = self._index_divs(df)
        today = datetime.date.today()
        yesterday = today - datetime.timedelta(days=1)
        final_results = {
            "trainers": self.calculate_roi_for_df(df, race_divs, min_bets=10),
            "today": self.get_period_stats(today, today),
            "yesterday": {"trainers": self.get_daily_stats_from_db(yesterday) or self.calculate_roi_for_df(df[pd.to_datetime(df['program_date']).dt.date == yesterday], race_divs, min_bets=1)},
            "last_updated": pd.Timestamp.now().strftime("%d/%m/%Y %H:%M")
        }
        with open(CACHE_FILE, "w") as f: json.dump(final_results, f, default=str)
        return final_results
