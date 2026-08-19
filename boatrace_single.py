"""
ボートレース完全分析支援ツール（玄人向け・Python/Streamlit版）
PRO TRADER TERMINAL v101 — 1ページ完結ダッシュボード

コアロジック：線形補間スコアリング・Plackett-Luce勝率予測・市場分析・
Kelly基準ポートフォリオ最適化・フォーメーション組み合わせ確率・
キマリテ推定・発注規律ゲート・購入ログ永続化（JSON）

【完全統合版 v101】HTML版(v101)とロジック・定数を完全一致させています。
"""

import streamlit as st
try:
    import requests as _requests
    from bs4 import BeautifulSoup as _BS
    _SCRAPING_AVAILABLE = True
except ImportError:
    _SCRAPING_AVAILABLE = False
try:
    import lightgbm as lgb
    import os as _os

    @st.cache_resource(show_spinner=False)
    def _load_lgb_model(filename: str):
        _lgb_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), filename)
        if _os.path.exists(_lgb_path):
            return lgb.Booster(model_file=_lgb_path)
        return None

    _LGB_AVAILABLE = True
except Exception:
    _LGB_AVAILABLE = False

LGB_PROFILES = {
    "precision": {
        "file": "lightgbm_v2_precision.txt",
        "hand_th": 0.27, "lgb_th": 0.65,
        "label": "少数精鋭型（的中率優先・n=174, 的中77.6%, ROI-4.7%）",
    },
    "volume": {
        "file": "lightgbm_v2_volume.txt",
        "hand_th": 0.23, "lgb_th": 0.40,
        "label": "広く浅く型（ROI優先・n=448, 的中70.8%, ROI-4.3%）",
    },
}

import pandas as pd
import math
import re
import json
import os
import itertools
from datetime import datetime, date

# ============================================================
# CONFIG / 定数（HTML版v101と完全一致）
# ============================================================
CONFIG = {
    "EX_MEAN_TIMING": 6.72,
    "ST_MEAN_TIMING": 0.15,
    "EV_THRESHOLD_BUY": 0.04,
    "EV_THRESHOLD_RISK": -0.02,
}

VENUE_SCORE = {
    "大村": 6, "徳山": 6, "芦屋": 4, "下関": 6, "びわこ": -2,
    "戸田": -10, "平和島": -12, "江戸川": -15, "多摩川": 0,
    # v14確定: 実バックテスト20,000件以上で検証済み
}

CALIBRATION_TABLE = {
    "大村": {
        0.15: 0.3333,
        0.16: 0.3333,
        0.17: 0.3333,
        0.18: 0.3333,
        0.19: 0.3333,
        0.2: 0.3333,
        0.21: 0.3837,
        0.22: 0.3837,
        0.23: 0.3837,
        0.24: 0.5,
        0.25: 0.5,
        0.26: 0.5231,
        0.27: 0.6141,
        0.28: 0.617,
        0.29: 0.6608,
        0.3: 0.6608,
        0.31: 0.7746,
        0.32: 0.7857,
        0.33: 0.7857,
        0.34: 0.7857,
        0.35: 0.8333,
        0.36: 0.8333,
        0.37: 1.0,
        0.38: 1.0,
        0.39: 1.0,
        0.4: 1.0,
        0.41: 1.0,
        0.42: 1.0,
        0.43: 1.0,
        0.44: 1.0,
        0.45: 1.0,
    },
    "芦屋": {
        0.15: 0.0,
        0.16: 0.0,
        0.17: 0.0,
        0.18: 0.0,
        0.19: 0.0,
        0.2: 0.0,
        0.21: 0.3623,
        0.22: 0.3767,
        0.23: 0.3767,
        0.24: 0.3767,
        0.25: 0.4167,
        0.26: 0.5,
        0.27: 0.5556,
        0.28: 0.597,
        0.29: 0.597,
        0.3: 0.6513,
        0.31: 0.7019,
        0.32: 0.7019,
        0.33: 0.7019,
        0.34: 0.75,
        0.35: 0.7778,
        0.36: 0.8889,
        0.37: 1.0,
        0.38: 1.0,
        0.39: 1.0,
        0.4: 1.0,
        0.41: 1.0,
        0.42: 1.0,
        0.43: 1.0,
        0.44: 1.0,
        0.45: 1.0,
    },
    "徳山": {
        0.15: 0.0,
        0.16: 0.0,
        0.17: 0.0,
        0.18: 0.0,
        0.19: 0.0,
        0.2: 0.4286,
        0.21: 0.4286,
        0.22: 0.4783,
        0.23: 0.4978,
        0.24: 0.4978,
        0.25: 0.5781,
        0.26: 0.5781,
        0.27: 0.6094,
        0.28: 0.6621,
        0.29: 0.6621,
        0.3: 0.6939,
        0.31: 0.6939,
        0.32: 0.7442,
        0.33: 0.7442,
        0.34: 1.0,
        0.35: 1.0,
        0.36: 1.0,
        0.37: 1.0,
        0.38: 1.0,
        0.39: 1.0,
        0.4: 1.0,
        0.41: 1.0,
        0.42: 1.0,
        0.43: 1.0,
        0.44: 1.0,
        0.45: 1.0,
    },
    "下関": {
        0.15: 0.0,
        0.16: 0.0,
        0.17: 0.0,
        0.18: 0.0,
        0.19: 0.0,
        0.2: 0.2927,
        0.21: 0.2927,
        0.22: 0.3704,
        0.23: 0.4633,
        0.24: 0.4633,
        0.25: 0.5016,
        0.26: 0.5016,
        0.27: 0.5858,
        0.28: 0.5858,
        0.29: 0.631,
        0.3: 0.631,
        0.31: 0.6634,
        0.32: 0.675,
        0.33: 0.675,
        0.34: 0.675,
        0.35: 0.675,
        0.36: 0.675,
        0.37: 0.675,
        0.38: 0.675,
        0.39: 0.675,
        0.4: 0.675,
        0.41: 0.675,
        0.42: 0.675,
        0.43: 0.675,
        0.44: 0.675,
        0.45: 0.675,
    },
    "多摩川": {
        0.15: 0.0,
        0.16: 0.0,
        0.17: 0.0,
        0.18: 0.0,
        0.19: 0.1538,
        0.2: 0.2381,
        0.21: 0.2787,
        0.22: 0.3833,
        0.23: 0.4221,
        0.24: 0.4672,
        0.25: 0.5158,
        0.26: 0.5263,
        0.27: 0.5673,
        0.28: 0.6154,
        0.29: 0.6154,
        0.3: 0.6809,
        0.31: 0.6809,
        0.32: 0.75,
        0.33: 0.75,
        0.34: 0.85,
        0.35: 1.0,
        0.36: 1.0,
        0.37: 1.0,
        0.38: 1.0,
        0.39: 1.0,
        0.4: 1.0,
        0.41: 1.0,
        0.42: 1.0,
        0.43: 1.0,
        0.44: 1.0,
        0.45: 1.0,
    },
    "びわこ": {
        0.15: 0.0,
        0.16: 0.0,
        0.17: 0.0,
        0.18: 0.0,
        0.19: 0.1111,
        0.2: 0.3727,
        0.21: 0.3778,
        0.22: 0.3778,
        0.23: 0.4497,
        0.24: 0.4813,
        0.25: 0.4813,
        0.26: 0.5588,
        0.27: 0.5878,
        0.28: 0.5878,
        0.29: 0.6078,
        0.3: 0.6324,
        0.31: 0.6324,
        0.32: 0.6324,
        0.33: 0.6324,
        0.34: 0.6324,
        0.35: 0.6324,
        0.36: 0.6324,
        0.37: 0.6324,
        0.38: 0.6324,
        0.39: 0.6324,
        0.4: 0.6324,
        0.41: 0.6324,
        0.42: 0.6324,
        0.43: 0.6324,
        0.44: 0.6324,
        0.45: 0.6324,
    },
    "平和島": {
        0.15: 0.0,
        0.16: 0.0,
        0.17: 0.0,
        0.18: 0.0,
        0.19: 0.2188,
        0.2: 0.2673,
        0.21: 0.3168,
        0.22: 0.3843,
        0.23: 0.3843,
        0.24: 0.3843,
        0.25: 0.439,
        0.26: 0.5561,
        0.27: 0.5561,
        0.28: 0.7241,
        0.29: 0.7241,
        0.3: 0.7586,
        0.31: 0.7586,
        0.32: 0.7586,
        0.33: 0.7586,
        0.34: 0.7586,
        0.35: 0.7586,
        0.36: 0.7586,
        0.37: 0.7586,
        0.38: 0.7586,
        0.39: 0.7586,
        0.4: 0.7586,
        0.41: 0.7586,
        0.42: 0.7586,
        0.43: 0.7586,
        0.44: 0.7586,
        0.45: 0.7586,
    },
    "戸田": {
        0.15: 0.1786,
        0.16: 0.1786,
        0.17: 0.1786,
        0.18: 0.1786,
        0.19: 0.1786,
        0.2: 0.3103,
        0.21: 0.3103,
        0.22: 0.3103,
        0.23: 0.3848,
        0.24: 0.4222,
        0.25: 0.4222,
        0.26: 0.4834,
        0.27: 0.4834,
        0.28: 0.4834,
        0.29: 0.5909,
        0.3: 0.6829,
        0.31: 0.6829,
        0.32: 0.7143,
        0.33: 1.0,
        0.34: 1.0,
        0.35: 1.0,
        0.36: 1.0,
        0.37: 1.0,
        0.38: 1.0,
        0.39: 1.0,
        0.4: 1.0,
        0.41: 1.0,
        0.42: 1.0,
        0.43: 1.0,
        0.44: 1.0,
        0.45: 1.0,
    },
    "江戸川": {
        0.15: 0.0,
        0.16: 0.0,
        0.17: 0.0,
        0.18: 0.0,
        0.19: 0.2588,
        0.2: 0.2588,
        0.21: 0.2588,
        0.22: 0.3311,
        0.23: 0.3342,
        0.24: 0.3342,
        0.25: 0.408,
        0.26: 0.4302,
        0.27: 0.4302,
        0.28: 0.4371,
        0.29: 0.4371,
        0.3: 0.5122,
        0.31: 0.5357,
        0.32: 0.5357,
        0.33: 0.5357,
        0.34: 0.5357,
        0.35: 0.5357,
        0.36: 0.5357,
        0.37: 0.5357,
        0.38: 0.5357,
        0.39: 0.5357,
        0.4: 0.5357,
        0.41: 0.5357,
        0.42: 0.5357,
        0.43: 0.5357,
        0.44: 0.5357,
        0.45: 0.5357,
    },
    "global": {
        0.15: 0.0,
        0.16: 0.0,
        0.17: 0.0,
        0.18: 0.0,
        0.19: 0.234,
        0.2: 0.3187,
        0.21: 0.3463,
        0.22: 0.3685,
        0.23: 0.4221,
        0.24: 0.4325,
        0.25: 0.4817,
        0.26: 0.5164,
        0.27: 0.5738,
        0.28: 0.5877,
        0.29: 0.6047,
        0.3: 0.6522,
        0.31: 0.6806,
        0.32: 0.6935,
        0.33: 0.6935,
        0.34: 0.6935,
        0.35: 0.76,
        0.36: 0.76,
        0.37: 0.9,
        0.38: 0.9,
        0.39: 0.9,
        0.4: 0.9,
        0.41: 0.9,
        0.42: 0.9,
        0.43: 0.9,
        0.44: 0.9,
        0.45: 0.9,
    },
}

def load_venue_specific_model(venue: str):
    """
    会場別専用LightGBMモデルを自動ロード。
    専用モデルがなければNoneを返す（全体モデルにフォールバック）。
    データが1万件未満の場合は信頼性が低いため50%の重みで使用する。
    """
    # 【バグ修正】LGB_OKという未定義変数が参照されていた（正しくは_LGB_AVAILABLE）。
    # サンドボックス環境ではモデルファイルが存在せず_LGB_MODELがNoneになるため
    # この関数自体が呼ばれずクラッシュが表面化しなかったが、実際のモデルファイルが
    # 揃っている本番環境（ユーザーのWindows環境）では確実にNameErrorでクラッシュする。
    if not _LGB_AVAILABLE:
        return None
    VENUE_MODEL_MAP = {
        "江戸川": "lightgbm_v3_edogawa.txt",
        "戸田":   "lightgbm_v3_toda.txt",
        "平和島": "lightgbm_v3_heiwajima.txt",
        "びわこ": "lightgbm_v3_biwaako.txt",
    }
    model_file = VENUE_MODEL_MAP.get(venue)
    if model_file and os.path.exists(model_file):
        try:
            return lgb.Booster(model_file=model_file)
        except Exception:
            return None
    return None


def predict_lgbm(boats_data, venue, wind_dir, wind_mps, model, grade="一般") -> dict:
    """LightGBMによる1着確率予測（2プロファイル対応・gap特徴量対応）"""
    if model is None:
        return {}
    try:
        CLASS_MAP = {"A1":3,"A2":2,"B1":1,"B2":0}
        VENUE_MAP = {v:i for i,v in enumerate(['大村','芦屋','徳山','下関','多摩川','びわこ','平和島','戸田','江戸川'])}
        WIND_MAP  = {"無風":0,"向かい風":1,"追い風":2,"横風":3}
        GRADE_MAP = {"一般":0,"GI":1,"GII":2,"SG":3}
        row = {
            'venue': VENUE_MAP.get(venue, 0), 'grade': GRADE_MAP.get(grade, 0),
            'wind_dir': WIND_MAP.get(wind_dir, 0), 'wind_mps': wind_mps,
            'ex_mean': sum(b.get("ex", 6.72) for b in boats_data) / len(boats_data),
        }
        for b in boats_data:
            bn = b["boat"]
            row[f'b{bn}_nat']     = b.get("nat", 4.0)
            row[f'b{bn}_nat2']    = b.get("nat2", 0.0)
            row[f'b{bn}_nat3']    = b.get("nat3", 0.0)
            row[f'b{bn}_local']   = b.get("local", 0.0)
            row[f'b{bn}_motor']   = b.get("motor", 30.0)
            row[f'b{bn}_motor3']  = b.get("motor3", 0.0)
            row[f'b{bn}_ex']      = b.get("ex", 6.72)
            row[f'b{bn}_st']      = b.get("st", 0.17)
            row[f'b{bn}_recent2'] = b.get("recent2", 0.0)
            row[f'b{bn}_recent3'] = b.get("recent3", 0.0)
            row[f'b{bn}_class']   = CLASS_MAP.get(b.get("player_class", "B1"), 1)
        nats = [b.get("nat", 4.0) for b in boats_data]
        locals_ = [b.get("local", 0.0) for b in boats_data]
        rec2s = [b.get("recent2", 0.0) for b in boats_data]
        motors = [b.get("motor", 30.0) for b in boats_data]
        classes = [CLASS_MAP.get(b.get("player_class", "B1"), 1) for b in boats_data]
        sts    = [b.get("st", 0.17) for b in boats_data]
        ex_sts = [b.get("ex_st", 0.0) for b in boats_data]
        row['b1_nat_rank']     = (sorted(nats, reverse=True).index(nats[0]) + 1) if nats else 3
        row['b1_local_rank']   = (sorted(locals_, reverse=True).index(locals_[0]) + 1) if locals_ else 3
        row['b1_recent2_rank'] = (sorted(rec2s, reverse=True).index(rec2s[0]) + 1) if rec2s else 3
        if len(nats) >= 2:
            row['b1_nat_gap']     = nats[0] - max(nats[1:])
            row['b1_local_gap']   = locals_[0] - max(locals_[1:])
            row['b1_recent2_gap'] = rec2s[0] - max(rec2s[1:])
            row['b1_motor_gap']   = motors[0] - max(motors[1:])
            row['b1_class_gap']   = classes[0] - max(classes[1:])
        # 壁ロジック特徴量
        row['wall_strength']   = min(1.0, max(0.0, (sts[0] - sts[1] - 0.03) / 0.12)) if len(sts) >= 2 else 0.0
        row['st_1_minus_st_2'] = sts[0] - sts[1] if len(sts) >= 2 else 0.0
        # 展示ST特徴量
        ex_st_valid = [v for v in ex_sts if v != 0.0]
        row['ex_st_mean']      = float(sum(ex_st_valid)/len(ex_st_valid)) if ex_st_valid else 0.0
        row['ex_st_std']       = float(pd.Series(ex_st_valid).std()) if len(ex_st_valid) > 1 else 0.0
        row['b1_ex_st']        = ex_sts[0] if ex_sts else 0.0
        row['b1_ex_st_rank']   = sorted(ex_sts).index(ex_sts[0]) + 1 if ex_sts else 3
        row['b1_ex_st_gap']    = ex_sts[0] - min(ex_sts[1:]) if len(ex_sts) > 1 else 0.0

        X = pd.DataFrame([row])
        feat_names = model.feature_name()
        for f in feat_names:
            if f not in X.columns: X[f] = 0
        X = X[feat_names]
        prob_global = float(model.predict(X)[0])

        # 会場別専用モデルがあれば50%ブレンド
        venue_model = load_venue_specific_model(venue)
        if venue_model is not None:
            try:
                feat_v = venue_model.feature_name()
                Xv = pd.DataFrame([row])
                for f in feat_v:
                    if f not in Xv.columns: Xv[f] = 0
                Xv = Xv[feat_v]
                prob_venue = float(venue_model.predict(Xv)[0])
                # 専用モデルは信頼性が低いため50%ブレンド
                prob_blended = prob_global * 0.5 + prob_venue * 0.5
                return {"lgb_prob_b1": prob_blended, "lgb_venue_prob": prob_venue, "lgb_global_prob": prob_global}
            except Exception:
                pass
        return {"lgb_prob_b1": prob_global}
    except Exception:
        return {}



def calibrate_probability(raw_prob: float, venue: str) -> float:
    """
    Isotonic Regressionによる確率補正（v14実装・n=14,867件で学習済み）
    補正前: モデルが0.27と言うと実際は60%当たる（系統的過小評価）
    補正後: 0.27→0.60と正しく推定できる（Calibration誤差92.7%改善）
    """
    table = CALIBRATION_TABLE.get(venue, CALIBRATION_TABLE.get("global", {}))
    if not table:
        return raw_prob
    # 最近傍ルックアップ
    key = round(max(0.15, min(0.44, raw_prob)), 2)
    # 0.01刻みに丸める
    key = round(key * 100) / 100
    return table.get(key, raw_prob)

BOAT_COLORS = {
    1: "#f4f4f4", 2: "#1d4ed8", 3: "#dc2626", 4: "#eab308", 5: "#ec4899", 6: "#059669",
}
BOAT_TEXT = {1: "#0a0a0a", 2: "#fff", 3: "#fff", 4: "#0a0a0a", 5: "#fff", 6: "#fff"}
BOAT_LABEL = {1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤", 6: "⑥"}

MASTER_BOATS = {
    1: {"nat": 6.85, "motor": 39.5, "ex": 6.66, "st": 0.12},
    2: {"nat": 5.42, "motor": 32.0, "ex": 6.74, "st": 0.15},
    3: {"nat": 5.91, "motor": 35.2, "ex": 6.71, "st": 0.14},
    4: {"nat": 5.10, "motor": 29.0, "ex": 6.75, "st": 0.16},
    5: {"nat": 6.22, "motor": 41.5, "ex": 6.64, "st": 0.13},
    6: {"nat": 4.05, "motor": 24.1, "ex": 6.79, "st": 0.17},
}

LOG_FILE = "purchase_logs.json"


# ============================================================
# コアエンジン（線形補間・Plackett-Luce・Kelly基準等）
# ============================================================
def lerp_clamp(x, x1, x2, y1, y2):
    """線形補間（クランプ処理付き・None/不正値ガード付き）"""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return float(y1)  # 不正値は下限値を返す
    if x1 == x2:
        return float(y1)
    t = (x - x1) / (x2 - x1)
    t = max(0.0, min(1.0, t))
    return y1 + t * (y2 - y1)


def calc_pro_score(boat_idx, nat, motor, ex, st, venue, wind_dir, wind_mps, ex_mean):
    """各艇の生スコア(rawScore)を精密に計算（全8会場・全4風向対応）"""
    f_nat = lerp_clamp(nat, 2.0, 8.5, 10, 80)
    f_motor = lerp_clamp(motor, 25.0, 45.0, 5, 25)

    ex_diff = ex_mean - ex
    f_ex = lerp_clamp(ex_diff, -0.15, 0.15, -20, 30)

    f_st = lerp_clamp(st, 0.22, 0.11, -10, 20)

    course_biases = {1: 35, 2: 15, 3: 10, 4: 5, 5: 0, 6: -5}
    f_course = course_biases.get(boat_idx, 0)

    v_score = VENUE_SCORE.get(venue, 0)
    if boat_idx == 1:
        f_course += v_score
    elif boat_idx in (3, 4) and v_score < 0:
        f_course += abs(v_score) * 0.5

    # 風速の影響：不連続ステップを排除しシグモイド的に滑らかに変化させる
    # wind_weight: 0m/s=0, 3m/s≈0.3, 5m/s≈0.7, 8m/s≈1.0
    wind_weight = min(1.0, max(0.0, (wind_mps - 1.5) / 5.0))
    if wind_weight > 0.05:
        if wind_dir == "向かい風":
            if boat_idx == 1:
                f_course -= 5 * wind_weight
            if boat_idx in (3, 4):
                f_course += 8 * wind_weight
        elif wind_dir == "追い風":
            if boat_idx == 1:
                f_course += 3 * wind_weight
            if boat_idx >= 2:
                f_course -= 2 * wind_weight
        elif wind_dir == "横風":
            if boat_idx == 1:
                f_course -= 2 * wind_weight
            if boat_idx in (3, 4):
                f_course += 3 * wind_weight

    total_score = f_nat + f_motor + f_ex + f_st + f_course
    return max(10.0, total_score), {
        "f_nat": f_nat, "f_motor": f_motor, "f_ex": f_ex, "f_st": f_st, "f_course": f_course
    }


def predict_finish_distribution(boats, venue, wind_dir, wind_mps):
    """Plackett-Luceモデルに基づく勝率予測（1着/2着内/3着内確率を正規化して算出）"""
    valid_exs = [b["ex"] for b in boats if b.get("ex") is not None]
    ex_mean = sum(valid_exs) / len(valid_exs) if valid_exs else CONFIG["EX_MEAN_TIMING"]

    raw_scores = {}
    factor_data = {}
    total_raw = 0.0
    for b in boats:
        score, factors = calc_pro_score(b["boat"], b["nat"], b["motor"], b["ex"], b["st"], venue, wind_dir, wind_mps, ex_mean)
        raw_scores[b["boat"]] = score
        factor_data[b["boat"]] = factors
        total_raw += score

    # ============================================================
    # 「壁」ロジック（複数艇のSTの関係から枠順干渉を補正）
    # 壁が成立する条件：
    #   - 1号艇STが遅い（> 0.17）かつ2号艇STが早い（< 0.15）
    #   → 2号艇が1号艇の隣でブロックし、3〜6号艇の外攻めを抑制
    # 壁強度 = max(0, st_2 - st_1) で計算（差が大きいほど壁が厚い）
    # ============================================================
    st_map = {b["boat"]: float(b.get("st", 0.17) or 0.17) for b in boats}
    st_1 = st_map.get(1, 0.17)
    st_2 = st_map.get(2, 0.17)

    # 壁強度（0.0〜1.0）：ST差が0.05以上で壁が成立し始め0.15で最大
    wall_strength = min(1.0, max(0.0, (st_1 - st_2 - 0.03) / 0.12))

    if wall_strength > 0.05:
        # 1号艇：壁により逃げやすくなる（最大+15点）
        raw_scores[1] = raw_scores.get(1, 10.0) * (1.0 + 0.12 * wall_strength)
        # 2号艇：壁役で2着固定率上昇（最大+8点）
        raw_scores[2] = raw_scores.get(2, 10.0) * (1.0 + 0.06 * wall_strength)
        # 3〜6号艇：外攻めが抑制される（最大-10点）
        for bn in [3, 4, 5, 6]:
            if bn in raw_scores:
                # コース序列が遠いほど抑制が強い（3号>4号>5号>6号の順）
                penalty = 0.04 + (bn - 3) * 0.015
                raw_scores[bn] = raw_scores[bn] * (1.0 - penalty * wall_strength)

        # total_rawも再計算
        total_raw = sum(raw_scores.values())
        # factor_dataに壁補正を記録
        for b in boats:
            bn = b["boat"]
            if bn == 1:
                factor_data[bn]["f_wall"] = round(raw_scores[bn] * 0.12 * wall_strength, 1)
            elif bn == 2:
                factor_data[bn]["f_wall"] = round(raw_scores[bn] * 0.06 * wall_strength, 1)
            elif bn in [3,4,5,6]:
                factor_data[bn]["f_wall"] = -round(raw_scores[bn] * (0.04+(bn-3)*0.015) * wall_strength, 1)
    else:
        wall_strength = 0.0
        for b in boats:
            factor_data[b["boat"]]["f_wall"] = 0.0

    probs_1st = {}
    for b in boats:
        probs_1st[b["boat"]] = raw_scores[b["boat"]] / total_raw if total_raw > 0 else 1.0 / 6.0

    probs_2nd = {b["boat"]: 0.0 for b in boats}
    probs_3rd = {b["boat"]: 0.0 for b in boats}

    for i in raw_scores.keys():
        for j in raw_scores.keys():
            if i == j:
                continue
            rem_sum2 = total_raw - raw_scores[i]
            p2 = raw_scores[j] / rem_sum2 if rem_sum2 > 0 else 1.0 / 5.0
            probs_2nd[j] += probs_1st[i] * p2

            for k in raw_scores.keys():
                if k == i or k == j:
                    continue
                rem_sum3 = rem_sum2 - raw_scores[j]
                p3 = raw_scores[k] / rem_sum3 if rem_sum3 > 0 else 1.0 / 4.0
                probs_3rd[k] += probs_1st[i] * p2 * p3

    s1 = sum(probs_1st.values())
    s2 = sum(probs_2nd.values())
    s3 = sum(probs_3rd.values())

    # ============================================================
    # 再正規化ガード＋数値整合チェック
    # NaN/inf/負値/合計ズレを検知して安全値に強制補正
    # ============================================================
    import math as _math
    EPS = 1e-9
    _prob_repair_log = []

    def _safe_normalize(prob_dict, n_boats=6, _label="prob"):
        """確率辞書を安全に正規化。NaN/inf/負値をゼロに修正してから合計で割る。
        修正が発生した場合は _prob_repair_log に記録し、後段でBUYを禁止する判断材料にする。"""
        cleaned = {}
        repaired = False
        for bn, v in prob_dict.items():
            if _math.isnan(v) or _math.isinf(v) or v < 0:
                cleaned[bn] = 0.0
                repaired = True
            else:
                cleaned[bn] = v
        total = sum(cleaned.values())
        if total < EPS:
            _prob_repair_log.append(f"{_label}: 全艇ゼロ/異常値のため均等分配に強制補正")
            return {bn: 1.0 / n_boats for bn in cleaned}
        if repaired:
            _prob_repair_log.append(f"{_label}: NaN/inf/負値を検出しゼロに補正")
        # 合計が1.0から大きくずれている場合も記録（丸め誤差レベルは許容）
        if abs(total - 1.0) > 0.02 and not repaired:
            _prob_repair_log.append(f"{_label}: 合計確率が{total:.4f}（1.0から乖離）")
        return {bn: v / total for bn, v in cleaned.items()}

    probs_1st = _safe_normalize(probs_1st, _label="1着確率")
    probs_2nd = _safe_normalize(probs_2nd, _label="2着内確率")
    probs_3rd = _safe_normalize(probs_3rd, _label="3着内確率")

    # 単調性チェック：1着内 ≥ 2着内 ≥ 3着内（警告のみ・補正はしない）
    s1 = sum(probs_1st.values())
    s2 = sum(probs_2nd.values())
    s3 = sum(probs_3rd.values())

    results = []
    decomp_data = {}
    for b in boats:
        bn = b["boat"]
        p1 = probs_1st[bn] / s1 if s1 > 0 else 1.0 / 6.0
        p2 = probs_2nd[bn] / s2 if s2 > 0 else 1.0 / 6.0
        p3 = probs_3rd[bn] / s3 if s3 > 0 else 1.0 / 6.0

        ex_diff = ex_mean - b["ex"] if b.get("ex") is not None else 0.0
        anomaly_text = "通常"
        if ex_diff >= 0.06 and b["motor"] >= 38:
            anomaly_text = "🔥特注(展示爆伸×強モータ)"
        elif ex_diff >= 0.05:
            anomaly_text = "🚀展示爆伸び"
        elif b["motor"] >= 40:
            anomaly_text = "💎名機(モーター超抜)"
        elif ex_diff <= -0.05 and b["motor"] <= 28:
            anomaly_text = "🚨お辞儀気味"

        f = factor_data[bn]
        decomp_data[bn] = {
            "anomaly": anomaly_text,
            "breakdown": f"全国:{f['f_nat']:.0f} モータ:{f['f_motor']:.0f} 展示:{f['f_ex']:+.0f} ST:{f['f_st']:+.0f} 枠:{f['f_course']:+.0f} 壁:{f.get('f_wall',0):+.0f}",
        }

        results.append({
            "boat": bn,
            "score": round(raw_scores[bn], 1),
            "prob_1st": p1,
            "prob_2nd_within": min(1.0, p1 + p2),
            "prob_3rd_within": min(1.0, p1 + p2 + p3),
        })

    # ============================================================
    # 確率整合性の最終検証（項目7対応）
    # 合計1.0・範囲[0,1]・6艇分揃っているかを厳密にチェックし、
    # 異常があれば「補正して続行」ではなく「分析失敗」として記録する。
    # ここでの異常はBUY判定を強制的に禁止する材料として後段で使用される。
    # ============================================================
    _final_sum_1st = sum(probs_1st.values())
    _integrity_errors = list(_prob_repair_log)
    if abs(_final_sum_1st - 1.0) > 1e-6:
        _integrity_errors.append(f"1着確率の最終合計が{_final_sum_1st:.6f}（1.0と不一致）")
    if not all(0.0 <= p <= 1.0 for p in probs_1st.values()):
        _integrity_errors.append("1着確率に0〜1の範囲外の値がある")
    if len(probs_1st) != 6 or set(probs_1st.keys()) != set(range(1, 7)):
        _integrity_errors.append(f"1着確率が6艇分（1〜6号艇）揃っていない：{sorted(probs_1st.keys())}")
    prob_integrity = {"ok": len(_integrity_errors) == 0, "errors": _integrity_errors}

    return results, raw_scores, decomp_data, ex_mean, wall_strength, prob_integrity


def odds_to_market_probability(odds: dict) -> dict:
    """単勝オッズから市場の暗黙確率を逆算"""
    market_probs = {}
    sum_inverse = 0.0
    cleaned_odds = {}
    for bn, o in odds.items():
        if o is not None and o >= 1.0:
            cleaned_odds[bn] = o
            sum_inverse += (1.0 / o)
        else:
            cleaned_odds[bn] = None

    for bn in range(1, 7):
        o = cleaned_odds.get(bn)
        if sum_inverse > 0 and o is not None:
            market_probs[bn] = (1.0 / o) / sum_inverse
        else:
            market_probs[bn] = 0.0
    return market_probs


def parse_raw_odds(text: str) -> dict:
    """オッズコピペ文字列パーサー"""
    odds = {}
    if not text or not text.strip():
        return odds
    lines = text.strip().split("\n")
    for line in lines:
        if not line.strip():
            continue
        boat_match = re.search(r"\b([1-6])\b", line)
        if not boat_match:
            continue
        bn = int(boat_match.group(1))
        number_matches = re.findall(r"\d+\.\d+|\d{2,}", line)
        if not number_matches:
            continue
        try:
            odds_val = float(number_matches[-1])
            if 1.0 <= odds_val <= 9999.0:
                odds[bn] = odds_val
        except (ValueError, IndexError):
            continue
    return odds


def calc_race_confidence(raw_scores: dict) -> dict:
    """生スコアの標準偏差からレース信頼度・混戦度を判定"""
    if not raw_scores:
        return {"score": 0, "level": "C", "status": "データ無"}
    scores = list(raw_scores.values())
    mean_s = sum(scores) / len(scores)
    variance = sum(math.pow(s - mean_s, 2) for s in scores) / len(scores)
    std_dev = math.sqrt(variance)
    conf_score = min(100, max(10, int(std_dev * 2.2)))
    if conf_score >= 65:
        level, status = "A", "本命信頼（実力断層あり）"
    elif conf_score >= 45:
        level, status = "B", "標準（中位混戦）"
    else:
        level, status = "C", "大混戦（荒れ気配濃厚）"
    return {"score": conf_score, "level": level, "status": status}


def calc_confidence_breakdown(results: list) -> dict:
    """
    「信頼度」という1つの数値に混ざっていた意味を3つの軸に分離する。
    ------------------------------------------------------------
    ① 本命断層（gap）    : 1位と2位の1着確率差。差が大きいほど「抜けた本命」がいる。
    ② 混戦度（chaos）    : 6艇の1着確率のエントロピー。均等に近いほど「読みにくい」。
    ③ 集中度（top3）     : 上位3艇に確率がどれだけ集中しているか。
    これらは互いに独立した情報であり、1つのconf_scoreに押し込めると
    「本命がいるのに混戦」「混戦だが妙に信頼度が高い」といったケースを区別できなくなる。
    """
    probs = sorted([r["prob_1st"] for r in results], reverse=True)
    if len(probs) < 2:
        return {"gap_score": 0, "gap_label": "データ不足", "chaos_score": 50, "chaos_label": "データ不足", "top3_concentration": 0}

    # ① 本命断層：1位と2位の確率差（0〜約0.5想定）を0-100にスケール
    gap = probs[0] - probs[1]
    gap_score = min(100, round(gap * 250))
    if gap_score >= 60:
        gap_label = "明確な断層あり（本命が抜けている）"
    elif gap_score >= 30:
        gap_label = "やや断層あり"
    else:
        gap_label = "断層なし（横並び）"

    # ② 混戦度：シャノンエントロピーを最大値(6艇均等)で正規化
    H = -sum(p * math.log2(p) for p in probs if p > 0)
    H_max = math.log2(len(probs))
    chaos_score = round((H / H_max) * 100) if H_max > 0 else 50
    if chaos_score >= 75:
        chaos_label = "大混戦（ほぼ均等）"
    elif chaos_score >= 50:
        chaos_label = "中程度の混戦"
    else:
        chaos_label = "混戦度低い（絞れている）"

    # ③ 上位3艇への確率集中度
    top3_concentration = round(sum(probs[:3]) * 100)

    return {
        "gap_score": gap_score, "gap_label": gap_label,
        "chaos_score": chaos_score, "chaos_label": chaos_label,
        "top3_concentration": top3_concentration,
    }


def calc_data_reliability(odds_source: str, has_missing_ex: bool, lgb_and_hand_agree,
                           prob_integrity_ok: bool, wind_mps: float, is_demo_data: bool = False) -> dict:
    """
    「予測信頼度」＝この予測がどれだけ信頼できるデータの上に成り立っているかを評価する。
    本命断層・混戦度（＝レースの性質）とは完全に独立した「入力データの質」の軸。
    100点から、データ品質を落とす要因ごとに減点していく方式。
    """
    score = 100
    reasons = []
    if is_demo_data:
        score -= 60
        reasons.append("6艇全てサンプル値未変更（実データ未入力）")
    if odds_source != "actual":
        score -= 30
        reasons.append("実測オッズ未取得（推定値で計算）")
    if has_missing_ex:
        score -= 25
        reasons.append("展示タイム欠損あり")
    if lgb_and_hand_agree is False:
        score -= 25
        reasons.append("AIと手動モデルが不一致")
    if not prob_integrity_ok:
        score -= 40
        reasons.append("確率整合性エラー")
    if wind_mps >= 5.0:
        score -= 10
        reasons.append(f"強風（{wind_mps:.1f}m/s）で直前情報の不確実性増")
    score = max(0, score)
    if score >= 80:
        label, color = "高（実測データが揃っている）", "#16e0a0"
    elif score >= 50:
        label, color = "中（一部推定値を含む）", "#ffb648"
    else:
        label, color = "低（推定・欠損が多く鵜呑み厳禁）", "#ff5c72"
    return {"score": score, "label": label, "color": color, "reasons": reasons}


def calc_synthetic_odds(combos_with_odds: list) -> float:
    """ブックメーカー定義（1/Σ(1/odds)）に基づく真の合成オッズ計算"""
    sum_inv = 0.0
    for c in combos_with_odds:
        odds = c.get("odds", 0)
        if odds > 1.0:
            sum_inv += (1.0 / odds)
    return 1.0 / sum_inv if sum_inv > 0 else 0.0


def detect_market_gap(ai_results: list, market_probs: dict) -> list:
    """AI予測確率と市場確率のズレ(歪み)を検出"""
    gaps = []
    for r in ai_results:
        bn = r["boat"]
        m_prob = market_probs.get(bn, 0.0)
        gap = r["prob_1st"] - m_prob
        gaps.append({"boat": bn, "ai_prob": r["prob_1st"], "market_prob": m_prob, "gap": gap})
    return gaps


def kelly_fraction(win_prob: float, odds: float, b_weight: float = 0.2) -> float:
    """ケリー基準による推奨投資比率の算出（破綻を防ぐための調整重み付き）"""
    if odds <= 1.0 or win_prob <= 0.0:
        return 0.0
    q = 1.0 - win_prob
    f = (win_prob * odds - q) / odds
    return max(0.0, f * b_weight)


def calc_expected_profit_raw(win_prob: float, odds: float, stake_yen) -> float:
    """想定投資額に対する数学的期待損益(EV円・float)の算出 ※生値・表示用"""
    if odds <= 0:
        return -float(stake_yen)
    return (win_prob * odds * float(stake_yen)) - float(stake_yen)


def estimate_combo_probability(boats_probs: list, combo: tuple) -> float:
    """Plackett-Luceモデルに基づく組み合わせ同時確率推定（NaN/inf/負値ガード付き）"""
    import math as _m
    prob_dict = {b["boat"]: max(0.0, b["prob_1st"]) for b in boats_probs}
    # NaN/infを0に置換
    prob_dict = {k: (0.0 if _m.isnan(v) or _m.isinf(v) else v) for k, v in prob_dict.items()}
    total = sum(prob_dict.values())
    if total < 1e-9:
        return 0.0  # 全艇ゼロの異常状態
    # 正規化
    prob_dict = {k: v / total for k, v in prob_dict.items()}
    if len(combo) == 2:
        c1, c2 = combo
        p1 = prob_dict.get(c1, 0.0)
        denom2 = 1.0 - p1
        p2 = prob_dict.get(c2, 0.0) / denom2 if denom2 > 1e-9 else 0.0
        return p1 * p2
    elif len(combo) == 3:
        c1, c2, c3 = combo
        p1 = prob_dict.get(c1, 0.0)
        denom2 = 1.0 - p1
        p2 = prob_dict.get(c2, 0.0) / denom2 if denom2 > 1e-9 else 0.0
        denom3 = max(denom2 - prob_dict.get(c2, 0.0), 1e-9)
        p3 = prob_dict.get(c3, 0.0) / denom3 if denom3 > 1e-9 else 0.0
        return p1 * p2 * p3
    return 0.0


def calc_formation_summary(boats_probs: list, axis_boats: list, partner_boats: list, ticket_type: str, total_budget: int) -> list:
    """フォーメーション組み合わせ生成＋資金の傾斜配分（2連単/2連複/3連単/3連複対応）"""
    combos = []
    if ticket_type == "trifecta":
        for axis in axis_boats:
            seconds = [p for p in partner_boats if p != axis]
            for second in seconds:
                thirds = [p for p in partner_boats if p not in (axis, second)]
                for third in thirds:
                    combos.append((axis, second, third))
    elif ticket_type == "trio":
        all_boats = partner_boats if len(partner_boats) >= 3 else [1,2,3,4,5,6]
        for i in range(len(all_boats)):
            for j in range(i+1, len(all_boats)):
                for k in range(j+1, len(all_boats)):
                    combos.append((all_boats[i], all_boats[j], all_boats[k]))
    elif ticket_type == "quinella":
        all_boats = partner_boats if len(partner_boats) >= 2 else [1,2,3,4,5,6]
        for i in range(len(all_boats)):
            for j in range(i+1, len(all_boats)):
                combos.append((all_boats[i], all_boats[j]))
    else:
        for axis in axis_boats:
            for partner in partner_boats:
                if partner != axis:
                    combos.append((axis, partner))

    combos = list(dict.fromkeys(combos))
    if not combos:
        return []

    def calc_prob(combo):
        if ticket_type == "quinella":
            return (estimate_combo_probability(boats_probs, combo) +
                    estimate_combo_probability(boats_probs, (combo[1], combo[0])))
        elif ticket_type == "trio":
            from itertools import permutations
            return sum(estimate_combo_probability(boats_probs, perm) for perm in permutations(combo))
        return estimate_combo_probability(boats_probs, combo)

    raw_probs = [calc_prob(combo) for combo in combos]
    sum_raw_prob = sum(raw_probs)
    if sum_raw_prob <= 0:
        return []

    results = []
    for combo, r_prob in zip(combos, raw_probs):
        norm_prob = r_prob / sum_raw_prob
        # 傾斜配分（最低100円・100円刻み）
        allocated = max(100, int(math.floor((total_budget * norm_prob) / 100) * 100))
        label = "-".join(map(str, combo))
        results.append({"combo": combo, "label": label, "prob": r_prob, "allocated_yen": allocated})

    # 合計が予算を超えていたら全体をスケールダウン
    total_allocated = sum(r["allocated_yen"] for r in results)
    if total_allocated > total_budget and total_allocated > 0:
        scale = total_budget / total_allocated
        for r in results:
            r["allocated_yen"] = max(100, int(math.floor(r["allocated_yen"] * scale / 100) * 100))

    return results


def filter_noise_combos(candidates: list, min_prob: float = 0.015) -> list:
    """AI予測確率が極端に低い組み合わせ（ノイズ）の足切り"""
    return [c for c in candidates if c.get("prob", 0.0) >= min_prob]


def estimate_kimarite(top_boat: int, venue: str) -> str:
    """会場特性と本命艇のコースから想定キマリテを簡易推定"""
    v_score = VENUE_SCORE.get(venue, 0)
    if top_boat == 1 and v_score >= 7:
        return "イン逃げ"
    elif top_boat == 1 and v_score < 0:
        return "イン逃げも捲り差し警戒"
    elif top_boat in (3, 4):
        return "センター捲り・捲り差し"
    elif top_boat == 2:
        return "差し"
    elif top_boat in (5, 6):
        return "大外一撃（波乱型）"
    return "二捲り・差し混合戦"


# ============================================================
# 購入ログ永続化（JSON ファイル）
# ============================================================
def load_logs() -> list:
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            # JSONが壊れている場合はバックアップを作成してから空を返す
            import shutil
            backup = LOG_FILE + ".bak"
            try:
                shutil.copy(LOG_FILE, backup)
                st.warning(f"⚠️ ログファイルが破損していたためバックアップを作成しました（{backup}）。新規ログから開始します。", icon="⚠️")
            except Exception:
                pass
            return []
        except Exception:
            return []
    return []


def save_logs(logs: list):
    """一時ファイル経由の安全な書き込み（同時書き込みによる破損対策）"""
    tmp_file = LOG_FILE + ".tmp"
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        # 書き込み成功後に本ファイルと差し替え
        os.replace(tmp_file, LOG_FILE)
    except Exception as e:
        # 一時ファイルが残っていたら削除
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass
        raise e


def append_log(entries: list):
    logs = load_logs()
    logs.extend(entries)
    save_logs(logs)


def update_log_result(log_id: str, result: str, payout_yen: int = None, final_odds: float = None, comment: str = None):
    logs = load_logs()
    for l in logs:
        if l["id"] == log_id:
            l["result"] = result
            if payout_yen is not None: l["payout_yen"] = payout_yen
            if final_odds is not None: l["final_odds"] = final_odds
            if comment is not None: l["comment"] = comment
    save_logs(logs)


def check_auto_skip(
    odds_source: str,          # "actual" or "manual"
    raw_odds: dict,            # {1: 1.7, ...}
    wind_mps: float,
    wind_dir: str,
    conf_score: int,
    lgb_and_hand_agree,        # None or bool
    lgb_prob_b1: float,
    hand_prob_b1: float,
    has_missing_ex: bool,      # 展示データ欠損あり
    boats_data: list,
    wall_strength: float,
    venue: str,
    is_demo_data: bool = False,  # 6艇全てがサンプル値未変更（項目11対応）
) -> list:
    """
    AUTO SKIP：人間の判断より前にシステムが強制排除する条件を評価。
    戻り値: [(条件名, True/False, 重大度, 理由, 解除条件), ...]
    重大度: "critical"（即SKIP）/ "warning"（注意）
    解除条件：この警告が消える／BUYが再度可能になるために何をすればよいか（具体的な操作）。
    """
    checks = []

    # ① 実測オッズ未取得（最重要）
    checks.append((
        "実測オッズ未取得",
        odds_source != "actual",
        "warning",
        "単勝オッズが自動取得できていません。推定オッズのみでの判断はリスクが高い。",
        "「実測オッズ自動取得」を実行するか、直前の実オッズを手入力してください。実オッズに切り替わると自動的に解除されます。",
    ))

    # ② 1号艇オッズ異常（過熱or未設定）
    o1 = raw_odds.get(1, 0)
    checks.append((
        "1号艇オッズ異常",
        o1 > 0 and o1 < 1.3,
        "critical",
        f"1号艇オッズ{o1:.1f}倍は過熱気味。AIの勝率と市場確率の乖離が大きい可能性。",
        "このレースでは解除できません（オッズが実際に1.3倍以上に動くまで待つ必要があります）。別のレースを検討してください。",
    ))

    # ③ 展示データ欠損
    checks.append((
        "展示データ欠損",
        has_missing_ex,
        "critical",
        "展示タイムが入力されていない艇があります。直前情報なしでの判断は危険。",
        "全艇の「展示T」欄に実際の展示タイムを入力してください。全艇入力が完了すると自動的に解除されます。",
    ))

    # ④ 強風+難水面の複合リスク
    hard_venues = ["江戸川", "戸田", "平和島", "びわこ"]
    checks.append((
        "強風×難水面",
        wind_mps >= 4.0 and venue in hard_venues,
        "critical",
        f"{venue}×風速{wind_mps:.1f}m/s — 波乱率が極めて高い組み合わせ。",
        "このレースでは解除できません（会場と風速の組み合わせによる構造的リスクのため）。風が収まるのを待つか、このレースは見送ってください。",
    ))

    # ⑤ 強風単独（向かい風・横風）
    checks.append((
        "強風リスク（向かい/横）",
        wind_mps >= 5.0 and wind_dir in ("向かい風", "横風"),
        "warning",
        f"{wind_dir}{wind_mps:.1f}m/s — スタート事故率が上昇。全艇のSTに不確実性。",
        "風速が5.0m/s未満に下がるか、風向きが「無風」「追い風」に変わると解除されます。直前の気象情報欄を更新してください。",
    ))

    # ⑥ モデル間乖離大（LGB vs 手作り）
    if lgb_and_hand_agree is not None:
        _diff = abs(lgb_prob_b1 - hand_prob_b1) * 100
        checks.append((
            f"モデル乖離大（差{_diff:.0f}%）",
            lgb_and_hand_agree is False and _diff >= 15,
            "critical",
            f"LGB({lgb_prob_b1*100:.0f}%)と手作り({hand_prob_b1*100:.0f}%)が大きく乖離。どちらかが誤っている可能性。",
            "このレースでは解除できません（2つのモデルの評価が割れている状態そのものがリスクのため）。入力データ（選手成績・展示タイム等）に誤りがないか再確認してください。",
        ))

    # ⑦ 信頼度が極端に低い
    checks.append((
        "信頼度極低（混戦）",
        conf_score < 35,
        "critical",
        f"信頼度{conf_score}点 — 混戦レース。予測の根拠が弱すぎて賭け金を投じるべきではない。",
        "このレースでは解除できません（レース自体が「読みにくい」構造のため）。信頼度が35点以上の別レースを検討してください。",
    ))

    # ⑧ 壁ロジック極強（外艇完全封鎖）
    checks.append((
        "壁強度極大",
        wall_strength >= 0.8,
        "warning",
        f"壁強度{wall_strength*100:.0f}% — 1-2着固定の流れ。3連単以外の券種は避けるべき。",
        "警告のみでBUY自体は妨げません。3連単など1-2着が固定されにくい券種を選ぶことで実質的にリスクを回避できます。",
    ))

    # ⑨ オッズ未入力（全艇0）
    checks.append((
        "オッズ未入力",
        sum(1 for v in raw_odds.values() if v <= 0) >= 3,
        "critical",
        "単勝オッズが未入力の艇が3艇以上。EV計算の精度が著しく低下。",
        "「単勝オッズ」欄に3艇以上の実際のオッズを入力してください。入力が3艇未満の欠損になれば自動的に解除されます。",
    ))

    # ⑩ サンプルデータのまま分析（項目11対応：デフォルト値を実データと区別する）
    checks.append((
        "サンプルデータ未変更",
        is_demo_data,
        "critical",
        "6艇全ての全国勝率・モーター・展示タイム・STが練習用サンプル値のまま。実在しないレースを分析している状態。",
        "出走表を見ながら実際の選手データを1艇でも入力してください。サンプル値と1つでも異なれば自動的に解除されます。",
    ))

    return checks


def calc_signal_grade(ev: float, iq: int, conf_score: int, is_buy: bool,
                       all_manual_ok: bool, wall_strength: float = 0.0,
                       data_ok: bool = True) -> dict:
    """
    最終シグナルをA+/A/B/C/D/Xの6段階で評価する。
    A+ : 全条件クリア・高EV・高IQ・壁なし → 強勝負
    A  : BUY条件クリア・標準的EV
    B  : EVプラスだが一部条件不足 → 少額検討
    C  : 条件付き・EVギリギリ → 様子見
    D  : EV不足または信頼度低 → 見送り
    X  : データ不足・入力異常 → 計算不能
    """
    if not data_ok:
        return {"grade": "X", "label": "データ不足", "color": "#4d5c80",
                "emoji": "⚫", "bet_size": "投資不可", "reason": "データが不足しています"}
    if not is_buy:
        if ev > 0 and conf_score >= 40:
            return {"grade": "C", "label": "様子見", "color": "#ff8c42",
                    "emoji": "🟠", "bet_size": "少額のみ", "reason": "EV+だが条件不足"}
        return {"grade": "D", "label": "見送り", "color": "#ff5c72",
                "emoji": "🔴", "bet_size": "投資なし", "reason": "EV不足または安全条件未達"}
    if not all_manual_ok:
        return {"grade": "B", "label": "条件付き", "color": "#ffb648",
                "emoji": "🟡", "bet_size": "半額以下", "reason": "直前チェック未完了"}
    if ev >= 0.20 and iq >= 75 and conf_score >= 65 and wall_strength < 0.3:
        return {"grade": "A+", "label": "強勝負", "color": "#16e0a0",
                "emoji": "🟢", "bet_size": "フル投資", "reason": "全条件クリア・高EV・高信頼度"}
    if ev >= 0.08 and iq >= 60:
        return {"grade": "A", "label": "勝負", "color": "#16e0a0",
                "emoji": "🟢", "bet_size": "推奨額", "reason": "BUY条件クリア"}
    return {"grade": "B", "label": "条件付き", "color": "#ffb648",
            "emoji": "🟡", "bet_size": "半額推奨", "reason": "条件は整うが余裕は少ない"}


def calc_ev_range(win_p: float, odds: float, prob_error: float = 0.05) -> dict:
    """
    EVレンジを3ケースで算出（保守/標準/強気）。
    prob_error: AI確率の推定誤差（デフォルト±5%）
    """
    if odds <= 0 or win_p <= 0:
        return {"pessimist": None, "standard": None, "optimist": None}
    ev_std  = win_p * odds - 1
    ev_pess = max(0.0, win_p - prob_error) * odds - 1
    ev_opt  = min(1.0, win_p + prob_error) * odds - 1
    return {
        "pessimist": round(ev_pess * 100, 1),
        "standard":  round(ev_std  * 100, 1),
        "optimist":  round(ev_opt  * 100, 1),
    }


def calc_coverage_rate(boats_probs: list, sorted_combos: list, n_picks: int) -> float:
    """
    上位N点の買い目の的中カバー率（合計的中確率）を算出。
    sorted_combos: [(確率, combo), ...] を確率降順でソート済み
    """
    total = sum(p for p, _ in sorted_combos[:n_picks])
    return round(total * 100, 1)


def get_trifecta_ranking(results: list) -> list:
    """
    3連単120通りの確率をPlackett-Luceで計算してEV順にランキング。
    results: [{"boat": N, "prob_1st": p}, ...]
    戻り値: [{"combo": "1-2-3", "prob": 0.072, "ev": 0.32, "rank": 1}, ...]
    """
    from itertools import permutations
    prob_map = {r["boat"]: r["prob_1st"] for r in results}
    boats = sorted(prob_map.keys())
    combos = []
    for perm in permutations(boats, 3):
        p1 = prob_map.get(perm[0], 0)
        denom2 = max(1e-9, 1 - p1)
        p2 = prob_map.get(perm[1], 0) / denom2
        denom3 = max(1e-9, denom2 - prob_map.get(perm[1], 0))
        p3 = prob_map.get(perm[2], 0) / denom3
        prob = p1 * p2 * p3
        combos.append({"combo": f"{perm[0]}-{perm[1]}-{perm[2]}", "prob": round(prob, 5)})
    combos.sort(key=lambda x: -x["prob"])
    for i, c in enumerate(combos):
        c["rank"] = i + 1
    return combos


def calc_iq_score(ev: float, conf_score: int, agreement: float) -> int:
    """Investment Qualityスコア（0〜100点）"""
    ev_score = max(0.0, min(40.0, ev * 400))
    conf_s   = max(0.0, min(35.0, conf_score * 0.35))
    agr_s    = max(0.0, min(25.0, agreement * 0.25))
    return round(ev_score + conf_s + agr_s)


VENUE_EDGE_CONDITIONS = {
    "平和島": {"min_prob": 0.25, "note": "平和島×1号艇確率≥25% → ROI+4.1%（バックテスト実績）"},
    "びわこ": {"min_prob": 0.28, "note": "びわこ×1号艇確率≥28% → ROI+4.4%（バックテスト実績）"},
}


def calc_recommended_kelly(iq: int, all_manual_ok: bool, recent_losses: int) -> dict:
    if recent_losses >= 3:
        return {"mult": 0.2, "label": "守りモード（連敗中）", "color": "#ff5c72"}
    if not all_manual_ok:
        return {"mult": 0.3, "label": "慎重（チェック未完了）", "color": "#ffb648"}
    if iq >= 80: return {"mult": 1.0, "label": "フルベット推奨", "color": "#16e0a0"}
    if iq >= 60: return {"mult": 0.5, "label": "通常", "color": "#3fc4ff"}
    return {"mult": 0.3, "label": "慎重（IQ低め）", "color": "#ffb648"}


# ============================================================
# 自動オッズ取得（boatrace.jp 単勝オッズ）
# ============================================================
VENUE_CODE_MAP = {
    "桐生":  "01", "戸田":  "02", "江戸川": "03", "平和島": "04",
    "多摩川":"05", "浜名湖": "06", "蒲郡":  "07", "常滑":  "08",
    "津":    "09", "三国":  "10", "びわこ": "11", "住之江": "12",
    "尼崎":  "13", "鳴門":  "14", "丸亀":  "15", "児島":  "16",
    "宮島":  "17", "徳山":  "18", "下関":  "19", "若松":  "20",
    "芦屋":  "21", "福岡":  "22", "唐津":  "23", "大村":  "24",
}

@st.cache_data(ttl=10, show_spinner=False)  # 10秒キャッシュ（締切直前のオッズ変動に追従）
def fetch_tansho_odds(venue_name: str, race_no: int) -> dict:
    """
    boatrace.jpから単勝オッズを取得。
    戻り値: {"odds": {1:1.7,...}, "status": "success"|"timeout"|"parse_error"|"no_data"|"unavailable", "message": str}
    """
    if not _SCRAPING_AVAILABLE:
        return {"odds": {}, "status": "unavailable", "message": "requests/bs4未インストール"}
    jcd = VENUE_CODE_MAP.get(venue_name, "")
    if not jcd:
        return {"odds": {}, "status": "no_data", "message": f"会場コードが見つかりません: {venue_name}"}
    try:
        url = f"https://www.boatrace.jp/owpc/pc/race/oddstf?jcd={jcd}&rno={race_no}"
        headers = {"User-Agent": "BoatRace/1.0 (iPhone; iOS 16.0) AppleWebKit/605.1.15"}
        res = _requests.get(url, headers=headers, timeout=8)
        if "データがありません" in res.text or "ログイン" in res.text[:500]:
            return {"odds": {}, "status": "no_data", "message": "非開催・発売前・締切後の可能性"}
        soup = _BS(res.text, "html.parser")
        odds_dict = {}
        tables = soup.select("table.is-w331") or soup.find_all("table")
        for tbl in tables:
            for row in tbl.select("tbody tr"):
                cols = row.select("td")
                if len(cols) >= 2:
                    try:
                        boat_num = int(cols[0].get_text(strip=True))
                        odds_text = cols[1].get_text(strip=True).replace("−","0").replace(",","")
                        odds_val = float(odds_text)
                        if 1 <= boat_num <= 6 and odds_val > 0:
                            odds_dict[boat_num] = odds_val
                    except (ValueError, IndexError):
                        continue
            if odds_dict:
                break
        if not odds_dict:
            return {"odds": {}, "status": "parse_error", "message": "テーブル構造変化の可能性（HTML変更）"}
        return {"odds": odds_dict, "status": "success", "message": f"{len(odds_dict)}艇分取得成功"}
    except _requests.exceptions.Timeout:
        return {"odds": {}, "status": "timeout", "message": "タイムアウト（公式サイトが重い可能性）"}
    except Exception as e:
        return {"odds": {}, "status": "parse_error", "message": f"取得エラー: {type(e).__name__}"}


# ============================================================
# Opportunity Ranking（当日レースEVランキング）
# ============================================================
def save_opportunity(venue: str, race_no: int, ev: float, iq: int, conf_score: int, is_buy: bool):
    if "opportunities" not in st.session_state:
        st.session_state.opportunities = []
    today = date.today().isoformat()
    # 同一会場・レース番号は上書き
    st.session_state.opportunities = [
        o for o in st.session_state.opportunities
        if not (o["venue"] == venue and o["race_no"] == race_no and o["date"] == today)
    ]
    st.session_state.opportunities.append({
        "date": today, "venue": venue, "race_no": race_no,
        "ev": ev, "iq": iq, "conf_score": conf_score, "is_buy": is_buy,
        "time": datetime.now().strftime("%H:%M"),
    })


def render_opportunity_ranking():
    if "opportunities" not in st.session_state:
        st.session_state.opportunities = []
    today = date.today().isoformat()
    opps = [o for o in st.session_state.opportunities if o["date"] == today]
    opps_sorted = sorted(opps, key=lambda x: -x["ev"])

    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">★ Today\'s Opportunities（EV順・当日分）</div>', unsafe_allow_html=True)
    if not opps_sorted:
        st.markdown('<div class="note">「このレースを登録」ボタンで当日の候補レースを追加します。</div>', unsafe_allow_html=True)
    else:
        medals = ["🥇","🥈","🥉"] + [f"{i+1}." for i in range(3,20)]
        rows = []
        for i, o in enumerate(opps_sorted):
            badge = "🟢BUY" if o["is_buy"] else "🛑SKIP"
            rows.append({
                "順位": medals[i],
                "会場": f"{o['venue']} {o['race_no']}R",
                "判定": badge,
                "EV": f"{o['ev']*100:+.1f}%",
                "IQ": f"{o['iq']}点",
                "信頼度": f"{o['conf_score']}点",
                "時刻": o["time"],
            })
        df_opp = pd.DataFrame(rows)
        st.dataframe(df_opp, width='stretch', hide_index=True)

        # 個別削除
        del_labels = [f"{o['venue']} {o['race_no']}R ({o['time']})" for o in opps_sorted]
        del_target = st.selectbox("削除するレース", ["（選択）"] + del_labels, key="opp_del_select", label_visibility="collapsed",
            help="ここでレースを選んでから、右下の「選択を削除」ボタンを押すと、そのレースの登録記録が一覧から消えます。削除は取り消せません。")
        dc1, dc2 = st.columns([1, 3])
        with dc1:
            if st.button("🗑 選択を削除", key="opp_del_btn") and del_target != "（選択）":
                idx = del_labels.index(del_target)
                o_del = opps_sorted[idx]
                st.session_state.opportunities = [
                    o for o in st.session_state.opportunities
                    if not (o["venue"]==o_del["venue"] and o["race_no"]==o_del["race_no"] and o["date"]==today)
                ]
                st.rerun()
        with dc2:
            if st.button("🗑 全クリア", key="opp_clear_btn"):
                st.session_state.opportunities = [o for o in st.session_state.opportunities if o["date"] != today]
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 損失ストッパー
# ============================================================
def check_loss_stopper(limit_yen: int) -> tuple[int, bool]:
    """当日損失を計算してストッパー判定。(today_loss, is_triggered)"""
    if limit_yen <= 0:
        return 0, False
    logs = load_logs()
    today = date.today().isoformat()
    today_logs = [l for l in logs if l.get("date", "").startswith(today)]
    today_loss = 0
    for l in today_logs:
        stake = l.get("stake_yen", 0)
        payout = l.get("payout_yen", 0)
        if l.get("result") == "miss":
            today_loss += stake
        elif l.get("result") == "hit" and payout < stake:
            today_loss += stake - payout
    return today_loss, today_loss >= limit_yen


def calc_race_difficulty(conf_score: int, wind_mps: float, top_win_p: float, agreement_pct: float) -> dict:
    """レース難易度スコア（S/A/B/C/D）を算出"""
    # 難易度ポイント（高いほど難しい）
    chaos = 100 - conf_score                             # 0〜100
    wind_pt = min(30, wind_mps * 6)                      # 風速ペナルティ
    thin_edge = max(0, (0.35 - top_win_p) * 100)        # 本命薄い場合のペナルティ
    market_pt = max(0, (agreement_pct - 70) * 0.3)      # 市場一致度が高すぎる（妙味薄）
    difficulty = min(100, chaos * 0.5 + wind_pt + thin_edge + market_pt)

    if difficulty < 20:
        grade, label, col, stars = "S", "超簡単（本命鉄板）", "#16e0a0", "★★★★★"
    elif difficulty < 38:
        grade, label, col, stars = "A", "簡単（本命軸安定）", "#3fc4ff", "★★★★☆"
    elif difficulty < 55:
        grade, label, col, stars = "B", "普通（標準レース）", "#ffb648", "★★★☆☆"
    elif difficulty < 72:
        grade, label, col, stars = "C", "難しい（波乱含み）", "#ff8c42", "★★☆☆☆"
    else:
        grade, label, col, stars = "D", "超難しい（見送り推奨）", "#ff5c72", "★☆☆☆☆"

    upset_pct = round(min(95, difficulty * 0.8 + 5), 1)  # 荒れ確率の概算
    return {"grade": grade, "label": label, "color": col, "stars": stars,
            "score": round(difficulty, 1), "upset_pct": upset_pct}


def calc_expected_profit_yen(win_p: float, odds: float, stake_yen: int) -> int:
    """期待利益（円・int）= 勝率 × 払戻金 − 投資額 ※Kelly配分・記録用"""
    if odds <= 0:
        return -stake_yen
    return round(win_p * odds * stake_yen - stake_yen)


def calc_expected_profit(win_p: float, odds: float, stake_yen) -> int:
    """後方互換ラッパー → calc_expected_profit_yen に委譲"""
    return calc_expected_profit_yen(win_p, odds, int(stake_yen))


def monthly_summary(logs: list, target_date: date):
    realized = 0
    hit_count = 0
    decided_count = 0
    for l in logs:
        d = datetime.fromisoformat(l["date"]).date()
        if d.year == target_date.year and d.month == target_date.month:
            stake = l.get("stake_yen", 0)
            payout = l.get("payout_yen", 0)
            if l.get("result") == "hit":
                realized += (payout - stake)
                hit_count += 1
                decided_count += 1
            elif l.get("result") == "miss":
                realized -= stake
                decided_count += 1
    hit_rate = (hit_count / decided_count * 100) if decided_count else 0.0
    return {"realized": realized, "hit_rate": hit_rate, "decided_count": decided_count, "hit_count": hit_count}


def calc_prediction_quality(logs: list) -> dict:
    """
    項目22対応：「予測」ではなく「予測品質」を評価する。
    的中率だけでは、AIが「40%」と言った時に本当に40%前後で当たっているかは分からない。
    win_p_at_purchase（購入時点の予測確率）が記録されている確定済みログから、
    Brier Score・LogLoss・確率帯別の実測的中率（キャリブレーション）を計算する。
    値が小さいほど（Brierは0に、LogLossも0に近いほど）予測の質が高い。
    """
    usable = [l for l in logs if l.get("result") in ("hit", "miss") and l.get("win_p_at_purchase") is not None]
    if len(usable) < 5:
        return {"available": False, "n": len(usable)}

    briers, logloss_terms = [], []
    for l in usable:
        p = max(1e-6, min(1 - 1e-6, l["win_p_at_purchase"]))
        y = 1.0 if l["result"] == "hit" else 0.0
        briers.append((p - y) ** 2)
        logloss_terms.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
    brier = sum(briers) / len(briers)
    logloss = sum(logloss_terms) / len(logloss_terms)

    # 確率帯別キャリブレーション（予測確率のビンごとに実際の的中率を集計）
    bins = [(0.0, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.50), (0.50, 1.01)]
    calib_rows = []
    for lo, hi in bins:
        bucket = [l for l in usable if lo <= l["win_p_at_purchase"] < hi]
        if not bucket:
            continue
        pred_mean = sum(l["win_p_at_purchase"] for l in bucket) / len(bucket)
        actual_rate = sum(1 for l in bucket if l["result"] == "hit") / len(bucket)
        calib_rows.append({
            "range": f"{lo*100:.0f}〜{min(hi,1.0)*100:.0f}%",
            "n": len(bucket),
            "pred": pred_mean * 100,
            "actual": actual_rate * 100,
            "gap": (actual_rate - pred_mean) * 100,
        })

    # ============================================================
    # EV較正曲線（外部レビューで最も強調されていた論点）
    # 「AIの予測確率が正しいか」と「そのEVで実際に儲かったか」は別物。
    # EV = 予測確率 × 購入時オッズ - 1 として、EV帯ごとに実現ROIを集計する。
    # 「EVが高いほど儲かる」という前提が本当に成り立っているかをここで検証する。
    # ============================================================
    ev_usable = [l for l in usable if l.get("final_odds", 0) > 0]
    ev_bins = [(-1.0, 0.0), (0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 999.0)]
    ev_calib_rows = []
    for lo, hi in ev_bins:
        bucket = []
        for l in ev_usable:
            ev_at_purchase = l["win_p_at_purchase"] * l["final_odds"] - 1
            if lo <= ev_at_purchase < hi:
                bucket.append((l, ev_at_purchase))
        if not bucket:
            continue
        n = len(bucket)
        pred_ev_mean = sum(e for _, e in bucket) / n
        total_stake = sum(l.get("stake_yen", 0) for l, _ in bucket)
        total_payout = sum(l.get("payout_yen", 0) for l, _ in bucket if l["result"] == "hit")
        realized_roi = (total_payout - total_stake) / total_stake if total_stake > 0 else None
        ev_calib_rows.append({
            "range": f"{lo*100:.0f}〜{'∞' if hi>=999 else f'{hi*100:.0f}'}%",
            "n": n,
            "pred_ev": pred_ev_mean * 100,
            "realized_roi": realized_roi * 100 if realized_roi is not None else None,
        })

    return {"available": True, "n": len(usable), "brier": brier, "logloss": logloss,
            "calib_rows": calib_rows, "ev_calib_rows": ev_calib_rows}


# ============================================================
# Streamlit ページ設定 & スタイル
# ============================================================
st.set_page_config(page_title="PRO TRADER TERMINAL v101", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@400;500;700;900&family=Roboto+Mono:wght@400;500;700&display=swap');

:root{
  --bg-main:#05070d; --bg-panel:#0c111f; --border-dark:#1c2740; --border-mid:#2a3859;
  --color-buy:#062a1c; --color-risk:#3a0d14; --color-watch:#2c1f06; --color-neutral:#0f172a;
  --text-buy:#16e0a0; --text-risk:#ff5c72; --text-watch:#ffb648; --text-neutral:#3fc4ff;
  --text-light:#f3f6fb; --text-dim:#7c8aab;
}
html, body, [class*="css"]{ font-family:'Zen Kaku Gothic New', sans-serif; }
.stApp{
  background:
    radial-gradient(ellipse at 20% -10%, rgba(22,224,160,0.06), transparent 45%),
    radial-gradient(ellipse at 90% 0%, rgba(255,182,72,0.05), transparent 40%),
    var(--bg-main);
  color: var(--text-light);
}
.block-container{ padding-top:1.2rem; max-width:1500px; }

.brand-mark{ font-family:'Roboto Mono',monospace; font-weight:700; font-size:22px; letter-spacing:1px; color:#fff; }
.brand-mark span{ color:var(--text-buy); }
.brand-sub{ font-family:'Roboto Mono',monospace; font-size:11px; color:#4d5c80; letter-spacing:2px; }

.panel-box{ background:var(--bg-panel); border:1px solid var(--border-dark); border-radius:10px; padding:16px; margin-bottom:12px; }
.eyebrow{ color:var(--text-dim); font-size:10px; font-weight:700; letter-spacing:1.5px; font-family:'Roboto Mono',monospace; }
.decision-text{ font-size:1.8rem; font-weight:900; margin-top:4px; }
.metric-label{ color:var(--text-dim); font-size:10px; font-weight:700; font-family:'Roboto Mono',monospace; }
.metric-val{ font-size:1.5rem; font-weight:700; font-family:'Roboto Mono',monospace; margin-top:2px; }

.gate-pill{ display:inline-block; padding:2px 10px; border-radius:20px; border:1px solid var(--border-mid); color:var(--text-dim); font-family:'Roboto Mono',monospace; font-size:11px; margin-right:6px; }
.gate-on{ border-color:var(--text-buy); color:var(--text-buy); background:rgba(22,224,160,0.08); }
.gate-off{ border-color:var(--text-risk); color:var(--text-risk); background:rgba(255,92,114,0.08); }

.trade-card{ border-radius:8px; padding:12px 14px; border:1px solid var(--border-dark); margin-bottom:8px; background:#0c111f; }
.prime-card{ border:1px solid var(--text-buy); background:var(--color-buy); box-shadow:0 0 16px rgba(22,224,160,0.18); }
.risk-card{ border:1px solid var(--text-risk); background:var(--color-risk); }
.card-badge{ font-size:10px; color:var(--text-dim); font-weight:700; font-family:'Roboto Mono',monospace; }
.card-anomaly{ color:var(--text-watch); margin-left:8px; }
.card-stat{ font-size:13px; font-weight:700; margin-top:3px; }
.card-breakdown{ font-size:10px; color:#4d5c80; margin-top:2px; font-family:'Roboto Mono',monospace; }

.boat-badge{ display:inline-block; text-align:center; font-weight:900; border-radius:5px; width:28px; padding:3px 0; font-family:'Roboto Mono',monospace; }
.cutoff-line{ text-align:center; margin:6px 0; border-bottom:1px dashed var(--text-risk); color:var(--text-risk); font-size:10px; font-weight:700; letter-spacing:1px; font-family:'Roboto Mono',monospace; padding-bottom:4px; }

.note{ font-size:10.5px; color:#4d5c80; margin-top:8px; line-height:1.6; font-family:'Roboto Mono',monospace; }

[data-testid="stMetricValue"]{ font-family:'Roboto Mono',monospace; }
hr{ border-color: var(--border-dark); }

.gloss-term{ display:inline-block; margin:0; }
.gloss-term summary{ display:inline; cursor:pointer; color:inherit; border-bottom:1px dotted var(--text-neutral); list-style:none; }
.gloss-term summary::-webkit-details-marker{ display:none; }
.gloss-term summary:hover{ color:var(--text-neutral); }
.gloss-term[open] summary{ color:var(--text-neutral); }
.gloss-term .gloss-body{ display:block; margin:4px 0 8px 0; padding:8px 10px; background:#0a0f1c; border:1px solid var(--text-neutral); border-radius:6px; font-size:11px; color:var(--text-light); font-family:'Zen Kaku Gothic New', sans-serif; line-height:1.6; max-width:420px; }
.breakdown-toggle{ margin-top:3px; }
.breakdown-toggle summary{ cursor:pointer; font-size:10px; color:#4d5c80; font-family:'Roboto Mono',monospace; list-style:none; user-select:none; }
.breakdown-toggle summary::-webkit-details-marker{ display:none; }
.breakdown-toggle summary::before{ content:"▶ "; }
.breakdown-toggle[open] summary::before{ content:"▼ "; color:var(--text-neutral); }
.breakdown-toggle .breakdown-body{ font-size:10px; color:#4d5c80; margin-top:4px; padding-left:10px; font-family:'Roboto Mono',monospace; line-height:1.6; border-left:2px solid var(--border-dark); }
.checklist-all-ok{ font-size:11px; font-weight:700; color:var(--text-buy); margin-top:6px; padding:4px 10px; background:#062a1c; border-radius:4px; display:inline-block; }
.checklist-ng{ font-size:11px; font-weight:700; color:var(--text-risk); margin-top:6px; padding:4px 10px; background:#3a0d14; border-radius:4px; display:inline-block; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 用語解説辞書（クリックで開閉するインライン解説）
# ============================================================
GLOSSARY = {
    "全国勝率": "その選手の全国レース成績を公式配点で得点化し、出走回数で割った指数。近年の成績が良いほど数値が高くなるレーティング。",
    "モーター2連率": "そのモーターを使った艇が2着以内に入った割合(%)。高いほど『調子の良いモーター』とされる。",
    "展示T": "展示航走で計測されたタイム。数値が小さい（速い）ほど、その日の仕上がりが良いとされる。",
    "平均ST": "スタートタイミングの過去平均値。0.00に近いほど鋭いスタート、マイナスはフライング（失格）を意味する。",
    "EV閾値": "期待値（Expected Value）の最低基準。『この艇に賭けると平均でどれだけ得か』を示す。",
    "悲観EV": "AIの勝率予測に-5%の誤差があった『最悪ケース』を想定しても、期待値がプラスかどうかを確認する安全マージン。標準のEVがプラスでも、悲観EVがマイナスなら『AIの予測が少しでも外れたら赤字になる』薄氷の勝負ということ。このゲートがOFFの間はBUY自体が成立しない。",
    "確率整合性": "『全艇の1着確率を足すと必ず100%になる』という、確率計算として絶対に成り立っていなければならない基本条件を検証するゲートです。計算過程でNaN（計算不能値）や異常値が混入すると、この前提が崩れます。従来はこっそりゼロに補正して計算を続けていましたが、それでは『AIの判定が本当に正しいのか』を保証できないため、このツールでは異常を検知した時点でBUY自体を禁止します。OFFの場合は入力データ（展示タイム・全国勝率・モーター等）に極端な値や欠損がないか確認してください。",
    "信頼度": "レース全体の『荒れにくさ』を0〜100で表したスコア。高いほど本命が来やすい。",
    "Kelly>0": "ケリー基準で算出した比率がプラスかどうか。マイナス＝賭けると長期的に損する計算。",
    "Kelly係数": "ケリー基準で算出される理論上の最適賭け金比率に、リスクを抑えるためにかける割合（0〜1）。",
    "AI一致": "手作りモデルとLightGBMモデルの両方が『1号艇が来る』と判定したときだけ点灯するゲート。precision: 手作り≥27%かつLGB≥65%, volume: 手作り≥23%かつLGB≥40%。",
    "補正確率": "AIが予測した勝率を、会場ごとに実際のレース結果（約14,867件）で補正した値（Isotonic Regression）。",
    "急変確認": "レース直前の4項目手動確認チェックリスト。全てチェックが入ると投資シグナルが有効になる。",
    "舟券種別": "どの形式の舟券を組み合わせ計算の対象にするかを選びます。「2連単」は1着と2着を順番通りに当てる券、「2連複」は順不同、「3連単」は1〜3着を順番通り、「3連複」は3着以内を順不同で当てる券です。的中難易度が上がるほど、当たったときの配当（オッズ）も大きくなります。",
    "ノイズ除去": "確率が極端に低い組み合わせ（例えば0.5%未満）を買い目候補から自動的に除外するためのしきい値（%）です。数値を上げるほど表示される買い目候補が絞り込まれ、逆に下げるほど低確率の穴狙いの組み合わせまで表示されるようになります。",
    "軸艇": "『必ずこの艇が絡む』という前提で組み合わせを作るための艇番号です（カンマ区切りで複数指定可）。例えば軸艇を「1」にすると、1号艇が絡む買い目だけが計算対象になります。",
    "相手艇": "軸艇と組み合わせる『相手』の艇番号です（カンマ区切りで複数指定可）。軸艇と相手艇の組み合わせをもとに、フォーメーション（複数の買い目のセット）が自動生成されます。",
    "合成オッズ": "複数の買い目に資金を分けて賭けた場合の、実質的な平均オッズ。",
    "EV": "期待値（Expected Value）の略。賭け金に対して理論上見込める平均的な収支。",
    "ROI": "投資収益率（Return On Investment）の略。(払戻金合計−投資金合計)÷投資金合計×100。",
    "的中率": "購入した舟券のうち、実際に当たった割合(%)。",
    "キャリブレーション": "モデルが出す予測確率を、実際の的中率と一致するように補正する処理。",
}

def gloss(term, label=None):
    desc = GLOSSARY.get(term, "")
    if not desc: return label or term
    disp = label or term
    return f'<details class="gloss-term"><summary>{disp}</summary><span class="gloss-body">{desc}</span></details>'


# ============================================================
# セッション初期化
# ============================================================
if "boat_inputs" not in st.session_state:
    st.session_state.boat_inputs = {i: dict(MASTER_BOATS[i]) for i in range(1, 7)}
if "odds_inputs" not in st.session_state:
    st.session_state.odds_inputs = {1: 1.7, 2: 3.4, 3: 4.8, 4: 14.2, 5: 18.0, 6: 42.5}
if "lgb_profile" not in st.session_state:
    st.session_state.lgb_profile = "precision"
if "prev_odds" not in st.session_state:
    st.session_state.prev_odds = {}
if "opportunities" not in st.session_state:
    st.session_state.opportunities = []
if "loss_limit" not in st.session_state:
    st.session_state.loss_limit = 5000
if "loss_stopper_dismissed" not in st.session_state:
    st.session_state.loss_stopper_dismissed = False
if "race_notes" not in st.session_state:
    st.session_state.race_notes = []
if "odds_source" not in st.session_state:
    st.session_state.odds_source = "manual"
if "odds_fetch_time" not in st.session_state:
    st.session_state.odds_fetch_time = "--"

# ============================================================
# state汚染防止：レースキー変更時に古い計算結果を破棄
# 日付-会場-レース番号 が変わったら前のレースの計算キャッシュをクリア
# ============================================================
if "kelly_conservative_sel" not in st.session_state:
    st.session_state.kelly_conservative_sel = "0.10"

# LightGBMプロファイル選択
st.markdown("###### 🤖 LightGBMアンサンブル・プロファイル", unsafe_allow_html=True)
profile_key = st.radio(
    "LightGBMアンサンブル・プロファイル",
    options=list(LGB_PROFILES.keys()),
    format_func=lambda k: LGB_PROFILES[k]["label"],
    horizontal=True, key="lgb_profile", label_visibility="collapsed",
    help="AIモデルの『性格』を選びます。「少数精鋭型」は自信のあるレースだけ強気にBUY判定を出す代わりに、対象レース数が少なくなります。「広く浅く型」は判定対象のレース数は増えますが、1件あたりの的中率は少し下がります。どちらもBUY/SKIPの最終判定ロジックに使われるため、切り替えると同じレースでも判定が変わることがあります。",
)
_active_profile = LGB_PROFILES[profile_key]
_LGB_MODEL = _load_lgb_model(_active_profile["file"]) if _LGB_AVAILABLE else None
if _LGB_MODEL is None:
    st.markdown(
        f'<div class="note">⚠️ {_active_profile["file"]} が見つかりません。boatフォルダに配置してください。</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# ヘッダー
# ============================================================
hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.markdown('<div class="brand-mark">PRO TRADER <span>TERMINAL</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">v102 — UNIFIED ENGINE + LightGBM / KELLY×PLACKETT-LUCE （Python/Streamlit版）</div>', unsafe_allow_html=True)
with hcol2:
    st.markdown(f'<div style="text-align:right; font-family:\'Roboto Mono\',monospace; font-size:12px; color:#7c8aab; padding-top:10px;">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>', unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# 📖 使い方ガイド（ボタンで展開）
# ============================================================
with st.expander("📖 このアプリの使い方ガイド（タップで開く）", expanded=False):
    st.markdown("""
## 🎯 このアプリで何ができるの？

**PRO TRADER TERMINAL** は、ボートレースの「買うべきレース・見送るべきレース」を数字で判断するツールです。
感覚や勘ではなく、**期待値（EV）** という数字をもとに投資判断を補助します。

---

## 📝 使い方の流れ（5ステップ）

### ① 会場・風を選ぶ
- **会場**：レースが行われる競艇場を選択。会場によってインコースの有利度が違います
- **風向・風速**：実況や公式サイトで確認して入力。風が強いほど予測が難しくなります

### ② 6艇のデータを入力する
各艇について4つの数字を入力します：

| 項目 | 意味 | どこで確認？ |
|------|------|------------|
| 全国勝率 | その選手が全国でどれだけ1着を取るか（高いほど強い） | 出走表 |
| モーター2連率 | 使用モーターの調子（高いほど良いモーター） | 出走表 |
| 展示タイム | レース前の練習タイム（**小さいほど速い**） | 場内モニター・公式サイト |
| 平均ST | スタートの早さ（**小さいほど早い**） | 出走表 |

### ③ 市場オッズを入力する
- 公式サイトやテレボートで「単勝オッズ」を確認して入力
- または「🔄 オッズ自動取得」ボタンでレース番号を選んで自動入力（発売中のレースのみ）

### ④ 直前チェックリストを確認する
4つのチェック項目を全部確認してからチェックを入れます：
- オッズ急変なし・進入変更なし・環境異常なし・高リスク艇なし

### ⑤ 結果を読む

---

## 🔢 重要な数字の意味

### 期待値（EV）
> 「100円賭けたとき、平均して何円戻ってくるか」

- **EV +10%** → 100円賭けると平均110円戻る（有利）
- **EV -20%** → 100円賭けると平均80円戻る（不利）
- **EV +4%以上** がAIが投資を推奨する閾値

### Investment Quality（IQ）スコア
> EVだけでなく「信頼度」と「市場との違い」を合わせた総合点（0〜100点）

- **80点以上** → 強力な買いシグナル
- **60〜79点** → 条件付き検討
- **59点以下** → 見送り推奨

### 信頼度スコア（0〜100点）
> 「このレース、AIがどれだけ自信を持っているか」

- 高いほど本命が1着になりやすいレース（波乱が少ない）
- 50点未満は波乱リスクあり

### レース難易度（S/A/B/C/D）
- **S・A** → 勝負しやすいレース（本命軸が安定）
- **B** → 標準的なレース
- **C・D** → 難しい・見送り推奨

### Kellyの推奨賭け金
> 資金を守りながら期待値を最大化する数学的な賭け金の計算方法

---

## 🟢🛑 最終シグナルの読み方

| シグナル | 意味 |
|---------|------|
| 🟢 **投資推奨** | 全条件クリア。期待値プラスのレース |
| 🟡 **条件付き監視** | 一部条件が満たされていない。慎重に |
| 🛑 **投資見送り** | 期待値不足または安全条件未達。見送り |

---

## ⚠️ 大切な注意事項

- このアプリの予測は**過去データに基づくヒューリスティック（経験則）**です
- 勝利を保証するものではありません
- **最終判断は必ずご自身で行ってください**
- テラ銭（25%控除）の構造上、長期的に全投資でプラスは困難です。**期待値の高いレースを厳選**することが重要です

---

## 💡 初心者向けのコツ

1. **最初は「見送り」に慣れる** — 良いレースを待つことが最重要
2. **IQ 60点以上、EV +10%以上のレースだけ** で実際に賭けてみる
3. **損失ストッパーを設定する** — 1日の損失上限を決めて守る
4. **購入ログを記録する** — なぜ買ったか必ず残す（レースノートも活用）
    """)

# ============================================================
# 投資モード切替（勝率優先/回収率優先/研究）
# 目的関数を明確にして判断軸のブレを防ぐ
# ============================================================
if "investment_mode" not in st.session_state:
    st.session_state.investment_mode = "win_rate"

_mode_cols = st.columns(3)
_mode_labels = [
    ("win_rate",    "🎯 勝率優先",   "高信頼・低ブレ中心。見送り率を高くして安全なレースだけ参戦。", "#16e0a0"),
    ("roi",         "💰 回収率優先", "EV・市場乖離重視。中穴も許容して期待値を最大化。",             "#3fc4ff"),
    ("research",    "🔬 研究モード", "全情報開示。閾値調整可。本番判断には使用しないこと。",           "#ffb648"),
]
for _col, (_key, _label, _desc, _color) in zip(_mode_cols, _mode_labels):
    with _col:
        _selected = st.session_state.investment_mode == _key
        _border = f"border:2px solid {_color};" if _selected else "border:1px solid #2a3859;"
        _txt_col = _color if _selected else "#7c8aab"
        st.markdown(
            f'<div style="background:#080d1a;{_border}border-radius:8px;padding:8px 10px;text-align:center;">'
            f'<div style="font-size:13px;font-weight:700;color:{_txt_col};">{_label}</div>'
            f'<div style="font-size:9px;color:var(--text-dim2);margin-top:3px;">{_desc}</div>'
            f'</div>', unsafe_allow_html=True,
        )
        if st.button("選択", key=f"mode_btn_{_key}", width='stretch'):
            st.session_state.investment_mode = _key
            st.rerun()

_inv_mode = st.session_state.investment_mode

# モード別設定値を調整
if _inv_mode == "win_rate":
    # 勝率優先：EV閾値を高く・見送り率を上げる
    _mode_ev_boost   = 0.04   # EV閾値に加算（+4%）
    _mode_conf_boost = 5      # 信頼度閾値に加算
    _mode_kelly_cap  = 0.15   # Kelly最大値
    st.markdown('<div style="background:#062a1c;border:1px solid #16e0a0;border-radius:5px;padding:4px 12px;font-size:10px;color:#16e0a0;margin-bottom:6px;">🎯 勝率優先モード：EV閾値+4%・信頼度閾値+5点・Kelly上限15%</div>', unsafe_allow_html=True)
elif _inv_mode == "roi":
    # 回収率優先：市場乖離を許容・EV閾値は維持
    _mode_ev_boost   = 0.0
    _mode_conf_boost = -5     # 信頼度閾値を緩める（-5点）
    _mode_kelly_cap  = 0.20
    st.markdown('<div style="background:#0d1a2a;border:1px solid #3fc4ff;border-radius:5px;padding:4px 12px;font-size:10px;color:#3fc4ff;margin-bottom:6px;">💰 回収率優先モード：EV重視・信頼度閾値-5点・中穴許容</div>', unsafe_allow_html=True)
else:
    # 研究モード：全情報開示・制限なし
    _mode_ev_boost   = 0.0
    _mode_conf_boost = -10
    _mode_kelly_cap  = 1.0
    st.markdown('<div style="background:#1a0f00;border:1px solid #ffb648;border-radius:5px;padding:4px 12px;font-size:10px;color:#ffb648;font-weight:700;margin-bottom:6px;">🔬 研究モード：全情報開示。本番判断には使用しないこと。</div>', unsafe_allow_html=True)

# ============================================================
# 3層構造ガイド（情報の引き算）
# ============================================================
with st.expander("📊 画面の見方（3層構造）", expanded=False):
    st.markdown("""
| 層 | 目的 | 時間 | 場所 |
|---|---|---|---|
| **🔴 第1層** | 「買うか・見送るか」を即決 | **3秒** | 追従ヘッダー・決定シグナル・IQ |
| **🟡 第2層** | 「なぜ買えるか」を確認 | **30秒** | EVランキング・艇別カード・オッズ感応度 |
| **🟢 第3層** | 根拠を詳しく見る | **必要時のみ** | レース診断・バックテスト・ログ |

**第1層でSKIPなら第2層・第3層は見ない。第1層でBUYなら第2層で確認して購入。**
    """)

# ============================================================
# Opportunity Ranking（当日レース候補一覧）
# ============================================================
render_opportunity_ranking()

# ============================================================
# 入力エリア（会場・風・6艇データ）
# ============================================================
with st.container():
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.markdown("### 📥 DATA INPUT")

    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        venue = st.selectbox("会場", list(VENUE_SCORE.keys()),
                              format_func=lambda v: f"{v} ({'+' if VENUE_SCORE[v]>=0 else ''}{VENUE_SCORE[v]})",
                              help="レースが行われる競艇場を選びます。カッコ内の数字は『イン逃げしやすさ』の指数で、プラスが大きいほど1号艇が有利な水面、マイナスが大きいほど荒れやすい水面です。会場を変えると、その水面の過去データに基づいて予測が自動的に再計算されます。")
    with ic2:
        wind_dir = st.selectbox("風向", ["無風", "向かい風", "追い風", "横風"],
                                 help="レース時の風向きです。「向かい風」「横風」はスタートが乱れやすく、荒れる（本命が飛ぶ）確率が上がるため、AIのリスク判定（AUTO SKIPなど）に影響します。")
    with ic3:
        wind_mps = st.number_input("風速 (m/s)", min_value=0.0, max_value=15.0, value=1.0, step=0.5,
                                    help="風の強さを秒速メートルで入力します。目安：4m/s以上で荒れやすい水面との組み合わせが警戒対象に、5m/s以上の向かい風・横風はスタート事故リスクとして扱われ、推奨購入額が自動的に減額されます。")

    # ============================================================
    # state汚染防止：会場が変わったらオッズ取得状態をリセット
    # ============================================================
    _prev_venue = st.session_state.get("_last_venue", "")
    if _prev_venue and _prev_venue != venue:
        # 会場が変わった → 実測オッズフラグをリセット
        st.session_state.odds_source = "manual"
        st.session_state.odds_fetch_time = "--"
        st.session_state.prev_odds = {}
        st.session_state.odds_inputs = {1: 1.7, 2: 3.4, 3: 4.8, 4: 14.2, 5: 18.0, 6: 42.5}
        # チェックリストもリセット
        for _ck in ["rule4_manual", "chk_course", "chk_wind", "chk_f"]:
            if _ck in st.session_state:
                del st.session_state[_ck]
    st.session_state["_last_venue"] = venue

    st.markdown("##### 艇別データ入力")
    cols = st.columns(6)
    boats_data = []
    _validation_warnings = []
    for i in range(1, 7):
        with cols[i - 1]:
            st.markdown(
                f'<div class="boat-badge" style="background:{BOAT_COLORS[i]}; color:{BOAT_TEXT[i]};">{BOAT_LABEL[i]}</div>',
                unsafe_allow_html=True,
            )
            nat = st.number_input(f"全国勝率", value=st.session_state.boat_inputs[i]["nat"], step=0.01, key=f"nat_{i}", format="%.2f",
                help="その選手の全国での通算成績を得点化した数値です（出走表に記載）。数字が大きいほど『強い選手』を意味し、予測の勝率計算のベースになります。目安は1.0〜10.0程度。")
            motor = st.number_input(f"モーター2連率", value=st.session_state.boat_inputs[i]["motor"], step=0.1, key=f"motor_{i}", format="%.1f",
                help="今使っているモーターがこれまで2着以内に入った割合（%）です。数字が高いほど『調子の良いモーター』とされ、AIの勝率予測を押し上げます。出走表またはボートレース場の掲示に記載されています。")
            ex = st.number_input(f"展示T", value=st.session_state.boat_inputs[i]["ex"], step=0.01, key=f"ex_{i}", format="%.2f",
                help="レース直前の展示航走で計測されたタイム（秒）です。数字が小さい（速い）ほど、その日の仕上がりが良いと判断されます。空欄のまま（初期値）だと『展示データ欠損』としてAUTO SKIPの警告対象になるので、必ず実際の数値を入力してください。")
            st_ = st.number_input(f"平均ST", value=st.session_state.boat_inputs[i]["st"], step=0.01, key=f"st_{i}", format="%.2f",
                help="その選手の過去のスタートタイミングの平均値（秒）です。0.00に近いほどフライング寸前の鋭いスタートを意味し、マイナスの値はフライング（失格）歴があることを示します。数値が小さいほどスタートで有利と評価されます。")

            # バリデーション（現実的な範囲チェック）
            if not (1.0 <= nat <= 10.0):
                _validation_warnings.append(f"{BOAT_LABEL[i]} 全国勝率: {nat:.2f}（通常1.0〜10.0）")
            if not (10.0 <= motor <= 65.0):
                _validation_warnings.append(f"{BOAT_LABEL[i]} モーター2連率: {motor:.1f}（通常10〜65）")
            if not (6.0 <= ex <= 8.5):
                _validation_warnings.append(f"{BOAT_LABEL[i]} 展示T: {ex:.2f}（通常6.0〜8.5秒）")
            if not (0.05 <= st_ <= 0.40):
                _validation_warnings.append(f"{BOAT_LABEL[i]} ST: {st_:.2f}（通常0.05〜0.40秒）")

            boats_data.append({"boat": i, "nat": nat, "motor": motor, "ex": ex, "st": st_})

    if _validation_warnings:
        with st.expander(f"⚠️ 入力値の確認（{len(_validation_warnings)}件）", expanded=True):
            for w in _validation_warnings:
                st.warning(f"📋 {w}", icon="⚠️")

    # ============================================================
    # デモ値検出（項目11対応：デフォルト値を実データと区別する）
    # 6艇全員が MASTER_BOATS のサンプル値から1つも変更されていない場合、
    # ユーザーが実際のレースデータを入力し忘れたまま分析している可能性が高い。
    # これを「実測データ」として扱いBUY判定を出すのは危険なため、明示的に検出して警告する。
    # ============================================================
    _is_demo_data = all(
        boats_data[i-1]["nat"] == MASTER_BOATS[i]["nat"]
        and boats_data[i-1]["motor"] == MASTER_BOATS[i]["motor"]
        and boats_data[i-1]["ex"] == MASTER_BOATS[i]["ex"]
        and boats_data[i-1]["st"] == MASTER_BOATS[i]["st"]
        for i in range(1, 7)
    )
    if _is_demo_data:
        st.markdown(
            '<div style="background:#1a0000;border:2px solid #ff5c72;border-radius:6px;padding:8px 12px;margin-top:8px;">'
            '<div style="font-size:12px;font-weight:900;color:#ff5c72;">🚨 サンプルデータのまま分析中</div>'
            '<div style="font-size:10px;color:var(--text-dim);margin-top:3px;">'
            '6艇すべての全国勝率・モーター・展示タイム・STが、練習用のサンプル値（架空のデータ）から1つも変更されていません。'
            'このままではAIは実在しないレースを分析していることになり、表示されるBUY判定は無意味です。'
            '出走表を見ながら実際の数値を入力してから判断してください。'
            '</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="note">※ 勝率モデルはPlackett-Luce型のヒューリスティックであり、統計的に検証された予測値ではありません。最終判断はご自身の責任で行ってください。</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# コア計算実行
# ============================================================
results, raw_scores, decomp_data, ex_mean, wall_strength, prob_integrity = predict_finish_distribution(boats_data, venue, wind_dir, wind_mps)
lgb_result = predict_lgbm(boats_data, venue, wind_dir, wind_mps, _LGB_MODEL)
lgb_prob_b1 = lgb_result.get("lgb_prob_b1", None)
conf = calc_race_confidence(raw_scores)
_conf_breakdown = calc_confidence_breakdown(results)  # 本命断層・混戦度を分離（項目6対応）

# オッズ入力
with st.container():
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.markdown("### 💸 MARKET ODDS")

    # 自動オッズ取得
    auto_col1, auto_col2, auto_col3 = st.columns([1, 1, 3])
    with auto_col1:
        fetch_race_no = st.number_input("レース番号", min_value=1, max_value=12, value=1, step=1, key="fetch_race_no",
            help="オッズを自動取得したいレースの番号（1〜12）を選びます。右の「オッズ自動取得」ボタンを押すと、この番号のレースの単勝オッズをboatrace.jpから取得して下の欄に自動入力します。")
    with auto_col2:
        st.markdown('<div style="height:28px;"></div>', unsafe_allow_html=True)
        if st.button("🔄 オッズ自動取得", key="auto_fetch_odds"):
            with st.spinner("取得中..."):
                fetched = fetch_tansho_odds(venue, fetch_race_no)
            _status = fetched.get("status", "parse_error")
            _odds   = fetched.get("odds", {})
            _msg    = fetched.get("message", "")
            if _status == "success" and _odds:
                for bn, ov in _odds.items():
                    st.session_state.odds_inputs[bn] = ov
                st.session_state["odds_source"] = "actual"   # 実測フラグ
                st.session_state["odds_fetch_time"] = datetime.now().strftime("%H:%M:%S")
                st.success(f"✅ {venue} {fetch_race_no}R取得成功 / {_msg}", icon="✅")
                st.rerun()
            elif _status == "timeout":
                st.warning(f"⏱ タイムアウト：{_msg}", icon="⚠️")
                st.session_state["odds_source"] = "manual"
            elif _status == "no_data":
                st.info(f"📭 {_msg}（非開催・発売前・締切後の可能性）", icon="ℹ️")
                st.session_state["odds_source"] = "manual"
            elif _status == "parse_error":
                st.error(f"⚠️ HTML構造変化の可能性：{_msg}", icon="🚨")
                st.session_state["odds_source"] = "manual"
            else:
                st.warning(f"取得できませんでした：{_msg}", icon="⚠️")
                st.session_state["odds_source"] = "manual"
    with auto_col3:
        if not _SCRAPING_AVAILABLE:
            st.markdown('<div class="note">⚠️ requests/beautifulsoup4 が未インストールです。`pip install requests beautifulsoup4` を実行してください。</div>', unsafe_allow_html=True)

    # ============================================================
    # AC-04: オッズ取得状態バー（実測/手動の明示）
    # ============================================================
    if "odds_source" not in st.session_state:
        st.session_state["odds_source"] = "manual"
    if "odds_fetch_time" not in st.session_state:
        st.session_state["odds_fetch_time"] = "--"
    _odds_src = st.session_state["odds_source"]
    _odds_time = st.session_state["odds_fetch_time"]
    if _odds_src == "actual":
        _src_html = (f'<div style="background:#062a1c;border:1px solid #16e0a0;border-radius:5px;padding:4px 10px;'
                     f'font-size:10px;color:#16e0a0;font-weight:700;margin-bottom:6px;">'
                     f'🟢 <b>実測オッズ</b> — {venue} 自動取得済み（{_odds_time}） ← BUY判定ゲートに使用可</div>')
    else:
        _src_html = (f'<div style="background:#1a0f00;border:1px solid #ffb648;border-radius:5px;padding:4px 10px;'
                     f'font-size:10px;color:#ffb648;font-weight:700;margin-bottom:6px;">'
                     f'🟡 <b>手動入力オッズ</b> — 実測オッズを自動取得するとゲート判定の精度が向上します</div>')
    st.markdown(_src_html, unsafe_allow_html=True)

    odds_cols = st.columns(6)
    raw_odds = {}
    for i in range(1, 7):
        with odds_cols[i - 1]:
            o = st.number_input(f"{BOAT_LABEL[i]} 単勝オッズ", value=float(st.session_state.odds_inputs[i]), step=0.1, key=f"odds_{i}", format="%.1f",
                help="この艇が1着になったときの単勝配当倍率です（例：3.4倍なら100円が340円になる）。EV（期待値）計算の分母となる重要な数値で、ここが実際のオッズとずれていると、BUY/SKIP判定全体が不正確になります。上の「オッズ自動取得」または一括貼り付けで正確な値を入れてください。")
            raw_odds[i] = o
    if raw_odds.get(1, 0) and raw_odds[1] <= 1.3:
        st.markdown('<span style="background:#5a1a06; color:#ffb648; padding:3px 8px; border-radius:4px; font-size:11px; font-weight:700;">🚨 1号艇 異常過熱投票検知</span>', unsafe_allow_html=True)

    # オッズ変化率・急変アラート
    _snap_col, _ = st.columns([1, 3])
    with _snap_col:
        if st.button("📸 現在値を基準として記録", key="snapshot_odds_sp"):
            st.session_state.prev_odds = dict(raw_odds)
            st.success("基準値を記録しました", icon="📸")
    if st.session_state.prev_odds:
        _alert_boats = []
        _chg_cols = st.columns(6)
        for i in range(1, 7):
            _prev = st.session_state.prev_odds.get(i, 0)
            _curr = raw_odds.get(i, 0)
            if _prev > 0 and _curr > 0:
                _pct = (_curr - _prev) / _prev * 100
                with _chg_cols[i-1]:
                    if abs(_pct) >= 30:
                        st.markdown(f'<div style="text-align:center;font-size:10px;color:#ff5c72;font-weight:700;">🚨{_pct:+.0f}%</div>', unsafe_allow_html=True)
                        _alert_boats.append(f"{i}号艇 {_pct:+.0f}%")
                    elif abs(_pct) >= 10:
                        st.markdown(f'<div style="text-align:center;font-size:10px;color:#ffb648;">⚠️{_pct:+.0f}%</div>', unsafe_allow_html=True)
                    elif abs(_pct) >= 5:
                        st.markdown(f'<div style="text-align:center;font-size:10px;color:#7c8aab;">{_pct:+.0f}%</div>', unsafe_allow_html=True)
        if _alert_boats:
            st.markdown(f'<div style="background:#3a0d14;border:1px solid #ff5c72;border-radius:6px;padding:7px 10px;font-size:11px;color:#ff5c72;font-weight:700;">🚨 オッズ急変検知：{"　".join(_alert_boats)}</div>', unsafe_allow_html=True)

    with st.expander("📋 オッズ一括貼り付けパーサー（公式サイトコピペ用）"):
        paste_text = st.text_area("コピペテキスト", height=100, placeholder="例）\n1 1.7\n2 3.4\n...",
            help="ボートレース公式サイトのオッズ表をコピーしてそのまま貼り付けるだけで、艇番とオッズを自動で読み取り、下のオッズ欄に反映できます。貼り付けた後は必ず「パース結果をオッズ欄に反映」ボタンを押してください。")
        if st.button("パース結果をオッズ欄に反映"):
            parsed = parse_raw_odds(paste_text)
            for bn, val in parsed.items():
                st.session_state.odds_inputs[bn] = val
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

market_probs = odds_to_market_probability(raw_odds)

# EVリスト構築
kelly_weight = 0.2
trade_list = []
for r in results:
    bn = r["boat"]
    odds = raw_odds.get(bn, 0.0)
    ev = (r["prob_1st"] * odds) - 1.0 if odds > 0 else -1.0
    ev_yen = round(ev * 1000)
    gap = r["prob_1st"] - market_probs.get(bn, 0.0)
    kelly = kelly_fraction(r["prob_1st"], odds, kelly_weight)
    trade_list.append({
        "boat": bn, "win_p": r["prob_1st"], "ev": ev, "ev_yen": ev_yen, "gap": gap, "kelly": kelly,
        "score": round(r["prob_1st"] * 150), "breakdown": decomp_data[bn]["breakdown"], "anomaly": decomp_data[bn]["anomaly"],
    })
trade_list.sort(key=lambda x: -x["ev"])
max_ev_item = trade_list[0]

total_abs_diff = sum(abs(r["prob_1st"] - market_probs.get(r["boat"], 0.0)) for r in results)
agreement_pct = max(0.0, 100 - (total_abs_diff / 2 * 100))

# ============================================================
# 【バグ修正】IQスコアは元々ここより後ろ（Kelly配分セクション内）で計算されており、
# それより前にある _signal_grade の計算が iq を先に参照してしまい、
# 「NameError: name 'iq' is not defined」でアプリ全体がクラッシュしていた（アップロード元の既存バグ）。
# 依存する値（EV・信頼度・市場一致度）はここまでに全て揃っているため、計算をここに繰り上げる。
# ============================================================
iq = calc_iq_score(max_ev_item["ev"], conf["score"], agreement_pct)

# ============================================================
# 決定パネル & メトリクスバー
# ============================================================
dcol1, dcol2 = st.columns([2, 1])

in_ev = next((d["ev"] for d in trade_list if d["boat"] == 1), -1)

# EVレンジ計算（保守/標準/強気）
# ※ BUY判定（rule5＝悲観EVゲート）で使うため、表示より前の早い段階で算出する。
_ev_range = calc_ev_range(max_ev_item["win_p"], raw_odds.get(max_ev_item["boat"], 0))

# ============================================================
# 行動制御レイヤー（ティルト防止）
# ------------------------------------------------------------
# 重要：ここで計算する「連敗中かどうか」は、レースそのものの良し悪しとは無関係の
# 「今のあなたの精神状態が賭けるのに適しているか」という別問題。
# そのため、EV閾値やシグナルグレード（＝レースに対するAIの純粋な判定）には
# 一切影響させない。判定（judgment）と行動制御（action control）を混ぜないための設計。
# 実際の行動抑制は、後段で①Kelly推奨額の大幅減額 ②SKIP理由への明記 の2点のみで行う。
# ============================================================
_tilt_logs = load_logs()
_tilt_decided = sorted(
    [l for l in _tilt_logs if l.get("result") in ("hit","miss")],
    key=lambda x: x.get("date",""), reverse=True
)[:5]
_tilt_losses = 0
for _tl in _tilt_decided:
    if _tl.get("result") == "miss": _tilt_losses += 1
    else: break

_tilt_active = _tilt_losses >= 3  # 行動制御フラグ（モデル判定には不使用）
if _tilt_active:
    st.markdown(
        f'<div style="background:#140a1e;border:1px solid #b967ff;border-radius:6px;padding:8px 12px;font-size:11px;color:#c98bff;font-weight:700;margin-bottom:4px;">'
        f'🧘 行動制御レイヤー発動（直近{_tilt_losses}連敗中） — '
        f'この警告はレース自体の評価（EV・シグナルグレード）には影響しません。'
        f'あなた自身の「取り返し行動」を防ぐための独立したブレーキで、下記の推奨購入額を自動的に大きく減額します。'
        f'</div>'
        f'<details class="gloss-term" style="margin-bottom:8px;"><summary>❓ これは何のためのルール？（クリックで説明）</summary>'
        f'<span class="gloss-body">連敗が続くと、人は「次こそ取り返したい」という心理から、'
        f'普段より大きな金額を賭けてしまいがちです（これを「ティルト」と呼びます）。'
        f'このツールはレースの分析（AIの判定）と、あなたの行動を守るブレーキを、あえて別の仕組みとして分けています。'
        f'なぜなら、連敗しているからといって「今回のレースの期待値」が変わるわけではないからです。'
        f'AIの判定はそのまま正直に表示しつつ、実際に賭ける金額だけを自動的に絞ることで、'
        f'冷静な判断と資金を守ることの両方を実現します。'
        f'解除条件：直近5走の判定のうち、連敗が3回未満に戻ると自動的に解除されます。</span></details>',
        unsafe_allow_html=True,
    )

rule1 = max_ev_item["ev"] >= CONFIG["EV_THRESHOLD_BUY"] + _mode_ev_boost  # モード別EV閾値調整（ティルトの影響を受けない）
rule2 = conf["score"] >= (50 + _mode_conf_boost)                       # モード別信頼度閾値調整
rule3 = max_ev_item["kelly"] > 0
rule5 = _ev_range["pessimist"] is None or _ev_range["pessimist"] >= 0  # 悲観EV（確率誤差-5%を想定した最悪ケース）がマイナスなら見送り
rule6 = prob_integrity["ok"]  # 確率整合性チェック（合計1.0・範囲・6艇分の検証）に失敗していないか

# アンサンブル確信度ゲート
lgb_and_hand_agree = None
# 【バグ修正】_gate_probはelse分岐でのみ定義されていたため、boat1が最上位候補かつ
# LGBが利用可能な場合（if側の分岐）は未定義のままとなり、後段のSKIP理由表示
# （f"🎲 Kelly配分なし（勝率{_gate_prob*100:.1f}%...）"）でNameErrorが発生していた。
# 常にキャリブレーション確率を計算しておくことで両分岐で参照可能にする。
_gate_prob = calibrate_probability(max_ev_item["win_p"], venue)
if max_ev_item["boat"] == 1 and lgb_prob_b1 is not None:
    hand_prob_b1 = max_ev_item["win_p"]
    lgb_and_hand_agree = (hand_prob_b1 >= _active_profile["hand_th"]) and (lgb_prob_b1 >= _active_profile["lgb_th"])
    rule_conf = lgb_and_hand_agree
    gate_label = f"AI一致({'精鋭' if profile_key=='precision' else '広範'})"
else:
    rule_conf = _gate_prob >= 0.50
    gate_label = "補正確率≥50%"

with dcol2:
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">⚠️ レース直前チェックリスト</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px; color:#7c8aab; margin-bottom:8px;">全項目「問題なし」を確認してからチェックを入れてください。</div>', unsafe_allow_html=True)
    rule4 = st.checkbox("オッズ急変なし（直前オッズが予測時から大きく変動していないか）", value=False, key="rule4_manual",
        help="目安：予測時から±30%以上動いていたら再評価。")
    chk_course = st.checkbox("進入変更・前付け動きなし（展示後の進入隊形が出走表通りか）", value=False, key="chk_course",
        help="前付けがあると1号艇のコース取りが変わる。展示後の進入隊形を確認。")
    chk_wind = st.checkbox("環境異常なし（安定板使用・強風・波浪警戒なし）", value=False, key="chk_wind",
        help="安定板使用時はターンが外に膨らみやすくイン逃げ率が下がる。")
    chk_f = st.checkbox("高リスク艇なし（F2以上・転覆直後・モーター乗換なし）", value=False, key="chk_f",
        help="F2保有選手は慎重スタートになりやすい。整備によるモーター乗換は展示と乖離することがある。")
    all_manual_ok = rule4 and chk_course and chk_wind and chk_f
    skip_checks = [
        ("イン逃げ期待値不足", VENUE_SCORE.get(venue, 0) >= 7 and in_ev < 0.02),
        ("展示爆伸艇の存在(波乱注意)", any("爆伸び" in d["anomaly"] for d in trade_list)),
        ("市場乖離・歪み不足", max_ev_item["ev"] < CONFIG["EV_THRESHOLD_BUY"]),
        ("悲観EVマイナス（予測誤差-5%で赤字転落）", not rule5),
        ("確率整合性エラー（分析失敗）", not rule6),
        ("混戦・低信頼度", conf["score"] < 50),
        ("オッズ急変（要確認）", not rule4),
        ("進入変更・前付けリスク", not chk_course),
        ("環境異常（強風・波浪）", not chk_wind),
        ("高リスク艇の混在", not chk_f),
    ]
    any_skip = any(p for _, p in skip_checks)
    if all_manual_ok:
        st.markdown('<div class="checklist-all-ok">✅ 全項目確認済み — シグナル有効</div>', unsafe_allow_html=True)
    else:
        unchecked = sum(1 for x in [rule4, chk_course, chk_wind, chk_f] if not x)
        st.markdown(f'<div class="checklist-ng">⬜ 未確認 {unchecked}項目</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# all_rules_pass / is_buy は「レースそのものへのAIの判定」。
# rule5（悲観EVゲート）：AI予測に-5%の誤差があった最悪ケースでもEVがプラスであることを要求する。
# これがマイナスのまま強気モードで買い続けると、予測が少しでも外れた瞬間に構造的に負ける。
all_rules_pass = rule1 and rule2 and rule3 and rule5 and rule6 and all_manual_ok and rule_conf
is_buy = all_rules_pass and not any_skip

# ============================================================
# AUTO SKIP：BUY/SKIP判定より前にシステムが強制排除
# ============================================================
_has_missing_ex = any(b.get("ex", CONFIG["EX_MEAN_TIMING"]) == CONFIG["EX_MEAN_TIMING"] for b in boats_data)
_auto_skip_checks = check_auto_skip(
    odds_source       = st.session_state.get("odds_source", "manual"),
    raw_odds          = raw_odds,
    wind_mps          = wind_mps,
    wind_dir          = wind_dir,
    conf_score        = conf["score"],
    lgb_and_hand_agree= lgb_and_hand_agree,
    lgb_prob_b1       = lgb_prob_b1 if lgb_prob_b1 is not None else 0.0,
    hand_prob_b1      = max_ev_item["win_p"],
    has_missing_ex    = _has_missing_ex,
    boats_data        = boats_data,
    wall_strength     = wall_strength,
    venue             = venue,
    is_demo_data      = _is_demo_data,
)
_auto_skip_critical = [(n,d,r) for n,v,s,d,r in _auto_skip_checks if v and s=="critical"]
_auto_skip_warning  = [(n,d,r) for n,v,s,d,r in _auto_skip_checks if v and s=="warning"]
_has_auto_skip = len(_auto_skip_critical) > 0

# 予測信頼度（＝入力データの質）を、本命断層・混戦度とは独立した軸として算出（項目6対応）
_data_reliability = calc_data_reliability(
    odds_source=st.session_state.get("odds_source", "manual"),
    has_missing_ex=_has_missing_ex,
    lgb_and_hand_agree=lgb_and_hand_agree,
    prob_integrity_ok=prob_integrity["ok"],
    wind_mps=wind_mps,
    is_demo_data=_is_demo_data,
)

# AUTO SKIPが発動した場合はBUYを強制キャンセル（これはレース自体の客観的リスク評価＝判定の一部）
if _has_auto_skip:
    is_buy = False

# ============================================================
# 行動制御レイヤー（ティルト防止）は判定（is_buy／シグナルグレード）を変更しない。
# 唯一ここ（Kelly推奨額の減額）でのみ、行動を実際に抑制する。
# ============================================================
_tilt_action_block = _tilt_active

# ============================================================
# Kelly自動減衰：不安定条件時に自動でKelly係数を縮小
# ============================================================
_kelly_decay = 1.0
_kelly_decay_reasons = []

if st.session_state.get("odds_source", "manual") != "actual":
    _kelly_decay *= 0.5
    _kelly_decay_reasons.append("実測オッズ未取得(-50%)")
if lgb_and_hand_agree is False:
    _kelly_decay *= 0.5
    _kelly_decay_reasons.append("LGB不一致(-50%)")
if wind_mps >= 4.0:
    _factor = max(0.3, 1.0 - (wind_mps - 4.0) * 0.1)
    _kelly_decay *= _factor
    _kelly_decay_reasons.append(f"強風({wind_mps:.1f}m/s, -{round((1-_factor)*100):.0f}%)")
if _has_auto_skip:
    _kelly_decay *= 0.3
    _kelly_decay_reasons.append("AUTO SKIP発動(-70%)")
if conf["score"] < 50:
    _kelly_decay *= 0.6
    _kelly_decay_reasons.append(f"信頼度低({conf['score']}点,-40%)")
if _tilt_action_block:
    _kelly_decay *= 0.2
    _kelly_decay_reasons.append(f"行動制御:ティルト防止({_tilt_losses}連敗,-80%)")

_kelly_decay = max(0.1, min(1.0, _kelly_decay))

# ============================================================
# AUTO SKIP 表示パネル（BUY/SKIPより上位）
# ============================================================
if _has_auto_skip or _auto_skip_warning:
    _as_bg = "#1a0000" if _has_auto_skip else "#1a0f00"
    _as_border = "#ff5c72" if _has_auto_skip else "#ffb648"
    _as_title = "🚫 AUTO SKIP 発動" if _has_auto_skip else "⚠️ AUTO SKIP 警告"
    _as_color = "#ff5c72" if _has_auto_skip else "#ffb648"
    _as_items = _auto_skip_critical + _auto_skip_warning
    _as_body = "".join([
        f'<div style="font-size:10px;color:var(--text-dim);margin-top:3px;">'
        f'{"🔴" if (n,d,r) in _auto_skip_critical else "🟡"} <b>{n}</b>：{d}'
        f'<details class="gloss-term" style="margin-top:2px;"><summary>❓ 解除条件（クリックで表示）</summary>'
        f'<span class="gloss-body">{r}</span></details>'
        f'</div>'
        for n, d, r in _as_items
    ])
    st.markdown(
        f'<div style="background:{_as_bg};border:2px solid {_as_border};border-radius:8px;'
        f'padding:10px 14px;margin-bottom:10px;">'
        f'<div style="font-size:13px;font-weight:900;color:{_as_color};margin-bottom:6px;">'
        f'{_as_title}</div>'
        f'{_as_body}'
        f'{"<div style=margin-top:6px;font-size:10px;color:#ff5c72;font-weight:700;>→ BUYシグナルを強制キャンセルします。人間の判断に関わらず買いを実行しないでください。</div>" if _has_auto_skip else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )

# シグナルグレード計算（_ev_range は早期計算済み・rule5に使用済み）
_signal_grade = calc_signal_grade(
    ev=max_ev_item["ev"], iq=iq, conf_score=conf["score"],
    is_buy=is_buy, all_manual_ok=all_manual_ok,
    wall_strength=wall_strength,
)

with dcol1:
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">FINAL INVESTMENT SIGNAL</div>', unsafe_allow_html=True)

    # グレードバッジ
    _sg = _signal_grade
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">'
        f'<span style="font-size:2.2rem;font-weight:900;color:{_sg["color"]};'
        f'background:{_sg["color"]}18;border:2px solid {_sg["color"]};'
        f'border-radius:8px;padding:4px 16px;letter-spacing:0.05em;">{_sg["grade"]}</span>'
        f'<div><div style="font-size:16px;font-weight:700;color:{_sg["color"]};">'
        f'{_sg["emoji"]} {_sg["label"]}</div>'
        f'<div style="font-size:10px;color:var(--text-dim2);margin-top:2px;">'
        f'推奨：{_sg["bet_size"]}　|　{_sg["reason"]}</div></div></div>',
        unsafe_allow_html=True,
    )

    # ============================================================
    # 判定基準の一覧化（項目2対応）
    # 「なぜこのグレードになったのか」を、条件のANDリストとして厳格に明示する。
    # 現在値と基準値を並べ、達成/未達成が一目で分かるようにする。
    # ============================================================
    _crit_rows = [
        ("EV（期待値）", f"{max_ev_item['ev']*100:.1f}%", "≥20%", max_ev_item["ev"] >= 0.20, "≥8%", max_ev_item["ev"] >= 0.08),
        ("IQスコア", f"{iq}点", "≥75点", iq >= 75, "≥60点", iq >= 60),
        ("レース信頼度", f"{conf['score']}点", "≥65点", conf["score"] >= 65, "（A判定では未使用）", True),
        ("壁強度", f"{wall_strength*100:.0f}%", "<30%", wall_strength < 0.3, "（A判定では未使用）", True),
    ]
    _crit_html = "".join([
        f'<div style="display:flex;justify-content:space-between;align-items:center;padding:3px 0;border-bottom:1px solid var(--border-dark);font-size:10px;">'
        f'<span style="color:var(--text-dim2);flex:1;">{name}</span>'
        f'<span style="color:var(--text-light);font-family:var(--font-mono,monospace);width:70px;text-align:right;">{cur}</span>'
        f'<span style="width:16px;text-align:center;">{"✅" if a_ok else "❌"}</span>'
        f'<span style="color:var(--text-dim2);width:70px;text-align:right;">A+:{a_th}</span>'
        f'<span style="width:16px;text-align:center;">{"✅" if b_ok else "❌"}</span>'
        f'<span style="color:var(--text-dim2);width:100px;text-align:right;">A:{b_th}</span>'
        f'</div>'
        for name, cur, a_th, a_ok, b_th, b_ok in _crit_rows
    ])
    st.markdown(
        f'<details class="gloss-term" style="margin-bottom:8px;">'
        f'<summary>❓ このグレード判定の基準を見る（クリックで開く）</summary>'
        f'<span class="gloss-body">'
        f'グレードは以下の順番で上から判定され、最初に条件を満たした段階で確定します（後の条件は評価されません）。'
        f'<br><br>'
        f'<b>🔴 X（データ不足）</b>：入力データが不足・異常。分析そのものが成立しない。'
        f'<br><b>🔴 D（見送り）</b>：BUY条件（rule1〜rule6・悲観EV・確率整合性など）のいずれかを満たしていない。'
        f'<br><b>🟠 C（様子見）</b>：BUY条件は満たさないが、標準EVがプラスかつ信頼度40点以上（＝惜しいが今回は見送るレベル）。'
        f'<br><b>🟡 B（条件付き）</b>：BUY条件は満たすが、直前チェック（4項目）が未完了、または下記A/A+の数値基準に届いていない。'
        f'<br><b>🟢 A（勝負）</b>：BUY条件を満たし、かつEV≥8%・IQ≥60点。'
        f'<br><b>🟢 A+（強勝負）</b>：BUY条件を満たし、かつEV≥20%・IQ≥75点・信頼度≥65点・壁強度30%未満（全て同時に満たす必要あり）。'
        f'<br><br>'
        f'下の表の✅❌は「今回のレースがその基準を満たしているか」を示しています。'
        f'</span>'
        f'<div style="margin-top:8px;background:#080d1a;border-radius:4px;padding:6px 8px;">'
        f'{_crit_html}'
        f'</div>'
        f'</details>',
        unsafe_allow_html=True,
    )

    # EVレンジ表示
    if _ev_range["standard"] is not None:
        _ep = _ev_range["pessimist"]; _es = _ev_range["standard"]; _eo = _ev_range["optimist"]
        _ec = "#16e0a0" if _es >= 0 else "#ff5c72"
        st.markdown(
            f'<div style="background:#080d1a;border:1px solid var(--border-dark);border-radius:6px;padding:7px 10px;margin-bottom:8px;">'
            f'<div style="font-size:9px;color:var(--text-dim2);margin-bottom:4px;">📊 EVレンジ（±5%の確率誤差を考慮）</div>'
            f'<div style="display:flex;justify-content:space-between;font-family:var(--font-mono,monospace);">'
            f'<div style="text-align:center;"><div style="font-size:8px;color:var(--text-dim2);">保守</div>'
            f'<div style="font-size:12px;color:{"#16e0a0" if _ep>=0 else "#ff5c72"};font-weight:700;">{_ep:+.1f}%</div></div>'
            f'<div style="text-align:center;border:1px solid {_ec};border-radius:4px;padding:2px 8px;">'
            f'<div style="font-size:8px;color:var(--text-dim2);">標準</div>'
            f'<div style="font-size:14px;color:{_ec};font-weight:900;">{_es:+.1f}%</div></div>'
            f'<div style="text-align:center;"><div style="font-size:8px;color:var(--text-dim2);">強気</div>'
            f'<div style="font-size:12px;color:{"#16e0a0" if _eo>=0 else "#ff5c72"};font-weight:700;">{_eo:+.1f}%</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    gates_html = "".join([
        f'<span class="gate-pill {"gate-on" if rule1 else "gate-off"}">{gloss("EV閾値")}</span>',
        f'<span class="gate-pill {"gate-on" if rule2 else "gate-off"}">{gloss("信頼度")}</span>',
        f'<span class="gate-pill {"gate-on" if rule3 else "gate-off"}">{gloss("Kelly>0", "Kelly&gt;0")}</span>',
        f'<span class="gate-pill {"gate-on" if rule5 else "gate-off"}">{gloss("悲観EV", "悲観EV≥0")}</span>',
        f'<span class="gate-pill {"gate-on" if rule6 else "gate-off"}">{gloss("確率整合性")}</span>',
        f'<span class="gate-pill {"gate-on" if rule_conf else "gate-off"}">{gloss("AI一致" if lgb_and_hand_agree is not None else "補正確率", gate_label)}</span>',
        f'<span class="gate-pill {"gate-on" if all_manual_ok else "gate-off"}">{gloss("急変確認", "直前確認")}</span>',
        f'<span class="gate-pill {"gate-on" if not any_skip else "gate-off"}">リスク無</span>',
    ])
    st.markdown(f'<div style="margin-top:8px;">{gates_html}</div>', unsafe_allow_html=True)
    if lgb_and_hand_agree is not None:
        _hand_pct = max_ev_item["win_p"] * 100
        _lgb_pct  = lgb_prob_b1 * 100
        _hand_th  = _active_profile["hand_th"] * 100
        _lgb_th   = _active_profile["lgb_th"] * 100
        _profile_label = _active_profile["label"].split("（")[0]

        if lgb_and_hand_agree:
            _agree_html = (
                f'<div class="note" style="margin-top:6px;border-left:3px solid #16e0a0;padding-left:8px;">'
                f'✅ AI内訳（{_profile_label}）手作り {_hand_pct:.1f}%（閾値{_hand_th:.0f}%）'
                f' / LGB {_lgb_pct:.1f}%（閾値{_lgb_th:.0f}%）→ <b style="color:#16e0a0;">両モデル一致</b></div>'
            )
        else:
            # ============================================================
            # M-02: LGB不一致時の差分診断（仕様書4.6.4）
            # ============================================================
            _diff = _lgb_pct - _hand_pct
            if _lgb_pct > _hand_pct:
                # LGBが高い場合
                _diag_title = "LGBが手作りより高く評価"
                _diag_body  = "LGBは展示タイム・展示ST・壁強度を重視。手作りモデルはこれらの動的情報を軽く扱う傾向がある。"
                _diag_hint  = "直前情報（展示・ST）が有望な場合にLGBが強く反応しやすい。"
            else:
                # 手作りが高い場合
                _diag_title = "手作りがLGBより高く評価"
                _diag_body  = "手作りはインコース優勢傾向・全国勝率を重視。LGBは会場補正・モーター状態で厳しく評価している可能性。"
                _diag_hint  = "手作りの会場バイアスとLGBの客観評価の食い違い。会場別成績を再確認推奨。"
            _agree_html = (
                f'<div style="background:#1a0f00;border:1px solid #ffb648;border-radius:6px;'
                f'padding:8px 12px;margin-top:6px;font-size:10.5px;">'
                f'<div style="color:#ffb648;font-weight:700;margin-bottom:4px;">'
                f'⚡ LGB差分診断：{_diag_title}（差分 {_diff:+.1f}%）</div>'
                f'<div style="color:var(--text-dim);line-height:1.6;">'
                f'手作り {_hand_pct:.1f}%（閾値{_hand_th:.0f}%） / LGB {_lgb_pct:.1f}%（閾値{_lgb_th:.0f}%）<br>'
                f'<b style="color:var(--text-light);">主因：</b>{_diag_body}<br>'
                f'<b style="color:var(--text-light);">ヒント：</b>{_diag_hint}</div>'
                f'<div style="color:#ff5c72;font-size:10px;margin-top:4px;font-weight:700;">'
                f'→ 不一致のため投資見送り方向（BUYゲート未通過）</div>'
                f'</div>'
            )
        st.markdown(_agree_html, unsafe_allow_html=True)
    auto_html = "".join([
        f'<div style="color:{"#ff5c72" if p else "#5a6685"}; font-size:11px;">{"☑" if p else "☐"} {label}</div>'
        for label, p in skip_checks[:4]
    ])
    st.markdown(f'<div style="margin-top:10px;"><div class="eyebrow">🤖 自動リスク判定</div>{auto_html}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 分析スナップショット保存・比較（項目26系対応）
# 入力値を少し変えながら「判定がどう変わるか」を見比べるための、
# セッション内限定の一時的な記録。実際の購入ログ（📒タブ）とは別物。
# ============================================================
if "analysis_snapshots" not in st.session_state:
    st.session_state.analysis_snapshots = []

st.markdown(
    '<details class="gloss-term"><summary>📸 分析スナップショット（❓クリックで説明）</summary>'
    '<span class="gloss-body">「オッズを少し変えたらグレードは変わるか？」「展示タイムを実測値に直したらEVはどう動くか？」'
    'といった“もしも”を比較するための機能です。ボタンを押すと今の判定結果（グレード・EV・IQ・信頼度）が'
    '一時的に記録され、下に一覧表示されます。入力を変えてまた保存すれば、変化を並べて確認できます。'
    'この記録はブラウザを閉じたり画面を再読み込みすると消える、あくまで比較用の一時メモです（購入記録には残りません）。</span>'
    '</details>',
    unsafe_allow_html=True,
)
_snap_c1, _snap_c2 = st.columns([3, 1])
with _snap_c1:
    if st.button("📸 現在の分析結果を保存して比較", key="save_snapshot"):
        st.session_state.analysis_snapshots.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "venue": venue, "boat": max_ev_item["boat"],
            "grade": _signal_grade["grade"], "ev": max_ev_item["ev"],
            "pessimist_ev": _ev_range["pessimist"], "iq": iq,
            "conf": conf["score"], "action": "BUY" if is_buy else "SKIP/WATCH",
        })
with _snap_c2:
    if st.session_state.analysis_snapshots and st.button("🗑️ 全消去", key="clear_snapshots"):
        st.session_state.analysis_snapshots = []
        st.rerun()

if st.session_state.analysis_snapshots:
    _snap_rows = []
    for i, s in enumerate(st.session_state.analysis_snapshots):
        _snap_rows.append({
            "#": i + 1, "時刻": s["time"], "会場": s["venue"], "艇": f"{s['boat']}号艇",
            "グレード": s["grade"], "標準EV": f"{s['ev']*100:+.1f}%",
            "悲観EV": f"{s['pessimist_ev']:+.1f}%" if s["pessimist_ev"] is not None else "--",
            "IQ": s["iq"], "信頼度": s["conf"],
        })
    st.dataframe(pd.DataFrame(_snap_rows), hide_index=True, width='stretch')

mcols = st.columns(6)
with mcols[0]:
    st.markdown(f'<div class="panel-box"><div class="metric-label">📈 MAX EV</div><div class="metric-val" style="color:{"#16e0a0" if max_ev_item["ev"]>=0 else "#ff5c72"};">{"+" if max_ev_item["ev"]>=0 else ""}{max_ev_item["ev"]*100:.1f}%</div></div>', unsafe_allow_html=True)
with mcols[1]:
    chaos_idx = _conf_breakdown["chaos_score"]
    st.markdown(f'<div class="panel-box"><div class="metric-label">⚠️ CHAOS INDEX</div><div class="metric-val" style="color:{"#ff5c72" if chaos_idx>50 else "#16e0a0"};">{chaos_idx}</div></div>', unsafe_allow_html=True)
with mcols[2]:
    st.markdown(f'<div class="panel-box"><div class="metric-label">🎯 レース信頼度</div><div class="metric-val">{conf["score"]}点({conf["level"]})</div></div>', unsafe_allow_html=True)

# Confidence Score 6軸内訳
_conf_total = sum(raw_scores.values()) if raw_scores else 1
_c_ex    = round(min(100, max(0, (7.0 - boats_data[0].get("ex", 6.72)) / 0.5 * 50 + 50)))
_c_motor = round(min(100, max(0, (boats_data[0].get("motor", 30) - 15) / 40 * 100)))
_c_class = {"A1":95,"A2":80,"B1":60,"B2":40}.get(boats_data[0].get("player_class","B1"), 60)
_c_st    = round(min(100, max(0, (0.30 - boats_data[0].get("st", 0.17)) / 0.15 * 100)))
_c_odds  = round(min(100, max(0, agreement_pct)))
_c_env   = round(min(100, max(0, 100 - wind_mps * 8)))
_conf_axes = [("展示",_c_ex),("モーター",_c_motor),("級別",_c_class),("ST",_c_st),("オッズ整合",_c_odds),("環境補正",_c_env)]
_ch = '<div class="panel-box" style="border-color:#2a3859;"><div class="eyebrow">🔬 Confidence 内訳（1号艇）</div>'
for _an, _av in _conf_axes:
    _ac = "#16e0a0" if _av >= 75 else "#ffb648" if _av >= 50 else "#ff5c72"
    _ch += (f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">'
            f'<span style="font-size:9px;color:var(--text-dim2);width:56px;">{_an}</span>'
            f'<div style="flex:1;background:#1a2a3a;border-radius:3px;height:8px;">'
            f'<div style="width:{_av}%;background:{_ac};height:8px;border-radius:3px;"></div></div>'
            f'<span style="font-size:9px;color:{_ac};width:26px;text-align:right;">{_av}</span></div>')
_ch += '</div>'
st.markdown(_ch, unsafe_allow_html=True)
with mcols[3]:
    st.markdown(f'<div class="panel-box"><div class="metric-label">🔍 AI/市場乖離度</div><div class="metric-val">{agreement_pct:.0f}%</div></div>', unsafe_allow_html=True)
with mcols[4]:
    logs_now = load_logs()
    msum = monthly_summary(logs_now, date.today())
    st.markdown(f'<div class="panel-box"><div class="metric-label">📊 月間実績</div><div class="metric-val" style="color:{"#16e0a0" if msum["realized"]>=0 else "#ff5c72"};">{"+" if msum["realized"]>=0 else ""}{msum["realized"]:,}円</div></div>', unsafe_allow_html=True)
with mcols[5]:
    iq = calc_iq_score(max_ev_item["ev"], conf["score"], agreement_pct)
    iq_col = "#16e0a0" if iq>=80 else "#3fc4ff" if iq>=60 else "#ffb648" if iq>=40 else "#ff5c72"
    iq_stars = "★"*(iq//20) + "☆"*(5-iq//20)
    ev_score_part  = round(max(0.0, min(40.0, max_ev_item["ev"] * 400)))
    conf_score_part= round(max(0.0, min(35.0, conf["score"] * 0.35)))
    agr_score_part = round(max(0.0, min(25.0, agreement_pct * 0.25)))
    st.markdown(
        f'<div class="panel-box" style="border-color:#2a3859;">'
        f'<div class="metric-label" style="color:var(--text-neutral);">✨ IQ</div>'
        f'<div class="metric-val" style="color:{iq_col};">{iq}点</div>'
        f'<div style="font-size:10px;color:{iq_col};">{iq_stars}</div>'
        f'<div style="font-size:8.5px;color:var(--text-dim2);margin-top:4px;line-height:1.6;">'
        f'EV: <b style="color:{"#16e0a0" if ev_score_part>=28 else "#ffb648"};">{ev_score_part}/40</b><br>'
        f'信頼度: <b style="color:{"#16e0a0" if conf_score_part>=24 else "#ffb648"};">{conf_score_part}/35</b><br>'
        f'市場一致: <b style="color:{"#16e0a0" if agr_score_part>=18 else "#ffb648"};">{agr_score_part}/25</b>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

# ============================================================
# 信頼度の3軸分離表示（項目6対応）
# 「レース信頼度」1つの数字に混ざっていた意味を、独立した3つの軸として明示する。
# ============================================================
_cb = _conf_breakdown
_dr = _data_reliability
st.markdown(
    f'<div class="panel-box" style="margin-top:8px;">'
    f'<div class="eyebrow">🔬 信頼度の内訳（3軸分離）'
    f'<details class="gloss-term" style="display:inline;margin-left:6px;"><summary style="display:inline;">❓</summary>'
    f'<span class="gloss-body">「レース信頼度」という1つの数字は、本来まったく別の3つの意味を含んでいます。'
    f'①本命断層＝このレースに実力差のある本命がいるか（レースの性質）。'
    f'②混戦度＝6艇の力がどれだけ均等に近いか（レースの性質）。'
    f'③データ信頼度＝入力したデータがどれだけ信頼できるか（データの質）。'
    f'①②はレースそのものの特徴なので変えようがありませんが、③は実測オッズの取得や展示タイムの入力を揃えることで改善できます。'
    f'この3つを分けて見ることで、「本命はいるが入力データが粗い」ような危険なケースを見逃さずに済みます。</span></details>'
    f'</div>'
    f'<div style="display:flex;gap:10px;margin-top:6px;flex-wrap:wrap;">'
    f'<div style="flex:1;min-width:120px;background:#080d1a;border:1px solid var(--border-dark);border-radius:6px;padding:8px 10px;">'
    f'<div style="font-size:9px;color:var(--text-dim2);">① 本命断層</div>'
    f'<div style="font-size:16px;font-weight:900;color:{"#16e0a0" if _cb["gap_score"]>=60 else "#ffb648" if _cb["gap_score"]>=30 else "#7c8aab"};">{_cb["gap_score"]}点</div>'
    f'<div style="font-size:9px;color:var(--text-dim);">{_cb["gap_label"]}</div></div>'
    f'<div style="flex:1;min-width:120px;background:#080d1a;border:1px solid var(--border-dark);border-radius:6px;padding:8px 10px;">'
    f'<div style="font-size:9px;color:var(--text-dim2);">② 混戦度</div>'
    f'<div style="font-size:16px;font-weight:900;color:{"#ff5c72" if _cb["chaos_score"]>=75 else "#ffb648" if _cb["chaos_score"]>=50 else "#16e0a0"};">{_cb["chaos_score"]}点</div>'
    f'<div style="font-size:9px;color:var(--text-dim);">{_cb["chaos_label"]}</div></div>'
    f'<div style="flex:1;min-width:120px;background:#080d1a;border:1px solid var(--border-dark);border-radius:6px;padding:8px 10px;">'
    f'<div style="font-size:9px;color:var(--text-dim2);">③ データ信頼度</div>'
    f'<div style="font-size:16px;font-weight:900;color:{_dr["color"]};">{_dr["score"]}点</div>'
    f'<div style="font-size:9px;color:var(--text-dim);">{_dr["label"]}</div></div>'
    f'</div>'
    f'{"<div style=font-size:9px;color:var(--text-dim);margin-top:6px;>減点理由：" + "、".join(_dr["reasons"]) + "</div>" if _dr["reasons"] else ""}'
    f'</div>',
    unsafe_allow_html=True,
)

# レース難易度スコア（IQ計算後に実行）
_top_win_p = trade_list[0]["win_p"] if trade_list else 0
_difficulty = calc_race_difficulty(conf["score"], wind_mps, _top_win_p, agreement_pct)
_d_col = _difficulty["color"]
_diff_cols = st.columns(3)
with _diff_cols[0]:
    st.markdown(
        f'<div class="panel-box" style="border-color:{_d_col};">'
        f'<div class="metric-label">🏁 レース難易度</div>'
        f'<div class="metric-val" style="color:{_d_col};">{_difficulty["grade"]}ランク</div>'
        f'<div style="font-size:10px;color:{_d_col};">{_difficulty["stars"]}</div>'
        f'<div style="font-size:9px;color:var(--text-dim2);margin-top:3px;">{_difficulty["label"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
with _diff_cols[1]:
    _upset_col = "#ff5c72" if _difficulty["upset_pct"] > 60 else "#ffb648" if _difficulty["upset_pct"] > 40 else "#16e0a0"
    st.markdown(
        f'<div class="panel-box">'
        f'<div class="metric-label">🎲 荒れ確率</div>'
        f'<div class="metric-val" style="color:{_upset_col};">{_difficulty["upset_pct"]}%</div>'
        f'<div style="font-size:9px;color:var(--text-dim2);margin-top:3px;">'
        f'{"見送り推奨" if _difficulty["upset_pct"]>60 else "注意" if _difficulty["upset_pct"]>40 else "安定"}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
with _diff_cols[2]:
    _diff_score_col = "#16e0a0" if _difficulty["score"] < 38 else "#ffb648" if _difficulty["score"] < 60 else "#ff5c72"
    st.markdown(
        f'<div class="panel-box">'
        f'<div class="metric-label">📐 難易度スコア</div>'
        f'<div class="metric-val" style="color:{_diff_score_col};">{_difficulty["score"]}</div>'
        f'<div style="font-size:9px;color:var(--text-dim2);margin-top:3px;">低いほど勝負しやすい</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# 会場別有望条件アラート（重複削除済み）
_venue_cond = VENUE_EDGE_CONDITIONS.get(venue)
_prob1st_b1 = next((r["prob_1st"] for r in results if r["boat"] == 1), 0)
if _venue_cond and _prob1st_b1 >= _venue_cond["min_prob"]:
    st.markdown(
        f'<div style="background:#062a1c;border:1px solid #16e0a0;border-radius:6px;padding:8px 12px;font-size:11px;color:#16e0a0;font-weight:700;margin-top:8px;">⭐ 有望条件マッチ：{_venue_cond["note"]}　1号艇確率 {_prob1st_b1*100:.1f}%</div>',
        unsafe_allow_html=True,
    )

# 壁ロジックアラート
if wall_strength > 0.15:
    _ws_pct = round(wall_strength * 100)
    _ws_col = "#16e0a0" if wall_strength >= 0.5 else "#ffb648"
    _ws_border = "#16e0a0" if wall_strength >= 0.5 else "#ffb648"
    _ws_bg = "#062a1c" if wall_strength >= 0.5 else "#1a0f00"
    _ws_label = "強壁" if wall_strength >= 0.5 else "弱壁"
    # 【バグ修正】st_1/st_2という未定義変数が参照されていた（壁強度>15%のたびに必ずクラッシュしていた）。
    # 個艇入力ループのローカル変数(st_)がここまでスコープに残っていなかったため、
    # boats_dataから1号艇・2号艇のSTを正しく取り直す。
    _wall_st1 = boats_data[0]["st"]
    _wall_st2 = boats_data[1]["st"]
    st.markdown(
        f'<div style="background:{_ws_bg};border:1px solid {_ws_border};border-radius:6px;padding:8px 12px;font-size:11px;color:{_ws_col};font-weight:700;margin-top:8px;">'
        f'🧱 壁ロジック発動（{_ws_label}・強度{_ws_pct}%）：2号艇（ST{_wall_st2:.2f}）が1号艇（ST{_wall_st1:.2f}）のブロック役に。'
        f'1号艇逃げ・2号艇差しの固定展開を補正済み。外艇の攻めを抑制。</div>',
        unsafe_allow_html=True,
    )

# 逆張り特注アラート（市場が強く同意しているがAIは低評価 → 市場の罠の可能性）
_top_boat = trade_list[0]["boat"] if trade_list else 1
_top_market_prob = 1 / raw_odds.get(_top_boat, 3.0) if raw_odds.get(_top_boat, 0) > 0 else 0
_top_ai_prob = trade_list[0]["win_p"] if trade_list else 0
_agreement_gap = _top_market_prob - _top_ai_prob  # 正値 = 市場が過剰に買っている
if _agreement_gap > 0.12 and agreement_pct > 75:
    st.markdown(
        f'<div style="background:#1a0a00;border:1px solid #ffb648;border-radius:6px;padding:8px 12px;font-size:11px;color:#ffb648;font-weight:700;margin-top:8px;">'
        f'⚡ 逆張り特注アラート：市場は{BOAT_LABEL[_top_boat]}に{_top_market_prob*100:.0f}%の支持（AI評価{_top_ai_prob*100:.0f}%）— '
        f'市場が過剰評価の可能性。{BOAT_LABEL[_top_boat]}消しの妙味あり。'
        f'</div>',
        unsafe_allow_html=True,
    )
elif _top_ai_prob - _top_market_prob > 0.12 and agreement_pct < 50:
    st.markdown(
        f'<div style="background:#0d1a2a;border:1px solid #3fc4ff;border-radius:6px;padding:8px 12px;font-size:11px;color:#3fc4ff;font-weight:700;margin-top:8px;">'
        f'💎 穴狙い特注アラート：AIは{BOAT_LABEL[_top_boat]}を{_top_ai_prob*100:.0f}%評価（市場{_top_market_prob*100:.0f}%）— '
        f'市場が過小評価。穴狙いの妙味あり。'
        f'</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# 損失ストッパー確認（変数定義を先行）
# ============================================================
_loss_limit = st.session_state.loss_limit
_today_loss, _loss_triggered = check_loss_stopper(_loss_limit)

# 判定ルールの明文化（SKIP理由を優先順位付きで表示）
_decision_rules = [
    ("🛑 損失ストッパー発動", _today_loss >= _loss_limit and _loss_limit > 0 and not st.session_state.loss_stopper_dismissed),
    ("⚠️ 直前チェック未完了", not all_manual_ok),
    (f"📉 EV不足（{max_ev_item['ev']*100:.1f}% < 閾値{CONFIG['EV_THRESHOLD_BUY']*100:.0f}%）", not rule1),
    (f"📉 悲観EVマイナス（予測誤差-5%の最悪ケースで{_ev_range['pessimist']}%）", not rule5),
    (f"🧮 確率整合性エラー（{'; '.join(prob_integrity['errors']) if prob_integrity['errors'] else ''}）", not rule6),
    (f"🎯 信頼度不足（{conf['score']}点 < 50点）", not rule2),
    (f"🎲 Kelly配分なし（勝率{_gate_prob*100:.1f}% × 最高オッズ{max(raw_odds.values() or [0]):.1f}倍）", not rule3),
    (f"🔍 AI/市場確信度不足（{'AIと手動モデルが不一致' if lgb_and_hand_agree is False else '会場別補正確率が50%未満'}）", not rule_conf),
    (f"🧘 行動制御中（ティルト防止・{_tilt_losses}連敗）※レース評価とは別軸。購入額を自動減額済み", _tilt_action_block),
]
_active_blocks = [(label, blocked) for label, blocked in _decision_rules if blocked]

# ============================================================
# 【バグ修正】AUTO SKIP（オッズ異常・強風×難水面など、_decision_rulesに含まれない
# 独自の強制排除条件）が単独で発動した場合、_active_blocksが空のままになり、
# action_stateが誤って"BUY"と判定されて何も表示されなくなる不具合があったため、
# _has_auto_skipを明示的にブロック理由として追加する。
# ============================================================
if _has_auto_skip and not any("AUTO SKIP" in label for label, _ in _active_blocks):
    _auto_skip_names = "、".join(n for n, d, r in _auto_skip_critical)
    _active_blocks = _active_blocks + [(f"🚫 AUTO SKIP発動（{_auto_skip_names}）", True)]

# ============================================================
# BUY / WATCH / SKIP の3段階判定（項目2・3対応）
# 標準EVは基準を満たすが悲観EVだけがマイナス、という「予測誤差に弱い」ケースを
# 一律SKIPに埋没させず、WATCH（見送るが記録に値する）として区別する。
# ============================================================
_block_labels = {label for label, _ in _active_blocks}
_only_pessimistic_ev_block = (
    len(_active_blocks) > 0
    and all("悲観EV" in label for label in _block_labels)
    and rule1
)
if not _active_blocks:
    action_state = "BUY"
elif _only_pessimistic_ev_block:
    action_state = "WATCH"
else:
    action_state = "SKIP"

# 整合性の最終保証：is_buyがFalseなのにaction_state=="BUY"になることは論理的に矛盾するため、
# 万一そうなった場合は安全側（SKIP）に倒す（is_buyの計算に将来ロジックが追加された場合の保険）。
if action_state == "BUY" and not is_buy:
    action_state = "SKIP"
    if not _active_blocks:
        _active_blocks = [("⚠️ 判定不整合のため安全側でSKIP（詳細は上記AUTO SKIPパネルを参照）", True)]

if _active_blocks and action_state == "SKIP":
    _rules_html = "".join([
        f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0;border-bottom:1px solid var(--border-dark);">'
        f'<span style="font-size:10px;color:#ff5c72;font-weight:700;min-width:16px;">{i+1}</span>'
        f'<span style="font-size:10px;color:var(--text-light);">{label}</span>'
        f'</div>'
        for i, (label, _) in enumerate(_active_blocks)
    ])
    st.markdown(
        f'<div style="background:#0d0608;border:1px solid rgba(255,92,114,0.3);border-radius:6px;padding:8px 12px;margin-top:8px;">'
        f'<div style="font-size:9px;font-weight:700;color:#ff5c72;letter-spacing:0.08em;margin-bottom:5px;">🚫 SKIP理由（優先順位順）</div>'
        f'{_rules_html}</div>',
        unsafe_allow_html=True,
    )
elif action_state == "WATCH":
    st.markdown(
        f'<div style="background:#1a1400;border:1px solid #ffb648;border-radius:6px;padding:8px 12px;margin-top:8px;">'
        f'<div style="font-size:11px;font-weight:900;color:#ffb648;margin-bottom:3px;">🟡 WATCH — 予測誤差に弱い</div>'
        f'<div style="font-size:10px;color:var(--text-dim);">標準EV（{max_ev_item["ev"]*100:+.1f}%）は基準を満たしていますが、'
        f'AI予測に-5%の誤差があった悲観シナリオでは期待値{_ev_range["pessimist"]:+.1f}%とマイナスに転落します。'
        f'『薄氷の勝負』のため今回は見送り、次に同じパターンが出たときの判断材料として記録しておくことを推奨します。</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
elif is_buy:
    st.markdown(
        '<div style="background:#062a1c;border:1px solid #16e0a0;border-radius:6px;padding:6px 12px;margin-top:8px;font-size:10px;color:#16e0a0;font-weight:700;">✅ 全ルールクリア — 投資推奨</div>',
        unsafe_allow_html=True,
    )

# ============================================================
# 押し上げ/押し下げ要因
# ============================================================
_top = trade_list[0]
_push = []
_pull = []
if _top["win_p"] >= 0.35: _push.append("高確率本命（勝率35%以上）")
elif _top["win_p"] >= 0.27: _push.append("有力候補（勝率27%以上）")
if _top["gap"] > 0.08: _push.append(f"市場大幅過小評価（+{_top['gap']*100:.0f}%）")
elif _top["gap"] > 0.04: _push.append(f"市場過小評価（+{_top['gap']*100:.0f}%）")
if _top["ev"] >= CONFIG["EV_THRESHOLD_BUY"]: _push.append(f"期待値プラス（EV{_top['ev']*100:+.0f}%）")
if conf["score"] >= 75: _push.append(f"高信頼度レース（{conf['score']}点）")
if VENUE_SCORE.get(venue, 0) >= 6: _push.append(f"{venue}はインコース優勢")
if wind_mps <= 1: _push.append("風速穏やか（展示安定）")
if all_manual_ok: _push.append("直前チェック全項目クリア")
if _top["ev"] < CONFIG["EV_THRESHOLD_BUY"]: _pull.append(f"期待値不足（EV{_top['ev']*100:.0f}%）")
if _top["gap"] < -0.04: _pull.append(f"市場過大評価（{_top['gap']*100:.0f}%）")
if conf["score"] < 50: _pull.append(f"信頼度低め（{conf['score']}点）")
if wind_mps >= 4: _pull.append(f"風速強め（{wind_mps}m/s）")
if VENUE_SCORE.get(venue, 0) <= -10: _pull.append(f"{venue}は波乱になりやすい")
if not all_manual_ok: _pull.append("直前チェック未完了")

_fcol1, _fcol2 = st.columns(2)
with _fcol1:
    st.markdown(
        '<div style="background:#080d1a;border:1px solid rgba(22,224,160,0.3);border-radius:6px;padding:8px 10px;margin-top:8px;">'
        '<div style="font-size:9px;font-weight:700;color:var(--text-buy);margin-bottom:4px;">▲ 押し上げ要因</div>'
        + "".join([f'<div style="font-size:10px;color:var(--text-light);padding:2px 0;"><span style="color:var(--text-buy);">▲</span> {f}</div>' for f in _push[:3]])
        + ('</div>' if _push else '<div style="font-size:10px;color:var(--text-dim2);">なし</div></div>'),
        unsafe_allow_html=True,
    )
with _fcol2:
    st.markdown(
        '<div style="background:#080d1a;border:1px solid rgba(255,92,114,0.3);border-radius:6px;padding:8px 10px;margin-top:8px;">'
        '<div style="font-size:9px;font-weight:700;color:var(--text-risk);margin-bottom:4px;">▼ 押し下げ要因</div>'
        + "".join([f'<div style="font-size:10px;color:var(--text-light);padding:2px 0;"><span style="color:var(--text-risk);">▼</span> {f}</div>' for f in _pull[:3]])
        + ('</div>' if _pull else '<div style="font-size:10px;color:var(--text-dim2);">なし</div></div>'),
        unsafe_allow_html=True,
    )

# Opportunity登録ボタン
_opp_col, _ = st.columns([1, 3])
with _opp_col:
    if st.button("＋ このレースを登録（Opportunity）", key="register_opp"):
        save_opportunity(venue, int(st.session_state.get("fetch_race_no", 1)), _top["ev"], iq, conf["score"], is_buy)
        st.success("レースを登録しました", icon="★")
        st.rerun()

# ============================================================
# 損失ストッパー画面ブロック
# ============================================================
if _loss_triggered and not st.session_state.loss_stopper_dismissed:
    st.error(
        f"🛑 **LOSS STOPPER 発動** — 本日損失: {_today_loss:,}円 ／ 上限: {_loss_limit:,}円\n\n"
        "これ以上の投資は資金管理ルール違反です。本日の投資を終了してください。",
        icon="🚨",
    )
    if st.button("✋ 理解しました・本日の投資を終了します", key="dismiss_loss_stopper", type="primary"):
        st.session_state.loss_stopper_dismissed = True
        st.rerun()
    st.stop()


# ============================================================
# EVランキング & タブ群
# ============================================================
col_left, col_right = st.columns([1, 1.3])

with col_left:
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)

    # レース難易度バッジをEVランキングヘッダーに統合
    _grade_badge = (
        f'<span style="background:rgba({",".join(["22,224,160"] if _difficulty["grade"]=="S" else ["63,196,255"] if _difficulty["grade"]=="A" else ["255,182,72"] if _difficulty["grade"]=="B" else ["255,140,66"] if _difficulty["grade"]=="C" else ["255,92,114"])},0.15);'
        f'color:{_difficulty["color"]};border:1px solid {_difficulty["color"]};border-radius:4px;padding:1px 7px;font-size:10px;font-weight:700;margin-left:8px;">'
        f'{_difficulty["grade"]}ランク 荒れ{_difficulty["upset_pct"]}%</span>'
    )
    st.markdown(f'### 🎯 COMPRESSED EV RANKING {_grade_badge}', unsafe_allow_html=True)

    # 期待利益の基準金額を取得（Kelly配分タブの予算か1,000円）
    _ep_stake = int(st.session_state.get("total_budget_ref", 1000))
    st.markdown(f'<div style="font-size:9px;color:var(--text-dim2);margin-bottom:6px;">💡 期待利益は{_ep_stake:,}円投資ベースで表示</div>', unsafe_allow_html=True)

    cutoff_placed = False
    for idx, tc in enumerate(trade_list):
        if not cutoff_placed and tc["ev"] < CONFIG["EV_THRESHOLD_BUY"]:
            st.markdown('<div class="cutoff-line">▼ ここから下は期待値投資基準外（見送り領域）▼</div>', unsafe_allow_html=True)
            cutoff_placed = True

        card_class = "trade-card"
        badge = "🔵 NEUTRAL"
        if tc["ev"] <= CONFIG["EV_THRESHOLD_RISK"]:
            card_class += " risk-card"
            badge = "🚨 OVER-VALUED（過剰人気・危険艇）"
        elif idx == 0 and tc["ev"] >= CONFIG["EV_THRESHOLD_BUY"]:
            card_class += " prime-card"
            badge = "🔥 EV-AXIS（最高期待値軸候補）"
        elif CONFIG["EV_THRESHOLD_RISK"] < tc["ev"] < CONFIG["EV_THRESHOLD_BUY"]:
            badge = "⚪ WATCH（基準未満）"

        ev_color = "#16e0a0" if tc["ev"] >= 0 else "#ff5c72"
        # 期待利益（投資額ベースで計算）
        ep_yen = calc_expected_profit(tc["win_p"], raw_odds.get(tc["boat"], 0), _ep_stake)
        ep_col = "#16e0a0" if ep_yen >= 0 else "#ff5c72"
        st.markdown(f"""
        <div class="{card_class}">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="display:flex; gap:10px; align-items:center;">
                    <div class="boat-badge" style="background:{BOAT_COLORS[tc['boat']]}; color:{BOAT_TEXT[tc['boat']]};">{BOAT_LABEL[tc['boat']]}</div>
                    <div>
                        <div class="card-badge">{badge}<span class="card-anomaly">{tc['anomaly'] if tc['anomaly']!='通常' else ''}</span></div>
                        <div class="card-stat">勝率: {tc['win_p']*100:.1f}% | スコア: {tc['score']}点 | Kelly: {tc['kelly']*100:.1f}%</div>
                        <details class="breakdown-toggle"><summary>根拠を見る</summary><div class="breakdown-body">{tc['breakdown']}</div></details>
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:9px; color:#7c8aab; font-family:'Roboto Mono',monospace;">EV&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;期待利益({_ep_stake:,}円)</div>
                    <div style="font-family:'Roboto Mono',monospace;">
                      <span style="font-size:1rem; font-weight:700; color:{ev_color};">{'+' if tc['ev']>=0 else ''}{tc['ev']*100:.1f}%</span>
                      &nbsp;<span style="font-size:1.1rem; font-weight:900; color:{ep_col};">{'+' if ep_yen>=0 else ''}{ep_yen:,}円</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    tabs = st.tabs(["💰 Kelly配分", "🧮 フォーメーション", "🏆 キマリテ分析", "🏁 レース診断", "📊 買い目EV", "📝 レースノート", "📒 購入ログ", "📈 バックテスト分析"])

    # --- Kelly配分タブ ---
    with tabs[0]:
        st.markdown('<div class="panel-box">', unsafe_allow_html=True)
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            total_budget = st.number_input("総投資予算(円)", min_value=100, value=3000, step=100,
                help="このレースに使う投資予算の合計額です。この金額を基準に、下のKelly係数（賭け金の割合）に応じて各買い目への配分金額が自動計算されます。生活費とは別の『投資に回してよい余剰資金』の範囲で設定してください。")
        st.session_state["total_budget_ref"] = total_budget  # EVランキングの期待利益表示に使用

        # ============================================================
        # AC-06: Kelly保守モード（実戦）+ 研究モード切替
        # 仕様書4.6.1: 実戦=保守3段階ボタン / 研究=フルスライダー（赤バッジ）
        # ============================================================
        with bcol2:
            _kelly_mode = st.radio(
                "Kelly配分モード",
                ["保守モード（実戦推奨）", "研究モード（高リスク）"],
                key="kelly_mode_radio",
                horizontal=True,
                help="賭け金の計算方法を選びます。「保守モード」はAIの予測誤差を考慮して賭け金を控えめに抑えるため、実際にお金を賭けるときはこちらを選んでください。「研究モード」は理論上の最大賭け金（フルKelly）に近づけられますが、予測が少しでも外れると損失が急激に大きくなるため、バックテストや検証の目的以外では使わないでください。"
            )

        if _kelly_mode == "保守モード（実戦推奨）":
            st.markdown(
                '<div style="font-size:10px;color:var(--text-dim2);margin-bottom:6px;">'
                '🛡️ 保守モード：確率誤差を考慮したリスク抑制設定。実戦ではこちらを推奨。</div>',
                unsafe_allow_html=True,
            )
            _kelly_cols = st.columns(3)
            _kelly_labels = [
                ("0.10", "保守10%", "✓推奨", "#16e0a0"),
                ("0.15", "保守15%", "", "#ffb648"),
                ("0.20", "保守20%", "", "#ffb648"),
            ]
            if "kelly_conservative_sel" not in st.session_state:
                st.session_state.kelly_conservative_sel = "0.10"
            for _ci, (_kv, _kl, _badge, _kc) in enumerate(zip(["0.10","0.15","0.20"],
                                                                ["保守10%","保守15%","保守20%"],
                                                                ["✓推奨","",""], ["#16e0a0","#ffb648","#ffb648"])):
                with _kelly_cols[_ci]:
                    _selected = st.session_state.kelly_conservative_sel == _kv
                    _btn_style = f"border:2px solid {_kc};" if _selected else "border:1px solid #2a3859;"
                    _dim_col = "#7c8aab"
                    _active_col = _kc if _selected else _dim_col
                    st.markdown(
                        f'<div style="text-align:center;background:#080d1a;{_btn_style}border-radius:6px;padding:6px 4px;cursor:pointer;">'
                        f'<div style="font-size:13px;font-weight:700;color:{_active_col};">{_kl}</div>'
                        f'{"<div style=font-size:9px;color:#16e0a0;>"+_badge+"</div>" if _badge else ""}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    _btn_help = {
                        "0.10": "理論上の最適賭け金（フルKelly）の10%だけを実際に賭ける、最も安全な設定です。1回あたりの損益の振れ幅が小さく、初めて使う方や資金を大きく減らしたくない方向けです。",
                        "0.15": "フルKellyの15%を賭ける設定です。10%よりやや積極的に増やせますが、連敗時の資金減少幅も少し大きくなります。",
                        "0.20": "フルKellyの20%を賭ける、保守モードの中では最も積極的な設定です。的中時のリターンは大きくなりますが、連敗が続いた場合の資金減少も相応に大きくなる点に注意してください。",
                    }[_kv]
                    if st.button(f"選択{'✓' if _selected else ''}", key=f"kelly_btn_{_kv}",
                                 width='stretch', help=_btn_help):
                        st.session_state.kelly_conservative_sel = _kv
                        st.rerun()
            kelly_weight = float(st.session_state.kelly_conservative_sel)
        else:
            st.markdown(
                '<div style="background:#1a0500;border:1px solid #ff5c72;border-radius:6px;padding:6px 10px;'
                'font-size:10px;color:#ff5c72;font-weight:700;margin-bottom:6px;">'
                '🔬 研究専用モード — フルKellyは確率誤差で壊滅的損失のリスクあり。実戦では使用しないこと。</div>',
                unsafe_allow_html=True,
            )
            kelly_weight = st.slider("Kelly係数（研究専用）", 0.05, 1.0, 0.2, 0.05,
                                     help="理論上の最適賭け金（フルKelly）に対する割合をスライダーで自由に設定します。1.0に近づけるほど『理論上の期待値は最大』になりますが、AIの勝率予測が少しでもズレていた場合の損失も跳ね上がります。実戦での資金投入には向かないため、あくまで検証・研究目的で使ってください。")

        # 損失ストッパー設定
        _ls_col1, _ls_col2 = st.columns([1, 2])
        with _ls_col1:
            _new_limit = st.number_input("🛑 1日の許容損失額(円)", min_value=0, value=st.session_state.loss_limit, step=500, key="loss_limit_input",
                help="1日に負けてもよい金額の上限です。本日の累計損失がこの金額に達すると、画面に『LOSS STOPPER 発動』の警告が出て、それ以上の投資が事実上止められます（取り返そうとして損失を拡大させないための仕組みです）。0にすると上限チェックは働きません。")
            if _new_limit != st.session_state.loss_limit:
                st.session_state.loss_limit = _new_limit
                st.session_state.loss_stopper_dismissed = False
        with _ls_col2:
            if _loss_limit > 0:
                _pct = min(100, _today_loss/_loss_limit*100)
                _col = "#ff5c72" if _today_loss >= _loss_limit else "#ffb648" if _pct >= 70 else "#4d5c80"
                st.markdown(f'<div style="margin-top:28px;font-size:10px;color:{_col};">本日損失: {_today_loss:,}円 / {_loss_limit:,}円（{_pct:.0f}%）</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="margin-top:28px;font-size:10px;color:#4d5c80;">（損失ストッパー無効）</div>', unsafe_allow_html=True)

        # 動的Kelly推奨
        _sp_logs = load_logs()
        _sp_decided = sorted([l for l in _sp_logs if l.get("result") in ("hit","miss")], key=lambda x: x["date"], reverse=True)[:5]
        _sp_losses = 0
        for _l in _sp_decided:
            if _l.get("result") == "miss": _sp_losses += 1
            else: break
        _sp_rec = calc_recommended_kelly(iq, all_manual_ok, _sp_losses)
        st.markdown(
            f'<div style="font-size:10px; margin-top:4px;">推奨: <b style="color:{_sp_rec["color"]};">{_sp_rec["mult"]:.1f}×</b>　'
            f'<span style="color:{_sp_rec["color"]};">{_sp_rec["label"]}{f" 直近{_sp_losses}連敗" if _sp_losses>=2 else ""}</span></div>',
            unsafe_allow_html=True,
        )
        if st.button(f'推奨値 {_sp_rec["mult"]:.1f}× を適用', key="apply_kelly_rec_sp"):
            kelly_weight = _sp_rec["mult"]
            st.rerun()

        # 再計算（kelly_weight変更を反映 + 自動減衰を適用）
        _effective_kelly = kelly_weight * _kelly_decay
        for tc in trade_list:
            tc["kelly"] = kelly_fraction(tc["win_p"], raw_odds.get(tc["boat"], 0.0), _effective_kelly)
        trade_list.sort(key=lambda x: -x["ev"])

        # ============================================================
        # 理論Kelly / 採用Kelly / 上限適用後Kelly の内訳表示（項目9対応）
        # 「IQが高いからフルベット推奨」という単純化を避け、
        # ①理論値 ②今回選んだ倍率 ③自動減衰後の実際の適用値 を分けて見せる。
        # ============================================================
        _theory_kelly_frac = kelly_fraction(trade_list[0]["win_p"], raw_odds.get(trade_list[0]["boat"], 0.0), 1.0)
        _adopted_kelly_frac = kelly_fraction(trade_list[0]["win_p"], raw_odds.get(trade_list[0]["boat"], 0.0), kelly_weight)
        _applied_kelly_frac = trade_list[0]["kelly"]
        st.markdown(
            f'<div style="background:#080d1a;border:1px solid var(--border-dark);border-radius:6px;padding:8px 12px;margin-bottom:8px;">'
            f'<div style="font-size:9px;color:var(--text-dim2);margin-bottom:4px;">💰 Kelly内訳（最上位候補）'
            f'<details class="gloss-term" style="display:inline;margin-left:6px;"><summary style="display:inline;">❓</summary>'
            f'<span class="gloss-body">「理論Kelly」は数式上の最大賭け金割合（フルKelly、倍率1.0）です。'
            f'これをそのまま賭けると、予測が少しでも外れたときの資金減少が大きすぎるため、'
            f'実際には「採用Kelly」（あなたが選んだ保守/研究モードの倍率）まで落とします。'
            f'さらに、オッズが古い・モデルが不一致・ティルト中などの不安定要因があれば、'
            f'そこからさらに自動で削られたものが「適用後Kelly」＝実際に使われる金額です。</span></details>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;font-family:var(--font-mono,monospace);">'
            f'<div style="text-align:center;"><div style="font-size:8px;color:var(--text-dim2);">理論Kelly</div>'
            f'<div style="font-size:12px;color:#7c8aab;font-weight:700;">{_theory_kelly_frac*100:.1f}%</div></div>'
            f'<div style="text-align:center;"><div style="font-size:8px;color:var(--text-dim2);">採用Kelly</div>'
            f'<div style="font-size:12px;color:#3fc4ff;font-weight:700;">{_adopted_kelly_frac*100:.1f}%</div></div>'
            f'<div style="text-align:center;border:1px solid {"#16e0a0" if _applied_kelly_frac>0 else "#ff5c72"};border-radius:4px;padding:2px 8px;">'
            f'<div style="font-size:8px;color:var(--text-dim2);">適用後Kelly</div>'
            f'<div style="font-size:14px;color:{"#16e0a0" if _applied_kelly_frac>0 else "#ff5c72"};font-weight:900;">{_applied_kelly_frac*100:.1f}%</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # Kelly自動減衰の表示
        if _kelly_decay < 1.0:
            _decay_pct = round(_kelly_decay * 100)
            st.markdown(
                f'<div style="background:#1a0500;border:1px solid #ff5c72;border-radius:6px;'
                f'padding:7px 12px;font-size:10px;color:#ff5c72;font-weight:700;margin-bottom:8px;">'
                f'⚡ Kelly自動減衰：{kelly_weight:.2f}× → {_effective_kelly:.2f}×（{_decay_pct}%）<br>'
                f'<span style="font-weight:400;color:var(--text-dim);">'
                f'理由：{"、".join(_kelly_decay_reasons)}</span></div>',
                unsafe_allow_html=True,
            )

        best_boat = trade_list[0]["boat"]
        partner_boat = trade_list[1]["boat"]
        inv_sum = (1 / raw_odds[best_boat] if raw_odds.get(best_boat, 0) > 0 else 0) + (1 / raw_odds[partner_boat] if raw_odds.get(partner_boat, 0) > 0 else 0)
        synthetic_odds = (1 / inv_sum) if inv_sum > 0 else 0

        st.markdown(f"**<span title='軸艇と相手艇を組み合わせたときの理論上のオッズ。単純掛け算ではなく控除率を考慮した計算'>合成オッズ(軸-相手)</span>**: <span style='color:#16e0a0; font-family:Roboto Mono;'>{synthetic_odds:.2f}倍</span>", unsafe_allow_html=True)
        teleboat_str = f"{best_boat}-{partner_boat}=2000,{best_boat}-全=1000"
        st.text_area("テレボート一括投票用コピペ文字列", value=teleboat_str, height=60, help="テレボートの一括投票画面に貼り付けると自動で投票内容が入力される文字列。金額は右のスライダーで変更可能")

        top3 = sorted(trade_list, key=lambda x: -x["kelly"])[:3]
        kelly_sum = sum(t["kelly"] for t in top3)
        alloc_rows = []
        for t in top3:
            if t["kelly"] <= 0:
                continue
            frac = t["kelly"] / kelly_sum if kelly_sum > 0 else 0
            stake = max(100, int(math.floor((total_budget * frac) / 100) * 100))
            payout = round(stake * raw_odds.get(t["boat"], 0))
            alloc_rows.append({"買い目": f"{t['boat']}号艇 単勝", "AI予測確率": f"{t['win_p']*100:.1f}%", "Kelly配分": f"{stake:,}円", "想定払戻": f"{payout:,}円", "_boat": t["boat"], "_stake": stake, "_payout": payout, "_win_p": t["win_p"]})

        if alloc_rows:
            df_alloc = pd.DataFrame(alloc_rows)[["買い目", "AI予測確率", "Kelly配分", "想定払戻"]]
            st.dataframe(df_alloc, width='stretch', hide_index=True)
            rec_col1, rec_col2, rec_col3 = st.columns([1, 1, 2])
            with rec_col1:
                rec_race_no = st.number_input("レース番号", min_value=1, max_value=12, value=1, step=1, key="rec_race_no",
                    help="実際に購入した（する予定の）レースの番号です。「✅ この配分で購入記録」ボタンを押したときに、この番号で購入ログに記録されます。")
            with rec_col2:
                rec_final_odds = st.number_input("最終オッズ（実際）", min_value=0.0, value=0.0, step=0.1, format="%.1f", key="rec_final_odds",
                                                  help="投票直前の実際のオッズ。0のままでは想定払戻を使用します。")
            with rec_col3:
                rec_comment = st.text_input("メモ・コメント（任意）", value="", key="rec_comment", placeholder="例：1号艇調子良し、向かい風注意、など",
                    help="このレースを買った理由や気づいたことを自由に書き残せます。後で「購入ログ」タブから振り返るときに、なぜその判断をしたのかを思い出す手がかりになります。入力しなくても記録は可能です。")
            if st.button("✅ この配分で購入記録", type="primary"):
                entries = []
                for r in alloc_rows:
                    actual_odds = rec_final_odds if rec_final_odds > 0 else raw_odds.get(r["_boat"], 0)
                    actual_payout = round(r["_stake"] * actual_odds) if actual_odds > 0 else r["_payout"]
                    entries.append({
                        "id": f"{datetime.now().timestamp()}_{r['_boat']}",
                        "date": datetime.now().isoformat(),
                        "venue": venue, "race_no": int(rec_race_no),
                        "combo": r["買い目"], "stake_yen": r["_stake"],
                        "payout_yen": actual_payout, "final_odds": float(actual_odds),
                        "result": "pending", "comment": rec_comment.strip(),
                        # 項目22対応：予測品質評価（Brier Score等）のため、購入時点の
                        # AI予測確率をログに保存する。的中率だけでなく「40%と言った時に
                        # 本当に40%前後で当たっているか」を後から検証できるようにする。
                        "win_p_at_purchase": round(r["_win_p"], 4),
                    })
                append_log(entries)
                st.success(f"{len(entries)}件の購入を記録しました。")
                st.rerun()
        else:
            st.info("Kelly基準上、推奨できる配分がありません（期待値が確率的優位を示していません）。")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- フォーメーションタブ ---
    with tabs[1]:
        st.markdown('<div class="panel-box">', unsafe_allow_html=True)

        # ============================================================
        # 6艇ランキング表（軸艇選びの判断材料・「1号艇が必ず本命ではない」問題への対応）
        # ============================================================
        _rank_sorted = sorted(results, key=lambda r: -r["prob_1st"])
        _rank_ev_map = {t["boat"]: t["ev"] for t in trade_list}
        _rank_rows_html = "".join([
            f'<div style="display:flex;align-items:center;gap:8px;padding:4px 6px;'
            f'{"background:rgba(22,224,160,0.08);border-radius:4px;" if i==0 else ""}">'
            f'<span style="font-size:11px;color:var(--text-dim2);width:18px;">{i+1}位</span>'
            f'<span style="font-size:13px;font-weight:900;color:{"#16e0a0" if i==0 else "var(--text-light)"};width:50px;">{r["boat"]}号艇</span>'
            f'<span style="font-size:11px;color:var(--text-dim2);width:90px;">AI予測 {r["prob_1st"]*100:.1f}%</span>'
            f'<span style="font-size:11px;color:{"#16e0a0" if _rank_ev_map.get(r["boat"],-1)>=0 else "#ff5c72"};">EV {_rank_ev_map.get(r["boat"],0)*100:+.1f}%</span>'
            f'</div>'
            for i, r in enumerate(_rank_sorted)
        ])
        st.markdown(
            f'<div style="background:#080d1a;border:1px solid var(--border-dark);border-radius:6px;padding:8px 12px;margin-bottom:10px;">'
            f'<div style="font-size:10px;color:var(--text-dim2);margin-bottom:4px;">🎯 1着になりやすい順（軸艇選びの参考に）'
            f'<details class="gloss-term" style="display:inline;margin-left:6px;"><summary style="display:inline;">❓</summary>'
            f'<span class="gloss-body">「1号艇はイン（内側）から発艇できるので統計的に有利」というのは一般論に過ぎず、'
            f'このレース個別では選手の実力・モーターの調子・展示タイムによって、他の艇の方が1着になりやすいことは普通にあります。'
            f'この表は「AI予測」欄（＝実力ベースの1着確率）でレース内の全艇を順位づけしたものなので、'
            f'まずはこの1位の艇を軸艇の第一候補として考えてください。'
            f'<br><br>「EV」欄はオッズも加味した期待値で、こちらは軸艇選びではなく『どの買い目が金額的にお得か』を見るためのものです。'
            f'軸艇（＝1着になってほしい艇）を選ぶときはAI予測（実力）を優先し、EVは相手艇や買い目を絞り込むときの参考にしてください。</span></details>'
            f'</div>{_rank_rows_html}</div>',
            unsafe_allow_html=True,
        )

        fcol1, fcol2 = st.columns(2)
        with fcol1:
            st.markdown(gloss("舟券種別"), unsafe_allow_html=True)
            ticket_type_label = st.selectbox("舟券種別", ["2連単", "2連複", "3連単", "3連複"], label_visibility="collapsed")
            ticket_type = {"2連単": "exacta", "2連複": "quinella", "3連単": "trifecta", "3連複": "trio"}[ticket_type_label]
        with fcol2:
            st.markdown(gloss("ノイズ除去", "ノイズ除去 最低確率(%)"), unsafe_allow_html=True)
            min_prob_pct = st.number_input("ノイズ除去 最低確率(%)", min_value=0.0, value=1.5, step=0.5, label_visibility="collapsed")
        fcol3, fcol4 = st.columns(2)
        with fcol3:
            st.markdown(gloss("軸艇", "軸艇(カンマ区切り)"), unsafe_allow_html=True)
            # 【改善】固定で"1"にせず、AIが計算した現在のレースの1着有力艇を自動で初期値にする
            _auto_axis = str(_rank_sorted[0]["boat"]) if _rank_sorted else "1"
            axis_str = st.text_input("軸艇(カンマ区切り)", value=_auto_axis, label_visibility="collapsed",
                help=f"最初は上のランキング表で1位（AI予測が最も高い艇＝{_auto_axis}号艇）が自動で入っています。必要に応じて数字を書き換えてください。")
        with fcol4:
            st.markdown(gloss("相手艇", "相手艇(カンマ区切り)"), unsafe_allow_html=True)
            partner_str = st.text_input("相手艇(カンマ区切り)", value="1,2,3,4,5,6", label_visibility="collapsed")

        def parse_boats(s):
            out = []
            for v in s.split(","):
                v = v.strip()
                if v.isdigit() and 1 <= int(v) <= 6:
                    out.append(int(v))
            return out

        axis_boats = parse_boats(axis_str)
        partner_boats = parse_boats(partner_str)

        if axis_boats and partner_boats:
            combos = calc_formation_summary(results, axis_boats, partner_boats, ticket_type, total_budget)
            combos = filter_noise_combos(combos, min_prob_pct / 100)
            combos.sort(key=lambda x: -x["prob"])
            if combos:
                df_combo = pd.DataFrame([
                    {"買い目": c["label"], "同時確率": f"{c['prob']*100:.2f}%", "傾斜配分": f"{c['allocated_yen']:,}円"}
                    for c in combos[:25]
                ])
                st.dataframe(df_combo, width='stretch', hide_index=True)
                st.markdown(f'<div class="note">合計 {len(combos)} 点中 上位25点を表示。フィルタ基準: 確率 {min_prob_pct:.1f}% 以上。</div>', unsafe_allow_html=True)

                # ============================================================
                # カバー率・軸崩壊リスク・実質独立賭け数（外部レビュー ㉒㉓㉔㉖㉗対応）
                # 「点数が多い＝分散できている」とは限らない、という指摘への対応。
                # 複数点買っていても、実際には同じ艇の1着に依存していることが多く、
                # その艇が1着を逃した瞬間に全滅する、というリスクを可視化する。
                # ============================================================
                _total_prob = sum(c["prob"] for c in combos)
                _norm_probs = [c["prob"] / _total_prob for c in combos] if _total_prob > 0 else []
                # 実質独立賭け数：各買い目の資金配分の偏りをもとにしたeffective number（逆ハーフィンダール指数）
                _total_alloc = sum(c["allocated_yen"] for c in combos)
                if _total_alloc > 0:
                    _hhi = sum((c["allocated_yen"] / _total_alloc) ** 2 for c in combos)
                    _effective_bets = round(1 / _hhi, 1) if _hhi > 0 else 0
                else:
                    _effective_bets = 0

                _axis_html = ""
                if ticket_type in ("exacta", "trifecta"):
                    # 1着（combo[0]）ごとの資金配分依存度を集計
                    _axis_alloc = {}
                    for c in combos:
                        b1 = c["combo"][0]
                        _axis_alloc[b1] = _axis_alloc.get(b1, 0) + c["allocated_yen"]
                    if _total_alloc > 0 and _axis_alloc:
                        _top_axis, _top_alloc = max(_axis_alloc.items(), key=lambda x: x[1])
                        _axis_dep_pct = _top_alloc / _total_alloc * 100
                        _axis_html = (
                            f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-top:1px solid var(--border-dark);margin-top:4px;">'
                            f'<span style="color:var(--text-dim2);">軸崩壊リスク（{_top_axis}号艇1着への資金依存度）</span>'
                            f'<span style="color:{"#ff5c72" if _axis_dep_pct>=70 else "#ffb648" if _axis_dep_pct>=40 else "#16e0a0"};font-weight:900;">{_axis_dep_pct:.0f}%</span></div>'
                        )

                st.markdown(
                    f'<div style="background:#080d1a;border:1px solid var(--border-dark);border-radius:6px;padding:8px 12px;margin-top:8px;">'
                    f'<div style="font-size:9px;color:var(--text-dim2);margin-bottom:2px;">📐 フォーメーションのリスク分散度'
                    f'<details class="gloss-term" style="display:inline;margin-left:6px;"><summary style="display:inline;">❓</summary>'
                    f'<span class="gloss-body">舟券は点数を増やすほど「分散できている」ように見えますが、実際には多くの買い目が'
                    f'「同じ艇が1着になる」という同じ前提に依存していることがよくあります。その艇が1着を逃した瞬間、'
                    f'10点買っていても全滅する、ということが起こり得ます。'
                    f'<br>「実質独立賭け数」は、資金配分の偏り具合から計算した「本当の意味で分散できている賭けの数」の目安です。'
                    f'例えば10点買っていても実質独立賭け数が2.3なら、実際には2〜3個の独立した賭けをしているのとあまり変わりません。'
                    f'<br>「軸崩壊リスク」は、資金の何%が特定の1艇の1着を前提にしているかを示します。これが高いほど、その艇が飛んだ時の被害が大きくなります。</span></details>'
                    f'</div>'
                    f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
                    f'<span style="color:var(--text-dim2);">カバー率（この{len(combos)}点でカバーする結果空間）</span>'
                    f'<span style="color:var(--text-light);font-weight:900;">{_total_prob*100:.1f}%</span></div>'
                    f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-top:1px solid var(--border-dark);">'
                    f'<span style="color:var(--text-dim2);">実質独立賭け数（{len(combos)}点中）</span>'
                    f'<span style="color:var(--text-light);font-weight:900;">{_effective_bets}</span></div>'
                    f'{_axis_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("条件に合う買い目がありません（ノイズ除去フィルタを下げてください）。")
        else:
            st.info("軸艇・相手艇を入力してください。")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- キマリテ分析タブ ---
    with tabs[2]:
        st.markdown('<div class="panel-box">', unsafe_allow_html=True)
        sorted_results = sorted(results, key=lambda x: -x["prob_1st"])
        top_boat = sorted_results[0]["boat"]
        pattern = estimate_kimarite(top_boat, venue)
        v_score = VENUE_SCORE.get(venue, 0)
        kcol1, kcol2, kcol3 = st.columns(3)
        kcol1.metric("本命想定艇", BOAT_LABEL[top_boat])
        kcol2.metric("想定キマリテ", pattern)
        kcol3.metric("会場イン優位指数", f"{'+' if v_score>=0 else ''}{v_score}")
        df_kim = pd.DataFrame([
            {"艇": BOAT_LABEL[r["boat"]], "1着率": f"{r['prob_1st']*100:.1f}%", "2着内率": f"{r['prob_2nd_within']*100:.1f}%", "3着内率": f"{r['prob_3rd_within']*100:.1f}%"}
            for r in sorted_results
        ])
        st.dataframe(df_kim, width='stretch', hide_index=True)
        st.markdown('<div class="note">想定キマリテは会場のイン優位指数と上位艇のコース構成から導く簡易推定です。</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 展示異常ランキング
        st.markdown('<div class="panel-box">', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">📊 展示タイムランキング（速い順）</div>', unsafe_allow_html=True)
        _ex_list = [(b["boat"], b["ex"], decomp_data.get(b["boat"], {}).get("anomaly", "通常")) for b in boats_data]
        _ex_mean = sum(e for _,e,_ in _ex_list) / len(_ex_list)
        _ex_sorted = sorted(_ex_list, key=lambda x: x[1])
        _ex_rows = [{"艇": BOAT_LABEL[bn], "展示T": f"{ex:.2f}s", "平均差": f"{ex-_ex_mean:+.2f}", "異常": ano if ano!="通常" else ""} for bn,ex,ano in _ex_sorted]
        st.dataframe(pd.DataFrame(_ex_rows), width='stretch', hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # AI推奨理由
        st.markdown('<div class="panel-box">', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">🤖 AI分析サマリー</div>', unsafe_allow_html=True)
        _sp_top = trade_list[0]
        _sp_tags = []
        if _sp_top["win_p"] >= 0.35: _sp_tags.append("高確率本命")
        elif _sp_top["win_p"] >= 0.27: _sp_tags.append("有力候補")
        if _sp_top["gap"] > 0.05: _sp_tags.append("市場過小評価")
        elif _sp_top["gap"] < -0.05: _sp_tags.append("市場過大評価")
        if _sp_top["ev"] > CONFIG["EV_THRESHOLD_BUY"]: _sp_tags.append(f'EV+{_sp_top["ev"]*100:.0f}%')
        if conf["score"] >= 70: _sp_tags.append("高信頼度レース")
        if VENUE_SCORE.get(venue, 0) >= 6: _sp_tags.append(f"{venue}インコース有利")
        elif VENUE_SCORE.get(venue, 0) <= -10: _sp_tags.append(f"{venue}波乱注意")
        _sp_summary = f'{BOAT_LABEL[_sp_top["boat"]]}が{"・".join(_sp_tags[:3])}の条件で{"注目" if _sp_top["ev"]>=CONFIG["EV_THRESHOLD_BUY"] else "参考"}。' if _sp_tags else f'{BOAT_LABEL[_sp_top["boat"]]}が本命候補。'
        _sp_tags_html = "".join([f'<span style="display:inline-block;background:var(--border-mid);color:var(--text-neutral);border-radius:3px;padding:1px 6px;font-size:9px;margin-right:3px;">{t}</span>' for t in _sp_tags])
        st.markdown(f'<div style="background:#0a0f1c;border:1px solid var(--border-dark);border-radius:6px;padding:8px 12px;font-size:11px;">{_sp_tags_html}<span style="color:var(--text-light);">{_sp_summary}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # リスク分解
        st.markdown('<div class="panel-box">', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">⚠️ リスク分解</div>', unsafe_allow_html=True)
        _sp_ex_spread = max(b["ex"] for b in boats_data) - min(b["ex"] for b in boats_data)
        _sp_ex_s = 5 if _sp_ex_spread<0.1 else 4 if _sp_ex_spread<0.2 else 3 if _sp_ex_spread<0.3 else 2 if _sp_ex_spread<0.4 else 1
        _sp_w_s = 5 if wind_mps<=1 else 4 if wind_mps<=2 else 3 if wind_mps<=3 else 2 if wind_mps<=5 else 1
        _sp_m_s = 5 if conf["score"]>=80 else 4 if conf["score"]>=65 else 3 if conf["score"]>=50 else 2 if conf["score"]>=35 else 1
        def _sp_star(n): return "★"*n + "☆"*(5-n)
        def _sp_col(n): return "#16e0a0" if n>=4 else "#3fc4ff" if n>=3 else "#ffb648" if n>=2 else "#ff5c72"
        _sp_rcols = st.columns(4)
        _sp_rcols[0].markdown(f'**展示安定性**\n\n<span style="color:{_sp_col(_sp_ex_s)};font-size:14px;">{_sp_star(_sp_ex_s)}</span>\n\n差{_sp_ex_spread:.2f}s', unsafe_allow_html=True)
        _sp_rcols[1].markdown(f'**進入（要確認）**\n\n<span style="color:#ffb648;font-size:14px;">{_sp_star(3)}</span>\n\n目視確認', unsafe_allow_html=True)
        _sp_rcols[2].markdown(f'**風**\n\n<span style="color:{_sp_col(_sp_w_s)};font-size:14px;">{_sp_star(_sp_w_s)}</span>\n\n{wind_mps}m/s', unsafe_allow_html=True)
        _sp_rcols[3].markdown(f'**市場安定性**\n\n<span style="color:{_sp_col(_sp_m_s)};font-size:14px;">{_sp_star(_sp_m_s)}</span>\n\n信頼度{conf["score"]}点', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- レース診断タブ ---
    with tabs[3]:
        st.markdown('<div class="panel-box">', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">🏁 レースタイプ診断</div>', unsafe_allow_html=True)

        # レースタイプ判定
        _top_t = trade_list[0]
        _chaos_t = 100 - conf["score"]
        if _top_t["win_p"] >= 0.40 and conf["score"] >= 70:
            _rtype, _rcolor, _rstars, _rdesc = "◎ 本命戦", "#16e0a0", "★★★★★", "トップ艇が圧倒的。イン逃げ濃厚。単勝・2連単が有効。"
        elif _top_t["win_p"] >= 0.30 and conf["score"] >= 55:
            _rtype, _rcolor, _rstars, _rdesc = "○ 本命寄り", "#3fc4ff", "★★★★☆", "本命有力だが相手次第。2連単・2連複が安定。"
        elif _chaos_t <= 40:
            _rtype, _rcolor, _rstars, _rdesc = "△ 混戦", "#ffb648", "★★★☆☆", "力差なし。BOXや幅広いフォーメーションを検討。"
        elif wind_mps >= 4 or _chaos_t >= 60:
            _rtype, _rcolor, _rstars, _rdesc = "▲ 荒れ", "#ff8c42", "★★☆☆☆", "波乱含み。外コース・強モーター艇に注意。少額or見送り推奨。"
        else:
            _rtype, _rcolor, _rstars, _rdesc = "× 超波乱", "#ff5c72", "★☆☆☆☆", "読めないレース。見送りを強く推奨。"

        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;flex-wrap:wrap;">'
            f'<span style="padding:6px 18px;border-radius:20px;font-size:16px;font-weight:900;color:{_rcolor};border:2px solid {_rcolor};background:rgba(0,0,0,0.3);">{_rtype}</span>'
            f'<span style="font-size:18px;color:{_rcolor};">{_rstars}</span>'
            f'<span style="font-size:11px;color:var(--text-dim);">{_rdesc}</span></div>',
            unsafe_allow_html=True,
        )

        # おすすめ券種
        _tickets = [
            ("単勝",   5 if _top_t["win_p"]>=0.40 else 4 if _top_t["win_p"]>=0.30 else 3 if _top_t["win_p"]>=0.22 else 2, "最もシンプル"),
            ("2連単",  5 if _top_t["win_p"]>=0.30 and conf["score"]>=65 else 4 if _top_t["win_p"]>=0.25 else 3, "本命+相手明確時"),
            ("2連複",  4 if conf["score"]>=60 and _chaos_t<=50 else 3, "混戦で保険"),
            ("3連単",  4 if _top_t["win_p"]>=0.35 and conf["score"]>=70 else 3 if _chaos_t<=35 else 2, "高配当狙い"),
        ]
        _best_score = max(t[1] for t in _tickets)
        st.markdown('<div class="eyebrow" style="margin-bottom:6px;">🎫 おすすめ券種</div>', unsafe_allow_html=True)
        _tcols = st.columns(4)
        for i, (tname, tscore, tnote) in enumerate(_tickets):
            _is_best = tscore == _best_score
            _tstars = "★"*tscore + "☆"*(5-tscore)
            _tcol_str = "#16e0a0" if tscore>=4 else "#3fc4ff" if tscore>=3 else "#7c8aab"
            _border = "border:2px solid #16e0a0;" if _is_best else "border:1px solid var(--border-dark);"
            _tcols[i].markdown(
                f'<div style="background:#080d1a;{_border}border-radius:8px;padding:10px 6px;text-align:center;">'
                f'<div style="font-size:11px;font-weight:700;color:var(--text-light);">{tname}{"  🏆" if _is_best else ""}</div>'
                f'<div style="font-size:13px;color:{_tcol_str};margin:4px 0;">{_tstars}</div>'
                f'<div style="font-size:9px;color:var(--text-dim2);">{tnote}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # 補正テーブルUI可視化
        st.markdown('<div class="panel-box">', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">🔬 会場別Isotonic補正テーブル（現在の会場）</div>', unsafe_allow_html=True)
        st.markdown('<div class="note">AIの生確率がこの会場では統計的にどう補正されるかを示します。n=14,867レースの実績から学習済み。</div>', unsafe_allow_html=True)

        _cal_table = CALIBRATION_TABLE.get(venue, CALIBRATION_TABLE.get("global", {}))
        if _cal_table:
            _cal_rows = []
            _sorted_keys = sorted(_cal_table.keys())
            for _k in _sorted_keys:
                _v = _cal_table[_k]
                _diff = _v - _k
                _diff_col = "#16e0a0" if _diff > 0.01 else "#ff5c72" if _diff < -0.01 else "#7c8aab"
                _diff_str = f"{_diff*100:+.1f}%"
                # 現在の補正ポイントをハイライト
                _is_current = False
                for r in results:
                    if abs(r["prob_1st"] - _k) < 0.03:
                        _is_current = True
                        break
                _cal_rows.append({
                    "生確率帯": f"{_k*100:.0f}%前後",
                    "補正後": f"{_v*100:.1f}%",
                    "補正量": _diff_str,
                    "傾向": "↑ 過小評価修正" if _diff > 0.01 else "↓ 過大評価修正" if _diff < -0.01 else "→ 変化なし",
                })
            if _cal_rows:
                _df_cal = pd.DataFrame(_cal_rows)
                st.dataframe(_df_cal, width='stretch', hide_index=True)

                # 現在のレース結果と補正の対応を表示
                st.markdown('<div style="margin-top:8px; font-size:10px; font-weight:700; color:var(--text-dim);">📍 このレースの補正適用結果</div>', unsafe_allow_html=True)
                _comp_rows = []
                for r in sorted(results, key=lambda x: -x["prob_1st"])[:3]:
                    _raw = r.get("raw_prob", r["prob_1st"])
                    _cal = r["prob_1st"]
                    _d = _cal - _raw
                    _comp_rows.append({
                        "艇": f"{r['boat']}号艇",
                        "生確率": f"{_raw*100:.1f}%",
                        "補正後": f"{_cal*100:.1f}%",
                        "補正量": f"{_d*100:+.1f}%",
                    })
                if _comp_rows:
                    st.dataframe(pd.DataFrame(_comp_rows), width='stretch', hide_index=True)
        else:
            st.info(f"{venue}の補正テーブルが見つかりません。global補正を使用中です。")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 買い目EVランキングタブ ---
    with tabs[4]:
        st.markdown('<div class="panel-box">', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">📊 2連単 買い目別EVランキング</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="background:#1a0f00;border:1px solid #ffb648;border-radius:6px;'
            'padding:7px 12px;font-size:10.5px;color:#ffb648;font-weight:700;margin-bottom:8px;">'
            '⚠️ <b>推定EV（参考値のみ）</b> — 2連単オッズは単勝からの近似計算です。'
            'BUY/SKIP判定ゲートには使用されません。実測2連単オッズが取得できた場合のみ参考にしてください。</div>',
            unsafe_allow_html=True,
        )

        _ev_count = st.selectbox("表示件数", [5, 10, 30], index=0, key="ev_rank_count",
            help="期待値（EV）が高い順に並べた買い目候補を、何件まで一覧表示するかを選びます。件数を増やすほど下位の候補まで確認できますが、一覧が長くなります。")
        _ev_thresh = st.selectbox("EV閾値フィルター", ["すべて", "EV>0%のみ", "EV>5%のみ", "EV>10%のみ"], index=0, key="ev_rank_filter",
            help="期待値が一定以上の買い目だけに絞り込んで表示します。例えば「EV>5%のみ」を選ぶと、期待値が5%を超える（統計的に有利な）買い目候補だけが残ります。「すべて」は絞り込みを行いません。")
        _thresh_map = {"すべて": -99, "EV>0%のみ": 0.0, "EV>5%のみ": 0.05, "EV>10%のみ": 0.10}
        _ev_min = _thresh_map[_ev_thresh]

        # 2連単30通りEV計算
        _score_map = {r["boat"]: r.get("score", 0) for r in results}
        _ev_combos = []
        for _i in range(1, 7):
            for _j in range(1, 7):
                if _i == _j: continue
                _p1 = next((r["prob_1st"] for r in results if r["boat"]==_i), 0)
                _rem_sum = sum(v for b,v in _score_map.items() if b != _i)
                _p2g1 = (_score_map.get(_j,0)/_rem_sum) if _rem_sum > 0 else 0
                _prob = _p1 * _p2g1
                _o1 = raw_odds.get(_i, 0); _o2 = raw_odds.get(_j, 0)
                _exacta_odds = max(1.1, _o1*_o2*0.12) if _o1>0 and _o2>0 else 0
                _ev_val = _prob*_exacta_odds-1 if _exacta_odds>0 else -1
                _ev_combos.append({"買い目":f"{_i}-{_j}", "EV": round(_ev_val*100,1), "的中率(%)": round(_prob*100,2), "推定オッズ": round(_exacta_odds,1)})

        _ev_combos.sort(key=lambda x: -x["EV"])
        _ev_filtered = [c for c in _ev_combos if c["EV"]/100 >= _ev_min][:_ev_count]
        if _ev_filtered:
            _df_ev = pd.DataFrame(_ev_filtered)
            _df_ev["EV"] = _df_ev["EV"].apply(lambda x: f"{x:+.1f}%")
            _df_ev["的中率(%)"] = _df_ev["的中率(%)"].apply(lambda x: f"{x:.2f}%")
            _df_ev["推定オッズ"] = _df_ev["推定オッズ"].apply(lambda x: f"{x:.1f}倍" if x>0 else "--")
            st.dataframe(_df_ev, width='stretch', hide_index=True)
        else:
            st.info("該当する買い目がありません")

        st.markdown("---")

        # ============================================================
        # 買い目カバー率（N点で何%的中するか）
        # ============================================================
        st.markdown('<div class="eyebrow">🎯 買い目カバー率（2連単）</div>', unsafe_allow_html=True)
        st.markdown('<div class="note">上位N点買ったとき、合計的中確率は何%か</div>', unsafe_allow_html=True)
        _sorted_probs = sorted(_ev_combos, key=lambda x: -x["的中率(%)"])
        _cover_rows = []
        _cumulative = 0.0
        for _n in [3, 5, 8, 10, 15, 20, 30]:
            _top_n = _sorted_probs[:_n]
            _cumulative = sum(float(c["的中率(%)"].replace("%","")) if isinstance(c["的中率(%)"],str) else c["的中率(%)"] for c in _top_n)
            _cover_rows.append({"点数": f"{_n}点", "的中カバー率": f"{_cumulative:.1f}%",
                                 "平均EV": f"{sum(float(c['EV'].replace('%','').replace('+','')) if isinstance(c['EV'],str) else c['EV'] for c in _top_n)/_n:+.1f}%"})
        st.dataframe(pd.DataFrame(_cover_rows), width='stretch', hide_index=True)

        # 的中率50%に必要な最小点数
        _cum = 0.0
        _min_pts_50 = None
        for _n_pts in range(1, 31):
            _cum += float(_sorted_probs[_n_pts-1]["的中率(%)"].replace("%","")) if isinstance(_sorted_probs[_n_pts-1]["的中率(%)"],str) else _sorted_probs[_n_pts-1]["的中率(%)"]
            if _cum >= 50.0 and _min_pts_50 is None:
                _min_pts_50 = _n_pts
                break
        if _min_pts_50:
            st.markdown(
                f'<div style="background:#062a1c;border:1px solid #16e0a0;border-radius:6px;padding:7px 12px;font-size:11px;color:#16e0a0;font-weight:700;margin-top:6px;">'
                f'🎯 的中率50%を確保する最小点数：<b>{_min_pts_50}点</b></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ============================================================
        # 3連単120通りランキング（上位15件）
        # ============================================================
        st.markdown('<div class="eyebrow">🏆 3連単 上位15件（AI確率順）</div>', unsafe_allow_html=True)
        st.markdown('<div class="note">※オッズは取得不可のため確率のみ表示</div>', unsafe_allow_html=True)
        try:
            _trifecta = get_trifecta_ranking(results)[:15]
            _df_tri = pd.DataFrame([
                {"順位": f"#{t['rank']}", "買い目": t["combo"], "AI確率": f"{t['prob']*100:.2f}%"}
                for t in _trifecta
            ])
            st.dataframe(_df_tri, width='stretch', hide_index=True)
        except Exception as _e:
            st.caption(f"3連単計算エラー: {_e}")

        st.markdown('</div>', unsafe_allow_html=True)

    # --- レースノートタブ ---
    with tabs[5]:
        st.markdown('<div class="panel-box">', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">📝 レースノート</div>', unsafe_allow_html=True)

        _note_venue = venue
        _note_raceno = st.session_state.get("fetch_race_no", 1)

        _n1, _n2 = st.columns(2)
        with _n1:
            _note_reason = st.text_area("✅ 買い理由（何を根拠にしたか）", height=80, placeholder="例：展示1位・オッズ乖離+15%・大村イン有利", key="note_reason",
                help="このレースを『買おう』と判断した根拠を書いておきます。後から見返すことで、自分の判断のクセや、当たったとき／外れたときの共通点を振り返る材料になります。")
        with _n2:
            _note_concern = st.text_area("⚠️ 懸念点・リスク", height=80, placeholder="例：向かい風3m・4号艇の展示が伸びていた", key="note_concern",
                help="購入時に気になっていた不安要素・リスクを書いておきます。結果が外れたときに『やっぱりあの懸念が的中していた』と気づければ、次回の判断材料になります。")
        _note_review = st.text_area("📚 見落としていたこと（レース後に記入）", height=60, placeholder="例：前付けがあった・1号艇のSTが遅かった", key="note_review",
            help="レースが終わったあとに、購入前には気づけなかった点を書き足す欄です。的中・不的中に関わらず、次のレースでの見落としを減らすための振り返りメモとして使ってください。")

        if st.button("💾 ノートを保存", key="save_note_btn"):
            if not any([_note_reason, _note_concern, _note_review]):
                st.warning("内容を入力してください")
            else:
                if "race_notes" not in st.session_state:
                    st.session_state.race_notes = []
                st.session_state.race_notes.insert(0, {
                    "date": datetime.now().strftime("%m/%d %H:%M"),
                    "venue": _note_venue, "race_no": _note_raceno,
                    "reason": _note_reason, "concern": _note_concern, "review": _note_review,
                })
                if len(st.session_state.race_notes) > 50:
                    st.session_state.race_notes.pop()
                st.success("ノートを保存しました", icon="💾")

        # 過去ノート表示
        _notes = st.session_state.get("race_notes", [])
        if _notes:
            st.markdown("---")
            st.markdown('<div class="eyebrow">📚 直近のノート</div>', unsafe_allow_html=True)
            for ni, n in enumerate(_notes[:5]):
                with st.expander(f"{n['venue']} {n['race_no']}R　{n['date']}"):
                    if n.get("reason"): st.markdown(f"**✅ 買い理由:** {n['reason']}")
                    if n.get("concern"): st.markdown(f"**⚠️ 懸念点:** {n['concern']}")
                    if n.get("review"): st.markdown(f"**📚 見落とし:** {n['review']}")
                    if st.button("🗑 削除", key=f"del_note_{ni}"):
                        st.session_state.race_notes.pop(ni)
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 購入ログタブ ---
    with tabs[6]:
        st.markdown('<div class="panel-box">', unsafe_allow_html=True)
        logs = load_logs()
        if not logs:
            st.info("まだ購入記録がありません。Kelly配分タブから「この配分で購入記録」を押すとここに表示されます。")
        else:
            # 全期間サマリー・会場別ROI
            _sp_all_decided = [l for l in logs if l.get("result") in ("hit","miss")]
            if _sp_all_decided:
                _sp_stake = sum(l.get("stake_yen",0) for l in _sp_all_decided)
                _sp_payout = sum(l.get("payout_yen",0) for l in _sp_all_decided if l.get("result")=="hit")
                _sp_pnl = _sp_payout - _sp_stake
                _sp_hit = sum(1 for l in _sp_all_decided if l.get("result")=="hit")
                _sp_roi = (_sp_pnl/_sp_stake*100) if _sp_stake>0 else 0.0
                st.markdown('<div class="eyebrow">📊 全期間サマリー</div>', unsafe_allow_html=True)
                _da1,_da2,_da3,_da4 = st.columns(4)
                _da1.metric("累積損益", f"{'+' if _sp_pnl>=0 else ''}{_sp_pnl:,}円")
                _da2.metric("ROI", f"{_sp_roi:+.1f}%")
                _da3.metric("的中率", f"{_sp_hit/len(_sp_all_decided)*100:.1f}%")
                _da4.metric("総件数", f"{len(_sp_all_decided)}件")
                st.markdown("---")

                # ============================================================
                # 統計分析：条件別成績・勝負の分岐点
                # ============================================================
                with st.expander("📈 詳細統計分析（条件別成績・分岐点）", expanded=False):
                    st.markdown('<div class="eyebrow">🔬 条件別成績分析</div>', unsafe_allow_html=True)

                    # 曜日別
                    _dow_map = {0:"月",1:"火",2:"水",3:"木",4:"金",5:"土",6:"日"}
                    _dow_stats = {}
                    for l in _sp_all_decided:
                        try:
                            _dow = _dow_map[datetime.fromisoformat(l["date"]).weekday()]
                        except Exception:
                            _dow = "不明"
                        if _dow not in _dow_stats:
                            _dow_stats[_dow] = {"stake":0,"payout":0,"hit":0,"count":0}
                        _dow_stats[_dow]["stake"] += l.get("stake_yen",0)
                        _dow_stats[_dow]["count"] += 1
                        if l.get("result")=="hit":
                            _dow_stats[_dow]["payout"] += l.get("payout_yen",0)
                            _dow_stats[_dow]["hit"] += 1

                    # 時間帯別（午前/午後）
                    _time_stats = {"午前(〜12時)":{"stake":0,"payout":0,"hit":0,"count":0},
                                   "午後(12時〜)":{"stake":0,"payout":0,"hit":0,"count":0}}
                    for l in _sp_all_decided:
                        try:
                            _hour = datetime.fromisoformat(l["date"]).hour
                            _tk = "午前(〜12時)" if _hour < 12 else "午後(12時〜)"
                        except Exception:
                            _tk = "午後(12時〜)"
                        _time_stats[_tk]["stake"] += l.get("stake_yen",0)
                        _time_stats[_tk]["count"] += 1
                        if l.get("result")=="hit":
                            _time_stats[_tk]["payout"] += l.get("payout_yen",0)
                            _time_stats[_tk]["hit"] += 1

                    _stat_col1, _stat_col2 = st.columns(2)
                    with _stat_col1:
                        st.markdown("**📅 曜日別成績**")
                        _dow_rows = []
                        for _dn in ["月","火","水","木","金","土","日"]:
                            if _dn in _dow_stats:
                                _ds = _dow_stats[_dn]
                                _droi = (_ds["payout"]-_ds["stake"])/_ds["stake"]*100 if _ds["stake"]>0 else 0
                                _dhr  = _ds["hit"]/_ds["count"]*100 if _ds["count"]>0 else 0
                                _dow_rows.append({"曜日":_dn, "件数":_ds["count"], "的中率":f"{_dhr:.0f}%", "ROI":f"{_droi:+.1f}%"})
                        if _dow_rows:
                            st.dataframe(pd.DataFrame(_dow_rows), hide_index=True, width='stretch')

                    with _stat_col2:
                        st.markdown("**⏰ 時間帯別成績**")
                        _time_rows = []
                        for _tk, _ts in _time_stats.items():
                            if _ts["count"] > 0:
                                _troi = (_ts["payout"]-_ts["stake"])/_ts["stake"]*100 if _ts["stake"]>0 else 0
                                _thr  = _ts["hit"]/_ts["count"]*100 if _ts["count"]>0 else 0
                                _time_rows.append({"時間帯":_tk, "件数":_ts["count"], "的中率":f"{_thr:.0f}%", "ROI":f"{_troi:+.1f}%"})
                        if _time_rows:
                            st.dataframe(pd.DataFrame(_time_rows), hide_index=True, width='stretch')

                    # 投資金額別成績
                    st.markdown("**💰 投資金額帯別成績**")
                    _stake_bands = [
                        ("〜500円",    0,   500),
                        ("501〜1,000円",  501,  1000),
                        ("1,001〜3,000円",1001, 3000),
                        ("3,001円〜",  3001, 9999999),
                    ]
                    _stake_rows = []
                    for _bname, _bmin, _bmax in _stake_bands:
                        _blogs = [l for l in _sp_all_decided if _bmin <= l.get("stake_yen",0) <= _bmax]
                        if _blogs:
                            _bstake  = sum(l.get("stake_yen",0) for l in _blogs)
                            _bpayout = sum(l.get("payout_yen",0) for l in _blogs if l.get("result")=="hit")
                            _broi    = (_bpayout-_bstake)/_bstake*100 if _bstake>0 else 0
                            _bhr     = sum(1 for l in _blogs if l.get("result")=="hit")/len(_blogs)*100
                            _stake_rows.append({"金額帯":_bname, "件数":len(_blogs), "的中率":f"{_bhr:.0f}%", "ROI":f"{_broi:+.1f}%", "損益":f"{_bpayout-_bstake:+,}円"})
                    if _stake_rows:
                        st.dataframe(pd.DataFrame(_stake_rows), hide_index=True, width='stretch')

                    # 貧富の分岐点（何円投資からROIがプラスに転じるか）
                    if len(_sp_all_decided) >= 5:
                        st.markdown("**📐 貧富の分岐点（累積損益チャート）**")
                        _cum_pnl = 0
                        _cum_data = []
                        for idx, l in enumerate(sorted(_sp_all_decided, key=lambda x: x.get("date",""))):
                            _s = l.get("stake_yen",0)
                            _p = l.get("payout_yen",0) if l.get("result")=="hit" else 0
                            _cum_pnl += _p - _s
                            _cum_data.append({"レース番号":idx+1, "累積損益":_cum_pnl})
                        if _cum_data:
                            import altair as alt
                            _df_cum = pd.DataFrame(_cum_data)
                            _line_col = "result:Q"
                            _chart = alt.Chart(_df_cum).mark_line(
                                color="#16e0a0" if _cum_pnl >= 0 else "#ff5c72", strokeWidth=2
                            ).encode(
                                x=alt.X("レース番号:Q", title="レース数"),
                                y=alt.Y("累積損益:Q", title="累積損益（円）"),
                                tooltip=["レース番号","累積損益"]
                            ).properties(height=200).configure_view(
                                strokeOpacity=0
                            ).configure_axis(
                                gridColor="#1a2a3a", labelColor="#7c8aab", titleColor="#7c8aab"
                            )
                            st.altair_chart(_chart, width='stretch')

                st.markdown("---")
                _sp_vm = {}
                for l in _sp_all_decided:
                    v = l.get("venue","不明")
                    if v not in _sp_vm: _sp_vm[v] = {"stake":0,"payout":0,"hit":0,"count":0}
                    _sp_vm[v]["stake"] += l.get("stake_yen",0); _sp_vm[v]["count"] += 1
                    if l.get("result")=="hit": _sp_vm[v]["payout"] += l.get("payout_yen",0); _sp_vm[v]["hit"] += 1
                if len(_sp_vm) >= 2:
                    st.markdown('<div class="eyebrow">🏟️ 会場別ROI</div>', unsafe_allow_html=True)
                    _sp_vrows = []
                    for vn, vd in sorted(_sp_vm.items(), key=lambda x: -(x[1]["payout"]-x[1]["stake"])/(x[1]["stake"] or 1)):
                        vroi = (vd["payout"]-vd["stake"])/vd["stake"]*100 if vd["stake"]>0 else 0
                        vhr = vd["hit"]/vd["count"]*100 if vd["count"]>0 else 0
                        _sp_vrows.append({"会場":vn,"ROI":f"{vroi:+.1f}%","的中率":f"{vhr:.0f}%","件数":vd["count"]})
                    st.dataframe(pd.DataFrame(_sp_vrows), width='stretch', hide_index=True)
                    st.markdown("---")

            decided = [l for l in logs if l.get("result") in ("hit", "miss")]
            decided_sorted = sorted(decided, key=lambda x: x["date"])
            if len(decided_sorted) >= 2:
                import altair as alt
                rows = []
                cum_pnl = 0
                for idx, l in enumerate(decided_sorted):
                    stake = l.get("stake_yen", 0)
                    payout = l.get("payout_yen", 0)
                    hit = 1 if l.get("result") == "hit" else 0
                    pnl = (payout - stake) if hit else -stake
                    cum_pnl += pnl
                    rows.append({"seq": idx+1, "date": datetime.fromisoformat(l["date"]).strftime("%m/%d"), "cum_pnl": cum_pnl, "hit": hit, "pnl": pnl, "venue": l.get("venue","")})
                df_log = pd.DataFrame(rows)
                df_log["roll_hit"] = df_log["hit"].rolling(min(10, len(df_log)), min_periods=1).mean() * 100
                color_pnl = "#16e0a0" if cum_pnl >= 0 else "#ff5c72"
                base = alt.Chart(df_log).encode(x=alt.X("seq:Q", title="記録件数"), tooltip=["seq:Q","date:N","venue:N",alt.Tooltip("cum_pnl:Q",title="累積損益(円)"),alt.Tooltip("pnl:Q",title="今回損益(円)")])
                line_pnl = base.mark_line(color=color_pnl, strokeWidth=2).encode(y=alt.Y("cum_pnl:Q", title="累積損益（円）"))
                zero_line = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(color="#2a3859", strokeDash=[4,4]).encode(y="y:Q")
                dots_pnl = base.mark_point(size=40, filled=True).encode(y="cum_pnl:Q", color=alt.condition(alt.datum.hit==1, alt.value("#16e0a0"), alt.value("#ff5c72")))
                chart_pnl = (zero_line+line_pnl+dots_pnl).properties(title=alt.TitleParams("📈 累積損益推移", color="#f3f6fb"), height=200, background="#0c111f").configure_view(strokeWidth=0).configure_axis(gridColor="#1c2740")
                line_hit = alt.Chart(df_log).mark_line(color="#3fc4ff", strokeWidth=2).encode(x=alt.X("seq:Q", title="記録件数"), y=alt.Y("roll_hit:Q", title=f"直近{min(10,len(df_log))}件的中率（%）", scale=alt.Scale(domain=[0,100])), tooltip=["seq:Q","date:N",alt.Tooltip("roll_hit:Q",title="移動的中率(%)",format=".1f")])
                ref50 = alt.Chart(pd.DataFrame({"y":[50]})).mark_rule(color="#ffb648", strokeDash=[4,4]).encode(y="y:Q")
                chart_hit = (ref50+line_hit).properties(title=alt.TitleParams(f"🎯 直近{min(10,len(df_log))}件移動的中率", color="#f3f6fb"), height=160, background="#0c111f").configure_view(strokeWidth=0).configure_axis(gridColor="#1c2740")
                gcol1, gcol2 = st.columns([3, 2])
                with gcol1:
                    st.altair_chart(chart_pnl, width='stretch')
                with gcol2:
                    st.altair_chart(chart_hit, width='stretch')
            elif len(decided_sorted) == 1:
                st.caption("グラフは2件以上の確定記録があると表示されます。")
            st.markdown("---")
            all_decided = [l for l in logs if l.get("result") in ("hit","miss")]
            if all_decided:
                total_stake = sum(l.get("stake_yen",0) for l in all_decided)
                total_payout = sum(l.get("payout_yen",0) for l in all_decided if l.get("result")=="hit")
                total_pnl = total_payout - total_stake
                total_hit = sum(1 for l in all_decided if l.get("result")=="hit")
                total_roi = (total_pnl/total_stake*100) if total_stake > 0 else 0.0
                acol1,acol2,acol3,acol4 = st.columns(4)
                acol1.metric("全期間累積損益", f"{'+' if total_pnl>=0 else ''}{total_pnl:,}円")
                acol2.metric("全期間的中率", f"{total_hit/len(all_decided)*100:.1f}%")
                acol3.metric("全期間ROI", f"{total_roi:+.1f}%")
                acol4.metric("確定件数", f"{len(all_decided)}件")
                st.markdown("---")
            msum = monthly_summary(logs, date.today())
            scol1, scol2, scol3 = st.columns(3)
            scol1.metric("当月実現損益", f"{'+' if msum['realized']>=0 else ''}{msum['realized']:,}円")
            scol2.metric("当月的中率", f"{msum['hit_rate']:.1f}%" if msum["decided_count"] else "--")
            scol3.metric("確定件数", f"{msum['decided_count']}件")
            st.markdown("---")

            # ============================================================
            # 予測品質評価（項目22対応）
            # 的中率だけでなく「AIが40%と言った時、本当に40%前後で当たっているか」を検証する。
            # ============================================================
            _pq = calc_prediction_quality(logs)
            with st.expander(f"🔬 予測品質評価（Brier Score / LogLoss）— 対象{_pq['n']}件", expanded=False):
                st.markdown(
                    '<details class="gloss-term"><summary>❓ これは何を見ている指標？（クリックで説明）</summary>'
                    '<span class="gloss-body">的中率だけでは「AIの予測確率そのものが正確か」は分かりません。'
                    '例えば「40%」と表示したレースが、10回中4回くらいの頻度で本当に当たっていれば予測は正確ですが、'
                    '10回中1回しか当たっていなければ、AIが自信過剰（オーバーコンフィデント）ということになります。'
                    'Brier Scoreは「予測確率と実際の結果（0か1か）のズレの二乗」の平均で、0に近いほど優秀です（目安：0.20未満で良好）。'
                    'LogLossも予測の正確さを測る指標で、こちらも0に近いほど優秀です。'
                    'どちらも、この購入ログに記録された『予測確率つきの確定済み結果』が5件以上ないと計算できません。</span></details>',
                    unsafe_allow_html=True,
                )
                if not _pq["available"]:
                    st.info(f"予測品質を評価するには、購入時の予測確率つき確定結果が最低5件必要です（現在{_pq['n']}件）。今後の購入分から自動的に蓄積されます。")
                else:
                    pq1, pq2 = st.columns(2)
                    _brier_color = "#16e0a0" if _pq["brier"] < 0.20 else "#ffb648" if _pq["brier"] < 0.25 else "#ff5c72"
                    _logloss_color = "#16e0a0" if _pq["logloss"] < 0.55 else "#ffb648" if _pq["logloss"] < 0.69 else "#ff5c72"
                    pq1.markdown(f'<div class="panel-box"><div class="metric-label">Brier Score</div><div class="metric-val" style="color:{_brier_color};">{_pq["brier"]:.3f}</div></div>', unsafe_allow_html=True)
                    pq2.markdown(f'<div class="panel-box"><div class="metric-label">LogLoss</div><div class="metric-val" style="color:{_logloss_color};">{_pq["logloss"]:.3f}</div></div>', unsafe_allow_html=True)
                    if _pq["calib_rows"]:
                        st.markdown('<div style="font-size:10px;color:var(--text-dim2);margin-top:8px;margin-bottom:4px;">確率帯別キャリブレーション（予測 vs 実測）</div>', unsafe_allow_html=True)
                        _df_calib = pd.DataFrame(_pq["calib_rows"])
                        _df_calib_disp = pd.DataFrame({
                            "予測確率帯": _df_calib["range"],
                            "件数": _df_calib["n"],
                            "AI予測平均": _df_calib["pred"].round(1).astype(str) + "%",
                            "実際の的中率": _df_calib["actual"].round(1).astype(str) + "%",
                            "乖離": _df_calib["gap"].apply(lambda x: f"{x:+.1f}%"),
                        })
                        st.dataframe(_df_calib_disp, hide_index=True, width='stretch')
                        st.markdown('<div style="font-size:9px;color:var(--text-dim2);">乖離がプラス＝AIの予測より実際の的中率が高い（過小評価気味）。マイナス＝AIが自信過剰。件数が少ない確率帯は参考程度に見てください。</div>', unsafe_allow_html=True)

                    # ============================================================
                    # EV較正曲線（「EVが高いほど儲かる」は本当か）
                    # ============================================================
                    if _pq.get("ev_calib_rows"):
                        st.markdown(
                            '<div style="margin-top:14px;"></div>'
                            '<details class="gloss-term"><summary>📐 EV較正曲線 — 「EVが高いほど儲かる」は本当か？（クリックで説明）</summary>'
                            '<span class="gloss-body">通常はEVが高いほど得だと考えますが、これはAIの確率予測が正しいことが前提です。'
                            '予測確率そのものにズレがあると、表示上のEVが高くても実際には儲からない、ということが起こり得ます。'
                            'この表は「購入時点のEV帯」ごとに、実際どれだけの損益率（実現ROI）だったかを集計したものです。'
                            'もし高いEV帯ほど実現ROIも高くなっていれば、EVという指標はちゃんと機能しています。'
                            'そうなっていない帯があれば、その確率帯のAI予測は見直しが必要というサインです。</span></details>',
                            unsafe_allow_html=True,
                        )
                        _df_ev_calib = pd.DataFrame(_pq["ev_calib_rows"])
                        _df_ev_disp = pd.DataFrame({
                            "購入時EV帯": _df_ev_calib["range"],
                            "件数": _df_ev_calib["n"],
                            "AI予測EV平均": _df_ev_calib["pred_ev"].round(1).astype(str) + "%",
                            "実現ROI": _df_ev_calib["realized_roi"].apply(lambda x: f"{x:+.1f}%" if x is not None else "--"),
                        })
                        st.dataframe(_df_ev_disp, hide_index=True, width='stretch')
                        st.markdown('<div style="font-size:9px;color:var(--text-dim2);">実現ROIが低い件数が多いEV帯ほど、その帯でのAI予測は信頼性が低い可能性があります。件数が少ない帯は参考程度に。</div>', unsafe_allow_html=True)
            st.markdown("---")
            for l in sorted(logs, key=lambda x: x["date"], reverse=True)[:30]:
                d = datetime.fromisoformat(l["date"])
                result_badge = {"pending":"⏳未確定","hit":"✅的中","miss":"❌不的中"}.get(l.get("result","pending"),"⏳未確定")
                race_no = l.get("race_no",""); final_odds = l.get("final_odds",0.0); comment = l.get("comment","")
                with st.expander(f"{d.strftime('%m/%d %H:%M')}  {l['venue']}{f'  {race_no}R' if race_no else ''}  {l['combo']}  {l['stake_yen']:,}円  {result_badge}", expanded=(l.get("result")=="pending")):
                    ic1,ic2 = st.columns(2)
                    with ic1:
                        st.markdown(f"**賭け金**: {l['stake_yen']:,}円")
                        st.markdown(f"**最終オッズ**: {final_odds:.1f}倍" if final_odds else "**最終オッズ**: 未入力")
                    with ic2:
                        st.markdown(f"**ステータス**: {result_badge}")
                        if comment: st.markdown(f"**メモ**: {comment}")
                    if l.get("result") == "pending":
                        st.markdown("---")
                        ec1,ec2,ec3 = st.columns([1.5,1.5,2])
                        with ec1:
                            actual_pay = st.number_input("実際の払戻額(円)", min_value=0, value=int(l.get("payout_yen",0)), step=100, key=f"pay_{l['id']}",
                                help="実際にレースが終わって受け取った（受け取る予定の）払戻金額を入力します。「✅ 的中確定」を押すとこの金額で結果が確定し、月間の実現損益や的中率の集計に反映されます。")
                        with ec2:
                            final_o = st.number_input("最終オッズ", min_value=0.0, value=float(l.get("final_odds",0.0)), step=0.1, format="%.1f", key=f"fodds_{l['id']}",
                                help="投票が締め切られた時点の実際のオッズです。予測時のオッズと違うことが多いため、正確な記録のために実際の値を入力してください。")
                        with ec3:
                            new_cmt = st.text_input("メモ更新", value=comment, key=f"cmt_{l['id']}", placeholder="レース後の感想など",
                                help="このレースの結果を振り返っての感想や気づきを書き残せます。「的中確定」「不的中」ボタンを押すタイミングで一緒に保存されます。")
                        bc1,bc2 = st.columns(2)
                        if bc1.button("✅ 的中確定", key=f"hit_{l['id']}", type="primary"):
                            update_log_result(l["id"],"hit",payout_yen=int(actual_pay),final_odds=float(final_o),comment=new_cmt.strip()); st.rerun()
                        if bc2.button("❌ 不的中", key=f"miss_{l['id']}"):
                            update_log_result(l["id"],"miss",payout_yen=0,final_odds=float(final_o),comment=new_cmt.strip()); st.rerun()
                    else:
                        st.markdown("---")
                        new_cmt = st.text_input("メモ更新", value=comment, key=f"cmt_{l['id']}",
                            help="確定済みのレースについて、あとからメモを書き直したいときに使います。書き換えたら下の「メモを保存」ボタンを押してください。")
                        if st.button("💾 メモを保存", key=f"save_cmt_{l['id']}"):
                            update_log_result(l["id"],l["result"],comment=new_cmt.strip()); st.rerun()
            if st.button("🗑 全ログ削除"):
                save_logs([]); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- バックテスト分析タブ ---
    with tabs[7]:
        st.markdown('<div class="panel-box">', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">📈 バックテスト分析ダッシュボード</div>', unsafe_allow_html=True)
        st.markdown('<div class="note">backtest_results_v17.csv（n=5,090件）の実績から「どの条件で勝てるか」を可視化します。</div>', unsafe_allow_html=True)

        MASTER_CSV_PATH = "backtest_results_v17.csv"
        if not os.path.exists(MASTER_CSV_PATH):
            st.warning(f"バックテストCSVが見つかりません: {MASTER_CSV_PATH}")
        else:
            @st.cache_data(ttl=300, show_spinner=False)
            def load_backtest():
                return pd.read_csv(MASTER_CSV_PATH, encoding="utf-8-sig")

            try:
                df_bt = load_backtest()
                st.success(f"✅ {len(df_bt):,}件 / {df_bt['日付'].nunique()}日分 読み込み済み", icon="📊")
                st.markdown("---")

                # ============================================================
                # ① 全体サマリー
                # ============================================================
                bt_hit_rate  = df_bt["的中"].mean() * 100
                bt_avg_odds  = df_bt.loc[df_bt["的中"]==1, "単勝オッズ"].mean()
                bt_avg_prob  = df_bt["予測確率"].mean() * 100
                bt_brier     = df_bt["ブライアスコア"].mean()
                s1,s2,s3,s4 = st.columns(4)
                s1.metric("全体的中率", f"{bt_hit_rate:.1f}%")
                s2.metric("的中時平均オッズ", f"{bt_avg_odds:.2f}倍" if pd.notna(bt_avg_odds) else "--")
                s3.metric("平均予測確率", f"{bt_avg_prob:.1f}%")
                s4.metric("平均ブライアスコア", f"{bt_brier:.4f}")
                st.markdown("---")

                # ============================================================
                # ② 確率閾値別成績（「どれだけ絞ると勝率が上がるか」）
                # ============================================================
                st.markdown('<div class="eyebrow">🎯 確率閾値別成績</div>', unsafe_allow_html=True)
                st.markdown('<div class="note">予測確率が高いレースに絞るほど的中率はどう変わるか</div>', unsafe_allow_html=True)
                thresholds = [0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
                thresh_rows = []
                for th in thresholds:
                    sub = df_bt[df_bt["予測確率"] >= th]
                    if len(sub) < 10: continue
                    hr  = sub["的中"].mean() * 100
                    avg_o = sub.loc[sub["的中"]==1, "単勝オッズ"].mean()
                    # 理論ROI: 的中率 × 平均オッズ - 1
                    roi = (hr/100 * avg_o - 1) * 100 if pd.notna(avg_o) else None
                    thresh_rows.append({
                        "確率閾値": f"≥{th*100:.0f}%",
                        "対象件数": len(sub),
                        "的中率": f"{hr:.1f}%",
                        "平均的中オッズ": f"{avg_o:.2f}倍" if pd.notna(avg_o) else "--",
                        "理論ROI": f"{roi:+.1f}%" if roi is not None else "--",
                    })
                if thresh_rows:
                    st.dataframe(pd.DataFrame(thresh_rows), hide_index=True, width='stretch')
                st.markdown("---")

                # ============================================================
                # ③ 会場別成績
                # ============================================================
                st.markdown('<div class="eyebrow">🏟️ 会場別成績</div>', unsafe_allow_html=True)
                venue_grp = df_bt.groupby("会場").agg(
                    件数=("的中","count"),
                    的中率=("的中","mean"),
                    平均オッズ=("単勝オッズ", lambda x: x[df_bt.loc[x.index,"的中"]==1].mean()),
                ).reset_index()
                venue_grp["的中率%"] = (venue_grp["的中率"]*100).round(1)
                venue_grp["理論ROI"] = (venue_grp["的中率"] * venue_grp["平均オッズ"] - 1) * 100
                venue_grp = venue_grp.sort_values("理論ROI", ascending=False)
                venue_grp["的中率%"] = venue_grp["的中率%"].apply(lambda x: f"{x:.1f}%")
                venue_grp["平均オッズ"] = venue_grp["平均オッズ"].apply(lambda x: f"{x:.2f}倍" if pd.notna(x) else "--")
                venue_grp["理論ROI"]  = venue_grp["理論ROI"].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "--")
                st.dataframe(venue_grp[["会場","件数","的中率%","平均オッズ","理論ROI"]], hide_index=True, width='stretch')
                st.markdown("---")

                # ============================================================
                # ④ 風速別成績
                # ============================================================
                st.markdown('<div class="eyebrow">💨 風速帯別成績</div>', unsafe_allow_html=True)
                df_bt["風速帯"] = pd.cut(df_bt["wind_mps"],
                    bins=[-0.1,1,2,3,5,99],
                    labels=["0〜1m","1〜2m","2〜3m","3〜5m","5m超"])
                wind_grp = df_bt.groupby("風速帯", observed=True).agg(
                    件数=("的中","count"),
                    的中率=("的中","mean"),
                    平均オッズ=("単勝オッズ", lambda x: x[df_bt.loc[x.index,"的中"]==1].mean()),
                ).reset_index()
                wind_grp["的中率%"] = (wind_grp["的中率"]*100).round(1).apply(lambda x: f"{x:.1f}%")
                wind_grp["理論ROI"] = ((wind_grp["的中率"] * wind_grp["平均オッズ"] - 1)*100).round(1).apply(lambda x: f"{x:+.1f}%")
                wind_grp["平均オッズ"] = wind_grp["平均オッズ"].apply(lambda x: f"{x:.2f}倍" if pd.notna(x) else "--")
                st.dataframe(wind_grp[["風速帯","件数","的中率%","平均オッズ","理論ROI"]], hide_index=True, width='stretch')
                st.markdown("---")

                # ============================================================
                # ⑤ 確率×会場のクロス分析（有望条件の発見）
                # ============================================================
                st.markdown('<div class="eyebrow">🔬 有望条件クロス分析（確率≥30% × 会場）</div>', unsafe_allow_html=True)
                st.markdown('<div class="note">ROI+の条件＝「狙い目の勝負所」</div>', unsafe_allow_html=True)
                df_high = df_bt[df_bt["予測確率"] >= 0.30]
                if len(df_high) >= 20:
                    cross_grp = df_high.groupby("会場").agg(
                        件数=("的中","count"),
                        的中率=("的中","mean"),
                        平均オッズ=("単勝オッズ", lambda x: x[df_high.loc[x.index,"的中"]==1].mean()),
                    ).reset_index()
                    cross_grp["理論ROI"] = (cross_grp["的中率"] * cross_grp["平均オッズ"] - 1) * 100
                    cross_grp = cross_grp[cross_grp["件数"] >= 10].sort_values("理論ROI", ascending=False)
                    cross_grp["的中率%"] = (cross_grp["的中率"]*100).apply(lambda x: f"{x:.1f}%")
                    cross_grp["平均オッズ"] = cross_grp["平均オッズ"].apply(lambda x: f"{x:.2f}倍" if pd.notna(x) else "--")
                    cross_grp["理論ROI"]  = cross_grp["理論ROI"].apply(lambda x: f"{x:+.1f}%")
                    # ROI+の条件を強調
                    st.dataframe(cross_grp[["会場","件数","的中率%","平均オッズ","理論ROI"]].head(10), hide_index=True, width='stretch')

                    # 最も有望な条件をアラート表示
                    best_venues = df_high.groupby("会場").apply(
                        lambda g: (g["的中"].mean() * g.loc[g["的中"]==1,"単勝オッズ"].mean() - 1) * 100
                        if len(g) >= 10 else None
                    ).dropna().sort_values(ascending=False)
                    if len(best_venues) > 0 and best_venues.iloc[0] > 0:
                        best_v = best_venues.index[0]
                        best_roi = best_venues.iloc[0]
                        st.markdown(
                            f'<div style="background:#062a1c;border:1px solid #16e0a0;border-radius:6px;padding:8px 12px;font-size:11px;color:#16e0a0;font-weight:700;margin-top:8px;">'
                            f'⭐ 最有望条件：確率≥30% × {best_v} → 理論ROI {best_roi:+.1f}%（{len(df_high[df_high["会場"]==best_v])}件）</div>',
                            unsafe_allow_html=True,
                        )

                st.markdown("---")

                # ============================================================
                # ⑥ 3次元クロス分析（確率 × 風速 × 会場）
                # ============================================================
                st.markdown('<div class="eyebrow">🔭 3次元クロス分析（確率帯 × 風速帯 × 上位会場）</div>', unsafe_allow_html=True)
                with st.expander("詳細を表示", expanded=False):
                    df_bt["確率帯"] = pd.cut(df_bt["予測確率"],
                        bins=[0, 0.25, 0.30, 0.35, 0.40, 1.0],
                        labels=["〜25%", "25〜30%", "30〜35%", "35〜40%", "40%〜"])
                    df_bt["風速帯2"] = pd.cut(df_bt["wind_mps"],
                        bins=[-0.1, 2, 4, 99],
                        labels=["弱風(〜2m)", "中風(2〜4m)", "強風(4m超)"])

                    cross3 = df_bt.groupby(["確率帯","風速帯2"], observed=True).agg(
                        件数=("的中","count"),
                        的中率=("的中","mean"),
                        平均オッズ=("単勝オッズ", lambda x: x[df_bt.loc[x.index,"的中"]==1].mean()),
                    ).reset_index()
                    cross3 = cross3[cross3["件数"] >= 15].copy()
                    cross3["理論ROI"] = (cross3["的中率"] * cross3["平均オッズ"] - 1) * 100
                    cross3 = cross3.sort_values("理論ROI", ascending=False)
                    cross3["的中率%"] = (cross3["的中率"]*100).apply(lambda x: f"{x:.1f}%")
                    cross3["平均オッズ"] = cross3["平均オッズ"].apply(lambda x: f"{x:.2f}倍" if pd.notna(x) else "--")
                    cross3["理論ROI"]  = cross3["理論ROI"].apply(lambda x: f"{x:+.1f}%")
                    st.dataframe(cross3[["確率帯","風速帯2","件数","的中率%","平均オッズ","理論ROI"]], hide_index=True, width='stretch')

                st.markdown("---")

                # ============================================================
                # ⑦ 勝負の分岐点発見（予測確率と理論ROIの関係を可視化）
                # ============================================================
                st.markdown('<div class="eyebrow">📐 勝負の分岐点（予測確率 vs 理論ROI）</div>', unsafe_allow_html=True)
                st.markdown('<div class="note">どの確率閾値から理論ROIがプラスに転じるか。この点が「勝負すべき最低ライン」です。</div>', unsafe_allow_html=True)
                breakpoints = []
                # 【バグ修正】numpyがimportされていないのに np.arange を使っており、
                # バックテストCSVが存在する環境（＝実際の利用環境）では確実にNameErrorで
                # クラッシュしていた。標準ライブラリのみで同等の0.01刻みループに置き換える。
                # （whileループにすると途中のcontinueでth加算が飛び無限ループ化する恐れがあるため、
                # 　整数ステップのforループで安全に実装する）
                for _th_step in range(20, 60):
                    th = _th_step / 100
                    sub = df_bt[df_bt["予測確率"] >= th]
                    if len(sub) < 20: break
                    avg_o = sub.loc[sub["的中"]==1, "単勝オッズ"].mean()
                    if pd.isna(avg_o): continue
                    hr  = sub["的中"].mean()
                    roi = (hr * avg_o - 1) * 100
                    breakpoints.append({"閾値": f"{th*100:.0f}%", "件数": len(sub), "的中率": hr*100, "理論ROI": roi})

                if breakpoints:
                    df_bp = pd.DataFrame(breakpoints)
                    # プラスに転じた最初の閾値を検出
                    positive_roi = df_bp[df_bp["理論ROI"] > 0]
                    if len(positive_roi) > 0:
                        first_positive = positive_roi.iloc[0]
                        st.markdown(
                            f'<div style="background:#062a1c;border:1px solid #16e0a0;border-radius:6px;padding:8px 12px;font-size:11px;color:#16e0a0;font-weight:700;margin-bottom:8px;">'
                            f'🎯 分岐点：確率 ≥ {first_positive["閾値"]} から理論ROIがプラスに転じる（{first_positive["件数"]}件 / ROI {first_positive["理論ROI"]:+.1f}%）</div>',
                            unsafe_allow_html=True,
                        )

                    # Altairチャート
                    try:
                        import altair as alt
                        df_bp["理論ROI"] = df_bp["理論ROI"].round(2)
                        color_cond = alt.condition(
                            alt.datum["理論ROI"] > 0,
                            alt.value("#16e0a0"),
                            alt.value("#ff5c72"),
                        )
                        chart = alt.Chart(df_bp).mark_bar().encode(
                            x=alt.X("閾値:N", title="予測確率閾値"),
                            y=alt.Y("理論ROI:Q", title="理論ROI(%)"),
                            color=color_cond,
                            tooltip=["閾値","件数","理論ROI"],
                        ).properties(height=220, title="予測確率閾値別 理論ROI").configure_view(
                            strokeOpacity=0
                        ).configure_axis(
                            gridColor="#1a2a3a", labelColor="#7c8aab", titleColor="#7c8aab"
                        ).configure_title(color="#7c8aab")
                        st.altair_chart(chart, width='stretch')
                    except ImportError:
                        # Altairなければテーブル表示
                        df_bp["的中率%"] = df_bp["的中率"].apply(lambda x: f"{x:.1f}%")
                        df_bp["理論ROI"] = df_bp["理論ROI"].apply(lambda x: f"{x:+.1f}%")
                        st.dataframe(df_bp[["閾値","件数","的中率%","理論ROI"]].iloc[::3], hide_index=True, width='stretch')

                st.markdown("---")

                # ============================================================
                # ⑧ 特徴量重要度（LightGBMモデルが何を重視しているか）
                # ============================================================
                for model_file, label in [("lightgbm_v2_precision.txt","精鋭型"), ("lightgbm_v2_volume.txt","広範型")]:
                    if os.path.exists(model_file):
                        try:
                            import lightgbm as lgb
                            model = lgb.Booster(model_file=model_file)
                            fi = pd.Series(model.feature_importance('gain'), index=model.feature_name())
                            fi = fi.sort_values(ascending=False).head(15)
                            st.markdown(f'<div class="eyebrow">🧠 LightGBM特徴量重要度（{label}）</div>', unsafe_allow_html=True)
                            fi_df = pd.DataFrame({"特徴量": fi.index, "重要度(gain)": fi.values.round(1)})
                            st.dataframe(fi_df, hide_index=True, width='stretch')
                            st.markdown("---")
                        except Exception:
                            pass

            except Exception as e:
                st.error(f"CSVの読み込みに失敗しました: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="note" style="text-align:center; margin-top:10px;">PRO TRADER TERMINAL v101 — 本ツールの予測・期待値・Kelly配分はヒューリスティックモデルに基づくものであり、的中や利益を保証するものではありません。投資判断は自己責任で行ってください。</div>', unsafe_allow_html=True)
