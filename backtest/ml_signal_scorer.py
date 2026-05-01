"""
ML信号打分器 — XGBoost三分类

从缠论买卖点中提取80维特征，训练三分类模型:
  大亏(<-3%) / 平(-3%~3%) / 大赢(>3%)

特征来源:
  1. 买点类型编码 (one-hot)
  2. 置信度
  3. 中枢结构 (宽度/位置/演化状态)
  4. MACD动力学 (背驰/面积/零轴)
  5. 均线特征 (MA5~MA250距离+斜率)
  6. 波动率 (ATR/RSI/历史波动率)
  7. 成交量 (量比/缩量程度)
  8. 趋势方向 (周线/走势类型)
  9. 市场环境 (大盘MA250状态)

用法:
  python -m backtest.ml_signal_scorer train   # 训练
  python -m backtest.ml_signal_scorer eval    # 评估
"""

import os
import sys
import json
import struct
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.hybrid_source import HybridSource
from indicator.market_environment import MarketEnvironment


MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_ml_models')
FEATURE_COLS_FILE = os.path.join(MODEL_DIR, 'feature_columns.json')

HOLD_DAYS = 5
BUY_TYPES = ['1buy', '2buy', '3buy', 'quasi2buy', 'sub1buy', '2b3bbuy']


def extract_features(
    signal: dict,
    daily_df: pd.DataFrame,
    weekly_dir: str = 'neutral',
    market_env: Optional[MarketEnvironment] = None,
) -> dict:
    """从信号和价格数据中提取ML特征"""
    features = {}

    closes = daily_df['close'].values
    volumes = daily_df['volume'].values
    highs = daily_df['high'].values
    lows = daily_df['low'].values
    n = len(closes)
    price = closes[-1]

    # 1. 买点类型 one-hot
    sig_type = signal.get('signal_type', 'unknown')
    for bt in BUY_TYPES:
        features[f'type_{bt}'] = int(sig_type == bt)

    # 2. 置信度
    conf = signal.get('confidence', 0.5)
    features['confidence'] = conf
    features['confidence_sq'] = conf ** 2

    # 3. 中枢结构
    pivot_info = signal.get('pivot_info', {})
    if isinstance(pivot_info, dict):
        zg = pivot_info.get('zg', 0)
        zd = pivot_info.get('zd', 0)
        width = zg - zd if zg > zd else 0.01
        features['pivot_width_pct'] = width / price * 100 if price > 0 else 0
        features['price_above_zg_pct'] = (price - zg) / zg * 100 if zg > 0 else 0
        features['price_above_zd_pct'] = (price - zd) / zd * 100 if zd > 0 else 0
    else:
        features['pivot_width_pct'] = 0
        features['price_above_zg_pct'] = 0
        features['price_above_zd_pct'] = 0
    features['stop_loss_dist_pct'] = 0
    sp = signal.get('stop_price', 0)
    if sp > 0 and price > 0:
        features['stop_loss_dist_pct'] = (price - sp) / price * 100

    # 4. MACD
    if n >= 35:
        ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        hist = (dif - dea) * 2
        features['macd_hist_last'] = float(hist.iloc[-1])
        features['macd_hist_prev'] = float(hist.iloc[-2]) if len(hist) >= 2 else 0
        features['macd_below_zero'] = int(hist.iloc[-1] < 0)
        features['macd_area_10'] = float(sum(abs(h) for h in hist.iloc[-10:]))
    else:
        features['macd_hist_last'] = 0
        features['macd_hist_prev'] = 0
        features['macd_below_zero'] = 0
        features['macd_area_10'] = 0

    # 5. 均线
    for period in [5, 10, 20, 60]:
        if n >= period:
            ma = np.mean(closes[-period:])
            features[f'ma{period}_dist_pct'] = (price - ma) / ma * 100 if ma > 0 else 0
            if n >= period + 5:
                ma_prev = np.mean(closes[-period - 5:-5])
                features[f'ma{period}_slope'] = (ma - ma_prev) / ma_prev * 100 if ma_prev > 0 else 0
            else:
                features[f'ma{period}_slope'] = 0
        else:
            features[f'ma{period}_dist_pct'] = 0
            features[f'ma{period}_slope'] = 0

    # 均线排列
    if n >= 60:
        ma5 = np.mean(closes[-5:])
        ma10 = np.mean(closes[-10:])
        ma20 = np.mean(closes[-20:])
        ma60 = np.mean(closes[-60:])
        features['ma_bullish_align'] = int(ma5 > ma10 > ma20 > ma60)
        features['ma_bearish_align'] = int(ma5 < ma10 < ma20 < ma60)
    else:
        features['ma_bullish_align'] = 0
        features['ma_bearish_align'] = 0

    # 6. 波动率
    if n >= 15:
        trs = []
        for i in range(1, min(15, n)):
            tr = max(highs[-i] - lows[-i],
                     abs(highs[-i] - closes[-i - 1]),
                     abs(lows[-i] - closes[-i - 1]))
            trs.append(tr)
        features['atr_14_pct'] = np.mean(trs) / price * 100 if price > 0 else 0
    else:
        features['atr_14_pct'] = 0

    if n >= 21:
        rets = np.diff(closes[-22:]) / closes[-22:-1]
        features['volatility_20'] = float(np.std(rets) * np.sqrt(252))
    else:
        features['volatility_20'] = 0

    # RSI
    if n >= 15:
        delta = np.diff(closes[-15:])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        avg_g = np.mean(gains)
        avg_l = np.mean(losses)
        features['rsi_14'] = 100 - 100 / (1 + avg_g / avg_l) if avg_l > 0 else 100
    else:
        features['rsi_14'] = 50

    # 7. 成交量
    if n >= 20:
        vol_5 = np.mean(volumes[-5:])
        vol_20 = np.mean(volumes[-20:])
        features['vol_ratio_5'] = float(volumes[-1] / vol_5) if vol_5 > 0 else 1
        features['vol_ratio_20'] = float(volumes[-1] / vol_20) if vol_20 > 0 else 1
        features['vol_shrink'] = float(vol_5 / vol_20) if vol_20 > 0 else 1
    else:
        features['vol_ratio_5'] = 1
        features['vol_ratio_20'] = 1
        features['vol_shrink'] = 1

    # 8. 周线方向
    features['weekly_up'] = int(weekly_dir == 'bull')
    features['weekly_down'] = int(weekly_dir == 'bear')

    # 9. 市场环境
    if market_env:
        features['env_bull'] = int(market_env.get_state() == 'BULL')
        features['env_bear'] = int(market_env.get_state() == 'BEAR')
    else:
        features['env_bull'] = 0
        features['env_bear'] = 0

    # 10. 交互特征
    features['conf_x_weekly_up'] = conf * features['weekly_up']
    features['conf_x_env_bull'] = conf * features.get('env_bull', 0)
    features['weekly_up_x_rsi'] = features['weekly_up'] * features['rsi_14']

    # 11. 近期涨幅
    for lookback in [3, 5, 10, 20]:
        if n > lookback:
            features[f'ret_{lookback}d'] = float((closes[-1] - closes[-1 - lookback]) / closes[-1 - lookback] * 100)
        else:
            features[f'ret_{lookback}d'] = 0

    # 12. 价格位置
    if n >= 20:
        high20 = max(highs[-20:])
        low20 = min(lows[-20:])
        features['dist_from_high20_pct'] = float((high20 - price) / price * 100)
        features['dist_from_low20_pct'] = float((price - low20) / price * 100)
    else:
        features['dist_from_high20_pct'] = 0
        features['dist_from_low20_pct'] = 0

    return features


def build_dataset(signals_data: List[dict], daily_map: dict,
                  market_env: Optional[MarketEnvironment] = None) -> pd.DataFrame:
    """从信号列表构建训练数据集"""
    rows = []
    for sig in signals_data:
        code = sig.get('code', '')
        sig_idx = sig.get('sig_idx', -1)
        if not code or sig_idx < 0 or code not in daily_map:
            continue
        df = daily_map[code]
        if sig_idx + HOLD_DAYS >= len(df):
            continue

        entry_price = sig.get('entry_price', df['close'].iloc[sig_idx])
        future_close = df['close'].iloc[sig_idx + HOLD_DAYS]
        ret_5d = (future_close - entry_price) / entry_price * 100

        # 最大回撤
        max_dd = 0
        for j in range(1, HOLD_DAYS + 1):
            if sig_idx + j < len(df):
                dd = (entry_price - df['low'].iloc[sig_idx + j]) / entry_price * 100
                max_dd = max(max_dd, dd)

        window = df.iloc[max(0, sig_idx - 100):sig_idx + 1]
        if len(window) < 30:
            continue

        weekly_dir = sig.get('weekly_trend', 'range')
        features = extract_features(sig, window, weekly_dir, market_env)
        features['code'] = code
        features['date'] = str(df.index[sig_idx]) if sig_idx < len(df) else ''
        features['ret_5d'] = round(ret_5d, 2)
        features['max_dd_5d'] = round(max_dd, 2)
        features['label_profit'] = int(ret_5d > 0)
        features['label_triple'] = 0 if ret_5d < -3 else (2 if ret_5d > 3 else 1)
        rows.append(features)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def train_model(df: pd.DataFrame):
    """训练XGBoost三分类模型 (自适应阈值)"""
    import xgboost as xgb

    os.makedirs(MODEL_DIR, exist_ok=True)

    df = df.sort_values('date').reset_index(drop=True)
    meta_cols = ['code', 'date', 'ret_5d', 'max_dd_5d', 'label_profit', 'label_triple']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_ret = df['ret_5d'].values

    # 自适应标签: 用训练集的分位数定义三分类阈值
    # 弱信号(<P33) / 普通(P33~P67) / 强信号(>P67)
    q33 = float(np.percentile(y_ret, 33))
    q67 = float(np.percentile(y_ret, 67))
    y_triple = np.where(y_ret < q33, 0, np.where(y_ret > q67, 2, 1))

    # 时序分割 70/15/15
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
    y_train, y_val, y_test = y_triple[:train_end], y_triple[train_end:val_end], y_triple[val_end:]
    y_ret_test = y_ret[val_end:]

    # 用训练集分位数重新标记(避免未来信息泄露)
    q33_train = float(np.percentile(y_ret[:train_end], 33))
    q67_train = float(np.percentile(y_ret[:train_end], 67))
    y_train = np.where(y_ret[:train_end] < q33_train, 0, np.where(y_ret[:train_end] > q67_train, 2, 1))
    y_val = np.where(y_ret[train_end:val_end] < q33_train, 0, np.where(y_ret[train_end:val_end] > q67_train, 2, 1))
    y_test = np.where(y_ret[val_end:] < q33_train, 0, np.where(y_ret[val_end:] > q67_train, 2, 1))

    # 保存feature columns和阈值
    with open(FEATURE_COLS_FILE, 'w') as f:
        json.dump(feature_cols, f)
    with open(os.path.join(MODEL_DIR, 'thresholds.json'), 'w') as f:
        json.dump({'q33': round(q33_train, 2), 'q67': round(q67_train, 2)}, f)

    # 三分类模型
    clf = xgb.XGBClassifier(
        num_class=3, objective='multi:softprob',
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=20, gamma=0.2,
        reg_alpha=0.2, reg_lambda=1.5,
        random_state=42, eval_metric='mlogloss',
        early_stopping_rounds=30, verbosity=0,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    proba = clf.predict_proba(X_test)
    p_strong = proba[:, 2]
    p_weak = proba[:, 0]

    print(f"\n{'='*60}")
    print(f"ML信号打分器 — 训练完成")
    print(f"{'='*60}")
    print(f"特征: {len(feature_cols)} 维 | 样本: {len(df)}")
    print(f"阈值: 弱信号<{q33_train:.2f}% / 强信号>{q67_train:.2f}%")
    print(f"分割: train={train_end} val={val_end-train_end} test={n-val_end}")
    print(f"三分类: 弱={sum(y_triple==0)} 普通={sum(y_triple==1)} 强={sum(y_triple==2)}")

    print(f"\n--- 按强信号概率筛选 ---")
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        mask = p_strong >= t
        if mask.sum() < 10:
            continue
        wr = (y_ret_test[mask] > 0).mean() * 100
        avg = y_ret_test[mask].mean()
        strong_rate = (y_ret_test[mask] > q67_train).mean() * 100
        print(f"  P(强)≥{t:.2f}: {mask.sum():4d}个 | 胜率{wr:.1f}% | 均收{avg:+.2f}% | 强信号率{strong_rate:.1f}%")

    print(f"\n--- 按弱信号概率排除 ---")
    for t in [0.25, 0.30, 0.35, 0.40]:
        mask = p_weak < t
        if mask.sum() < 10:
            continue
        wr = (y_ret_test[mask] > 0).mean() * 100
        avg = y_ret_test[mask].mean()
        weak_rate = (y_ret_test[mask] < q33_train).mean() * 100
        print(f"  P(弱)<{t:.2f}: {mask.sum():4d}个 | 胜率{wr:.1f}% | 均收{avg:+.2f}% | 弱信号率{weak_rate:.1f}%")

    # 组合筛选: 高强信号 + 低弱信号
    print(f"\n--- 组合筛选: P(强)≥0.45 且 P(弱)<0.30 ---")
    mask_combo = (p_strong >= 0.45) & (p_weak < 0.30)
    if mask_combo.sum() >= 10:
        wr = (y_ret_test[mask_combo] > 0).mean() * 100
        avg = y_ret_test[mask_combo].mean()
        strong_rate = (y_ret_test[mask_combo] > q67_train).mean() * 100
        baseline_avg = y_ret_test.mean()
        print(f"  筛选后: {mask_combo.sum()}个 | 胜率{wr:.1f}% | 均收{avg:+.2f}% | 强信号率{strong_rate:.1f}%")
        print(f"  基线:   {len(y_ret_test)}个 | 均收{baseline_avg:+.2f}%")
        print(f"  提升:   均收{(avg-baseline_avg):+.2f}%")

    # 特征重要性
    importance = clf.feature_importances_
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: -x[1])
    print(f"\n--- Top 15 特征重要性 ---")
    for i, (name, imp) in enumerate(feat_imp[:15]):
        print(f"  {i+1:2d}. {name:<30s} {imp:.4f}")

    # 保存模型
    clf.save_model(os.path.join(MODEL_DIR, 'clf_triple.json'))

    imp_path = os.path.join(MODEL_DIR, 'feature_importance.json')
    with open(imp_path, 'w', encoding='utf-8') as f:
        json.dump([{'feature': fn, 'importance': float(imp)} for fn, imp in feat_imp], f, indent=2)

    print(f"\n模型已保存至 {MODEL_DIR}/")
    return clf, feature_cols


def load_model():
    """加载已训练的模型"""
    import xgboost as xgb
    clf = xgb.XGBClassifier()
    clf.load_model(os.path.join(MODEL_DIR, 'clf_triple.json'))
    with open(FEATURE_COLS_FILE, 'r') as f:
        feature_cols = json.load(f)
    return clf, feature_cols


def predict_signal(signal: dict, daily_df: pd.DataFrame,
                   weekly_dir: str = 'neutral',
                   market_env: Optional[MarketEnvironment] = None) -> dict:
    """预测单个信号的质量"""
    import xgboost as xgb

    if not os.path.exists(os.path.join(MODEL_DIR, 'clf_triple.json')):
        return {'p_bigwin': 0.33, 'p_bigloss': 0.33, 'p_flat': 0.33, 'advice': '模型未训练'}

    clf, feature_cols = load_model()
    features = extract_features(signal, daily_df, weekly_dir, market_env)

    X = pd.DataFrame([{k: features.get(k, 0) for k in feature_cols}])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    proba = clf.predict_proba(X)[0]
    return {
        'p_bigloss': round(float(proba[0]), 3),
        'p_flat': round(float(proba[1]), 3),
        'p_bigwin': round(float(proba[2]), 3),
        'advice': '大赢概率高' if proba[2] > 0.4 else ('大亏风险' if proba[0] > 0.4 else '中性'),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML信号打分器')
    parser.add_argument('action', choices=['train', 'eval', 'predict'], help='操作')
    args = parser.parse_args()

    if args.action == 'train':
        print("请通过 backtest_e2e_2buy_1sell.py 的 --ml-train 参数训练模型")
    elif args.action == 'eval':
        print("评估功能开发中...")
    elif args.action == 'predict':
        print("请使用 predict_signal() 函数预测单只股票")
