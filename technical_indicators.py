# technical_indicators.py
import pandas as pd
import numpy as np

def calculate_ichimoku_components(df, tenkan_period=9, kijun_period=26, senkou_b_period=52):
    """一目均衡表の主要構成要素を計算する（先行スパンのシフトも考慮）"""
    df_ichimoku = df.copy()

    # 転換線
    high_tenkan = df_ichimoku['High'].rolling(window=tenkan_period, min_periods=1).max()
    low_tenkan = df_ichimoku['Low'].rolling(window=tenkan_period, min_periods=1).min()
    df_ichimoku['tenkan_sen'] = (high_tenkan + low_tenkan) / 2

    # 基準線
    high_kijun = df_ichimoku['High'].rolling(window=kijun_period, min_periods=1).max()
    low_kijun = df_ichimoku['Low'].rolling(window=kijun_period, min_periods=1).min()
    df_ichimoku['kijun_sen'] = (high_kijun + low_kijun) / 2

    # 先行スパンA (生の値を計算)
    df_ichimoku['senkou_span_a_raw'] = (df_ichimoku['tenkan_sen'] + df_ichimoku['kijun_sen']) / 2
    # 先行スパンA (基準線期間分未来へシフト)
    df_ichimoku['senkou_span_a'] = df_ichimoku['senkou_span_a_raw'].shift(kijun_period)

    # 先行スパンB (生の値を計算)
    high_senkou_b = df_ichimoku['High'].rolling(window=senkou_b_period, min_periods=1).max()
    low_senkou_b = df_ichimoku['Low'].rolling(window=senkou_b_period, min_periods=1).min()
    df_ichimoku['senkou_span_b_raw'] = (high_senkou_b + low_senkou_b) / 2
    # 先行スパンB (基準線期間分未来へシフト)
    df_ichimoku['senkou_span_b'] = df_ichimoku['senkou_span_b_raw'].shift(kijun_period)

    # 遅行スパン (基準線期間分過去へシフト)
    df_ichimoku['chikou_span'] = df_ichimoku['Close'].shift(-kijun_period)
    
    # strategy.py でチャート描画やシグナル生成に使うため、シフト前のスパンも返すようにする
    # (カラム名が重複しないように注意)
    # senkou_span_a_raw と senkou_span_b_raw は既に df_ichimoku に含まれている
    
    return df_ichimoku[['tenkan_sen', 'kijun_sen', 
                        'senkou_span_a', 'senkou_span_b', 
                        'chikou_span', 
                        'senkou_span_a_raw', 'senkou_span_b_raw']] # 必要なカラムのみ返す

# 他のテクニカル指標計算関数もここに追加可能