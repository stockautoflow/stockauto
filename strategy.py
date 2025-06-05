# strategy.py (修正版 - 一目均衡表の個別条件設定に対応)

import pandas as pd
import numpy as np
import talib # TA-Lib がインストールされている必要があります
import warnings
import os
import datetime # datetime をインポート
try:
    import yaml # PyYAMLライブラリ
except ImportError:
    # logging は base.py で設定されるため、ここでは print を使用
    print("[Strategy Critical Error] PyYAML library is not installed. Please install it by running: pip install pyyaml")
    YAML_AVAILABLE = False
else:
    YAML_AVAILABLE = True

# technical_indicators.py から一目均衡表の計算関数をインポート
from technical_indicators import calculate_ichimoku_components

def flatten_dict(d, parent_key='', sep='_'):
    """ネストした辞書をフラットな辞書に変換する。キーは親キーと結合され、大文字になる。"""
    items = []
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key.upper(), v))
    return dict(items)

def load_strategy_config_yaml(filename="config.yaml"):
    """
    YAML設定ファイルから戦略パラメータを読み込む。
    ファイルが存在しない、空、または読み込みエラー時は警告を出し、空のフラット辞書を返す。
    """
    if not YAML_AVAILABLE:
        print(f"[Strategy Warning] PyYAML is not available. Cannot load '{filename}'. Returning empty parameters.")
        return {}

    config_params_nested = {}
    if not os.path.exists(filename):
        print(f"[Strategy Warning] Config file '{filename}' not found. All strategy parameters must be defined in this file.")
    else:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                loaded_yaml = yaml.safe_load(f)
                if loaded_yaml and isinstance(loaded_yaml, dict):
                    config_params_nested = loaded_yaml
                elif not loaded_yaml:
                    print(f"[Strategy Warning] Config file '{filename}' is empty.")
                else:
                    print(f"[Strategy Warning] Config file '{filename}' does not contain a valid YAML dictionary.")
        except yaml.YAMLError as ye:
            print(f"[Strategy Error] Error parsing YAML file '{filename}': {ye}. Returning empty parameters.")
        except Exception as e:
            print(f"[Strategy Error] Failed to load config file '{filename}': {e}. Returning empty parameters.")

    return flatten_dict(config_params_nested)

def get_merged_param(key, params_from_backtester, strategy_primary_params_loaded,
                     is_period=False, is_float=False, is_activation_flag=False, is_bool=False):
    key_upper = key.upper()
    val_from_config = strategy_primary_params_loaded.get(key_upper)
    val_from_fw = params_from_backtester.get(key)
    if val_from_fw is None:
        val_from_fw = params_from_backtester.get(key_upper)

    val = val_from_config
    if val is None:
        val = val_from_fw

    if val is None:
        print(f"[Strategy Param Warning] Parameter '{key_upper}' not found. Using fallback.")
        if is_activation_flag: return 0
        if is_bool: return False
        if is_period: return 20
        if is_float: return 0.0
        return None

    try:
        if is_activation_flag: return int(float(val))
        if is_bool:
            if isinstance(val, bool): return val
            if isinstance(val, str): return val.lower() in ['true', '1', 'yes', 'on']
            return bool(int(float(val)))
        if is_period: return int(float(val))
        if is_float: return float(val)
        return val
    except (ValueError, TypeError) as e:
        print(f"[Strategy Param Warning] Conversion error for '{key_upper}', value '{val}': {e}. Using fallback.")
        if is_activation_flag: return 0
        if is_bool: return False
        if is_period: return 20
        if is_float: return 0.0
        return None

def calculate_vwap(df, period='D'):
    df_copy = df.copy()
    if isinstance(df_copy.index, pd.DatetimeIndex):
        if df_copy.index.tzinfo is None: df_copy.index = df_copy.index.tz_localize('Asia/Tokyo', ambiguous='infer')
        elif str(df_copy.index.tzinfo) != 'Asia/Tokyo': df_copy.index = df_copy.index.tz_convert('Asia/Tokyo')
    if 'Volume' not in df_copy.columns or df_copy['Volume'].isnull().all():
        df_copy['VWAP_daily'] = np.nan; return df_copy
    df_copy['TypicalPrice'] = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3
    df_copy['TPxV'] = df_copy['TypicalPrice'] * df_copy['Volume']
    if not isinstance(df_copy.index, pd.DatetimeIndex) or df_copy.empty or df_copy['Volume'].sum() == 0:
        df_copy['VWAP_daily'] = np.nan
    else:
        try:
            daily_grouper = df_copy.index.normalize()
            df_copy['CumVol_daily'] = df_copy.groupby(daily_grouper)['Volume'].transform('cumsum')
            df_copy['CumTPxV_daily'] = df_copy.groupby(daily_grouper)['TPxV'].transform('cumsum')
            df_copy['VWAP_daily'] = np.where(df_copy['CumVol_daily'] != 0, df_copy['CumTPxV_daily'] / df_copy['CumVol_daily'], np.nan)
            df_copy['VWAP_daily'] = df_copy.groupby(daily_grouper)['VWAP_daily'].transform(lambda x: x.ffill())
        except Exception as e: df_copy['VWAP_daily'] = np.nan
    for col_to_drop in ['TypicalPrice', 'TPxV', 'CumVol_daily', 'CumTPxV_daily']:
        if col_to_drop in df_copy.columns: df_copy.drop(col_to_drop, axis=1, inplace=True)
    return df_copy

def calculate_indicators(df_exec, df_context, params_from_backtester, loaded_strategy_params):
    def _get_param(key_name, is_period=False, is_float=False, is_activation_flag=False):
        return get_merged_param(key_name, params_from_backtester, loaded_strategy_params,
                                is_period, is_float, is_activation_flag)
    df_c = df_context.copy(); df_e = df_exec.copy()

    # 環境認識足の指標計算
    ema_short_p_ctx_gc = _get_param('EMA_SETTINGS_CONTEXT_PERIOD_SHORT_GC', is_period=True)
    ema_long_p_ctx_gc = _get_param('EMA_SETTINGS_CONTEXT_PERIOD_LONG_GC', is_period=True)
    adx_ctx_p = _get_param('ADX_SETTINGS_CONTEXT_PERIOD', is_period=True)
    atr_ctx_p_chart = _get_param('ATR_SETTINGS_PERIOD_CONTEXT', is_period=True)
    bb_period_ctx = _get_param('BB_SETTINGS_PERIOD_CTX', is_period=True)
    bb_nbdev_ctx_for_orig = _get_param('BB_SETTINGS_NBDEV_CTX', is_float=True)
    if bb_nbdev_ctx_for_orig is None: bb_nbdev_ctx_for_orig = 2.0
    sma1_p_ctx = _get_param('SMA_SETTINGS_CONTEXT_PERIOD_1', is_period=True)
    sma2_p_ctx = _get_param('SMA_SETTINGS_CONTEXT_PERIOD_2', is_period=True)
    atr_ctx_p_for_sl_val = _get_param('ATR_SETTINGS_STOP_PERIOD_CONTEXT')
    atr_ctx_p_for_sl = None
    if atr_ctx_p_for_sl_val is not None:
        try: atr_ctx_p_for_sl = int(float(atr_ctx_p_for_sl_val))
        except ValueError: pass

    if not df_c.empty:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if ema_short_p_ctx_gc and len(df_c['Close']) >= ema_short_p_ctx_gc: df_c[f'EMA{ema_short_p_ctx_gc}_ctx'] = talib.EMA(df_c['Close'], timeperiod=ema_short_p_ctx_gc)
            else: df_c[f'EMA{ema_short_p_ctx_gc}_ctx'] = np.nan
            if ema_long_p_ctx_gc and len(df_c['Close']) >= ema_long_p_ctx_gc: df_c[f'EMA{ema_long_p_ctx_gc}_ctx'] = talib.EMA(df_c['Close'], timeperiod=ema_long_p_ctx_gc)
            else: df_c[f'EMA{ema_long_p_ctx_gc}_ctx'] = np.nan
            
            if adx_ctx_p and len(df_c['High']) >= adx_ctx_p * 2:
                df_c['ADX_ctx'] = talib.ADX(df_c['High'], df_c['Low'], df_c['Close'], timeperiod=adx_ctx_p)
                df_c['PLUS_DI_ctx'] = talib.PLUS_DI(df_c['High'], df_c['Low'], df_c['Close'], timeperiod=adx_ctx_p)
                df_c['MINUS_DI_ctx'] = talib.MINUS_DI(df_c['High'], df_c['Low'], df_c['Close'], timeperiod=adx_ctx_p)
            else:
                df_c['ADX_ctx'] = np.nan
                df_c['PLUS_DI_ctx'] = np.nan
                df_c['MINUS_DI_ctx'] = np.nan

            if sma1_p_ctx and len(df_c['Close']) >= sma1_p_ctx: df_c[f'SMA{sma1_p_ctx}_ctx'] = talib.SMA(df_c['Close'], timeperiod=sma1_p_ctx)
            else: df_c[f'SMA{sma1_p_ctx}_ctx'] = np.nan
            if sma2_p_ctx and len(df_c['Close']) >= sma2_p_ctx: df_c[f'SMA{sma2_p_ctx}_ctx'] = talib.SMA(df_c['Close'], timeperiod=sma2_p_ctx)
            else: df_c[f'SMA{sma2_p_ctx}_ctx'] = np.nan
            if atr_ctx_p_chart and len(df_c['High']) >= atr_ctx_p_chart : df_c[f'ATR_{atr_ctx_p_chart}_CTX_Chart'] = talib.ATR(df_c['High'], df_c['Low'], df_c['Close'], timeperiod=atr_ctx_p_chart)
            else: df_c[f'ATR_{atr_ctx_p_chart}_CTX_Chart'] = np.nan
            if atr_ctx_p_for_sl and atr_ctx_p_for_sl != atr_ctx_p_chart and len(df_c['High']) >= atr_ctx_p_for_sl: df_c[f'ATR_{atr_ctx_p_for_sl}_ctx'] = talib.ATR(df_c['High'], df_c['Low'], df_c['Close'], timeperiod=atr_ctx_p_for_sl)
            elif atr_ctx_p_for_sl and atr_ctx_p_for_sl == atr_ctx_p_chart : df_c[f'ATR_{atr_ctx_p_for_sl}_ctx'] = df_c.get(f'ATR_{atr_ctx_p_chart}_CTX_Chart')
            elif atr_ctx_p_for_sl: df_c[f'ATR_{atr_ctx_p_for_sl}_ctx'] = np.nan

            if bb_period_ctx and len(df_c['Close']) >= bb_period_ctx:
                df_c['BB_Middle_ctx'] = talib.SMA(df_c['Close'], timeperiod=bb_period_ctx)
                std_dev_ctx = talib.STDDEV(df_c['Close'], timeperiod=bb_period_ctx, nbdev=1)
                for i in range(1, 4):
                    df_c[f'BB_Upper_ctx_{i}dev'] = df_c['BB_Middle_ctx'] + std_dev_ctx * i
                    df_c[f'BB_Lower_ctx_{i}dev'] = df_c['BB_Middle_ctx'] - std_dev_ctx * i
                df_c['BB_Upper_ctx'] = df_c['BB_Middle_ctx'] + std_dev_ctx * bb_nbdev_ctx_for_orig
                df_c['BB_Lower_ctx'] = df_c['BB_Middle_ctx'] - std_dev_ctx * bb_nbdev_ctx_for_orig
            else:
                for col_bb in ['BB_Middle_ctx', 'BB_Upper_ctx', 'BB_Lower_ctx'] + [f'BB_Upper_ctx_{i}dev' for i in range(1,4)] + [f'BB_Lower_ctx_{i}dev' for i in range(1,4)]:
                    df_c[col_bb] = np.nan
        df_c = calculate_vwap(df_c); df_c.rename(columns={'VWAP_daily': 'VWAP_daily_ctx'}, inplace=True)
    else:
        for p, col_base in [(ema_short_p_ctx_gc, "EMA"), (ema_long_p_ctx_gc, "EMA"), (sma1_p_ctx, "SMA"), (sma2_p_ctx, "SMA")]:
            df_c[f'{col_base}{p if p else ""}_ctx'] = np.nan
        df_c['ADX_ctx'] = np.nan; df_c['PLUS_DI_ctx'] = np.nan; df_c['MINUS_DI_ctx'] = np.nan
        df_c[f'ATR_{atr_ctx_p_chart if atr_ctx_p_chart else ""}_CTX_Chart'] = np.nan
        if atr_ctx_p_for_sl: df_c[f'ATR_{atr_ctx_p_for_sl}_ctx'] = np.nan
        for col_bb in ['BB_Middle_ctx', 'BB_Upper_ctx', 'BB_Lower_ctx'] + [f'BB_Upper_ctx_{i}dev' for i in range(1,4)] + [f'BB_Lower_ctx_{i}dev' for i in range(1,4)]:
            df_c[col_bb] = np.nan
        df_c['VWAP_daily_ctx'] = np.nan

    # 実行足の指標計算
    stoch_k_e = _get_param('STOCH_SETTINGS_K_EXEC',is_period=True); stoch_d_e = _get_param('STOCH_SETTINGS_D_EXEC',is_period=True); stoch_smooth_e = _get_param('STOCH_SETTINGS_SMOOTH_EXEC',is_period=True)
    macd_fast_e=_get_param('MACD_SETTINGS_FAST_EXEC',is_period=True); macd_slow_e=_get_param('MACD_SETTINGS_SLOW_EXEC',is_period=True); macd_signal_e=_get_param('MACD_SETTINGS_SIGNAL_EXEC',is_period=True)
    macd_hist_ema_period_exec = _get_param('MACD_SETTINGS_HIST_EMA_PERIOD_EXEC', is_period=True)
    bb_period_e = _get_param('BB_SETTINGS_PERIOD_EXEC', is_period=True); bb_nbdev_e_for_orig = _get_param('BB_SETTINGS_NBDEV_EXEC', is_float=True)
    if bb_nbdev_e_for_orig is None: bb_nbdev_e_for_orig = 2.0
    atr_exec_p_chart = _get_param('ATR_SETTINGS_PERIOD_EXEC', is_period=True)
    sma1_p_exec = _get_param('SMA_SETTINGS_EXEC_PERIOD_1', is_period=True); sma2_p_exec = _get_param('SMA_SETTINGS_EXEC_PERIOD_2', is_period=True)
    ema_s_p_exec_chart = _get_param('EMA_SETTINGS_SHORT_EXEC_CHART', is_period=True); ema_l_p_exec_chart = _get_param('EMA_SETTINGS_LONG_EXEC_CHART', is_period=True)
    # 一目均衡表のパラメータ取得 (実行足用)
    tenkan_p_exec = _get_param('ICHIMOKU_SETTINGS_TENKAN_SEN_PERIOD_EXEC', is_period=True)
    kijun_p_exec = _get_param('ICHIMOKU_SETTINGS_KIJUN_SEN_PERIOD_EXEC', is_period=True)
    senkou_b_p_exec = _get_param('ICHIMOKU_SETTINGS_SENKOU_SPAN_B_PERIOD_EXEC', is_period=True)


    if not df_e.empty:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for p, col_name_base, suffix in [(ema_s_p_exec_chart,"EMA","_exec"), (ema_l_p_exec_chart,"EMA","_exec"), (sma1_p_exec,"SMA","_exec"), (sma2_p_exec,"SMA","_exec")]:
                if p and len(df_e['Close']) >= p: df_e[f'{col_name_base}{p}{suffix}'] = getattr(talib, col_name_base)(df_e['Close'], timeperiod=p)
                else: df_e[f'{col_name_base}{p}{suffix}'] = np.nan
            min_len_stoch = 0
            if stoch_k_e and stoch_d_e and stoch_smooth_e: min_len_stoch = max(stoch_k_e, stoch_d_e) + stoch_smooth_e -1
            if all(p is not None for p in [stoch_k_e, stoch_d_e, stoch_smooth_e]) and len(df_e['Close']) >= min_len_stoch:
                df_e['STOCH_K_exec'], df_e['STOCH_D_exec'] = talib.STOCH(df_e['High'], df_e['Low'], df_e['Close'], fastk_period=stoch_k_e, slowk_period=stoch_d_e, slowk_matype=0, slowd_period=stoch_smooth_e, slowd_matype=0)
            else: df_e['STOCH_K_exec'], df_e['STOCH_D_exec'] = np.nan, np.nan
            min_len_macd = 0
            if macd_slow_e and macd_signal_e: min_len_macd = macd_slow_e + macd_signal_e -1 
            if all(p is not None for p in [macd_fast_e, macd_slow_e, macd_signal_e]) and len(df_e['Close']) >= min_len_macd :
                df_e['MACD_exec'], df_e['MACDsignal_exec'], df_e['MACDhist_exec'] = talib.MACD(df_e['Close'], fastperiod=macd_fast_e, slowperiod=macd_slow_e, signalperiod=macd_signal_e)
            else: df_e['MACD_exec'], df_e['MACDsignal_exec'], df_e['MACDhist_exec'] = np.nan, np.nan, np.nan
            if macd_hist_ema_period_exec and 'MACDhist_exec' in df_e.columns and df_e['MACDhist_exec'].notna().any() and len(df_e['MACDhist_exec'].dropna()) >= macd_hist_ema_period_exec:
                df_e[f'MACDhist_EMA_exec'] = talib.EMA(df_e['MACDhist_exec'], timeperiod=macd_hist_ema_period_exec)
            else:
                df_e[f'MACDhist_EMA_exec'] = np.nan
            
            # 一目均衡表の計算 (実行足)
            if all(p is not None for p in [tenkan_p_exec, kijun_p_exec, senkou_b_p_exec]) and \
               len(df_e['Close']) >= max(tenkan_p_exec, kijun_p_exec, senkou_b_p_exec):
                ichimoku_df = calculate_ichimoku_components(
                    df_e, 
                    tenkan_period=tenkan_p_exec, 
                    kijun_period=kijun_p_exec, 
                    senkou_b_period=senkou_b_p_exec
                )
                df_e['tenkan_sen_exec'] = ichimoku_df['tenkan_sen']
                df_e['kijun_sen_exec'] = ichimoku_df['kijun_sen']
                df_e['senkou_span_a_exec'] = ichimoku_df['senkou_span_a']
                df_e['senkou_span_b_exec'] = ichimoku_df['senkou_span_b']
                df_e['chikou_span_exec'] = ichimoku_df['chikou_span']
                df_e['senkou_span_a_raw_exec'] = ichimoku_df['senkou_span_a_raw']
                df_e['senkou_span_b_raw_exec'] = ichimoku_df['senkou_span_b_raw']
            else:
                df_e['tenkan_sen_exec'] = np.nan
                df_e['kijun_sen_exec'] = np.nan
                df_e['senkou_span_a_exec'] = np.nan
                df_e['senkou_span_b_exec'] = np.nan
                df_e['chikou_span_exec'] = np.nan
                df_e['senkou_span_a_raw_exec'] = np.nan
                df_e['senkou_span_b_raw_exec'] = np.nan

            if bb_period_e and len(df_e['Close']) >= bb_period_e:
                df_e['BB_Middle_exec'] = talib.SMA(df_e['Close'], timeperiod=bb_period_e)
                std_dev_exec = talib.STDDEV(df_e['Close'], timeperiod=bb_period_e, nbdev=1)
                for i in range(1, 4):
                    df_e[f'BB_Upper_exec_{i}dev'] = df_e['BB_Middle_exec'] + std_dev_exec * i
                    df_e[f'BB_Lower_exec_{i}dev'] = df_e['BB_Middle_exec'] - std_dev_exec * i
                df_e['BB_Upper_exec'] = df_e['BB_Middle_exec'] + std_dev_exec * bb_nbdev_e_for_orig
                df_e['BB_Lower_exec'] = df_e['BB_Middle_exec'] - std_dev_exec * bb_nbdev_e_for_orig
            else:
                for col_bb_e in ['BB_Middle_exec', 'BB_Upper_exec', 'BB_Lower_exec'] + [f'BB_Upper_exec_{i}dev' for i in range(1,4)] + [f'BB_Lower_exec_{i}dev' for i in range(1,4)]:
                    df_e[col_bb_e] = np.nan
            if atr_exec_p_chart and len(df_e['High']) >= atr_exec_p_chart: df_e[f'ATR_{atr_exec_p_chart}_EXEC_Chart'] = talib.ATR(df_e['High'], df_e['Low'], df_e['Close'], timeperiod=atr_exec_p_chart)
            else: df_e[f'ATR_{atr_exec_p_chart}_EXEC_Chart'] = np.nan
        df_e = calculate_vwap(df_e); df_e.rename(columns={'VWAP_daily': 'VWAP_daily_exec'}, inplace=True)
    else:
        for p, col_base, suffix in [(ema_s_p_exec_chart,"EMA","_exec"), (ema_l_p_exec_chart,"EMA","_exec"), (sma1_p_exec,"SMA","_exec"), (sma2_p_exec,"SMA","_exec")]:
            df_e[f'{col_base}{p if p else ""}{suffix}'] = np.nan
        for col_nan_e in ['STOCH_K_exec', 'STOCH_D_exec', 'MACD_exec', 'MACDsignal_exec', 'MACDhist_exec', 
                          f'MACDhist_EMA_exec', 
                          'tenkan_sen_exec', 'kijun_sen_exec', 'senkou_span_a_exec', 'senkou_span_b_exec', 'chikou_span_exec',
                          'senkou_span_a_raw_exec', 'senkou_span_b_raw_exec',
                          'BB_Middle_exec', 'BB_Upper_exec', 'BB_Lower_exec', 
                          f'ATR_{atr_exec_p_chart if atr_exec_p_chart else ""}_EXEC_Chart', 'VWAP_daily_exec']:
            df_e[col_nan_e] = np.nan
        for i in range(1,4): df_e[f'BB_Upper_exec_{i}dev']=np.nan; df_e[f'BB_Lower_exec_{i}dev']=np.nan

    df_c_reindexed = pd.DataFrame(index=df_e.index)
    if not df_c.empty:
        df_c_reindexed = df_c.reindex(df_e.index, method='ffill')

    ctx_cols_to_merge_base = []
    if ema_short_p_ctx_gc: ctx_cols_to_merge_base.append(f'EMA{ema_short_p_ctx_gc}_ctx')
    if ema_long_p_ctx_gc: ctx_cols_to_merge_base.append(f'EMA{ema_long_p_ctx_gc}_ctx')
    if sma1_p_ctx: ctx_cols_to_merge_base.append(f'SMA{sma1_p_ctx}_ctx')
    if sma2_p_ctx: ctx_cols_to_merge_base.append(f'SMA{sma2_p_ctx}_ctx')
    ctx_cols_to_merge_base.extend(['ADX_ctx', 'PLUS_DI_ctx', 'MINUS_DI_ctx'])
    if atr_ctx_p_chart: ctx_cols_to_merge_base.append(f'ATR_{atr_ctx_p_chart}_CTX_Chart')
    ctx_cols_to_merge_base.extend(['VWAP_daily_ctx', 'BB_Middle_ctx', 'BB_Upper_ctx', 'BB_Lower_ctx'])
    for i in range(1, 4): ctx_cols_to_merge_base.append(f'BB_Upper_ctx_{i}dev'); ctx_cols_to_merge_base.append(f'BB_Lower_ctx_{i}dev')
    
    valid_ctx_cols_for_its_suffix = [col for col in ctx_cols_to_merge_base if col and col in df_c_reindexed.columns]

    df_merged = df_e.copy()

    if not df_c_reindexed.empty:
        for col in valid_ctx_cols_for_its_suffix:
            if col in df_c_reindexed.columns:
                df_merged[col + '_ITS'] = df_c_reindexed[col]
        
        if 'Close' in df_c_reindexed.columns:
            df_merged['Close_ctx_ITS'] = df_c_reindexed['Close']
        else:
            df_merged['Close_ctx_ITS'] = np.nan
            
        if atr_ctx_p_for_sl:
            atr_col_for_sl_name_orig = f'ATR_{atr_ctx_p_for_sl}_ctx'
            if atr_col_for_sl_name_orig in df_c_reindexed.columns:
                df_merged[atr_col_for_sl_name_orig] = df_c_reindexed[atr_col_for_sl_name_orig]
            elif atr_col_for_sl_name_orig not in df_merged.columns:
                df_merged[atr_col_for_sl_name_orig] = np.nan
    else: 
        for col_base_name in ctx_cols_to_merge_base:
             if (col_base_name + '_ITS') not in df_merged.columns:
                 df_merged[col_base_name + '_ITS'] = np.nan
        if 'Close_ctx_ITS' not in df_merged.columns:
            df_merged['Close_ctx_ITS'] = np.nan
        if atr_ctx_p_for_sl:
            atr_col_for_sl_name_orig = f'ATR_{atr_ctx_p_for_sl}_ctx'
            if atr_col_for_sl_name_orig not in df_merged.columns:
                 df_merged[atr_col_for_sl_name_orig] = np.nan
                 
    return df_merged

def generate_signals(df, params_from_backtester, loaded_strategy_params):
    signals = pd.Series(index=df.index, data=0.0)
    if df.empty:
        df['Signal'] = signals
        return df

    def _get_param(key_name, is_period=False, is_float=False, is_activation_flag=False):
        return get_merged_param(key_name, params_from_backtester, loaded_strategy_params,
                                is_period, is_float, is_activation_flag)

    # 環境認識パラメータ
    ema_short_p_ctx = _get_param('EMA_SETTINGS_CONTEXT_PERIOD_SHORT_GC', is_period=True)
    ema_long_p_ctx = _get_param('EMA_SETTINGS_CONTEXT_PERIOD_LONG_GC', is_period=True)
    adx_threshold = _get_param('ADX_SETTINGS_CONTEXT_THRESHOLD', is_float=True)
    
    use_ema_cross_ctx = _get_param('ACTIVATION_FLAGS_CONTEXT_TREND_DIR_EMA_CROSS_ACTIVE', is_activation_flag=True)
    use_adx_ctx = _get_param('ACTIVATION_FLAGS_CONTEXT_TREND_STR_ADX_ACTIVE', is_activation_flag=True)
    use_vwap_trend_ctx = _get_param('ACTIVATION_FLAGS_CONTEXT_PRICE_VWAP_RELATION_ACTIVE', is_activation_flag=True)
    use_dmi_trend_ctx = _get_param('ACTIVATION_FLAGS_CONTEXT_TREND_DIR_DMI_ACTIVE', is_activation_flag=True)

    # エントリーパラメータ
    stoch_os = _get_param('STOCH_SETTINGS_OVERSOLD_LEVEL', is_float=True)
    stoch_ob = _get_param('STOCH_SETTINGS_OVERBOUGHT_LEVEL', is_float=True)
    use_stoch_entry = _get_param('ACTIVATION_FLAGS_ENTRY_STOCH_ACTIVE', is_activation_flag=True)
    use_macd_entry = _get_param('ACTIVATION_FLAGS_ENTRY_MACD_ACTIVE', is_activation_flag=True)
    use_bb_middle_entry = _get_param('ACTIVATION_FLAGS_ENTRY_BB_MIDDLE_ACTIVE', is_activation_flag=True)
    use_macd_hist_ema_entry = _get_param('ACTIVATION_FLAGS_ENTRY_MACD_HIST_EMA_ACTIVE', is_activation_flag=True)
    
    # 一目均衡表の個別条件フラグ
    use_ichimoku_entry = _get_param('ACTIVATION_FLAGS_ENTRY_ICHIMOKU_ACTIVE', is_activation_flag=True)
    use_tenkan_kijun_cross = _get_param('ACTIVATION_FLAGS_ENTRY_ICHIMOKU_TENKAN_KIJUN_CROSS_ACTIVE', is_activation_flag=True)
    use_kumo_breakout = _get_param('ACTIVATION_FLAGS_ENTRY_ICHIMOKU_KUMO_BREAKOUT_ACTIVE', is_activation_flag=True)
    use_chikou_cross = _get_param('ACTIVATION_FLAGS_ENTRY_ICHIMOKU_CHIKOU_CROSS_ACTIVE', is_activation_flag=True)
    kijun_p_exec_for_chikou_shift = _get_param('ICHIMOKU_SETTINGS_KIJUN_SEN_PERIOD_EXEC', is_period=True)


    # カラム名定義
    ema_short_col_its = f'EMA{ema_short_p_ctx}_ctx_ITS' if ema_short_p_ctx else None
    ema_long_col_its = f'EMA{ema_long_p_ctx}_ctx_ITS' if ema_long_p_ctx else None
    adx_col_its = 'ADX_ctx_ITS'
    vwap_col_ctx_its = 'VWAP_daily_ctx_ITS'
    price_col_ctx_its = 'Close_ctx_ITS'
    plus_di_col_its = 'PLUS_DI_ctx_ITS'
    minus_di_col_its = 'MINUS_DI_ctx_ITS'

    stoch_k_col = 'STOCH_K_exec'; stoch_d_col = 'STOCH_D_exec'
    macd_line_col = 'MACD_exec'; macd_signal_col = 'MACDsignal_exec'
    macd_hist_col = 'MACDhist_exec'
    macd_hist_ema_col = 'MACDhist_EMA_exec'
    bb_middle_col = 'BB_Middle_exec'; close_col = 'Close' # 実行足Close

    tenkan_sen_col = 'tenkan_sen_exec'
    kijun_sen_col = 'kijun_sen_exec'
    current_senkou_span_a_col = 'senkou_span_a_raw_exec'
    current_senkou_span_b_col = 'senkou_span_b_raw_exec'
    chikou_span_col = 'chikou_span_exec'

    required_cols_check_map = {
        "EMA_CTX_SHORT_ITS": (ema_short_col_its, use_ema_cross_ctx > 0), 
        "EMA_CTX_LONG_ITS": (ema_long_col_its, use_ema_cross_ctx > 0),
        "ADX_CTX_ITS": (adx_col_its, use_adx_ctx > 0),
        "VWAP_CTX_ITS": (vwap_col_ctx_its, use_vwap_trend_ctx > 0),
        "PRICE_CTX_ITS": (price_col_ctx_its, use_vwap_trend_ctx > 0),
        "PLUS_DI_CTX_ITS": (plus_di_col_its, use_dmi_trend_ctx > 0),
        "MINUS_DI_CTX_ITS": (minus_di_col_its, use_dmi_trend_ctx > 0),
        "STOCH_K_EXEC": (stoch_k_col, use_stoch_entry > 0),
        "STOCH_D_EXEC": (stoch_d_col, use_stoch_entry > 0), 
        "MACD_EXEC": (macd_line_col, use_macd_entry > 0),
        "MACD_SIGNAL_EXEC": (macd_signal_col, use_macd_entry > 0), 
        "BB_MIDDLE_EXEC": (bb_middle_col, use_bb_middle_entry > 0),
        "CLOSE_EXEC": (close_col, use_bb_middle_entry > 0 or use_ichimoku_entry > 0),
        "MACD_HIST_EXEC": (macd_hist_col, use_macd_hist_ema_entry > 0),
        "MACD_HIST_EMA_EXEC": (macd_hist_ema_col, use_macd_hist_ema_entry > 0),
        "TENKAN_SEN_EXEC": (tenkan_sen_col, use_ichimoku_entry > 0 and use_tenkan_kijun_cross > 0),
        "KIJUN_SEN_EXEC": (kijun_sen_col, use_ichimoku_entry > 0 and use_tenkan_kijun_cross > 0),
        "CURRENT_SENKOU_A_EXEC": (current_senkou_span_a_col, use_ichimoku_entry > 0 and use_kumo_breakout > 0),
        "CURRENT_SENKOU_B_EXEC": (current_senkou_span_b_col, use_ichimoku_entry > 0 and use_kumo_breakout > 0),
        "CHIKOU_SPAN_EXEC": (chikou_span_col, use_ichimoku_entry > 0 and use_chikou_cross > 0),
    }
    missing_or_all_nan_cols = []
    for display_name, (col_name, is_rule_active) in required_cols_check_map.items():
        if is_rule_active:
            if col_name is None or col_name not in df.columns:
                missing_or_all_nan_cols.append(f"{display_name} (column missing: expected '{col_name}')")
            elif df[col_name].isnull().all():
                missing_or_all_nan_cols.append(f"{display_name} (column '{col_name}' is all NaN)")
    
    if missing_or_all_nan_cols:
        print(f"[Strategy Signal Warning] Missing or all NaN for required active columns: {', '.join(missing_or_all_nan_cols)}. Signals will be 0.")
        df['Signal'] = 0.0
        return df

    # 環境認識フェーズの条件
    ctx_mandatory_long_cond = pd.Series(True, index=df.index)
    ctx_mandatory_short_cond = pd.Series(True, index=df.index)
    ctx_optional_long_cond_list = []
    ctx_optional_short_cond_list = []

    if use_ema_cross_ctx > 0 and ema_short_col_its and ema_long_col_its and \
       ema_short_col_its in df.columns and ema_long_col_its in df.columns:
        ema_s = df[ema_short_col_its].ffill().bfill(); ema_l = df[ema_long_col_its].ffill().bfill()
        valid_ema = ema_s.notna() & ema_l.notna()
        ema_trend_up = pd.Series(False, index=df.index); ema_trend_down = pd.Series(False, index=df.index)
        if valid_ema.any():
            ema_trend_up[valid_ema] = ema_s[valid_ema] > ema_l[valid_ema]
            ema_trend_down[valid_ema] = ema_s[valid_ema] < ema_l[valid_ema]
        if use_ema_cross_ctx == 2:
            ctx_mandatory_long_cond &= ema_trend_up
            ctx_mandatory_short_cond &= ema_trend_down
        elif use_ema_cross_ctx == 1:
            ctx_optional_long_cond_list.append(ema_trend_up)
            ctx_optional_short_cond_list.append(ema_trend_down)

    if use_adx_ctx > 0 and adx_col_its and adx_col_its in df.columns:
        adx = df[adx_col_its].fillna(0)
        current_adx_threshold = adx_threshold if adx_threshold is not None else -1
        adx_strong = adx > current_adx_threshold
        if use_adx_ctx == 2:
            ctx_mandatory_long_cond &= adx_strong
            ctx_mandatory_short_cond &= adx_strong
        elif use_adx_ctx == 1:
            ctx_optional_long_cond_list.append(adx_strong)
            ctx_optional_short_cond_list.append(adx_strong)

    if use_vwap_trend_ctx > 0 and vwap_col_ctx_its in df.columns and price_col_ctx_its in df.columns:
        vwap_ctx = df[vwap_col_ctx_its].ffill().bfill()
        price_ctx = df[price_col_ctx_its].ffill().bfill()
        valid_vwap_price = vwap_ctx.notna() & price_ctx.notna()
        vwap_trend_up = pd.Series(False, index=df.index)
        vwap_trend_down = pd.Series(False, index=df.index)
        if valid_vwap_price.any():
            vwap_trend_up[valid_vwap_price] = price_ctx[valid_vwap_price] > vwap_ctx[valid_vwap_price]
            vwap_trend_down[valid_vwap_price] = price_ctx[valid_vwap_price] < vwap_ctx[valid_vwap_price]
        if use_vwap_trend_ctx == 2:
            ctx_mandatory_long_cond &= vwap_trend_up
            ctx_mandatory_short_cond &= vwap_trend_down
        elif use_vwap_trend_ctx == 1:
            ctx_optional_long_cond_list.append(vwap_trend_up)
            ctx_optional_short_cond_list.append(vwap_trend_down)

    if use_dmi_trend_ctx > 0 and plus_di_col_its in df.columns and minus_di_col_its in df.columns:
        plus_di = df[plus_di_col_its].ffill().bfill()
        minus_di = df[minus_di_col_its].ffill().bfill()
        valid_dmi_data = plus_di.notna() & minus_di.notna()
        dmi_trend_up = pd.Series(False, index=df.index)
        dmi_trend_down = pd.Series(False, index=df.index)
        if valid_dmi_data.any():
            dmi_trend_up[valid_dmi_data] = plus_di[valid_dmi_data] > minus_di[valid_dmi_data]
            dmi_trend_down[valid_dmi_data] = minus_di[valid_dmi_data] > plus_di[valid_dmi_data]
        if use_dmi_trend_ctx == 2:
            ctx_mandatory_long_cond &= dmi_trend_up
            ctx_mandatory_short_cond &= dmi_trend_down
        elif use_dmi_trend_ctx == 1:
            ctx_optional_long_cond_list.append(dmi_trend_up)
            ctx_optional_short_cond_list.append(dmi_trend_down)

    final_ctx_optional_long_cond = pd.Series(True, index=df.index)
    if ctx_optional_long_cond_list:
        final_ctx_optional_long_cond = pd.concat(ctx_optional_long_cond_list, axis=1).any(axis=1)
    final_ctx_optional_short_cond = pd.Series(True, index=df.index)
    if ctx_optional_short_cond_list:
        final_ctx_optional_short_cond = pd.concat(ctx_optional_short_cond_list, axis=1).any(axis=1)

    final_env_long_ok = ctx_mandatory_long_cond & final_ctx_optional_long_cond
    final_env_short_ok = ctx_mandatory_short_cond & final_ctx_optional_short_cond

    # エントリーフェーズの条件
    entry_mandatory_long_cond = pd.Series(True, index=df.index); entry_optional_long_cond_list = []
    entry_mandatory_short_cond = pd.Series(True, index=df.index); entry_optional_short_cond_list = []

    if use_stoch_entry > 0 and stoch_k_col in df.columns and stoch_d_col in df.columns:
        k = df[stoch_k_col].ffill().bfill().fillna(50); d = df[stoch_d_col].ffill().bfill().fillna(50)
        k_prev = k.shift(1).ffill().bfill().fillna(50); d_prev = d.shift(1).ffill().bfill().fillna(50)
        stoch_gc_cond = (k_prev <= d_prev) & (k > d); stoch_dc_cond = (k_prev >= d_prev) & (k < d)
        current_stoch_os = stoch_os if stoch_os is not None else 0; current_stoch_ob = stoch_ob if stoch_ob is not None else 100
        stoch_buy_trigger = stoch_gc_cond & (k <= current_stoch_os); stoch_sell_trigger = stoch_dc_cond & (k >= current_stoch_ob)
        if use_stoch_entry == 2: entry_mandatory_long_cond &= stoch_buy_trigger; entry_mandatory_short_cond &= stoch_sell_trigger
        elif use_stoch_entry == 1: entry_optional_long_cond_list.append(stoch_buy_trigger); entry_optional_short_cond_list.append(stoch_sell_trigger)
    
    if use_macd_entry > 0 and macd_line_col in df.columns and macd_signal_col in df.columns:
        macd_l = df[macd_line_col].ffill().bfill().fillna(0); macd_s = df[macd_signal_col].ffill().bfill().fillna(0)
        macd_buy_trigger = macd_l > macd_s; macd_sell_trigger = macd_l < macd_s 
        if use_macd_entry == 2: entry_mandatory_long_cond &= macd_buy_trigger; entry_mandatory_short_cond &= macd_sell_trigger
        elif use_macd_entry == 1: entry_optional_long_cond_list.append(macd_buy_trigger); entry_optional_short_cond_list.append(macd_sell_trigger)

    if use_bb_middle_entry > 0 and bb_middle_col in df.columns and close_col in df.columns:
        close_val = df[close_col].ffill().bfill(); bb_mid = df[bb_middle_col].ffill().bfill()
        valid_bb_mid_data = close_val.notna() & bb_mid.notna()
        bb_buy_trigger = pd.Series(False, index=df.index); bb_sell_trigger = pd.Series(False, index=df.index)
        if valid_bb_mid_data.any(): bb_buy_trigger[valid_bb_mid_data] = close_val[valid_bb_mid_data] <= bb_mid[valid_bb_mid_data]; bb_sell_trigger[valid_bb_mid_data] = close_val[valid_bb_mid_data] >= bb_mid[valid_bb_mid_data]
        if use_bb_middle_entry == 2: entry_mandatory_long_cond &= bb_buy_trigger; entry_mandatory_short_cond &= bb_sell_trigger
        elif use_bb_middle_entry == 1: entry_optional_long_cond_list.append(bb_buy_trigger); entry_optional_short_cond_list.append(bb_sell_trigger)

    if use_macd_hist_ema_entry > 0 and macd_hist_col in df.columns and macd_hist_ema_col in df.columns:
        hist = df[macd_hist_col].ffill().bfill().fillna(0)
        hist_ema = df[macd_hist_ema_col].ffill().bfill().fillna(0)
        hist_prev = hist.shift(1).ffill().bfill().fillna(0)
        hist_ema_prev = hist_ema.shift(1).ffill().bfill().fillna(0)
        hist_ema_buy_trigger = (hist_prev <= hist_ema_prev) & (hist > hist_ema)
        hist_ema_sell_trigger = (hist_prev >= hist_ema_prev) & (hist < hist_ema)
        if use_macd_hist_ema_entry == 2:
            entry_mandatory_long_cond &= hist_ema_buy_trigger
            entry_mandatory_short_cond &= hist_ema_sell_trigger
        elif use_macd_hist_ema_entry == 1:
            entry_optional_long_cond_list.append(hist_ema_buy_trigger)
            entry_optional_short_cond_list.append(hist_ema_sell_trigger)

    if use_ichimoku_entry > 0:
        ichimoku_buy_cond_agg = pd.Series(True, index=df.index)
        ichimoku_sell_cond_agg = pd.Series(True, index=df.index)

        if use_tenkan_kijun_cross > 0 and all(c in df.columns for c in [tenkan_sen_col, kijun_sen_col]):
            tenkan = df[tenkan_sen_col].ffill().bfill()
            kijun = df[kijun_sen_col].ffill().bfill()
            tenkan_cross_kijun_buy = (tenkan.shift(1) <= kijun.shift(1)) & (tenkan > kijun)
            tenkan_cross_kijun_sell = (tenkan.shift(1) >= kijun.shift(1)) & (tenkan < kijun)
            ichimoku_buy_cond_agg &= tenkan_cross_kijun_buy
            ichimoku_sell_cond_agg &= tenkan_cross_kijun_sell

        if use_kumo_breakout > 0 and all(c in df.columns for c in [current_senkou_span_a_col, current_senkou_span_b_col, close_col]):
            ichimoku_close = df[close_col].ffill().bfill()
            current_span_a = df[current_senkou_span_a_col].ffill().bfill()
            current_span_b = df[current_senkou_span_b_col].ffill().bfill()
            current_kumo_upper = pd.Series(np.where(current_span_a > current_span_b, current_span_a, current_span_b), index=df.index)
            current_kumo_lower = pd.Series(np.where(current_span_a < current_span_b, current_span_a, current_span_b), index=df.index)
            price_above_current_kumo = ichimoku_close > current_kumo_upper
            price_below_current_kumo = ichimoku_close < current_kumo_lower
            ichimoku_buy_cond_agg &= price_above_current_kumo
            ichimoku_sell_cond_agg &= price_below_current_kumo

        if use_chikou_cross > 0 and close_col in df.columns and kijun_p_exec_for_chikou_shift is not None:
            close_price_for_chikou_comparison = df[close_col].shift(kijun_p_exec_for_chikou_shift)
            chikou_break_up = (df[close_col].shift(1) <= close_price_for_chikou_comparison.shift(1)) & \
                              (df[close_col] > close_price_for_chikou_comparison)
            chikou_break_down = (df[close_col].shift(1) >= close_price_for_chikou_comparison.shift(1)) & \
                                (df[close_col] < close_price_for_chikou_comparison)
            ichimoku_buy_cond_agg &= chikou_break_up
            ichimoku_sell_cond_agg &= chikou_break_down

        if use_ichimoku_entry == 2:
            entry_mandatory_long_cond &= ichimoku_buy_cond_agg
            entry_mandatory_short_cond &= ichimoku_sell_cond_agg
        elif use_ichimoku_entry == 1:
            entry_optional_long_cond_list.append(ichimoku_buy_cond_agg)
            entry_optional_short_cond_list.append(ichimoku_sell_cond_agg)

    final_entry_optional_long_cond = pd.Series(True, index=df.index)
    if entry_optional_long_cond_list:
        final_entry_optional_long_cond = pd.concat(entry_optional_long_cond_list, axis=1).any(axis=1)
    final_entry_optional_short_cond = pd.Series(True, index=df.index)
    if entry_optional_short_cond_list:
        final_entry_optional_short_cond = pd.concat(entry_optional_short_cond_list, axis=1).any(axis=1)

    final_entry_long_ok = entry_mandatory_long_cond & final_entry_optional_long_cond
    final_entry_short_ok = entry_mandatory_short_cond & final_entry_optional_short_cond
    
    buy_signal_series = final_env_long_ok & final_entry_long_ok
    sell_signal_series = final_env_short_ok & final_entry_short_ok
    
    signals.loc[buy_signal_series] = 1.0
    signals.loc[sell_signal_series] = -1.0
    signals.loc[buy_signal_series & sell_signal_series] = 0.0
    
    df['Signal'] = signals.fillna(0.0)
    return df

def determine_exit_conditions(current_position, current_bar_data, prev_bar_data,
                              params_from_backtester, loaded_strategy_params,
                              entry_price=None, current_bar_time=None):
    if current_position == 0: return False, "", None

    merged_params_for_exit = loaded_strategy_params.copy()
    if params_from_backtester:
        flat_fw_params = {k.upper() if isinstance(k, str) else k: v for k,v in params_from_backtester.items()}
        merged_params_for_exit.update(flat_fw_params)

    def _get_exit_param(key_name_upper, is_float=False, is_activation_flag=False, is_bool=False):
        val = merged_params_for_exit.get(key_name_upper.upper())
        if val is None:
            print(f"[Strategy Exit Param Warning] Param '{key_name_upper}' not found. Using fallback.")
            if is_activation_flag: return 0
            if is_bool: return False
            if is_float: return 0.0
            return None
        try:
            if is_activation_flag: return int(float(val))
            if is_bool:
                if isinstance(val, bool): return val
                if isinstance(val, str): return val.lower() in ['true', '1', 'yes', 'on']
                return bool(int(float(val)))
            if is_float: return float(val)
            return val
        except (ValueError, TypeError):
            print(f"[Strategy Exit Param Warning] Conversion error for '{key_name_upper}', val '{val}'. Using fallback.")
            if is_activation_flag: return 0
            if is_bool: return False
            if is_float: return 0.0
            return None
            
    exit_reason_list = []; exit_now = False

    sl_stoch_active = _get_exit_param('ACTIVATION_FLAGS_STOP_LOSS_STOCH_REVERSE_ACTIVE', is_activation_flag=True)
    sl_macd_active = _get_exit_param('ACTIVATION_FLAGS_STOP_LOSS_MACD_REVERSE_ACTIVE', is_activation_flag=True)
    sl_bb_middle_active = _get_exit_param('ACTIVATION_FLAGS_STOP_LOSS_BB_MIDDLE_REVERSE_ACTIVE', is_activation_flag=True)
    sl_atr_multiple = _get_exit_param('ACTIVATION_FLAGS_STOP_LOSS_ATR_MULTIPLE', is_float=True)
    sl_macd_hist_ema_active = _get_exit_param('ACTIVATION_FLAGS_STOP_LOSS_MACD_HIST_EMA_REVERSE_ACTIVE', is_activation_flag=True)
    
    atr_period_for_sl_ctx_val = _get_exit_param('ATR_SETTINGS_STOP_PERIOD_CONTEXT')
    atr_period_for_sl_ctx = None
    if atr_period_for_sl_ctx_val is not None:
        try: atr_period_for_sl_ctx = int(float(atr_period_for_sl_ctx_val))
        except ValueError: pass
    atr_col_for_sl = f'ATR_{atr_period_for_sl_ctx}_ctx' if atr_period_for_sl_ctx is not None else None

    if sl_stoch_active > 0:
        s_k = current_bar_data.get('STOCH_K_exec'); s_d = current_bar_data.get('STOCH_D_exec')
        if s_k is not None and s_d is not None and not (np.isnan(s_k) or np.isnan(s_d)):
            if current_position == 1 and s_k < s_d: exit_reason_list.append("SL:Stoch_DC")
            elif current_position == -1 and s_k > s_d: exit_reason_list.append("SL:Stoch_GC")
    if sl_macd_active > 0:
        m_line = current_bar_data.get('MACD_exec'); m_signal = current_bar_data.get('MACDsignal_exec')
        if m_line is not None and m_signal is not None and not (np.isnan(m_line) or np.isnan(m_signal)):
            if current_position == 1 and m_line < m_signal: exit_reason_list.append("SL:MACD_DC")
            elif current_position == -1 and m_line > m_signal: exit_reason_list.append("SL:MACD_GC")
    if sl_bb_middle_active > 0:
        close_price = current_bar_data.get('Close'); bb_middle = current_bar_data.get('BB_Middle_exec')
        if close_price is not None and bb_middle is not None and not (np.isnan(close_price) or np.isnan(bb_middle)):
            if current_position == 1 and close_price < bb_middle: exit_reason_list.append("SL:BB_Mid割れ")
            elif current_position == -1 and close_price > bb_middle: exit_reason_list.append("SL:BB_Mid超え")
    
    if sl_atr_multiple is not None and sl_atr_multiple > 0 and entry_price is not None and atr_col_for_sl and atr_col_for_sl in current_bar_data:
        current_atr_val = current_bar_data.get(atr_col_for_sl); current_low = current_bar_data.get('Low'); current_high = current_bar_data.get('High')
        if current_atr_val is not None and not np.isnan(current_atr_val) and current_atr_val > 0 and \
           current_low is not None and not np.isnan(current_low) and \
           current_high is not None and not np.isnan(current_high):
            if current_position == 1:
                sl_price_atr = entry_price - (current_atr_val * sl_atr_multiple)
                if current_low <= sl_price_atr: exit_reason_list.append(f"SL:ATR Hit (L)")
            elif current_position == -1:
                sl_price_atr = entry_price + (current_atr_val * sl_atr_multiple)
                if current_high >= sl_price_atr: exit_reason_list.append(f"SL:ATR Hit (S)")

    if sl_macd_hist_ema_active > 0:
        hist_curr = current_bar_data.get('MACDhist_exec')
        hist_ema_curr = current_bar_data.get('MACDhist_EMA_exec')
        if hist_curr is not None and hist_ema_curr is not None and not (np.isnan(hist_curr) or np.isnan(hist_ema_curr)):
            if current_position == 1 and hist_curr < hist_ema_curr:
                exit_reason_list.append("SL:MACDhist<EMA")
            elif current_position == -1 and hist_curr > hist_ema_curr:
                exit_reason_list.append("SL:MACDhist>EMA")

    if exit_reason_list:
        exit_now = True
    else:
        tp_stoch_active = _get_exit_param('ACTIVATION_FLAGS_TAKE_PROFIT_STOCH_EXTREME_REVERSE_ACTIVE', is_activation_flag=True)
        tp_macd_hist_active = _get_exit_param('ACTIVATION_FLAGS_TAKE_PROFIT_MACD_HIST_REVERSE_ACTIVE', is_activation_flag=True)
        tp_bb_touch_active = _get_exit_param('ACTIVATION_FLAGS_TAKE_PROFIT_BB_EXTREME_TOUCH_ACTIVE', is_activation_flag=True)
        tp_macd_hist_ema_active = _get_exit_param('ACTIVATION_FLAGS_TAKE_PROFIT_MACD_HIST_EMA_REVERSE_ACTIVE', is_activation_flag=True)
        stoch_ob_level = _get_exit_param('STOCH_SETTINGS_OVERBOUGHT_LEVEL', is_float=True)
        stoch_os_level = _get_exit_param('STOCH_SETTINGS_OVERSOLD_LEVEL', is_float=True)

        if tp_stoch_active > 0 and stoch_ob_level is not None and stoch_os_level is not None:
            s_k_curr = current_bar_data.get('STOCH_K_exec'); s_k_prev = prev_bar_data.get('STOCH_K_exec')
            if s_k_curr is not None and s_k_prev is not None and not (np.isnan(s_k_curr) or np.isnan(s_k_prev)):
                if current_position == 1 and s_k_prev >= stoch_ob_level and s_k_curr < s_k_prev: exit_reason_list.append("TP:Stoch_OB_Reverse")
                elif current_position == -1 and s_k_prev <= stoch_os_level and s_k_curr > s_k_prev: exit_reason_list.append("TP:Stoch_OS_Reverse")
        
        if tp_macd_hist_active > 0:
            m_hist_curr = current_bar_data.get('MACDhist_exec'); m_hist_prev = prev_bar_data.get('MACDhist_exec')
            if m_hist_curr is not None and m_hist_prev is not None and not (np.isnan(m_hist_curr) or np.isnan(m_hist_prev)):
                if current_position == 1 and m_hist_prev > 0 and m_hist_curr < m_hist_prev : exit_reason_list.append("TP:MACD_Hist_PeakOut")
                elif current_position == -1 and m_hist_prev < 0 and m_hist_curr > m_hist_prev: exit_reason_list.append("TP:MACD_Hist_PeakOut")
        
        if tp_bb_touch_active > 0:
            bb_upper = current_bar_data.get('BB_Upper_exec'); bb_lower = current_bar_data.get('BB_Lower_exec')
            high_price = current_bar_data.get('High'); low_price = current_bar_data.get('Low')
            if bb_upper is not None and bb_lower is not None and high_price is not None and low_price is not None and \
               not (np.isnan(bb_upper) or np.isnan(bb_lower) or np.isnan(high_price) or np.isnan(low_price)):
                if current_position == 1 and high_price >= bb_upper: exit_reason_list.append("TP:BB_UpTouch")
                elif current_position == -1 and low_price <= bb_lower: exit_reason_list.append("TP:BB_LowTouch")

        if tp_macd_hist_ema_active > 0:
            hist_ema_curr = current_bar_data.get('MACDhist_EMA_exec')
            hist_ema_prev = prev_bar_data.get('MACDhist_EMA_exec')
            if hist_ema_curr is not None and hist_ema_prev is not None and not (np.isnan(hist_ema_curr) or np.isnan(hist_ema_prev)):
                if current_position == 1 and hist_ema_curr < hist_ema_prev:
                    exit_reason_list.append("TP:MACDhist_EMA_TurnDn")
                elif current_position == -1 and hist_ema_curr > hist_ema_prev:
                    exit_reason_list.append("TP:MACDhist_EMA_TurnUp")
        if exit_reason_list:
            exit_now = True

    if not exit_now and current_bar_time is not None:
        considers_force_exit_val = _get_exit_param('ACTIVATION_FLAGS_STRATEGY_CONSIDERS_FORCE_EXIT_TIME', is_activation_flag=True)
        force_exit_time_obj = merged_params_for_exit.get('PARSED_FORCE_EXIT_TIME_OBJ')

        if considers_force_exit_val > 0 and force_exit_time_obj and isinstance(force_exit_time_obj, datetime.time):
            if current_bar_time >= force_exit_time_obj:
                if not exit_reason_list:
                    exit_reason_list.append(f"強制決済({force_exit_time_obj.strftime('%H:%M')})")
                exit_now = True
                
    exit_price_candidate = None
    final_exit_reason = ", ".join(sorted(list(set(exit_reason_list)))) if exit_reason_list else ""
    
    return exit_now, final_exit_reason, exit_price_candidate