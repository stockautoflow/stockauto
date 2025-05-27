# base.py (メタ情報ファイル保存機能追加版)
import pandas as pd
import numpy as np
import datetime
import math
import os
import warnings
import logging
import sys
import glob
import importlib
import re
import json # JSONを扱うためにインポート

import chart

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

BASE_DIR = '.'
CSV_OUTPUT_BASE_DIR = "data"
RESULTS_DIR = os.path.join(BASE_DIR, "Backtest")
LOG_DIR = os.path.join(RESULTS_DIR, "log")
CHART_OUTPUT_DIR = os.path.join(RESULTS_DIR, "Charts")

INITIAL_CAPITAL = 70_000_000
RISK_PER_TRADE = 0.005
COMMISSION_RATE = 0.0005
SLIPPAGE = 0.0002

def load_csv_data(filepath):
    try:
        df = pd.read_csv(filepath, index_col='datetime', parse_dates=True)
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is None: df = df.tz_localize('Asia/Tokyo', ambiguous='infer')
            elif str(df.index.tz) != 'Asia/Tokyo': df = df.tz_convert('Asia/Tokyo')
        else: raise ValueError("datetime列が正しくDatetimeIndexとしてパースされませんでした。")
        df.columns = [col.capitalize() for col in df.columns]
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"必要なOHLCVカラムが不足しています: {filepath}")
        for col in required_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=required_cols)
        df['Volume'] = df['Volume'].fillna(0).astype(np.int64)
        return df
    except FileNotFoundError: logging.error(f"    ファイルが見つかりません: {filepath}"); raise
    except Exception as e: logging.error(f"    CSVファイルの読み込みまたは前処理中にエラーが発生しました ({os.path.basename(filepath)}): {e}"); raise

def get_codes_from_data_dir(data_dir_path, interval_str_for_glob, date_str_for_glob):
    logging.info(f"データフォルダから銘柄コードリストを推測中: {data_dir_path} (指定日付: {date_str_for_glob}, 足種パターン: {interval_str_for_glob})")
    codes = set()
    try:
        if not os.path.isdir(data_dir_path): raise FileNotFoundError(f"データフォルダが見つかりません: {data_dir_path}")
        pattern_specific_date = f"????_{interval_str_for_glob}_{date_str_for_glob}.csv"
        logging.debug(f"  検索パターン (指定日付): {pattern_specific_date}")
        file_list_specific_date = glob.glob(os.path.join(data_dir_path, pattern_specific_date))
        file_list_to_process = file_list_specific_date
        used_wildcard_date = False
        if not file_list_specific_date:
            pattern_any_date = f"????_{interval_str_for_glob}_*.csv"
            logging.debug(f"  指定日付のファイルなし。代替検索パターン (日付ワイルドカード): {pattern_any_date}")
            file_list_any_date = glob.glob(os.path.join(data_dir_path, pattern_any_date))
            if file_list_any_date:
                logging.info(f"  指定日付 '{date_str_for_glob}' の '{interval_str_for_glob}' ファイルが見つからなかったため、日付部分が異なるファイルも対象とします。")
                file_list_to_process = file_list_any_date; used_wildcard_date = True
            else: raise FileNotFoundError(f"フォルダ '{data_dir_path}' 内に '{pattern_specific_date}' または (代替として) '{pattern_any_date}' に一致するファイルが見つかりませんでした。")
        for fpath in file_list_to_process:
            fname = os.path.basename(fpath)
            parts = fname.split('_')
            if len(parts) > 0 and parts[0].isdigit() and len(parts[0]) >= 4: codes.add(parts[0][:4])
        sorted_codes = sorted(list(codes))
        if not sorted_codes: raise ValueError(f"ファイル名から4桁の銘柄コードを抽出できませんでした。{' (日付ワイルドカード使用)' if used_wildcard_date else ''}")
        logging.info(f"データフォルダ内のファイル名から {len(sorted_codes)} 個の銘柄コードを推測しました。{' (日付ワイルドカード使用)' if used_wildcard_date else ''}")
        return sorted_codes
    except FileNotFoundError as fnf: logging.error(f"エラー: {fnf}"); return []
    except ValueError as ve: logging.error(f"エラー: {ve}"); return []
    except Exception as e: logging.error(f"銘柄リスト推測中に予期せぬエラーが発生しました: {e}"); return []

def log_perf_summary(stock_code, df_processed_data, final_equity_val, total_pnl_val, max_drawdown_val,
                     total_trades_val, win_rate_val, profit_factor_val, avg_winning_trade_val, avg_losing_trade_val):
    logging.info(f"--- パフォーマンス指標 (銘柄: {stock_code}) ---")
    if df_processed_data is not None and not df_processed_data.empty and isinstance(df_processed_data.index, pd.DatetimeIndex) and len(df_processed_data.index) > 1:
        logging.info(f"  期間: {df_processed_data.index[0].strftime('%Y-%m-%d %H:%M')} - {df_processed_data.index[-1].strftime('%Y-%m-%d %H:%M')}")
    else: logging.info("  期間: N/A (データ不足)")
    logging.info(f"  最終資産: {final_equity_val:,.0f} 円 (初期資産: {INITIAL_CAPITAL:,.0f} 円)")
    logging.info(f"  総損益 (PnL): {total_pnl_val:,.0f} 円")
    if INITIAL_CAPITAL > 0: logging.info(f"  リターン率: {(final_equity_val / INITIAL_CAPITAL - 1) * 100:.2f}%")
    else: logging.info("  リターン率: N/A (初期資産が0のため計算不可)")
    logging.info(f"  最大ドローダウン: {max_drawdown_val:.2f}%" if not np.isnan(max_drawdown_val) else "最大ドローダウン: N/A")
    logging.info(f"  総トレード数: {total_trades_val}")
    logging.info(f"  勝率: {win_rate_val:.2f}%")
    logging.info(f"  プロフィットファクター: {profit_factor_val:.2f}" if profit_factor_val != np.inf else "プロフィットファクター: inf")
    logging.info(f"  平均利益（勝ちトレード）: {avg_winning_trade_val:,.0f} 円")
    logging.info(f"  平均損失（負けトレード）: {avg_losing_trade_val:,.0f} 円")

def run_backtest(strategy_module_name, config_filepath,
                 exec_interval_min, context_interval_min, date_str,
                 output_chart_flag, target_chart_codes_list,
                 invalid_chart_codes_from_cli):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    if output_chart_flag:
        os.makedirs(CHART_OUTPUT_DIR, exist_ok=True)

    execution_start_time = datetime.datetime.now() # バックテスト開始時刻を記録
    now_str_for_files = execution_start_time.strftime('%Y%m%d%H%M%S')
    interval_exec_str_format = f"{exec_interval_min}m"
    interval_context_str_format = f"{context_interval_min}m"

    safe_strategy_name_for_file = strategy_module_name.replace('.', '_')

    log_filename_format = f"BacktestLog_{safe_strategy_name_for_file}_{interval_exec_str_format}{interval_context_str_format}_{date_str}_{now_str_for_files}.log"
    log_filepath_format = os.path.join(LOG_DIR, log_filename_format)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] (%(module)s:%(funcName)s:%(lineno)d) %(message)s',
        handlers=[
            logging.FileHandler(log_filepath_format, encoding='utf-8')
        ]
    )
    print(f"--- バックテスト処理開始 (ログファイル: {log_filepath_format}) ---")

    logging.info(f"=== バックテストスクリプト開始 (Strategy: {strategy_module_name}, Config: {config_filepath}) ===")
    logging.info(f"実行足: {interval_exec_str_format}, 環境足: {interval_context_str_format}, データ日付: {date_str}")

    if invalid_chart_codes_from_cli:
        logging.warning(f"コマンドラインでのチャート対象銘柄指定において、以下の不正な形式のコードが検出されました（これらはグラフ出力対象から除外されています）: {', '.join(invalid_chart_codes_from_cli)}")

    if output_chart_flag:
        if not target_chart_codes_list:
            logging.info("グラフ出力: 全ての銘柄のグラフを出力します。")
        else:
            logging.info(f"グラフ出力: 指定された銘柄のグラフを出力します: {', '.join(target_chart_codes_list)}")
    else:
        logging.info("グラフ出力: グラフは出力されません。")

    try:
        strategy_module = importlib.import_module(strategy_module_name)
        logging.info(f"戦略モジュール '{strategy_module_name}' の読み込みに成功しました。")
    except ModuleNotFoundError:
        logging.error(f"エラー: 戦略モジュール '{strategy_module_name}' が見つかりません。sys.path を確認してください。")
        print(f"エラー (base): 戦略モジュール '{strategy_module_name}' が見つかりません。", file=sys.stderr)
        raise
    except Exception as e:
        logging.error(f"エラー: 戦略モジュール '{strategy_module_name}' の読み込み中にエラー: {e}")
        print(f"エラー (base): 戦略モジュール '{strategy_module_name}' の読み込み中にエラー: {e}", file=sys.stderr)
        raise

    loaded_strategy_params_for_meta = {} # メタ情報用に元の形式も保持したい場合 (ここではフラット化されたものを使用)
    try:
        # strategy.py の load_strategy_config_yaml はフラット化された辞書を返す
        loaded_strategy_params = strategy_module.load_strategy_config_yaml(config_filepath)
        if not loaded_strategy_params:
            logging.warning(f"設定ファイル '{config_filepath}' からパラメータが読み込めませんでした。戦略定義のデフォルト値（あれば）で続行します。")
        else:
            logging.info(f"設定ファイル '{config_filepath}' から戦略パラメータを読み込みました。")
            loaded_strategy_params_for_meta = loaded_strategy_params # 保存用にコピー
    except Exception as e:
        logging.error(f"設定ファイル '{config_filepath}' の読み込み処理中に予期せぬエラー: {e}")
        print(f"エラー (base): 設定ファイル '{config_filepath}' の読み込み中にエラー: {e}", file=sys.stderr)
        raise

    current_strategy_params = {
        'interval_exec': exec_interval_min,
        'interval_context': context_interval_min,
    }

    DATA_DIR_PATH = os.path.join(BASE_DIR, f"{CSV_OUTPUT_BASE_DIR}_{date_str}")
    logging.info(f"データ読み込みフォルダ: {os.path.abspath(DATA_DIR_PATH)}")
    logging.info(f"結果CSV保存先ベースフォルダ: {os.path.abspath(RESULTS_DIR)}")
    if output_chart_flag:
        logging.info(f"チャート保存先フォルダ: {os.path.abspath(CHART_OUTPUT_DIR)}")

    logging.info(f"バックテスタから戦略モジュールに渡される基本パラメータ (current_strategy_params): {current_strategy_params}")
    logging.info(f"初期資金: {INITIAL_CAPITAL:,.0f}円, 1トレードリスク(目安): {RISK_PER_TRADE*100:.2f}%, 手数料率: {COMMISSION_RATE*100:.4f}%, スリッページ: {SLIPPAGE*100:.4f}%")

    entry_start_time_str = strategy_module.get_merged_param('TRADING_HOURS_ENTRY_START_TIME_STR', current_strategy_params, loaded_strategy_params)
    entry_end_time_str = strategy_module.get_merged_param('TRADING_HOURS_ENTRY_END_TIME_STR', current_strategy_params, loaded_strategy_params)
    force_exit_time_str_for_log = strategy_module.get_merged_param('TRADING_HOURS_FORCE_EXIT_TIME_STR', current_strategy_params, loaded_strategy_params)

    try:
        if entry_start_time_str is None or entry_end_time_str is None:
            raise ValueError("エントリー開始時刻または終了時刻が設定されていません。")
        entry_start_time_obj = datetime.datetime.strptime(entry_start_time_str, '%H:%M:%S').time()
        entry_end_time_obj = datetime.datetime.strptime(entry_end_time_str, '%H:%M:%S').time()
        if force_exit_time_str_for_log:
            current_strategy_params['PARSED_FORCE_EXIT_TIME_OBJ'] = datetime.datetime.strptime(force_exit_time_str_for_log, '%H:%M:%S').time()
    except (TypeError, ValueError) as e_time:
        logging.error(f"エラー: 取引時間の設定値が不正です: {e_time} (START: {entry_start_time_str}, END: {entry_end_time_str}, FORCE_EXIT: {force_exit_time_str_for_log})。")
        print(f"エラー (base): 取引時間の設定値が不正です。ログを確認してください。", file=sys.stderr)
        raise

    stock_codes_list_all = []
    try:
        stock_codes_list_all = get_codes_from_data_dir(DATA_DIR_PATH, interval_exec_str_format, date_str)
        if not stock_codes_list_all:
            logging.info(f"実行足 '{interval_exec_str_format}' で銘柄発見できず。環境足 '{interval_context_str_format}' で再試行。")
            stock_codes_list_all = get_codes_from_data_dir(DATA_DIR_PATH, interval_context_str_format, date_str)
        if not stock_codes_list_all:
            raise ValueError(f"データフォルダ {DATA_DIR_PATH} (日付 {date_str}) から銘柄コードを推測できませんでした。処理対象銘柄なし。")
        NUM_STOCKS_TO_PROCESS = len(stock_codes_list_all)
        logging.info(f"処理対象として {NUM_STOCKS_TO_PROCESS} 個の銘柄コードを特定: {stock_codes_list_all}")
    except (FileNotFoundError, ValueError) as e_codes:
        logging.error(f"エラー: {e_codes}")
        print(f"エラー (base): {e_codes}", file=sys.stderr)
        return
    except Exception as e_codes_generic:
        logging.error(f"銘柄リスト推測中に予期せぬエラー: {e_codes_generic}", exc_info=True)
        print(f"エラー (base): 銘柄リスト推測中に予期せぬエラーが発生しました。", file=sys.stderr)
        raise

    print(f"--- 全 {NUM_STOCKS_TO_PROCESS} 銘柄のバックテスト処理ループ開始 ---")
    results_summary_list = []
    all_trades_log_list = []
    total_stocks_processed_count = 0
    processed_data_period_start = None # 全銘柄を通じたデータ期間の開始
    processed_data_period_end = None   # 全銘柄を通じたデータ期間の終了

    for stock_idx, stock_code_val in enumerate(stock_codes_list_all):
        total_stocks_processed_count += 1
        progress_str = f"銘柄 {stock_code_val} ({total_stocks_processed_count}/{NUM_STOCKS_TO_PROCESS})"
        print(f"  Processing: {stock_code_val} ({total_stocks_processed_count}/{NUM_STOCKS_TO_PROCESS})... ", end="", flush=True)
        logging.info(f"\n  --- {progress_str} のバックテスト処理開始 ---")

        filepath_exec_val, filename_exec_val = None, ""
        specific_exec_file_path = os.path.join(DATA_DIR_PATH, f"{stock_code_val}_{interval_exec_str_format}_{date_str}.csv")
        if os.path.exists(specific_exec_file_path): filepath_exec_val = specific_exec_file_path
        else:
            alt_exec_files_list = glob.glob(os.path.join(DATA_DIR_PATH, f"{stock_code_val}_{interval_exec_str_format}_*.csv"))
            if alt_exec_files_list: filepath_exec_val = alt_exec_files_list[0]; logging.warning(f"    実行足: 指定日付ファイルなし。代替ファイル '{os.path.basename(filepath_exec_val)}' 使用。")
        if filepath_exec_val: filename_exec_val = os.path.basename(filepath_exec_val)
        else:
            logging.error(f"  {progress_str}: 実行足CSVファイル未検出。スキップ。")
            results_summary_list.append({'銘柄コード': stock_code_val, 'エラー': f'実行足CSVなし'})
            print("NG (実行足CSVなし)")
            continue

        filepath_context_val, filename_context_val = None, ""
        specific_context_file_path = os.path.join(DATA_DIR_PATH, f"{stock_code_val}_{interval_context_str_format}_{date_str}.csv")
        if os.path.exists(specific_context_file_path): filepath_context_val = specific_context_file_path
        else:
            alt_context_files_list = glob.glob(os.path.join(DATA_DIR_PATH, f"{stock_code_val}_{interval_context_str_format}_*.csv"))
            if alt_context_files_list: filepath_context_val = alt_context_files_list[0]; logging.warning(f"    環境足: 指定日付ファイルなし。代替ファイル '{os.path.basename(filepath_context_val)}' 使用。")
        
        if filepath_context_val:
            filename_context_val = os.path.basename(filepath_context_val)
        else:
            logging.error(f"  {progress_str}: 環境足CSVファイル未検出。スキップ。")
            results_summary_list.append({'銘柄コード': stock_code_val, 'エラー': f'環境足CSVなし'})
            print("NG (環境足CSVなし)")
            continue
        logging.info(f"    使用ファイル: 実行足='{filename_exec_val}', 環境足='{filename_context_val}'")

        df_exec_ohlc, df_context_ohlc, df_with_signals = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        current_capital = INITIAL_CAPITAL; current_equity = INITIAL_CAPITAL
        current_position = 0; current_entry_price = 0.0; current_shares = 0
        trade_history_for_stock = []; equity_curve_for_stock = [INITIAL_CAPITAL]; entry_datetime_obj = None

        try:
            df_exec_ohlc = load_csv_data(filepath_exec_val)
            df_context_ohlc = load_csv_data(filepath_context_val)

            if df_exec_ohlc.empty: raise ValueError(f"実行足データが空です: {filepath_exec_val}")
            if df_context_ohlc.empty: raise ValueError(f"環境足データが空です: {filepath_context_val}")

            # 全体のデータ期間を更新 (最初の銘柄またはより古い日付があれば更新)
            current_min_date = df_exec_ohlc.index.min()
            current_max_date = df_exec_ohlc.index.max()
            if processed_data_period_start is None or current_min_date < processed_data_period_start:
                processed_data_period_start = current_min_date
            if processed_data_period_end is None or current_max_date > processed_data_period_end:
                processed_data_period_end = current_max_date


            df_merged_indicators = strategy_module.calculate_indicators(
                df_exec_ohlc.copy(),
                df_context_ohlc.copy(),
                current_strategy_params,
                loaded_strategy_params
            )
            if df_merged_indicators is None or df_merged_indicators.empty:
                raise ValueError("strategy.calculate_indicators が有効なDataFrameを返しませんでした。")

            df_with_signals = strategy_module.generate_signals(
                df_merged_indicators.copy(),
                current_strategy_params,
                loaded_strategy_params
            )
            if df_with_signals is None or df_with_signals.empty or 'Signal' not in df_with_signals.columns:
                raise ValueError("strategy.generate_signals が 'Signal' カラムを含む有効なDataFrameを返しませんでした。")
            if not all(col in df_with_signals.columns for col in ['Open', 'High', 'Low', 'Close']):
                raise ValueError("指標計算後のDataFrameにOHLCカラムが不足しています。")
            if len(df_with_signals) < 2:
                logging.warning(f"  {progress_str}: シグナル生成後のデータが2行未満です。スキップします。")
                results_summary_list.append({ '銘柄コード': stock_code_val, 'エラー': 'データ不足(シグナル後2行未満)'})
                print("NG (データ不足)")
                continue

            for i in range(1, len(df_with_signals) -1):
                current_bar_index = df_with_signals.index[i]
                current_bar_time = current_bar_index.time()
                current_bar_data_series = df_with_signals.iloc[i]
                prev_bar_data_series = df_with_signals.iloc[i-1]
                next_bar_open_for_exit = df_with_signals['Open'].iloc[i+1]
                signal_prev_bar = df_with_signals['Signal'].iloc[i-1] if not pd.isna(df_with_signals['Signal'].iloc[i-1]) else 0

                if current_position != 0:
                    exit_now, exit_reason_from_strategy, _ = strategy_module.determine_exit_conditions(
                        current_position,
                        current_bar_data_series,
                        prev_bar_data_series,
                        current_strategy_params,
                        loaded_strategy_params,
                        current_entry_price,
                        current_bar_time
                    )
                    if exit_now:
                        exit_price_val = next_bar_open_for_exit
                        pnl_per_share = (exit_price_val - current_entry_price) if current_position == 1 else (current_entry_price - exit_price_val)
                        gross_pnl = current_shares * pnl_per_share
                        commission_exit = current_shares * exit_price_val * COMMISSION_RATE
                        net_pnl = gross_pnl - commission_exit
                        current_capital += net_pnl
                        trade_history_for_stock.append({
                            '銘柄コード': stock_code_val, 'entry_date': entry_datetime_obj,
                            'exit_date': df_with_signals.index[i+1],
                            'type': 'Long' if current_position == 1 else 'Short',
                            'entry_price': current_entry_price, 'exit_price': exit_price_val,
                            'shares': current_shares, 'pnl': net_pnl, 'exit_type': exit_reason_from_strategy
                        })
                        logging.info(f"    {df_with_signals.index[i+1]}: {exit_reason_from_strategy} {'L' if current_position==1 else 'S'}@E:{current_entry_price:.2f} X:{exit_price_val:.2f}, PnL:{net_pnl:,.0f}, Shares:{current_shares}, Cap:{current_capital:,.0f}")
                        current_position = 0; current_entry_price = 0.0; current_shares = 0; entry_datetime_obj = None

                entry_bar_open_price = df_with_signals['Open'].iloc[i]
                is_within_entry_time = (current_bar_time >= entry_start_time_obj and current_bar_time < entry_end_time_obj)

                if current_position == 0 and signal_prev_bar != 0 and is_within_entry_time:
                    entry_price_candidate = entry_bar_open_price * (1 + SLIPPAGE * signal_prev_bar)
                    risk_width_per_share_example = entry_price_candidate * 0.02
                    if risk_width_per_share_example <= 0: risk_width_per_share_example = entry_price_candidate * 0.01
                    target_risk_amount = current_equity * RISK_PER_TRADE
                    shares_to_trade = 0
                    if risk_width_per_share_example > 0:
                        shares_to_trade = int(target_risk_amount / risk_width_per_share_example)
                        shares_to_trade = (shares_to_trade // 100) * 100
                    if shares_to_trade == 0:
                        logging.debug(f"    {current_bar_index}: エントリーシグナル ({'Long' if signal_prev_bar == 1 else 'Short'}) だが株数0で見送り。RiskWidth(仮): {risk_width_per_share_example:.2f}")
                    else:
                        trade_cost = entry_price_candidate * shares_to_trade
                        commission_entry = trade_cost * COMMISSION_RATE
                        required_capital_for_trade = trade_cost + commission_entry
                        can_trade = True
                        if signal_prev_bar == 1 and current_capital < required_capital_for_trade :
                            logging.debug(f"    {current_bar_index}: Longエントリー資金不足。必要:{required_capital_for_trade:,.0f} > 現在:{current_capital:,.0f}"); can_trade = False
                        if can_trade:
                            current_position = int(signal_prev_bar); current_entry_price = entry_price_candidate
                            current_shares = shares_to_trade; entry_datetime_obj = current_bar_index
                            current_capital -= commission_entry
                            logging.info(f"    {entry_datetime_obj}: エントリー {'L' if current_position==1 else 'S'}@P:{current_entry_price:.2f}, Shares:{current_shares}, Cost:{trade_cost:,.0f}, Comm:{commission_entry:,.0f}, Cap(after comm):{current_capital:,.0f}")
                unrealized_pnl = 0
                current_bar_close_price = df_with_signals['Close'].iloc[i]
                if current_position == 1: unrealized_pnl = current_shares * (current_bar_close_price - current_entry_price)
                elif current_position == -1: unrealized_pnl = current_shares * (current_entry_price - current_bar_close_price)
                current_equity = current_capital + unrealized_pnl
                equity_curve_for_stock.append(current_equity)

            if current_position != 0 and len(df_with_signals) > 0 :
                final_bar_index = df_with_signals.index[-1]
                final_exit_price = df_with_signals['Open'].iloc[-1]
                exit_reason_final = "データ終了強制決済(最終足始値)"
                pnl_per_share_final = (final_exit_price - current_entry_price) if current_position == 1 else (current_entry_price - final_exit_price)
                gross_pnl_final = current_shares * pnl_per_share_final
                commission_final = current_shares * final_exit_price * COMMISSION_RATE
                net_pnl_final = gross_pnl_final - commission_final
                current_capital += net_pnl_final
                trade_history_for_stock.append({
                    '銘柄コード': stock_code_val, 'entry_date': entry_datetime_obj, 'exit_date': final_bar_index,
                    'type': 'Long' if current_position == 1 else 'Short', 'entry_price': current_entry_price,
                    'exit_price': final_exit_price, 'shares': current_shares, 'pnl': net_pnl_final, 'exit_type': exit_reason_final
                })
                logging.info(f"    {final_bar_index}: {exit_reason_final} {'L' if current_position==1 else 'S'}@E:{current_entry_price:.2f} X:{final_exit_price:.2f}, PnL:{net_pnl_final:,.0f}, Shares:{current_shares}, Cap:{current_capital:,.0f}")
                equity_curve_for_stock.append(current_capital)

            final_equity_for_stock = current_capital
            logging.info(f"  {progress_str}: バックテスト完了。最終確定資産: {final_equity_for_stock:,.0f} 円")
            trade_log_df_for_stock = pd.DataFrame(trade_history_for_stock)
            s_total_trades=0; s_win_rate=0.0; s_total_pnl=0.0; s_pf=0.0; s_avg_win=0.0; s_avg_loss=0.0; s_max_dd=np.nan
            if not trade_log_df_for_stock.empty:
                s_total_trades = len(trade_log_df_for_stock)
                winning_trades_df = trade_log_df_for_stock[trade_log_df_for_stock['pnl'] > 0]
                losing_trades_df = trade_log_df_for_stock[trade_log_df_for_stock['pnl'] <= 0]
                num_winning_trades = len(winning_trades_df); num_losing_trades = len(losing_trades_df)
                s_win_rate = (num_winning_trades / s_total_trades) * 100 if s_total_trades > 0 else 0.0
                s_total_pnl = trade_log_df_for_stock['pnl'].sum()
                gross_profit = winning_trades_df['pnl'].sum(); gross_loss = abs(losing_trades_df['pnl'].sum())
                s_pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
                s_avg_win = gross_profit / num_winning_trades if num_winning_trades > 0 else 0.0
                s_avg_loss = gross_loss / num_losing_trades if num_losing_trades > 0 else 0.0
                if len(equity_curve_for_stock) > 1:
                    equity_series = pd.Series(equity_curve_for_stock); peak = equity_series.cummax()
                    drawdown = (equity_series - peak) / peak
                    if drawdown.notna().any() and (INITIAL_CAPITAL > 0 and not peak.empty and peak.iloc[0] != 0):
                        s_max_dd = abs(drawdown.min() * 100) if drawdown.min() < 0 else 0.0
                    else: s_max_dd = 0.0
                else: s_max_dd = 0.0
                log_perf_summary(stock_code_val, df_with_signals, final_equity_for_stock, s_total_pnl, s_max_dd, s_total_trades, s_win_rate, s_pf, s_avg_win, s_avg_loss)
                results_summary_list.append({'銘柄コード': stock_code_val, '総損益': s_total_pnl, 'PF': s_pf, '勝率(%)': s_win_rate, 'トレード数': s_total_trades, '最大DD(%)': s_max_dd, '平均利益': s_avg_win, '平均損失': s_avg_loss, 'エラー': None})
                if not trade_log_df_for_stock.empty: all_trades_log_list.append(trade_log_df_for_stock)
            else:
                logging.info(f"  {progress_str}: トレード実行なし。")
                results_summary_list.append({'銘柄コード': stock_code_val, '総損益': 0, 'PF': 0, '勝率(%)': 0, 'トレード数': 0, '最大DD(%)': 0.0, '平均利益': 0, '平均損失': 0, 'エラー': 'トレードなし'})
            print("OK")

            if output_chart_flag:
                if not target_chart_codes_list or stock_code_val in target_chart_codes_list:
                    chart_base_filename_parts = [
                        safe_strategy_name_for_file, # ファイル名にはsafeな名前を使用
                        f"{interval_exec_str_format}{interval_context_str_format}",
                        date_str,
                        stock_code_val
                    ]
                    logging.info(f"  {progress_str}: チャート出力処理を開始します。")
                    try:
                        chart.plot_chart_for_stock(
                            df_context_ohlc,
                            df_exec_ohlc,
                            df_with_signals,
                            trade_log_df_for_stock,
                            stock_code_val,
                            loaded_strategy_params,
                            CHART_OUTPUT_DIR,
                            chart_base_filename_parts
                        )
                        logging.info(f"  {progress_str}: チャートを出力しました。")
                    except Exception as e_chart:
                        logging.error(f"  {progress_str}: チャート出力中にエラーが発生しました: {e_chart}", exc_info=True)
                        print(f"NG (チャート出力エラー: {stock_code_val})")
        except FileNotFoundError as fnf_stock:
            logging.error(f"  {progress_str}: ファイル未検出: {fnf_stock}。スキップ。")
            results_summary_list.append({ '銘柄コード': stock_code_val, 'エラー': f'ファイルなし: {os.path.basename(str(fnf_stock))}' })
            print(f"NG (FNF: {os.path.basename(str(fnf_stock))})")
        except ValueError as ve_stock:
            logging.error(f"  {progress_str}: データ処理/設定エラー: {ve_stock}。スキップ。")
            results_summary_list.append({ '銘柄コード': stock_code_val, 'エラー': f'データ処理/設定エラー: {ve_stock}' })
            print(f"NG (データエラー: {stock_code_val})")
        except Exception as e_stock:
            logging.error(f"  {progress_str}: 予期せぬエラー ({type(e_stock).__name__}: {e_stock})。スキップ。", exc_info=True)
            results_summary_list.append({ '銘柄コード': stock_code_val, 'エラー': f'予期せぬエラー: {type(e_stock).__name__}'})
            print(f"NG (エラー: {type(e_stock).__name__} in {stock_code_val})")
        logging.info(f"  --- {progress_str} のバックテスト処理終了 ---")

    # --- 全体結果とメタ情報の出力 ---
    logging.info(f"\n===== 全 {total_stocks_processed_count} 銘柄のバックテストが完了しました =====")
    print(f"\n===== 全 {total_stocks_processed_count} 銘柄のバックテストが完了しました =====")
    interval_combination_str = f"{interval_exec_str_format}{interval_context_str_format}"

    # メタ情報を作成
    meta_info = {
        "execution_datetime": execution_start_time.isoformat(),
        "strategy_module": strategy_module_name,
        "config_file": config_filepath,
        "execution_interval_minutes": exec_interval_min,
        "context_interval_minutes": context_interval_min,
        "data_date": date_str,
        "chart_output_enabled": output_chart_flag,
        "chart_target_codes": target_chart_codes_list if target_chart_codes_list else "ALL",
        "initial_capital": INITIAL_CAPITAL,
        "risk_per_trade": RISK_PER_TRADE,
        "commission_rate": COMMISSION_RATE,
        "slippage": SLIPPAGE,
        "processed_data_period_start": processed_data_period_start.isoformat() if processed_data_period_start else None,
        "processed_data_period_end": processed_data_period_end.isoformat() if processed_data_period_end else None,
        "loaded_strategy_parameters": loaded_strategy_params_for_meta # config.yamlからロードされたパラメータ
    }
    meta_output_filename = f"BacktestMeta_{safe_strategy_name_for_file}_{interval_combination_str}_{date_str}_{now_str_for_files}.json"
    meta_output_filepath = os.path.join(RESULTS_DIR, meta_output_filename)
    try:
        with open(meta_output_filepath, 'w', encoding='utf-8') as f:
            json.dump(meta_info, f, ensure_ascii=False, indent=4)
        logging.info(f"メタ情報JSON出力完了: {meta_output_filepath}")
        print(f"メタ情報JSON出力: {meta_output_filepath}")
    except Exception as e_json:
        logging.error(f"エラー: メタ情報JSON出力失敗: {e_json}")
        print(f"エラー (base): メタ情報JSON出力失敗: {e_json}", file=sys.stderr)


    if all_trades_log_list:
        all_trades_combined_df = pd.concat(all_trades_log_list, ignore_index=True)
        detail_output_filename = f"BacktestDetail_{safe_strategy_name_for_file}_{interval_combination_str}_{date_str}_{now_str_for_files}.csv"
        detail_output_filepath = os.path.join(RESULTS_DIR, detail_output_filename)
        detail_columns_order = ['銘柄コード','entry_date', 'exit_date', 'type', 'entry_price', 'exit_price', 'shares', 'pnl', 'exit_type']
        detail_columns_present = [col for col in detail_columns_order if col in all_trades_combined_df.columns]
        try:
            all_trades_combined_df[detail_columns_present].to_csv(detail_output_filepath, sep=',', index=False, encoding='utf-8-sig', float_format='%.2f')
            logging.info(f"詳細トレード履歴CSV出力完了: {detail_output_filepath}")
            print(f"\n詳細トレード履歴CSV出力: {detail_output_filepath}")
        except Exception as e_csv_detail:
            logging.error(f"エラー: 詳細トレード履歴CSV出力失敗: {e_csv_detail}")
            print(f"エラー (base): 詳細トレード履歴CSV出力失敗: {e_csv_detail}", file=sys.stderr)
    else:
        logging.warning("詳細トレード履歴なし。CSV未作成。")
        print("詳細トレード履歴なし。CSV未作成。")

    if results_summary_list:
        summary_df = pd.DataFrame(results_summary_list)
        summary_output_filename = f"BacktestSummary_{safe_strategy_name_for_file}_{interval_combination_str}_{date_str}_{now_str_for_files}.csv"
        summary_output_filepath = os.path.join(RESULTS_DIR, summary_output_filename)
        expected_summary_cols = ['銘柄コード', '総損益', 'PF', '勝率(%)', 'トレード数', '最大DD(%)', '平均利益', '平均損失', 'エラー']
        for col in expected_summary_cols:
            if col not in summary_df.columns: summary_df[col] = np.nan
        summary_df = summary_df[expected_summary_cols]
        numeric_cols_to_format = ['総損益', 'PF', '勝率(%)', '最大DD(%)', '平均利益', '平均損失']
        for col in numeric_cols_to_format:
            if col in summary_df.columns: summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')
        if 'PF' in summary_df.columns: summary_df['PF'] = summary_df['PF'].replace([np.inf, -np.inf], 'inf')
        if '総損益' in summary_df.columns: summary_df['総損益'] = summary_df['総損益'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A')
        if 'PF' in summary_df.columns: summary_df['PF'] = summary_df['PF'].apply(lambda x: f"{x:.2f}" if pd.notna(x) and x != 'inf' else ('inf' if x == 'inf' else 'N/A'))
        if '勝率(%)' in summary_df.columns: summary_df['勝率(%)'] = summary_df['勝率(%)'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A')
        if '最大DD(%)' in summary_df.columns: summary_df['最大DD(%)'] = summary_df['最大DD(%)'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A')
        if '平均利益' in summary_df.columns: summary_df['平均利益'] = summary_df['平均利益'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A')
        if '平均損失' in summary_df.columns: summary_df['平均損失'] = summary_df['平均損失'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A')
        summary_df = summary_df.fillna('N/A')
        try:
            summary_df.to_csv(summary_output_filepath, sep=',', index=False, encoding='utf-8-sig')
            logging.info(f"サマリーCSV出力完了: {summary_output_filepath}")
            print(f"サマリーCSV出力: {summary_output_filepath}")
        except Exception as e_csv_summary:
            logging.error(f"エラー: サマリーCSV出力失敗: {e_csv_summary}")
            print(f"エラー (base): サマリーCSV出力失敗: {e_csv_summary}", file=sys.stderr)
    else:
        logging.warning("結果サマリーなし。CSV未作成。")
        print("結果サマリーなし。CSV未作成。")

    logging.info("バックテストスクリプト処理完了。")
    print("\nスクリプト処理完了。ログファイルを確認してください。")