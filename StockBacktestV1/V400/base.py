# base.py
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt # chart.py に移動するが、base で直接使わないように注意
import math
import os
import warnings
import logging
import sys
import glob
import importlib

# chart.py と strategy.py の関数をインポート
import chart # chart.py 内の関数を呼び出す際に chart.plot_chart_for_stock のようにする
# strategy モジュールは動的に読み込むので、ここでは直接インポートしない

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# === グローバル設定 (backtest_fw.py から移植) ===
BASE_DIR = '.'
CSV_OUTPUT_BASE_DIR = "data" # backtest_cli.py から渡される date_str と組み合わせて使用
RESULTS_DIR = os.path.join(BASE_DIR, "Backtest")
LOG_DIR = os.path.join(RESULTS_DIR, "log")
CHART_OUTPUT_DIR = os.path.join(RESULTS_DIR, "Charts") # chart.py で使用

# --- バックテスト基本設定 (backtest_fw.py から移植) ---
INITIAL_CAPITAL = 70_000_000
RISK_PER_TRADE = 0.005
COMMISSION_RATE = 0.0005
SLIPPAGE = 0.0002

# === ヘルパー関数 (backtest_fw.py から移植) ===
def load_csv_data(filepath):
    # logging.debug(f"    CSV読み込み開始: {filepath}") # ロギングは run_backtest 内で設定後に行う
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
        df['Volume'] = df['Volume'].fillna(0).astype(np.int64) # np.int64 に修正
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


# === メイン処理関数 ===
def run_backtest(strategy_module_name, exec_interval_min, context_interval_min, date_str,
                 output_chart_flag, target_chart_codes_list):
    """
    バックテストを実行するメイン関数。
    backtest_cli.py から呼び出される。
    """
    print("\n--- バックテスト基盤処理開始 ---")
    # --- ロギング設定 ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    if output_chart_flag:
        os.makedirs(CHART_OUTPUT_DIR, exist_ok=True)

    now_str_for_files = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    interval_exec_str_format = f"{exec_interval_min}m"
    interval_context_str_format = f"{context_interval_min}m"

    log_filename_format = f"BacktestLog_{strategy_module_name}_{interval_exec_str_format}{interval_context_str_format}_{date_str}_{now_str_for_files}.log"
    log_filepath_format = os.path.join(LOG_DIR, log_filename_format)
    # 既存のハンドラをクリア
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] (%(module)s:%(funcName)s:%(lineno)d) %(message)s', # モジュール名と関数名も表示
        handlers=[
            logging.FileHandler(log_filepath_format, encoding='utf-8'),
            #logging.StreamHandler(sys.stdout) # 標準出力にも出す場合
        ]
    )
    print(f"ログはファイルに出力されます: {log_filepath_format}")
    logging.info(f"ログファイルパス: {log_filepath_format}")

    # --- 戦略モジュールの読み込み ---
    try:
        strategy_module = importlib.import_module(strategy_module_name)
        logging.info(f"戦略モジュール '{strategy_module_name}.py' の読み込みに成功しました。")
    except ModuleNotFoundError:
        logging.error(f"エラー: strategyファイル '{strategy_module_name}.py' が見つかりません。")
        print(f"エラー: strategyファイル '{strategy_module_name}.py' が見つかりません。", file=sys.stderr)
        return # または sys.exit(1)
    except Exception as e:
        logging.error(f"エラー: strategyファイル '{strategy_module_name}.py' の読み込み中にエラー: {e}")
        print(f"エラー: strategyファイル '{strategy_module_name}.py' の読み込み中にエラー: {e}", file=sys.stderr)
        return

    # --- current_strategy_params の準備 ---
    # base.py から strategy.py に渡す動的なパラメータ
    current_strategy_params = {
        'interval_exec': exec_interval_min,
        'interval_context': context_interval_min,
        # config.yaml にあるべきだが、FW側で上書きまたは動的に設定したいものがあればここに追加
        # 例: 'TRADING_HOURS_FORCE_EXIT_TIME_STR': '14:55:00' # FW側で強制したい場合など
    }
    # strategy.py 内の STRATEGY_PRIMARY_PARAMS (config.yamlからロードされたもの) とマージして使う

    logging.info(f"=== バックテストスクリプト開始 (Strategy: {strategy_module_name}) ===")
    logging.info(f"実行足: {interval_exec_str_format}, 環境足: {interval_context_str_format}, データ日付: {date_str}")
    DATA_DIR_PATH = os.path.join(BASE_DIR, f"{CSV_OUTPUT_BASE_DIR}_{date_str}")
    logging.info(f"データ読み込みフォルダ: {os.path.abspath(DATA_DIR_PATH)}")
    logging.info(f"結果CSV保存先ベースフォルダ: {os.path.abspath(RESULTS_DIR)}")
    if output_chart_flag:
        logging.info(f"チャート保存先フォルダ: {os.path.abspath(CHART_OUTPUT_DIR)}")
        if target_chart_codes_list:
            logging.info(f"チャートは指定された銘柄のみ出力: {target_chart_codes_list}")
        else:
            logging.info(f"チャートは全銘柄出力。")

    logging.info(f"バックテスタから戦略モジュールに渡される基本パラメータ (current_strategy_params): {current_strategy_params}")
    logging.info(f"初期資金: {INITIAL_CAPITAL:,.0f}円, 1トレードリスク(目安): {RISK_PER_TRADE*100:.2f}%, 手数料率: {COMMISSION_RATE*100:.4f}%, スリッページ: {SLIPPAGE*100:.4f}%")

    # 取引時間関連のパラメータを strategy.py の get_merged_param を通じて取得
    # strategy_module.STRATEGY_PRIMARY_PARAMS は strategy.py 内で config.yaml から読み込まれた辞書
    entry_start_time_str = strategy_module.get_merged_param('TRADING_HOURS_ENTRY_START_TIME_STR', current_strategy_params, strategy_module.STRATEGY_PRIMARY_PARAMS)
    entry_end_time_str = strategy_module.get_merged_param('TRADING_HOURS_ENTRY_END_TIME_STR', current_strategy_params, strategy_module.STRATEGY_PRIMARY_PARAMS)
    force_exit_time_str_for_log = strategy_module.get_merged_param('TRADING_HOURS_FORCE_EXIT_TIME_STR', current_strategy_params, strategy_module.STRATEGY_PRIMARY_PARAMS)

    try:
        if entry_start_time_str is None or entry_end_time_str is None:
            raise ValueError("エントリー開始時刻または終了時刻が設定されていません。config.yaml または strategy.py のデフォルト値を確認してください。")
        entry_start_time_obj = datetime.datetime.strptime(entry_start_time_str, '%H:%M:%S').time()
        entry_end_time_obj = datetime.datetime.strptime(entry_end_time_str, '%H:%M:%S').time()
        # 強制決済時刻もここでパースしておく（strategy.py に渡すため）
        if force_exit_time_str_for_log:
            current_strategy_params['parsed_force_exit_time_obj'] = datetime.datetime.strptime(force_exit_time_str_for_log, '%H:%M:%S').time()

    except (TypeError, ValueError) as e_time:
        logging.error(f"エラー: エントリー開始/終了/強制決済時刻のフォーマットまたは値が不正です: {e_time} (START: {entry_start_time_str}, END: {entry_end_time_str}, FORCE_EXIT: {force_exit_time_str_for_log})。config.yaml または strategy.py のデフォルト値を確認してください。")
        print(f"エラー: 取引時間の設定値が不正です。ログを確認してください。", file=sys.stderr)
        return
    logging.info(f"取引時間設定: エントリー許可 {entry_start_time_obj.strftime('%H:%M:%S')} - {entry_end_time_obj.strftime('%H:%M:%S')}, 強制決済は戦略側で {force_exit_time_str_for_log} を考慮")


    # --- 銘柄リストの取得 (backtest_fw.py から移植) ---
    stock_codes_list = []
    try:
        stock_codes_list = get_codes_from_data_dir(DATA_DIR_PATH, interval_exec_str_format, date_str)
        if not stock_codes_list:
            logging.info(f"実行足 '{interval_exec_str_format}' で銘柄発見できず。環境足 '{interval_context_str_format}' で再試行。")
            stock_codes_list = get_codes_from_data_dir(DATA_DIR_PATH, interval_context_str_format, date_str)
        if not stock_codes_list:
            raise ValueError(f"データフォルダ {DATA_DIR_PATH} (日付 {date_str}) から銘柄コードを推測できませんでした。")
        NUM_STOCKS_TO_PROCESS = len(stock_codes_list)
        logging.info(f"処理対象として {NUM_STOCKS_TO_PROCESS} 個の銘柄コードを特定: {stock_codes_list}")
    except (FileNotFoundError, ValueError) as e_codes:
        logging.error(f"エラー: {e_codes}")
        print(f"エラー: 銘柄リストの取得に失敗しました。ログを確認してください。", file=sys.stderr)
        return
    except Exception as e_codes_generic:
        logging.error(f"銘柄リスト推測中に予期せぬエラー: {e_codes_generic}", exc_info=True)
        print(f"エラー: 銘柄リストの取得中に予期せぬエラーが発生しました。ログを確認してください。", file=sys.stderr)
        return

    # --- バックテスト処理ループ (backtest_fw.py から移植・調整) ---
    logging.info(f"--- 全 {NUM_STOCKS_TO_PROCESS} 銘柄のバックテスト処理ループ開始 ---")
    print(f"--- 全 {NUM_STOCKS_TO_PROCESS} 銘柄のバックテスト処理ループ開始 ---")
    results_summary_list = []
    all_trades_log_list = []
    total_stocks_processed_count = 0

    for stock_idx, stock_code_val in enumerate(stock_codes_list):
        total_stocks_processed_count += 1
        progress_str = f"銘柄 {stock_code_val} ({total_stocks_processed_count}/{NUM_STOCKS_TO_PROCESS})"
        print(f"  Processing: {stock_code_val} ({total_stocks_processed_count}/{NUM_STOCKS_TO_PROCESS})... ", end="", flush=True)
        logging.info(f"\n  --- {progress_str} のバックテスト処理開始 ---")

        # ... (filepath_exec_val, filepath_context_val の取得ロジックは backtest_fw.py と同様) ...
        filepath_exec_val, filename_exec_val = None, ""
        specific_exec_file_path = os.path.join(DATA_DIR_PATH, f"{stock_code_val}_{interval_exec_str_format}_{date_str}.csv")
        if os.path.exists(specific_exec_file_path): filepath_exec_val = specific_exec_file_path
        else:
            alt_exec_files_list = glob.glob(os.path.join(DATA_DIR_PATH, f"{stock_code_val}_{interval_exec_str_format}_*.csv"))
            if alt_exec_files_list: filepath_exec_val = alt_exec_files_list[0]; logging.warning(f"    実行足: 指定日付ファイルなし。代替ファイル '{os.path.basename(filepath_exec_val)}' 使用。")
        if filepath_exec_val: filename_exec_val = os.path.basename(filepath_exec_val)
        else: logging.error(f"  {progress_str}: 実行足CSVファイル未検出。スキップ。"); results_summary_list.append({'銘柄コード': stock_code_val, 'エラー': f'実行足CSVなし'}); print("NG (実行足CSVなし)"); continue

        filepath_context_val, filename_context_val = None, ""
        specific_context_file_path = os.path.join(DATA_DIR_PATH, f"{stock_code_val}_{interval_context_str_format}_{date_str}.csv")
        if os.path.exists(specific_context_file_path): filepath_context_val = specific_context_file_path
        else:
            alt_context_files_list = glob.glob(os.path.join(DATA_DIR_PATH, f"{stock_code_val}_{interval_context_str_format}_*.csv"))
            if alt_context_files_list: filepath_context_val = alt_context_files_list[0]; logging.warning(f"    環境足: 指定日付ファイルなし。代替ファイル '{os.path.basename(filepath_context_val)}' 使用。")
        if filepath_context_val: filename_context_val = os.path.basename(filepath_context_val)
        else: logging.error(f"  {progress_str}: 環境足CSVファイル未検出。スキップ。"); results_summary_list.append({'銘柄コード': stock_code_val, 'エラー': f'環境足CSVなし'}); print("NG (環境足CSVなし)"); continue
        logging.info(f"    使用ファイル: 実行足='{filename_exec_val}', 環境足='{filename_context_val}'")


        df_exec_ohlc, df_context_ohlc, df_with_signals = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        current_capital = INITIAL_CAPITAL; current_equity = INITIAL_CAPITAL
        current_position = 0; current_entry_price = 0.0; current_shares = 0
        trade_history_for_stock = []; equity_curve_for_stock = [INITIAL_CAPITAL]; entry_datetime_obj = None

        try:
            df_exec_ohlc = load_csv_data(filepath_exec_val)
            df_context_ohlc = load_csv_data(filepath_context_val)
            if df_exec_ohlc.empty: raise ValueError(f"実行足データが空: {filepath_exec_val}")
            if df_context_ohlc.empty: logging.warning(f"環境足データが空: {filepath_context_val} (処理は続行)") # 環境足が空でも実行足があれば処理できる場合がある


            # --- strategy.py の関数呼び出し ---
            # current_strategy_params には FW から渡したい動的な値（足種など）が入る
            # strategy_module.STRATEGY_PRIMARY_PARAMS は config.yaml から読み込まれた静的な値
            # calculate_indicators や generate_signals 内でこれらがマージされて使われる想定
            df_merged_indicators = strategy_module.calculate_indicators(
                df_exec_ohlc.copy(),
                df_context_ohlc.copy(),
                current_strategy_params # FWからの動的パラメータ
            )
            if df_merged_indicators is None or df_merged_indicators.empty:
                raise ValueError("strategy.calculate_indicators が有効なDataFrameを返しませんでした。")

            df_with_signals = strategy_module.generate_signals(
                df_merged_indicators.copy(),
                current_strategy_params # FWからの動的パラメータ
            )
            if df_with_signals is None or df_with_signals.empty or 'Signal' not in df_with_signals.columns:
                raise ValueError("strategy.generate_signals が 'Signal' カラムを含む有効なDataFrameを返しませんでした。")
            if not all(col in df_with_signals.columns for col in ['Open', 'High', 'Low', 'Close']):
                raise ValueError("指標計算後のDataFrameにOHLCカラムが不足しています。")
            if len(df_with_signals) < 2: # 少なくとも2行ないと次の足の始値参照でエラーになる
                logging.warning(f"  {progress_str}: シグナル生成後のデータが2行未満です。スキップします。")
                results_summary_list.append({ '銘柄コード': stock_code_val, 'エラー': 'データ不足(シグナル後2行未満)'})
                print("NG (データ不足)")
                continue

            # --- 売買シミュレーションループ (backtest_fw.py から移植) ---
            for i in range(1, len(df_with_signals) -1): # 最終行の一つ手前まで（決済に翌日始値を使うため）
                current_bar_index = df_with_signals.index[i]
                current_bar_time = current_bar_index.time() # datetime.time オブジェクト
                current_bar_data_series = df_with_signals.iloc[i]
                prev_bar_data_series = df_with_signals.iloc[i-1] # 決済判断用に前バーの情報も渡す
                next_bar_open_for_exit = df_with_signals['Open'].iloc[i+1] # 決済価格として使用
                signal_prev_bar = df_with_signals['Signal'].iloc[i-1] if not pd.isna(df_with_signals['Signal'].iloc[i-1]) else 0


                # ポジションがある場合、決済条件を確認
                if current_position != 0:
                    # strategy.determine_exit_conditions を呼び出す
                    # current_strategy_params には FWからの動的パラメータと、パース済みの強制決済時刻も含む
                    exit_now, exit_reason_from_strategy, _ = strategy_module.determine_exit_conditions(
                        current_position,
                        current_bar_data_series, # 現在のバーの情報
                        prev_bar_data_series,    # 1つ前のバーの情報
                        current_strategy_params, # FWからの動的パラメータ (パース済み強制決済時刻含む)
                        current_entry_price,     # 現在の建玉価格
                        current_bar_time         # 現在のバーの時刻 (timeオブジェクト)
                    )

                    if exit_now:
                        exit_price_val = next_bar_open_for_exit # 翌足始値で決済
                        pnl_per_share = (exit_price_val - current_entry_price) if current_position == 1 else (current_entry_price - exit_price_val)
                        gross_pnl = current_shares * pnl_per_share
                        commission_exit = current_shares * exit_price_val * COMMISSION_RATE
                        net_pnl = gross_pnl - commission_exit
                        current_capital += net_pnl
                        trade_history_for_stock.append({
                            '銘柄コード': stock_code_val, 'entry_date': entry_datetime_obj,
                            'exit_date': df_with_signals.index[i+1], # 決済が成立した足のタイムスタンプ
                            'type': 'Long' if current_position == 1 else 'Short',
                            'entry_price': current_entry_price, 'exit_price': exit_price_val,
                            'shares': current_shares, 'pnl': net_pnl, 'exit_type': exit_reason_from_strategy
                        })
                        logging.info(f"    {df_with_signals.index[i+1]}: {exit_reason_from_strategy} {'L' if current_position==1 else 'S'}@E:{current_entry_price:.2f} X:{exit_price_val:.2f}, PnL:{net_pnl:,.0f}, Shares:{current_shares}, Cap:{current_capital:,.0f}")
                        current_position = 0; current_entry_price = 0.0; current_shares = 0; entry_datetime_obj = None


                # ポジションがない場合、エントリー条件を確認 (前バーのシグナルと現在のバーの時刻で判断)
                entry_bar_open_price = df_with_signals['Open'].iloc[i] # 当該足の始値でエントリー
                is_within_entry_time = (current_bar_time >= entry_start_time_obj and current_bar_time < entry_end_time_obj)

                if current_position == 0 and signal_prev_bar != 0 and is_within_entry_time:
                    # signal_prev_bar は前バーの終値で計算されたシグナル
                    # エントリーは現在のバーの始値 (entry_bar_open_price) で行う
                    entry_price_candidate = entry_bar_open_price * (1 + SLIPPAGE * signal_prev_bar) # スリッページ考慮

                    # 資金管理ロジック (backtest_fw.py から移植)
                    # ATR を使ったリスク幅の計算などは strategy.py 側で行うか、
                    # または、ここで strategy.py からリスク幅を取得する関数を呼び出す設計も考えられる。
                    # ここでは簡易的に固定比率でリスク幅を仮定 (より高度なものは strategy 側で)
                    risk_width_per_share_example = entry_price_candidate * 0.02 # 仮。実際はATR等で計算
                    if risk_width_per_share_example <= 0: risk_width_per_share_example = entry_price_candidate * 0.01 # 0割防止

                    target_risk_amount = current_equity * RISK_PER_TRADE # current_equity を使う
                    shares_to_trade = 0
                    if risk_width_per_share_example > 0 :
                        shares_to_trade = int(target_risk_amount / risk_width_per_share_example)
                        shares_to_trade = (shares_to_trade // 100) * 100 # 100株単位
                    if shares_to_trade == 0:
                        logging.debug(f"    {current_bar_index}: エントリーシグナル ({'Long' if signal_prev_bar == 1 else 'Short'}) だが株数0で見送り。RiskWidth(仮): {risk_width_per_share_example:.2f}, TargetRiskAmt:{target_risk_amount:,.0f}, EntryPrice:{entry_price_candidate:.2f}")
                    else:
                        trade_cost = entry_price_candidate * shares_to_trade
                        commission_entry = trade_cost * COMMISSION_RATE
                        required_capital_for_trade = trade_cost + commission_entry # Long の場合はこの額が必要

                        can_trade = True
                        # Short の場合の証拠金チェックは省略 (現物買いのみを想定)
                        if signal_prev_bar == 1 and current_capital < required_capital_for_trade :
                            logging.debug(f"    {current_bar_index}: Longエントリー資金不足。必要:{required_capital_for_trade:,.0f} > 現在:{current_capital:,.0f}")
                            can_trade = False

                        if can_trade:
                            current_position = int(signal_prev_bar)
                            current_entry_price = entry_price_candidate
                            current_shares = shares_to_trade
                            entry_datetime_obj = current_bar_index # エントリーしたバーのタイムスタンプ

                            current_capital -= commission_entry # エントリー手数料を資本から引く
                            logging.info(f"    {entry_datetime_obj}: エントリー {'L' if current_position==1 else 'S'}@P:{current_entry_price:.2f}, Shares:{current_shares}, Cost:{trade_cost:,.0f}, Comm:{commission_entry:,.0f}, Cap(after comm):{current_capital:,.0f}")

                # 資産評価 (Equity Curve の計算)
                unrealized_pnl = 0
                current_bar_close_price = df_with_signals['Close'].iloc[i] # 現在のバーの終値で評価
                if current_position == 1: unrealized_pnl = current_shares * (current_bar_close_price - current_entry_price)
                elif current_position == -1: unrealized_pnl = current_shares * (current_entry_price - current_bar_close_price)
                current_equity = current_capital + unrealized_pnl
                equity_curve_for_stock.append(current_equity)
            # --- ループ終了 ---

            # データ最終日でポジションが残っている場合、強制決済 (backtest_fw.py から移植)
            # ループが len(df_with_signals) - 1 までなので、最終行 df_with_signals.iloc[-1] の始値で決済
            if current_position != 0 and len(df_with_signals) > 0 : # df_with_signals が空でないことを確認
                final_bar_index = df_with_signals.index[-1] # 最後のバーのインデックス
                final_exit_price = df_with_signals['Open'].iloc[-1] # 最後のバーの始値で決済
                exit_reason_final = "データ終了強制決済(最終足始値)"

                pnl_per_share_final = (final_exit_price - current_entry_price) if current_position == 1 else (current_entry_price - final_exit_price)
                gross_pnl_final = current_shares * pnl_per_share_final
                commission_final = current_shares * final_exit_price * COMMISSION_RATE # 決済手数料
                net_pnl_final = gross_pnl_final - commission_final
                current_capital += net_pnl_final # 確定損益を資本に反映

                trade_history_for_stock.append({
                    '銘柄コード': stock_code_val, 'entry_date': entry_datetime_obj,
                    'exit_date': final_bar_index, # 決済が成立した足のタイムスタンプ
                    'type': 'Long' if current_position == 1 else 'Short',
                    'entry_price': current_entry_price, 'exit_price': final_exit_price,
                    'shares': current_shares, 'pnl': net_pnl_final, 'exit_type': exit_reason_final
                })
                logging.info(f"    {final_bar_index}: {exit_reason_final} {'L' if current_position==1 else 'S'}@E:{current_entry_price:.2f} X:{final_exit_price:.2f}, PnL:{net_pnl_final:,.0f}, Shares:{current_shares}, Cap:{current_capital:,.0f}")
                # 最終決済後のエクイティも記録
                equity_curve_for_stock.append(current_capital) # ポジションクローズ後の確定資本


            # --- パフォーマンス計算 (backtest_fw.py から移植) ---
            final_equity_for_stock = current_capital # 最終的な確定資本
            logging.info(f"  {progress_str}: バックテスト完了。最終確定資産: {final_equity_for_stock:,.0f} 円")

            trade_log_df_for_stock = pd.DataFrame(trade_history_for_stock)
            s_total_trades=0; s_win_rate=0.0; s_total_pnl=0.0; s_pf=0.0; s_avg_win=0.0; s_avg_loss=0.0; s_max_dd=np.nan

            if not trade_log_df_for_stock.empty:
                s_total_trades = len(trade_log_df_for_stock)
                winning_trades_df = trade_log_df_for_stock[trade_log_df_for_stock['pnl'] > 0]
                losing_trades_df = trade_log_df_for_stock[trade_log_df_for_stock['pnl'] <= 0] # 0も負けに含む
                num_winning_trades = len(winning_trades_df)
                num_losing_trades = len(losing_trades_df)

                s_win_rate = (num_winning_trades / s_total_trades) * 100 if s_total_trades > 0 else 0.0
                s_total_pnl = trade_log_df_for_stock['pnl'].sum()
                gross_profit = winning_trades_df['pnl'].sum()
                gross_loss = abs(losing_trades_df['pnl'].sum())
                s_pf = gross_profit / gross_loss if gross_loss > 0 else np.inf # 負けがない場合は無限大
                s_avg_win = gross_profit / num_winning_trades if num_winning_trades > 0 else 0.0
                s_avg_loss = gross_loss / num_losing_trades if num_losing_trades > 0 else 0.0 # 平均損失は正の値で

                if len(equity_curve_for_stock) > 1:
                    equity_series = pd.Series(equity_curve_for_stock)
                    peak = equity_series.cummax()
                    drawdown = (equity_series - peak) / peak
                    # drawdown の計算で peak が0の場合や equity_series が全て同じ値の場合の NaN/inf を避ける
                    if drawdown.notna().any() and (INITIAL_CAPITAL > 0 and not peak.empty and peak.iloc[0] != 0):
                        s_max_dd = abs(drawdown.min() * 100) if drawdown.min() < 0 else 0.0
                    else:
                        s_max_dd = 0.0 # ドローダウンが発生しなかった、または計算不能
                else:
                    s_max_dd = 0.0 # トレードが1回以下ではドローダウンは0

                log_perf_summary(stock_code_val, df_with_signals, final_equity_for_stock, s_total_pnl, s_max_dd,
                                 s_total_trades, s_win_rate, s_pf, s_avg_win, s_avg_loss)
                results_summary_list.append({
                    '銘柄コード': stock_code_val, '総損益': s_total_pnl, 'PF': s_pf,
                    '勝率(%)': s_win_rate, 'トレード数': s_total_trades, '最大DD(%)': s_max_dd,
                    '平均利益': s_avg_win, '平均損失': s_avg_loss, 'エラー': None
                })
                if not trade_log_df_for_stock.empty:
                    all_trades_log_list.append(trade_log_df_for_stock)
            else:
                logging.info(f"  {progress_str}: トレード実行なし。")
                results_summary_list.append({
                    '銘柄コード': stock_code_val, '総損益': 0, 'PF': 0, '勝率(%)': 0,
                    'トレード数': 0, '最大DD(%)': 0.0, '平均利益': 0, '平均損失': 0, 'エラー': 'トレードなし'
                })
            print("OK")

            # --- チャート出力 (chart.py の関数呼び出し) ---
            if output_chart_flag:
                if not target_chart_codes_list or stock_code_val in target_chart_codes_list:
                    chart_base_filename_parts = [
                        strategy_module_name,
                        f"{interval_exec_str_format}{interval_context_str_format}",
                        date_str,
                        stock_code_val
                    ]
                    logging.info(f"  {progress_str}: チャート出力処理を開始します。")
                    # strategy_module.STRATEGY_PRIMARY_PARAMS (config.yamlからロードされた辞書) を渡す
                    chart.plot_chart_for_stock(
                        df_context_ohlc,        # 元の環境足OHLC
                        df_exec_ohlc,           # 元の実行足OHLC
                        df_with_signals,        # シグナルと指標が含まれるDF (実行足ベース)
                        trade_log_df_for_stock, # この銘柄のトレード履歴
                        stock_code_val,
                        strategy_module.STRATEGY_PRIMARY_PARAMS, # config.yaml の内容 (フラット化された辞書)
                        CHART_OUTPUT_DIR,
                        chart_base_filename_parts
                    )
                    logging.info(f"  {progress_str}: チャートを出力しました。")
                else:
                    logging.info(f"  {progress_str}: チャート出力対象外のためスキップ ({stock_code_val} not in {target_chart_codes_list})。")

        except FileNotFoundError as fnf_stock:
            logging.error(f"  {progress_str}: ファイル未検出: {fnf_stock}。スキップ。")
            results_summary_list.append({ '銘柄コード': stock_code_val, 'エラー': f'ファイルなし: {os.path.basename(str(fnf_stock))}' })
            print(f"NG (FNF: {os.path.basename(str(fnf_stock))})")
        except ValueError as ve_stock:
            logging.error(f"  {progress_str}: データ/設定エラー: {ve_stock}。スキップ。")
            results_summary_list.append({ '銘柄コード': stock_code_val, 'エラー': f'データ/設定エラー: {ve_stock}' })
            print(f"NG (ValueErr)") # エラーメッセージが長くなる可能性があるのでValueErrのみ表示
        except Exception as e_stock:
            logging.error(f"  {progress_str}: 予期せぬエラー ({type(e_stock).__name__}: {e_stock})。スキップ。", exc_info=True)
            results_summary_list.append({ '銘柄コード': stock_code_val, 'エラー': f'予期せぬエラー: {type(e_stock).__name__}'})
            print(f"NG (エラー: {type(e_stock).__name__})")
        logging.info(f"  --- {progress_str} のバックテスト処理終了 ---")

    # --- 全体結果の出力 (backtest_fw.py から移植) ---
    print(f"\n===== 全 {total_stocks_processed_count} 銘柄のバックテストが完了しました =====")
    logging.info(f"\n===== 全 {total_stocks_processed_count} 銘柄のバックテストが完了しました =====")
    interval_combination_str = f"{interval_exec_str_format}{interval_context_str_format}"

    if all_trades_log_list:
        all_trades_combined_df = pd.concat(all_trades_log_list, ignore_index=True)
        detail_output_filename = f"BacktestDetail_{strategy_module_name}_{interval_combination_str}_{date_str}_{now_str_for_files}.csv"
        detail_output_filepath = os.path.join(RESULTS_DIR, detail_output_filename)
        # カラム順序を定義
        detail_columns_order = ['銘柄コード','entry_date', 'exit_date', 'type', 'entry_price', 'exit_price', 'shares', 'pnl', 'exit_type']
        # 存在するカラムのみで順序を適用
        detail_columns_present = [col for col in detail_columns_order if col in all_trades_combined_df.columns]
        try:
            all_trades_combined_df[detail_columns_present].to_csv(detail_output_filepath, sep=',', index=False, encoding='utf-8-sig', float_format='%.2f')
            logging.info(f"詳細トレード履歴CSV出力完了: {detail_output_filepath}")
            print(f"\n詳細トレード履歴CSV出力: {detail_output_filepath}")
        except Exception as e_csv_detail:
            logging.error(f"エラー: 詳細トレード履歴CSV出力失敗: {e_csv_detail}")
            print(f"エラー: 詳細トレード履歴CSV出力失敗: {e_csv_detail}", file=sys.stderr)
    else:
        logging.warning("詳細トレード履歴なし。CSV未作成。")
        print("詳細トレード履歴なし。CSV未作成。")

    if results_summary_list:
        summary_df = pd.DataFrame(results_summary_list)
        summary_output_filename = f"BacktestSummary_{strategy_module_name}_{interval_combination_str}_{date_str}_{now_str_for_files}.csv"
        summary_output_filepath = os.path.join(RESULTS_DIR, summary_output_filename)
        # サマリーの期待されるカラムと順序
        expected_summary_cols = ['銘柄コード', '総損益', 'PF', '勝率(%)', 'トレード数', '最大DD(%)', '平均利益', '平均損失', 'エラー']
        # DataFrameに存在しない期待カラムがあればNaNで追加
        for col in expected_summary_cols:
            if col not in summary_df.columns:
                summary_df[col] = np.nan
        summary_df = summary_df[expected_summary_cols] # カラム順序を整える

        # 数値カラムのフォーマット (backtest_fw.py から移植)
        numeric_cols_to_format = ['総損益', 'PF', '勝率(%)', '最大DD(%)', '平均利益', '平均損失']
        for col in numeric_cols_to_format:
            if col in summary_df.columns:
                summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce') # まず数値型に

        if 'PF' in summary_df.columns: summary_df['PF'] = summary_df['PF'].replace([np.inf, -np.inf], 'inf') # inf を文字列 'inf' に

        # 表示用のフォーマット (文字列化するのでCSV保存直前に行う)
        # applymap や apply を使って条件分岐でフォーマット
        if '総損益' in summary_df.columns: summary_df['総損益'] = summary_df['総損益'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A')
        if 'PF' in summary_df.columns: summary_df['PF'] = summary_df['PF'].apply(lambda x: f"{x:.2f}" if pd.notna(x) and x != 'inf' else ('inf' if x == 'inf' else 'N/A'))
        if '勝率(%)' in summary_df.columns: summary_df['勝率(%)'] = summary_df['勝率(%)'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A')
        if '最大DD(%)' in summary_df.columns: summary_df['最大DD(%)'] = summary_df['最大DD(%)'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A')
        if '平均利益' in summary_df.columns: summary_df['平均利益'] = summary_df['平均利益'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A')
        if '平均損失' in summary_df.columns: summary_df['平均損失'] = summary_df['平均損失'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A') # 正の値で表示

        summary_df = summary_df.fillna('N/A') # 残りのNaNも'N/A'で埋める

        try:
            summary_df.to_csv(summary_output_filepath, sep=',', index=False, encoding='utf-8-sig')
            logging.info(f"サマリーCSV出力完了: {summary_output_filepath}")
            print(f"サマリーCSV出力: {summary_output_filepath}")
        except Exception as e_csv_summary:
            logging.error(f"エラー: サマリーCSV出力失敗: {e_csv_summary}")
            print(f"エラー: サマリーCSV出力失敗: {e_csv_summary}", file=sys.stderr)
    else:
        logging.warning("結果サマリーなし。CSV未作成。")
        print("結果サマリーなし。CSV未作成。")

    logging.info("バックテストスクリプト処理完了。")
    print("\nスクリプト処理完了。ログファイルを確認してください。")

# このファイルが直接実行されることは想定しないが、テスト用に残しても良い
# if __name__ == "__main__":
#     # テスト呼び出しの例
#     run_backtest(
#         strategy_module_name='strategy',
#         exec_interval_min=5,
#         context_interval_min=60,
#         date_str='20230104', # 仮の日付
#         output_chart_flag=True,
#         target_chart_codes_list=['1332'] # 仮の銘柄
#     )