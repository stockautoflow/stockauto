# === スクリプトの使い方 ===
"""
Yahoo Finance API を使用して日経225構成銘柄の株価データを取得し、
日付と足種ごとにCSVファイルとして保存するスクリプト。

【前提】
  - スクリプトと同じディレクトリに `nikkei_codes.txt` が必要です。
    (日経225の4桁銘柄コードが1行に1つずつ記載されたUTF-8テキストファイル)
  - 必要なライブラリ (`requests`, `pandas`, `numpy`) がインストールされていること。
    (例: pip install requests pandas numpy)

【実行方法】
  1. 引数なしで実行: 定義された全ての足種 (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo) を取得します。
     例: python DataFetch_Arg.py

  2. 引数ありで実行: 指定した足種のみを取得します（複数指定可）。
     例: python DataFetch_Arg.py 1 5      (1分足と5分足を取得)
     例: python DataFetch_Arg.py 60 D     (60分足と日足を取得)

【指定可能な足種引数 (大文字/小文字区別なし)】
  分足: 1, 2, 5, 15, 30, 60, 90
  時間足: H (または 1H)
  日足: D (または 1D)
  週足: W (または 1W, 1WK)
  月足: M (または 1MO)
  3ヶ月足: 3MO (または 3M)
  (注意: 2m, 90m, 3mo はYahoo Financeで正式にサポートされていない可能性があります)

【出力】
  - ログ: ./DataFetch/DataFetchLog_yyyymmdd_hhmmss.log
  - CSVデータ: ./data_yyyymmdd/銘柄コード_足種_yyyymmdd.csv
"""

# === ライブラリのインポート ===
print("--- ライブラリインポート ---")
import pandas as pd
import numpy as np
import datetime
# import talib # 不要
# import matplotlib.pyplot as plt # 不要
import math
import os
import warnings
import logging
import requests # Yahoo Finance API アクセスに必要
import time
import sys

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


# === グローバル設定 ===
print("--- グローバル設定 ---")
BASE_DIR = '.'
STOCK_CODE_LIST_FILE = "nikkei_codes.txt"
CSV_OUTPUT_BASE_DIR = "data"
LOG_OUTPUT_DIR = os.path.join(BASE_DIR, "DataFetch") # ログ出力先フォルダ名

# Yahoo Finance API用パラメータ
interval_map = {
    "1": "1m", "2": "2m", "5": "5m", "15": "15m", "30": "30m",
    "60": "60m", "90": "90m", "H": "1h", "1H": "1h", "D": "1d", "1D": "1d",
    "W": "1wk", "1W": "1wk", "1WK": "1wk", "M": "1mo", "1MO": "1mo",
    "3M": "3mo", "3MO": "3mo"
}
range_map = {
    "1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d", "30m": "60d",
    "60m": "730d", "90m": "730d", "1h": "730d",
    "1d": "max", "1wk": "max", "1mo": "max", "3mo": "max"
}
DEFAULT_INTERVALS_API = list(interval_map.values())

# --- Rate Limit 対策パラメータ ---
SLEEP_TIME = 3.0
MAX_RETRIES = 3
RETRY_DELAY = 5

# === ステップ0: 引数処理と取得対象インターバルの決定 ===
print("--- ステップ0: 引数処理 ---")
intervals_to_fetch_api = []
intervals_to_fetch_user = []
process_all_intervals = False

if len(sys.argv) == 1:
    process_all_intervals = True
    intervals_to_fetch_api = DEFAULT_INTERVALS_API
    api_to_user_map = {}; processed_values = set()
    for k, v in interval_map.items():
        if v not in processed_values: api_to_user_map[v] = k; processed_values.add(v)
    for api_interval in intervals_to_fetch_api:
        user_display = api_to_user_map.get(api_interval, api_interval)
        if user_display.isdigit(): user_display += 'm'
        if user_display not in intervals_to_fetch_user: intervals_to_fetch_user.append(user_display)
    print(f"引数がないため、定義された全ての足種を取得します: {', '.join(intervals_to_fetch_user)}")
else:
    valid_args = True; user_intervals_from_args = []; api_intervals_from_args = []
    for arg in sys.argv[1:]:
        interval_key = arg.upper()
        if interval_key in interval_map:
            api_interval = interval_map[interval_key]
            user_display = arg + ('m' if arg.isdigit() else '')
            if api_interval not in api_intervals_from_args:
                 api_intervals_from_args.append(api_interval)
                 intervals_to_fetch_user.append(user_display)
        else: print(f"エラー: 引数 '{arg}' は無効な足種です。"); valid_args = False
    if not valid_args or not api_intervals_from_args:
        allowed_intervals_display = ", ".join(interval_map.keys()) + " (分:数字, H:時, D:日, W:週, M:月, 3M:3ヶ月)"; print(f"エラー: 有効な足種引数が指定されていません。"); print(f"指定可能なInterval: {allowed_intervals_display}"); exit()
    intervals_to_fetch_api = api_intervals_from_args
    print(f"指定された足種を取得します: {', '.join(intervals_to_fetch_user)}")


# === ログ・データフォルダ作成 ===
today_str = datetime.datetime.now().strftime('%Y%m%d')
os.makedirs(LOG_OUTPUT_DIR, exist_ok=True)
now_str_log = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"DataFetchLog_{now_str_log}.log"
log_filepath = os.path.join(LOG_OUTPUT_DIR, log_filename)
csv_output_dir = os.path.join(BASE_DIR, f"{CSV_OUTPUT_BASE_DIR}_{today_str}")
os.makedirs(csv_output_dir, exist_ok=True)

# --- ログ設定 (ファイルハンドラのみ) ---
for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO,format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(log_filepath, encoding='utf-8')])
print(f"ログファイル: {log_filepath}"); print(f"CSV保存先フォルダ: {os.path.abspath(csv_output_dir)}")
logging.info(f"=== データ取得スクリプト開始 (取得対象: {', '.join(intervals_to_fetch_user)}) ==="); logging.info(f"ログファイル: {log_filepath}"); logging.info(f"CSV保存先フォルダ: {os.path.abspath(csv_output_dir)}")


# === ヘルパー関数 ===
def get_codes_from_file(filepath):
    """テキストファイルから銘柄コード読み込み"""
    logging.info(f"銘柄コードリストをファイルから取得中: {filepath}")
    print(f"銘柄コードリストをファイルから取得中: {filepath}")
    # ★★★ 修正箇所: codes = [] を try ブロックの中に移動 ★★★
    try:
        codes = [] # tryブロックの開始直後で初期化
        if not os.path.exists(filepath): raise FileNotFoundError(f"銘柄コードファイルが見つかりません: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f: codes = [line.strip() for line in f if line.strip().isdigit() and len(line.strip()) == 4]
        logging.info(f"ファイルから {len(codes)} 個の4桁コードを読み込みました。"); print(f"ファイルから {len(codes)} 個の4桁コードを読み込みました。")
        if not codes: logging.error("有効な銘柄コード読込不可。"); return []
        return sorted(list(set(codes)))
    except FileNotFoundError as fnf: logging.error(fnf); print(f"エラー: {fnf}"); return []
    except Exception as e: logging.error(f"銘柄コードファイル読込エラー: {e}"); print(f"銘柄コードファイル読込エラー: {e}"); return []

def fetch_yahoo_data(ticker: str, range_str: str, interval_str: str) -> pd.DataFrame | None:
    """Yahoo Financeからデータを取得・整形する関数 (Rate Limit対策強化版)"""
    logging.info(f"  銘柄 {ticker} | 期間 {range_str} | 間隔 {interval_str} | データ取得開始...")
    base_url = 'https://query1.finance.yahoo.com/v7/finance/chart/'; url = f"{base_url}{ticker}?range={range_str}&interval={interval_str}&indicators=quote&includeTimestamps=true"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    df = None
    for attempt in range(MAX_RETRIES):
        response = None
        try:
            response = requests.get(url, headers=headers, timeout=15); response.raise_for_status(); data = response.json()
            chart_data = data.get('chart', {}); result_list = chart_data.get('result', [])
            if not result_list or result_list[0] is None: logging.warning(f"    -> 銘柄 {ticker} ({interval_str}) データ構造無し"); return None
            result = result_list[0]; timestamps = result.get('timestamp'); indicators = result.get('indicators', {}).get('quote', [{}])[0]
            if not indicators: logging.warning(f"    -> 銘柄 {ticker} ({interval_str}) OHLCVデータ無し"); return None
            opens=indicators.get('open'); highs=indicators.get('high'); lows=indicators.get('low'); closes=indicators.get('close'); volumes=indicators.get('volume')
            if not all([isinstance(l, list) for l in [timestamps, opens, highs, lows, closes, volumes]]): logging.warning(f"    -> 銘柄 {ticker} ({interval_str}) 必要要素不足"); return None
            if not timestamps: logging.warning(f"    -> 銘柄 {ticker} ({interval_str}) タイムスタンプ空"); return None
            expected_length = len(timestamps); lists_to_check = [opens, highs, lows, closes, volumes]
            if not all(isinstance(lst, list) and len(lst) == expected_length for lst in lists_to_check if lst is not None): logging.warning(f"    -> 銘柄 {ticker} ({interval_str}) データ長不一致またはNone混入"); return None
            if expected_length == 0: logging.warning(f"    -> 銘柄 {ticker} ({interval_str}) データ件数0"); return None
            logging.info(f"    -> データ取得成功 ({expected_length}件)")
            df = pd.DataFrame({'Timestamp': timestamps,'Open': opens,'High': highs,'Low': lows,'Close': closes,'Volume': volumes})
            df['datetime'] = pd.to_datetime(df['Timestamp'], unit='s', utc=True); df = df.set_index(pd.DatetimeIndex(df['datetime'])); df = df.drop(columns=['Timestamp', 'datetime']); df = df.tz_convert('Asia/Tokyo'); df.index.name = 'datetime'
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']: df[col] = pd.to_numeric(df[col], errors='coerce')
            initial_len = len(df); df = df.dropna(); removed_rows = initial_len - len(df); df['Volume'] = df['Volume'].fillna(0).astype(np.int64)
            logging.info(f"    -> {ticker} ({interval_str}): DataFrame整形完了 (最終 {len(df)}件)")
            return df # 成功したらループ脱出
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                if attempt < MAX_RETRIES - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt); logging.warning(f"    -> HTTP 429 エラー ({ticker}, {interval_str})。{wait_time:.1f}秒待機してリトライ... ({attempt + 2}/{MAX_RETRIES})"); print(f"    -> Rate Limit (429). Waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    # continue は不要、ループの次へ自動で進む
                else:
                    # ★★★ 修正箇所: tryブロックを分ける ★★★
                    logging.error(f"    -> HTTP 429 エラー ({ticker}, {interval_str})。リトライ上限({MAX_RETRIES}回)到達。")
                    try: # response.textへのアクセスもエラーになる可能性があるためtryで囲む
                        logging.error(f"    -> Response Content:\n{response.text}")
                    except Exception as log_e:
                        logging.error(f"    -> Response Content取得/ログ出力エラー: {log_e}")
                    return None # リトライ失敗
            else:
                logging.error(f"    -> HTTPエラー ({ticker}, {interval_str}, Status: {e.response.status_code if e.response is not None else 'N/A'})")
                try:
                    if response is not None: logging.error(f"    -> Response Content:\n{response.text}")
                    else: logging.error(f"    -> Response object is None.")
                except Exception as log_e: logging.error(f"    -> Response Content取得エラー: {log_e}")
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f"    -> ネットワークエラー ({ticker}, {interval_str}): {e}")
            if attempt < MAX_RETRIES - 1: wait_time = RETRY_DELAY * (2 ** attempt); logging.warning(f"    -> ネットワークエラー。{wait_time:.1f}秒待機してリトライ... ({attempt + 2}/{MAX_RETRIES})"); print(f"    -> Network Error. Waiting {wait_time:.1f}s..."); time.sleep(wait_time)
            else: logging.error(f"    -> ネットワークエラー。リトライ上限({MAX_RETRIES}回)到達。"); return None
        except Exception as e: logging.error(f"    -> 予期せぬエラー ({ticker}, {interval_str}): {e}", exc_info=False); return None
    # リトライループ失敗
    logging.error(f"    -> 銘柄 {ticker} ({interval_str}) データ取得最終失敗。")
    return None


# === ステップ3: メイン処理ループ ===
logging.info("--- メイン処理開始 ---")
print("\n--- メイン処理開始 ---")

# 1. 銘柄リスト取得
stock_list_filepath = os.path.join(BASE_DIR, STOCK_CODE_LIST_FILE)
nikkei_codes = get_codes_from_file(stock_list_filepath)
if not nikkei_codes: logging.error(f"処理終了。"); print("エラー: 銘柄コード取得失敗。処理終了。"); exit()

# 2. データ保存フォルダ作成済み

# 3. 各銘柄のデータを取得・保存
total_codes_to_process = min(len(nikkei_codes), 225)
success_counts = {interval: 0 for interval in intervals_to_fetch_user}
failed_tickers = {interval: [] for interval in intervals_to_fetch_user}

logging.info(f"--- 全 {total_codes_to_process} 銘柄のループ処理開始 ---")
print(f"--- 全 {total_codes_to_process} 銘柄のループ処理開始 ---")

for i, code in enumerate(nikkei_codes[:total_codes_to_process]):
    logging.info(f"--- 銘柄 {i+1}/{total_codes_to_process}: {code} の処理開始 ---")
    print(f"Processing: {code} ({i+1}/{total_codes_to_process})... ", end="")
    yahoo_ticker = f"{code}.T"
    success_flags_per_stock = []

    # --- 指定された各足種についてループ ---
    for idx, interval_api in enumerate(intervals_to_fetch_api):
        interval_user = intervals_to_fetch_user[idx]
        current_range = range_map.get(interval_api, '1y')

        logging.info(f"  [{interval_user} データ]")
        df_current = fetch_yahoo_data(yahoo_ticker, current_range, interval_api)

        if df_current is not None and not df_current.empty:
            filename = f"{code}_{interval_user}_{today_str}.csv"
            filepath = os.path.join(csv_output_dir, filename)
            try:
                df_current.to_csv(filepath, index=True, encoding='utf_8_sig')
                logging.info(f"  -> {interval_user} データを保存しました: {filepath}")
                if interval_user in success_counts: success_counts[interval_user] += 1
                else: success_counts[interval_user] = 1
                success_flags_per_stock.append(True)
            except Exception as e:
                logging.error(f"  -> {interval_user} データのCSV保存中にエラー: {e}")
                if interval_user not in failed_tickers: failed_tickers[interval_user] = []
                failed_tickers[interval_user].append(yahoo_ticker)
                success_flags_per_stock.append(False)
        else:
            if interval_user not in failed_tickers: failed_tickers[interval_user] = []
            failed_tickers[interval_user].append(yahoo_ticker)
            success_flags_per_stock.append(False)

        # 足種間の短い待機は削除
        # if idx < len(intervals_to_fetch_api) - 1: time.sleep(0.1)

    # コンソールに進捗表示 (OK / NG)
    if all(success_flags_per_stock): print("OK")
    else: print("NG (一部失敗)")

    # --- 銘柄間のRate Limit対策待機 ---
    logging.info(f"  -> {SLEEP_TIME}秒待機...")
    time.sleep(SLEEP_TIME)

# === 処理完了メッセージ ===
logging.info("\n==============================")
logging.info("=== 全銘柄のデータ取得処理完了 ===")
logging.info(f"処理対象銘柄数: {total_codes_to_process}")
print("\n=============================="); print("=== 全銘柄のデータ取得処理完了 ==="); print(f"処理対象銘柄数: {total_codes_to_process}")
for interval_user_str, count in success_counts.items():
    logging.info(f"{interval_user_str} データ保存成功数: {count}"); print(f"{interval_user_str} データ保存成功数: {count}")
has_failures = any(len(lst) > 0 for lst in failed_tickers.values())
if has_failures:
    logging.warning("以下の銘柄/足種のデータ取得または保存に失敗しました:")
    for interval_user_str, tickers in failed_tickers.items():
        if tickers: log_limit = 20; logging.warning(f"  [{interval_user_str} failures ({len(tickers)}件)]:");
        for failed in tickers[:log_limit]: logging.warning(f"    - {failed}")
        if len(tickers) > log_limit: logging.warning(f"    ...他 {len(tickers) - log_limit} 件")
logging.info(f"ログファイル: {log_filepath}"); logging.info(f"CSV保存先フォルダ: {os.path.abspath(csv_output_dir)}")
print(f"\nスクリプト処理完了。詳細はログファイルを確認してください: {log_filepath}"); print(f"CSVファイルはフォルダ '{os.path.abspath(csv_output_dir)}' に保存されました。")