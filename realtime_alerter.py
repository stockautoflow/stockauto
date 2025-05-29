# realtime_alerter.py
import time
import logging
import os
import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import datetime
import pandas as pd
import numpy as np # <--- 追加
import schedule
import shutil
import tempfile
import glob # <--- 追加: ファイル一覧取得のため
import re   # <--- 追加: ファイル名からの銘柄コード抽出のため

# --- matplotlib バックエンド設定 ---
import matplotlib
matplotlib.use('Agg')
# ---------------------------------

import state_manager
import data_processor
import strategy
import alert_sender
import converter
import chart

# --- ロギング設定 ---
LOG_DIR = "log"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    print(f"ログディレクトリを作成しました: {os.path.abspath(LOG_DIR)}")

log_filename = time.strftime("realtime_alerter_log_%Y%m%d_%H%M%S.log")
log_filepath = os.path.join(LOG_DIR, log_filename)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d %(funcName)s) %(message)s',
    handlers=[
        logging.FileHandler(log_filepath, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# ---------------------------------------------------------

# --- 設定ファイルのパス定義 ---
REALTIME_CONFIG_FILENAME = "realtime_config.yaml"
STRATEGY_CONFIG_FILENAME = "config.yaml"

def load_yaml_config(filepath):
    """YAMLファイルを読み込んで辞書として返す"""
    logger.debug(f"設定ファイル読み込み試行: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            if not config_data:
                logger.warning(f"設定ファイル '{filepath}' が空、または有効なYAMLデータを含んでいません。")
                return {}
            logger.info(f"設定ファイル '{filepath}' を正常に読み込みました。")
            return config_data
    except FileNotFoundError:
        logger.error(f"エラー: 設定ファイル '{filepath}' が見つかりません。")
        return None
    except yaml.YAMLError as e:
        logger.error(f"エラー: 設定ファイル '{filepath}' のYAML解析に失敗しました: {e}")
        return None
    except Exception as e:
        logger.error(f"エラー: 設定ファイル '{filepath}' の読み込み中に予期せぬエラーが発生しました: {e}", exc_info=True)
        return None

# --- CSV変換ジョブ関数 ---
def excel_to_csv_conversion_job(excel_path, csv_output_path):
    """ExcelからCSVへの変換を実行するジョブ (コピーを作成して処理)"""
    logger.info(f"定期Excel→CSV変換ジョブ実行開始: 元ファイル '{excel_path}'")
    if not os.path.exists(excel_path):
        logger.error(f"指定されたExcelファイルが見つかりません: {excel_path}。今回のCSV変換をスキップします。")
        return
    if not ensure_directory_exists(csv_output_path):
        logger.error(f"CSV出力先ディレクトリ '{csv_output_path}' の準備に失敗。今回のCSV変換を中止します。")
        return

    temp_excel_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsm", prefix="excel_copy_") as tmp:
            temp_excel_file = tmp.name
        
        logger.debug(f"元Excelファイルを一時ファイルにコピー中: '{excel_path}' -> '{temp_excel_file}'")
        shutil.copy2(excel_path, temp_excel_file)
        logger.debug(f"コピー完了。一時Excelファイルを使用してCSV変換を実行: '{temp_excel_file}' -> '{csv_output_path}'")

        converter.create_csv_from_excel(temp_excel_file, csv_output_path) #
        logger.info(f"定期Excel→CSV変換ジョブ完了 (詳細はconverterのログを確認)。元ファイル: '{excel_path}'")

    except FileNotFoundError as e_fnf:
        logger.error(f"Excelファイルのコピー中にエラー（ファイルが見つからない等）: {e_fnf}", exc_info=True)
    except Exception as e_conv:
        logger.error(f"定期Excel→CSV変換ジョブ中にエラー: {e_conv}", exc_info=True)
    finally:
        if temp_excel_file and os.path.exists(temp_excel_file):
            try:
                os.remove(temp_excel_file)
                logger.debug(f"一時Excelファイルを削除しました: '{temp_excel_file}'")
            except Exception as e_remove:
                logger.error(f"一時Excelファイル '{temp_excel_file}' の削除中にエラー: {e_remove}")


# --- ★ 新規追加: データディレクトリから銘柄コードを抽出 ---
def get_stock_codes_from_data_directory(directory_path):
    """
    指定されたディレクトリ内のCSVファイル名から4桁の銘柄コードを抽出する。
    ファイル名は "XXXX_*.csv" (XXXXは4桁の数字) の形式を期待。
    """
    if not os.path.isdir(directory_path):
        logger.warning(f"銘柄コード抽出対象のディレクトリが見つかりません: {directory_path}")
        return []
    
    stock_codes = set()
    # パターンに合致するCSVファイルのみを検索
    # 例: 1234_1m_20230101.csv や 5678_ohlcv.csv など
    for filepath in glob.glob(os.path.join(directory_path, "*.csv")):
        filename = os.path.basename(filepath)
        # ファイル名の先頭が4桁の数字で、その後にアンダースコアが続くパターン
        match = re.match(r"(\d{4})_.*\.csv", filename)
        if match:
            stock_codes.add(match.group(1))
            
    if not stock_codes:
        logger.warning(f"ディレクトリ '{directory_path}' 内に処理対象となる株価ファイルが見つかりませんでした。(期待するファイル名形式: XXXX_*.csv)")
    
    sorted_codes = sorted(list(stock_codes))
    logger.info(f"ディレクトリ '{directory_path}' から {len(sorted_codes)} 個のユニークな銘柄コードを抽出しました: {sorted_codes}")
    return sorted_codes
# ---------------------------------------------------------


# --- ファイルイベントハンドラ ---
class PriceDataFileEventHandler(FileSystemEventHandler):
    # --- ★ 修正: コンストラクタに watched_stock_codes を追加 ---
    def __init__(self, realtime_config, strategy_config_params, dynamic_watched_stock_codes):
        super().__init__()
        self.realtime_config = realtime_config
        self.strategy_config = strategy_config_params
        # self.watched_stock_codes = realtime_config.get('watched_stock_codes', []) # 削除
        self.watched_stock_codes = dynamic_watched_stock_codes # ★ 起動時に生成されたリストを使用
        # ---------------------------------------------------------
        self.state_file_dir = realtime_config.get('state_file_directory')
        if not self.state_file_dir:
            logger.error("状態ファイルディレクトリが realtime_config に設定されていません。")
            self.state_file_dir = "default_runtime_state"
            ensure_directory_exists(self.state_file_dir)

        self.chart_output_on_alert = realtime_config.get('output_chart_on_alert', False) #
        self.chart_output_dir = realtime_config.get('chart_output_directory')          #
        if self.chart_output_on_alert and not self.chart_output_dir:
            logger.warning("アラート時チャート出力が有効ですが、出力先ディレクトリ(chart_output_directory)が未設定です。チャートは出力されません。")
            self.chart_output_on_alert = False
        elif self.chart_output_on_alert:
            ensure_directory_exists(self.chart_output_dir)


        logger.info(f"監視対象の銘柄コード (動的生成): {self.watched_stock_codes}") # ログメッセージ更新
        logger.info(f"状態ファイルディレクトリ: {os.path.abspath(self.state_file_dir)}")
        if self.chart_output_on_alert:
            logger.info(f"アラート時チャート出力先: {os.path.abspath(self.chart_output_dir)}")


        self.realtime_params_for_strategy = {
            'interval_exec': self.realtime_config.get('execution_interval_minutes', 1),
            'interval_context': self.realtime_config.get('context_interval_minutes', 5),
            'TRADING_HOURS_FORCE_EXIT_TIME_STR': self.strategy_config.get(
                'TRADING_HOURS_FORCE_EXIT_TIME_STR',
                self.strategy_config.get('TRADING_HOURS', {}).get('FORCE_EXIT_TIME_STR')
            )
        }
        if self.realtime_params_for_strategy['TRADING_HOURS_FORCE_EXIT_TIME_STR']:
            try:
                self.realtime_params_for_strategy['PARSED_FORCE_EXIT_TIME_OBJ'] = datetime.datetime.strptime(
                    self.realtime_params_for_strategy['TRADING_HOURS_FORCE_EXIT_TIME_STR'], '%H:%M:%S'
                ).time()
            except ValueError as e:
                logger.error(f"強制決済時刻のパース失敗: {e}。設定値: {self.realtime_params_for_strategy['TRADING_HOURS_FORCE_EXIT_TIME_STR']}")
                self.realtime_params_for_strategy['PARSED_FORCE_EXIT_TIME_OBJ'] = None
        else:
             self.realtime_params_for_strategy['PARSED_FORCE_EXIT_TIME_OBJ'] = None


    def on_modified(self, event):
        if event.is_directory:
            return

        filepath = event.src_path
        filename = os.path.basename(filepath)
        logger.debug(f"ファイル変更検知: {filepath}")

        # --- ★ 修正: ファイル名から銘柄コードを抽出し、watched_stock_codes でチェック ---
        stock_code_match = None
        if filepath.endswith(".csv"):
            # ファイル名から銘柄コードを抽出 (例: "1234_data.csv" -> "1234")
            match = re.match(r"(\d{4})_.*\.csv", filename)
            if match:
                extracted_code = match.group(1)
                if extracted_code in self.watched_stock_codes: # 動的に生成されたリストでチェック
                    stock_code_match = extracted_code
                else:
                    logger.debug(f"ファイル '{filename}' の銘柄コード '{extracted_code}' は監視対象外です。")
            else:
                logger.debug(f"ファイル '{filename}' は期待される命名規則 (XXXX_*.csv) に一致しません。")
        # --------------------------------------------------------------------------
        
        if stock_code_match:
            logger.info(f"処理対象ファイル更新: {filename} (銘柄コード: {stock_code_match})")
            self.process_stock_data(stock_code_match, filepath)
        else:
            logger.debug(f"無視するファイル変更 (CSVでないか、対象銘柄コードでない、または命名規則不一致): {filename}")

    def on_created(self, event):
        if event.is_directory:
            return
        logger.debug(f"ファイル作成検知: {event.src_path}")
        # ★ 新規ファイル作成時も同様に処理対象か判定する
        #   ここでは、新しい銘柄ファイルが追加されても、
        #   起動時にスキャンした self.watched_stock_codes に含まれていなければ処理されない。
        #   もし、実行中に新規追加された銘柄も処理したい場合は、
        #   watched_stock_codes を動的に更新する仕組みが必要。
        #   今回は起動時スキャンのみと仮定。
        self.on_modified(event) # on_modified と同じロジックで判定・処理


    def process_stock_data(self, stock_code, filepath):
        logger.info(f"--- 銘柄 {stock_code} のデータ処理開始 ({filepath}) ---")
        try:
            current_time_obj = datetime.datetime.now().time()
            current_datetime_for_filename = datetime.datetime.now()

            current_state = state_manager.load_state(stock_code, self.state_file_dir) #
            logger.debug(f"銘柄 {stock_code}: 状態ロード完了。ポジション: {current_state.get('position')}")

            df_exec_raw, df_context_raw, updated_state_after_data_prep = data_processor.load_and_prepare_data_for_strategy( #
                filepath,
                self.strategy_config,
                current_state.copy()
            )
            current_state = updated_state_after_data_prep

            if df_exec_raw is None or df_exec_raw.empty:
                logger.warning(f"銘柄 {stock_code}: 実行足の準備に失敗。処理を中断。")
                state_manager.save_state(stock_code, current_state, self.state_file_dir) #
                return
            
            if df_context_raw is None:
                logger.warning(f"銘柄 {stock_code}: 環境認識足の準備に失敗。空のDataFrameで代替します。")
                df_context_raw = pd.DataFrame()

            logger.info(f"銘柄 {stock_code}: 元実行足 {len(df_exec_raw)}本、元環境足 {len(df_context_raw)}本を準備完了。")

            indicators_df = strategy.calculate_indicators( #
                df_exec_raw.copy(),
                df_context_raw.copy(),
                self.realtime_params_for_strategy,
                self.strategy_config
            )

            df_with_signals_for_chart = pd.DataFrame() 
            if indicators_df is None or indicators_df.empty:
                logger.warning(f"銘柄 {stock_code}: 指標計算結果が空です。シグナル生成をスキップ。")
                latest_signal_val = 0
                df_with_signals_for_chart = df_exec_raw.copy() 
            else:
                signals_df = strategy.generate_signals( #
                    indicators_df.copy(),
                    self.realtime_params_for_strategy,
                    self.strategy_config
                )
                if not signals_df.empty and 'Signal' in signals_df.columns and not signals_df['Signal'].empty:
                    latest_signal_val = signals_df['Signal'].iloc[-1]
                else:
                    logger.warning(f"銘柄 {stock_code}: シグナル生成結果が空、またはSignalカラムが存在しません。")
                    latest_signal_val = 0
                df_with_signals_for_chart = signals_df.copy() 
            

            logger.info(f"銘柄 {stock_code}: シグナル判定結果 (from strategy.py): {latest_signal_val}")

            alert_needed = False
            alert_message_subject = ""
            alert_message_body = ""
            trade_event_for_chart = None
            new_position = current_state.get("position", "none")
            position_before_exit = current_state.get("position") 

            current_position_val = 0
            if new_position == "long": current_position_val = 1
            elif new_position == "short": current_position_val = -1

            if current_position_val != 0:
                if indicators_df is not None and not indicators_df.empty and len(indicators_df) > 0:
                    current_bar_data_for_exit = indicators_df.iloc[-1]
                    prev_bar_data_for_exit = indicators_df.iloc[-2] if len(indicators_df) >= 2 else pd.Series(dtype='object')
                    exit_now, exit_reason, _ = strategy.determine_exit_conditions( #
                        current_position_val, current_bar_data_for_exit, prev_bar_data_for_exit,
                        self.realtime_params_for_strategy, self.strategy_config,
                        current_state.get("entry_price"), current_time_obj
                    )
                    if exit_now:
                        alert_needed = True
                        new_position = "none"
                        exit_price_dummy = df_exec_raw['Close'].iloc[-1] if not df_exec_raw.empty else current_state.get("entry_price", "N/A")
                        alert_message_subject = f"【決済アラート】銘柄: {stock_code}"
                        alert_message_body = (
                            f"銘柄コード: {stock_code}\n"
                            f"シグナル: 決済 ({('ロング' if current_position_val == 1 else 'ショート')}ポジション)\n"
                            f"理由: {exit_reason}\n"
                            f"推定決済価格: 約{exit_price_dummy}\n"
                            f"時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        logger.info(alert_message_body)
                        trade_event_for_chart = {
                            'type': ('Long' if current_position_val == 1 else 'Short'), 
                            'entry_date': pd.to_datetime(current_state.get("entry_datetime"), errors='coerce'),
                            'entry_price': current_state.get("entry_price"),
                            'exit_date': df_exec_raw.index[-1] if not df_exec_raw.empty else pd.NaT,
                            'exit_price': exit_price_dummy,
                            'exit_type': exit_reason
                        }
                        # 決済時はエントリー情報をクリアする前に、チャート用に決済直前の情報を保持
                        current_state["position_before_exit_for_chart"] = position_before_exit
                        current_state["entry_price_before_exit_for_chart"] = current_state.get("entry_price")
                        current_state["entry_datetime_before_exit_for_chart"] = current_state.get("entry_datetime")
                        
                        current_state["entry_price"] = None
                        current_state["entry_datetime"] = None
                else:
                    logger.warning(f"銘柄 {stock_code}: ポジション保有中だが、決済条件判定のための指標データが不足。")

            if not alert_needed:
                if current_state.get("position") == "none":
                    entry_price_for_chart = df_exec_raw['Close'].iloc[-1] if not df_exec_raw.empty else None
                    entry_datetime_for_chart = df_exec_raw.index[-1] if not df_exec_raw.empty else pd.NaT
                    entry_datetime_for_state = entry_datetime_for_chart.isoformat() if pd.notna(entry_datetime_for_chart) else datetime.datetime.now().isoformat()
                    
                    signal_type_str = ""
                    trade_event_for_chart_type = ""
                    if latest_signal_val == 1:
                        signal_type_str = "新規買い"
                        new_position = "long"
                        trade_event_for_chart_type = "Long"
                    elif latest_signal_val == -1:
                        signal_type_str = "新規売り"
                        new_position = "short"
                        trade_event_for_chart_type = "Short"

                    if signal_type_str:
                        alert_needed = True
                        alert_message_subject = f"【{signal_type_str}アラート】銘柄: {stock_code}"
                        alert_message_body = (
                            f"銘柄コード: {stock_code}\n"
                            f"シグナル: {signal_type_str}\n"
                            f"推定エントリー価格: 約{entry_price_for_chart}\n"
                            f"時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        trade_event_for_chart = {
                            'type': trade_event_for_chart_type,
                            'entry_date': entry_datetime_for_chart,
                            'entry_price': entry_price_for_chart,
                            'exit_date': pd.NaT,
                            'exit_price': np.nan,
                            'exit_type': 'N/A_Open'
                        }
                        current_state["entry_price"] = entry_price_for_chart
                        current_state["entry_datetime"] = entry_datetime_for_state
            
            if new_position != current_state.get("position"):
                if new_position == "none" and position_before_exit != "none":
                    pass # 決済時のエントリー情報はtrade_event_for_chart生成時に利用済み
                logger.info(f"銘柄 {stock_code}: ポジション変更 {current_state.get('position')} -> {new_position}")
                current_state["position"] = new_position

            current_state["last_signal_generated"] = float(latest_signal_val)

            if alert_needed:
                logger.info(f"銘柄 {stock_code}: アラート送信実行！ メッセージ: {alert_message_body}")
                current_state["last_alert_datetime"] = datetime.datetime.now().isoformat()
                
                email_sent_successfully = alert_sender.send_email( #
                    alert_message_subject,
                    alert_message_body
                )
                if not email_sent_successfully:
                    logger.error(f"銘柄 {stock_code}: メール送信に失敗しました。詳細はalert_senderのログを確認してください。")

                if self.chart_output_on_alert and self.chart_output_dir:
                    logger.info(f"銘柄 {stock_code}: アラート発生のためチャート出力を試みます。")
                    try:
                        trade_history_for_chart_list = []
                        if trade_event_for_chart:
                            processed_event_for_chart = {
                                '銘柄コード': stock_code,
                                'type': trade_event_for_chart.get('type'),
                                'entry_date': pd.to_datetime(trade_event_for_chart.get('entry_date'), errors='coerce'),
                                'entry_price': trade_event_for_chart.get('entry_price'),
                                'exit_date': pd.to_datetime(trade_event_for_chart.get('exit_date'), errors='coerce'),
                                'exit_price': trade_event_for_chart.get('exit_price'),
                                'shares': np.nan,
                                'pnl': np.nan,
                                'exit_type': trade_event_for_chart.get('exit_type')
                            }
                            # 決済イベントの場合、元のエントリー情報を state から取得して追加（チャート表示用）
                            if trade_event_for_chart.get('type') != 'Long' and trade_event_for_chart.get('type') != 'Short' and trade_event_for_chart.get('exit_type'): # 決済イベントの場合
                                processed_event_for_chart['entry_date'] = pd.to_datetime(current_state.get("entry_datetime_before_exit_for_chart"), errors='coerce')
                                processed_event_for_chart['entry_price'] = current_state.get("entry_price_before_exit_for_chart")
                                # 元のポジションタイプも復元
                                processed_event_for_chart['type'] = current_state.get("position_before_exit_for_chart")

                            trade_history_for_chart_list.append(processed_event_for_chart)
                        
                        trade_history_df_for_chart = pd.DataFrame(trade_history_for_chart_list)
                        
                        tz_to_align = df_exec_raw.index.tz
                        if tz_to_align:
                            if 'entry_date' in trade_history_df_for_chart.columns and pd.api.types.is_datetime64_any_dtype(trade_history_df_for_chart['entry_date']):
                                trade_history_df_for_chart['entry_date'] = trade_history_df_for_chart['entry_date'].apply(
                                    lambda x: x.tz_localize(None).tz_localize(tz_to_align, ambiguous='infer', nonexistent='NaT') if pd.notna(x) and x.tzinfo is None else (x.tz_convert(tz_to_align) if pd.notna(x) and x.tzinfo is not None and x.tzinfo != tz_to_align else x)
                                )
                            if 'exit_date' in trade_history_df_for_chart.columns and pd.api.types.is_datetime64_any_dtype(trade_history_df_for_chart['exit_date']):
                                trade_history_df_for_chart['exit_date'] = trade_history_df_for_chart['exit_date'].apply(
                                    lambda x: x.tz_localize(None).tz_localize(tz_to_align, ambiguous='infer', nonexistent='NaT') if pd.notna(x) and x.tzinfo is None else (x.tz_convert(tz_to_align) if pd.notna(x) and x.tzinfo is not None and x.tzinfo != tz_to_align else x)
                                )


                        strategy_name_for_chart = os.path.splitext(os.path.basename(STRATEGY_CONFIG_FILENAME))[0]
                        interval_exec_str = f"{self.realtime_params_for_strategy['interval_exec']}m"
                        interval_context_str = f"{self.realtime_params_for_strategy['interval_context']}m"
                        
                        base_filename_parts = [
                            strategy_name_for_chart,
                            f"{interval_exec_str}{interval_context_str}",
                            current_datetime_for_filename.strftime('%Y%m%d%H%M%S'),
                            stock_code
                        ]

                        chart.plot_chart_for_stock( #
                            df_context_raw if df_context_raw is not None and not df_context_raw.empty else pd.DataFrame(),
                            df_exec_raw if df_exec_raw is not None and not df_exec_raw.empty else pd.DataFrame(),
                            df_with_signals_for_chart if df_with_signals_for_chart is not None and not df_with_signals_for_chart.empty else pd.DataFrame(),
                            trade_history_df_for_chart,
                            stock_code,
                            self.strategy_config,
                            self.chart_output_dir,
                            base_filename_parts
                        )
                        logger.info(f"銘柄 {stock_code}: チャートを出力しました。")
                    except Exception as e_chart:
                        logger.error(f"銘柄 {stock_code}: チャート出力中にエラー: {e_chart}", exc_info=True)
            else:
                logger.info(f"銘柄 {stock_code}: アラート条件に合致せず。")

            # 決済時に保存した一時的な情報をクリーンアップ (任意)
            current_state.pop("position_before_exit_for_chart", None)
            current_state.pop("entry_price_before_exit_for_chart", None)
            current_state.pop("entry_datetime_before_exit_for_chart", None)

            current_state['stock_code'] = stock_code
            state_manager.save_state(stock_code, current_state, self.state_file_dir) #
            logger.debug(f"銘柄 {stock_code}: 状態保存完了。")

            logger.info(f"--- 銘柄 {stock_code} のデータ処理完了 ---")

        except Exception as e:
            logger.error(f"銘柄 {stock_code} の処理中にエラーが発生しました: {e}", exc_info=True)


def ensure_directory_exists(dir_path):
    """指定されたディレクトリが存在しない場合、作成する"""
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            logger.info(f"ディレクトリを作成しました: {os.path.abspath(dir_path)}")
        except OSError as e:
            logger.error(f"ディレクトリ '{dir_path}' の作成に失敗しました: {e}")
            return False
    return True

def main():
    logger.info("リアルタイム株価アラートシステムを開始します...")

    realtime_config = load_yaml_config(REALTIME_CONFIG_FILENAME) #
    strategy_config_params = strategy.load_strategy_config_yaml(STRATEGY_CONFIG_FILENAME) #


    if realtime_config is None:
        logger.fatal(f"リアルタイム設定ファイル '{REALTIME_CONFIG_FILENAME}' の読み込みに失敗しました。処理を終了します。")
        return
    if strategy_config_params is None:
        logger.fatal(f"戦略設定ファイル '{STRATEGY_CONFIG_FILENAME}' の読み込みに失敗しました。処理を終了します。")
        return
    if not realtime_config:
        logger.fatal(f"リアルタイム設定ファイル '{REALTIME_CONFIG_FILENAME}' が空か無効です。処理を終了します。")
        return
    if not strategy_config_params:
        logger.fatal(f"戦略設定ファイル '{STRATEGY_CONFIG_FILENAME}' が空か、ロード後に有効なパラメータがありません。処理を終了します。")
        return

    excel_file_for_periodic_conversion = realtime_config.get('excel_file_path') #
    csv_output_dir_for_periodic_conversion = realtime_config.get('csv_output_directory_for_converter') #
    conversion_interval_min = realtime_config.get('excel_conversion_interval_minutes', 1) #

    if realtime_config.get('enable_periodic_excel_conversion', False): #
        if excel_file_for_periodic_conversion and csv_output_dir_for_periodic_conversion:
            logger.info(
                f"定期Excel→CSV変換を {conversion_interval_min} 分ごとにスケジュールします: "
                f"'{excel_file_for_periodic_conversion}' -> '{csv_output_dir_for_periodic_conversion}'"
            )
            
            schedule.every(conversion_interval_min).minutes.do(
                excel_to_csv_conversion_job,
                excel_path=excel_file_for_periodic_conversion,
                csv_output_path=csv_output_dir_for_periodic_conversion
            )
        else:
            logger.warning(
                "定期Excel→CSV変換が有効ですが、設定（excel_file_path または csv_output_directory_for_converter）が"
                "不十分なためスケジュールされません。"
            )
    else:
        logger.info("定期Excel→CSV変換は無効に設定されています。")

    data_dir_to_watch = realtime_config.get('data_directory_to_watch') #
    state_file_dir = realtime_config.get('state_file_directory') #

    if not data_dir_to_watch:
        logger.error(f"設定ファイル '{REALTIME_CONFIG_FILENAME}' に 'data_directory_to_watch' が指定されていません。処理を終了します。")
        return
    if not state_file_dir:
        logger.error(f"設定ファイル '{REALTIME_CONFIG_FILENAME}' に 'state_file_directory' が指定されていません。処理を終了します。")
        return

    if not ensure_directory_exists(data_dir_to_watch):
        logger.error(f"監視対象ディレクトリ '{data_dir_to_watch}' の準備に失敗しました。処理を終了します。")
        return
    if not ensure_directory_exists(state_file_dir):
        logger.error(f"状態ファイル保存ディレクトリ '{state_file_dir}' の準備に失敗しました。処理を終了します。")
        return

    logger.info(f"監視対象ディレクトリ: {os.path.abspath(data_dir_to_watch)}")
    logger.info(f"状態ファイル保存ディレクトリ: {os.path.abspath(state_file_dir)}")

    # --- ★ 修正: 起動時に監視対象ディレクトリから銘柄リストを動的に生成 ---
    dynamic_watched_codes = get_stock_codes_from_data_directory(data_dir_to_watch)
    if not dynamic_watched_codes:
        logger.warning(f"監視対象ディレクトリ '{data_dir_to_watch}' から処理可能な銘柄ファイルが見つかりませんでした。ファイル監視は開始しますが、実質的に何も処理されません。")
    # --------------------------------------------------------------------

    # --- ★ 修正: PriceDataFileEventHandler に動的銘柄リストを渡す ---
    event_handler = PriceDataFileEventHandler(realtime_config, strategy_config_params, dynamic_watched_codes)
    # --------------------------------------------------------------
    observer = Observer()
    observer.schedule(event_handler, data_dir_to_watch, recursive=False) # recursive=False のまま
    observer.start()
    logger.info("ファイル監視を開始しました。Ctrl+Cで終了します。")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("キーボード割り込みを受信しました。監視を終了します。")
    except Exception as e:
        logger.error(f"予期せぬエラーにより監視ループが停止しました: {e}", exc_info=True)
    finally:
        schedule.clear()
        observer.stop()
        observer.join()
        logger.info("リアルタイム株価アラートシステムを終了しました。")

if __name__ == "__main__":
    main()