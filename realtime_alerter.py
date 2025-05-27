# realtime_alerter.py
import time
import logging
import os
import yaml # PyYAMLライブラリ
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import datetime
import pandas as pd

import state_manager
import data_processor
import strategy # strategy.py をインポート
import alert_sender # alert_sender.py をインポート

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
            if not config_data: # ファイルは存在するが中身が空、またはYAMLとして無効な場合
                logger.warning(f"設定ファイル '{filepath}' が空、または有効なYAMLデータを含んでいません。")
                return {} # 空の辞書を返す
            logger.info(f"設定ファイル '{filepath}' を正常に読み込みました。")
            return config_data
    except FileNotFoundError:
        logger.error(f"エラー: 設定ファイル '{filepath}' が見つかりません。")
        return None # ファイルが存在しない場合はNoneを返す
    except yaml.YAMLError as e:
        logger.error(f"エラー: 設定ファイル '{filepath}' のYAML解析に失敗しました: {e}")
        return None # 解析失敗時もNoneを返す
    except Exception as e:
        logger.error(f"エラー: 設定ファイル '{filepath}' の読み込み中に予期せぬエラーが発生しました: {e}", exc_info=True)
        return None # その他のエラーでもNoneを返す

# --- ファイルイベントハンドラ ---
class PriceDataFileEventHandler(FileSystemEventHandler):
    def __init__(self, realtime_config, strategy_config_params): # strategy_config を strategy_config_params に変更
        super().__init__()
        self.realtime_config = realtime_config
        self.strategy_config = strategy_config_params # strategy.load_strategy_config_yaml からロードされたもの
        self.watched_stock_codes = realtime_config.get('watched_stock_codes', [])
        self.state_file_dir = realtime_config.get('state_file_directory')
        if not self.state_file_dir:
            logger.error("状態ファイルディレクトリが realtime_config に設定されていません。")
            self.state_file_dir = "default_runtime_state" # フォールバック例
            ensure_directory_exists(self.state_file_dir)

        # email_config の保持を削除 (alert_sender が直接 email_credentials.yaml を読むため)
        # logger.info("メール設定は email_credentials.yaml から直接読み込まれます。")

        logger.info(f"監視対象の銘柄コード: {self.watched_stock_codes}")
        logger.info(f"状態ファイルディレクトリ: {os.path.abspath(self.state_file_dir)}")

        # リアルタイム処理用のパラメータを準備
        self.realtime_params_for_strategy = {
            'interval_exec': self.realtime_config.get('execution_interval_minutes', 1),
            'interval_context': self.realtime_config.get('context_interval_minutes', 5),
            'TRADING_HOURS_FORCE_EXIT_TIME_STR': self.strategy_config.get( # self.strategy_config から取得
                'TRADING_HOURS_FORCE_EXIT_TIME_STR',
                self.strategy_config.get('TRADING_HOURS', {}).get('FORCE_EXIT_TIME_STR') # ネスト対応
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

        stock_code_match = None
        for code in self.watched_stock_codes:
            if filename.startswith(str(code)):
                stock_code_match = str(code)
                break
        
        if stock_code_match:
            logger.info(f"処理対象ファイル更新: {filename} (銘柄コード: {stock_code_match})")
            self.process_stock_data(stock_code_match, filepath)
        else:
            logger.debug(f"無視するファイル変更: {filename}")

    def on_created(self, event):
        if event.is_directory:
            return
        logger.debug(f"ファイル作成検知: {event.src_path}")
        self.on_modified(event)


    def process_stock_data(self, stock_code, filepath):
        logger.info(f"--- 銘柄 {stock_code} のデータ処理開始 ({filepath}) ---")
        try:
            current_time_obj = datetime.datetime.now().time()

            current_state = state_manager.load_state(stock_code, self.state_file_dir)
            logger.debug(f"銘柄 {stock_code}: 状態ロード完了。ポジション: {current_state.get('position')}")

            df_exec, df_context, updated_state = data_processor.load_and_prepare_data_for_strategy(
                filepath,
                self.strategy_config, # 戦略パラメータ (config.yaml の内容)
                current_state.copy()
            )
            current_state = updated_state

            if df_exec is None or df_exec.empty:
                logger.warning(f"銘柄 {stock_code}: 実行足の準備に失敗。処理を中断。")
                state_manager.save_state(stock_code, current_state, self.state_file_dir)
                return
            
            if df_context is None:
                logger.warning(f"銘柄 {stock_code}: 環境認識足の準備に失敗。空のDataFrameで代替します。")
                df_context = pd.DataFrame()

            logger.info(f"銘柄 {stock_code}: 実行足 {len(df_exec)}本、環境認識足 {len(df_context)}本を準備完了。")

            indicators_df = strategy.calculate_indicators(
                df_exec.copy(),
                df_context.copy(),
                self.realtime_params_for_strategy,
                self.strategy_config # 戦略パラメータ (config.yaml の内容)
            )

            if indicators_df is None or indicators_df.empty:
                logger.warning(f"銘柄 {stock_code}: 指標計算結果が空です。シグナル生成をスキップ。")
                latest_signal_val = 0
            else:
                signals_df = strategy.generate_signals(
                    indicators_df.copy(),
                    self.realtime_params_for_strategy,
                    self.strategy_config
                )
                if not signals_df.empty and 'Signal' in signals_df.columns and not signals_df['Signal'].empty:
                    latest_signal_val = signals_df['Signal'].iloc[-1]
                else:
                    logger.warning(f"銘柄 {stock_code}: シグナル生成結果が空、またはSignalカラムが存在しません。")
                    latest_signal_val = 0

            logger.info(f"銘柄 {stock_code}: シグナル判定結果 (from strategy.py): {latest_signal_val}")

            alert_needed = False
            alert_message_subject = ""
            alert_message_body = ""
            new_position = current_state.get("position", "none")
            current_position_val = 0
            if new_position == "long": current_position_val = 1
            elif new_position == "short": current_position_val = -1

            if current_position_val != 0:
                if not indicators_df.empty and not indicators_df.iloc[-1].empty:
                    current_bar_data_for_exit = indicators_df.iloc[-1]
                    prev_bar_data_for_exit = indicators_df.iloc[-2] if len(indicators_df) >= 2 else pd.Series(dtype='object')
                    exit_now, exit_reason, _ = strategy.determine_exit_conditions(
                        current_position_val, current_bar_data_for_exit, prev_bar_data_for_exit,
                        self.realtime_params_for_strategy, self.strategy_config,
                        current_state.get("entry_price"), current_time_obj
                    )
                    if exit_now:
                        alert_needed = True
                        new_position = "none"
                        exit_price_dummy = df_exec['Close'].iloc[-1] if not df_exec.empty else current_state.get("entry_price", "N/A")
                        alert_message_subject = f"【決済アラート】銘柄: {stock_code}"
                        alert_message_body = (
                            f"銘柄コード: {stock_code}\n"
                            f"シグナル: 決済 ({('ロング' if current_position_val == 1 else 'ショート')}ポジション)\n"
                            f"理由: {exit_reason}\n"
                            f"推定決済価格: 約{exit_price_dummy}\n"
                            f"時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        logger.info(alert_message_body)
                        current_state["entry_price"] = None
                        current_state["entry_datetime"] = None
                else:
                    logger.warning(f"銘柄 {stock_code}: ポジション保有中だが、決済条件判定のための指標データが不足。")

            if not alert_needed:
                if current_state.get("position") == "none":
                    entry_price_dummy = df_exec['Close'].iloc[-1] if not df_exec.empty else None
                    entry_datetime_dummy = df_exec.index[-1].isoformat() if not df_exec.empty else datetime.datetime.now().isoformat()
                    if latest_signal_val == 1:
                        alert_needed = True
                        new_position = "long"
                        alert_message_subject = f"【新規買いアラート】銘柄: {stock_code}"
                        alert_message_body = (
                            f"銘柄コード: {stock_code}\n"
                            f"シグナル: 新規買い\n"
                            f"推定エントリー価格: 約{entry_price_dummy}\n"
                            f"時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        current_state["entry_price"] = entry_price_dummy
                        current_state["entry_datetime"] = entry_datetime_dummy
                    elif latest_signal_val == -1:
                        alert_needed = True
                        new_position = "short"
                        alert_message_subject = f"【新規売りアラート】銘柄: {stock_code}"
                        alert_message_body = (
                            f"銘柄コード: {stock_code}\n"
                            f"シグナル: 新規売り\n"
                            f"推定エントリー価格: 約{entry_price_dummy}\n"
                            f"時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        current_state["entry_price"] = entry_price_dummy
                        current_state["entry_datetime"] = entry_datetime_dummy

            if new_position != current_state.get("position"):
                logger.info(f"銘柄 {stock_code}: ポジション変更 {current_state.get('position')} -> {new_position}")
                current_state["position"] = new_position
            current_state["last_signal_generated"] = float(latest_signal_val)

            if alert_needed:
                logger.info(f"銘柄 {stock_code}: アラート送信実行！ メッセージ: {alert_message_body}")
                current_state["last_alert_datetime"] = datetime.datetime.now().isoformat()
                # alert_sender.send_email の呼び出しから email_config (SMTPサーバー情報) の引数を削除
                # alert_sender が email_credentials.yaml から全てのメール設定を読み込む
                email_sent_successfully = alert_sender.send_email(
                    alert_message_subject,
                    alert_message_body
                )
                if not email_sent_successfully:
                    logger.error(f"銘柄 {stock_code}: メール送信に失敗しました。詳細はalert_senderのログを確認してください。")
            else:
                logger.info(f"銘柄 {stock_code}: アラート条件に合致せず。")

            current_state['stock_code'] = stock_code
            state_manager.save_state(stock_code, current_state, self.state_file_dir)
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

    realtime_config = load_yaml_config(REALTIME_CONFIG_FILENAME)
    # strategy.py の関数を使って戦略設定をロードし、フラット化する
    strategy_config_params = strategy.load_strategy_config_yaml(STRATEGY_CONFIG_FILENAME)


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

    data_dir_to_watch = realtime_config.get('data_directory_to_watch')
    state_file_dir = realtime_config.get('state_file_directory')

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

    event_handler = PriceDataFileEventHandler(realtime_config, strategy_config_params)
    observer = Observer()
    observer.schedule(event_handler, data_dir_to_watch, recursive=False)
    observer.start()
    logger.info("ファイル監視を開始しました。Ctrl+Cで終了します。")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("キーボード割り込みを受信しました。監視を終了します。")
    except Exception as e:
        logger.error(f"予期せぬエラーにより監視ループが停止しました: {e}", exc_info=True)
    finally:
        observer.stop()
        observer.join()
        logger.info("リアルタイム株価アラートシステムを終了しました。")

if __name__ == "__main__":
    main()