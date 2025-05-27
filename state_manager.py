# state_manager.py
import json
import os
import logging
import datetime

logger = logging.getLogger(__name__)

# --- Default State Structure ---
def get_initial_state(stock_code):
    """
    Returns the initial state structure for a new stock.
    """
    return {
        "stock_code": str(stock_code),
        "position": "none",  # "none", "long", "short"
        "entry_price": None,
        "entry_datetime": None, # ISO format string
        "last_processed_bar_datetime": None, # ISO format string for the last bar's datetime
        # OHLCV history will be lists of dictionaries or pandas DataFrame.to_dict('records')
        # For simplicity here, we'll keep them as lists of lists [timestamp, o, h, l, c, v]
        # Timestamps should be ISO format strings or Unix timestamps for JSON compatibility
        "ohlcv_history_exec": [], # To store execution interval candles
        "ohlcv_history_context": [], # To store context interval candles
        "last_signal_generated": None, # e.g., "buy", "sell", "hold"
        "last_alert_datetime": None, # ISO format string
        "custom_data": {} # For any other strategy-specific state
    }

def _get_state_filepath(stock_code, state_file_dir):
    """
    Constructs the filepath for a stock's state JSON file.
    """
    if not stock_code:
        raise ValueError("Stock code cannot be empty.")
    if not state_file_dir:
        raise ValueError("State file directory cannot be empty.")
    return os.path.join(state_file_dir, f"state_{str(stock_code)}.json")

def load_state(stock_code, state_file_dir):
    """
    Loads the state for a given stock code from a JSON file.
    If the file doesn't exist, returns an initial state.
    """
    filepath = _get_state_filepath(stock_code, state_file_dir)
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
                logger.info(f"銘柄 {stock_code}: 状態ファイルをロードしました: {filepath}")
                # Potentially add validation here to ensure loaded state has all necessary keys
                # For now, we assume the loaded state is valid or will be handled by the caller
                return state_data
        else:
            logger.info(f"銘柄 {stock_code}: 状態ファイルが見つかりません。初期状態で開始します: {filepath}")
            return get_initial_state(stock_code)
    except json.JSONDecodeError as e:
        logger.error(f"銘柄 {stock_code}: 状態ファイル '{filepath}' のJSONデコードエラー: {e}。初期状態で開始します。")
        return get_initial_state(stock_code)
    except Exception as e:
        logger.error(f"銘柄 {stock_code}: 状態ファイル '{filepath}' の読み込み中に予期せぬエラー: {e}。初期状態で開始します。")
        return get_initial_state(stock_code)

def save_state(stock_code, state_data, state_file_dir):
    """
    Saves the state for a given stock code to a JSON file.
    """
    if not isinstance(state_data, dict):
        logger.error(f"銘柄 {stock_code}: 保存する状態データが辞書型ではありません。保存をスキップします。")
        return False

    filepath = _get_state_filepath(stock_code, state_file_dir)
    try:
        # Ensure the directory exists (it should be created by realtime_alerter.py, but good to double check)
        os.makedirs(state_file_dir, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=4, ensure_ascii=False, default=str) # default=str for datetime objects
        logger.info(f"銘柄 {stock_code}: 状態ファイルを保存しました: {filepath}")
        return True
    except Exception as e:
        logger.error(f"銘柄 {stock_code}: 状態ファイル '{filepath}' の保存中にエラーが発生しました: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    # --- Example Usage and Testing ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    TEST_STATE_DIR = "test_runtime_state"
    if not os.path.exists(TEST_STATE_DIR):
        os.makedirs(TEST_STATE_DIR)

    test_code_1 = "9999"
    test_code_2 = "7777"

    # 1. Load initial state (file doesn't exist yet)
    logger.info(f"\n--- {test_code_1}: 初期状態のロード ---")
    state1 = load_state(test_code_1, TEST_STATE_DIR)
    print(json.dumps(state1, indent=4, ensure_ascii=False))

    # 2. Modify and save state
    logger.info(f"\n--- {test_code_1}: 状態変更と保存 ---")
    state1["position"] = "long"
    state1["entry_price"] = 1500.0
    state1["entry_datetime"] = datetime.datetime.now().isoformat()
    state1["ohlcv_history_exec"].append([datetime.datetime.now().isoformat(), 1490, 1510, 1480, 1505, 10000])
    state1["last_processed_bar_datetime"] = datetime.datetime.now().isoformat()
    save_state(test_code_1, state1, TEST_STATE_DIR)

    # 3. Load saved state
    logger.info(f"\n--- {test_code_1}: 保存された状態のロード ---")
    loaded_state1 = load_state(test_code_1, TEST_STATE_DIR)
    print(json.dumps(loaded_state1, indent=4, ensure_ascii=False))
    assert loaded_state1["position"] == "long"
    assert len(loaded_state1["ohlcv_history_exec"]) == 1

    # 4. Test another stock (initial state)
    logger.info(f"\n--- {test_code_2}: 別の銘柄の初期状態ロード ---")
    state2 = load_state(test_code_2, TEST_STATE_DIR)
    print(json.dumps(state2, indent=4, ensure_ascii=False))
    assert state2["stock_code"] == test_code_2
    assert state2["position"] == "none"

    # 5. Test invalid load (e.g., corrupted JSON - manual step needed to test this)
    # To test this, manually create a corrupted JSON file like state_XXXX.json in TEST_STATE_DIR

    logger.info("\n--- state_manager.py テスト完了 ---")
    # Clean up test directory if desired
    # import shutil
    # shutil.rmtree(TEST_STATE_DIR)
    # logger.info(f"テストディレクトリをクリーンアップしました: {TEST_STATE_DIR}")