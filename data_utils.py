# data_utils.py
import pandas as pd
import numpy as np
import logging # loggingをインポート
import os # osをインポート (os.path.basename用)

# loggerオブジェクトを取得 (このモジュール用)
logger = logging.getLogger(__name__)

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
    except FileNotFoundError:
        logger.error(f"    ファイルが見つかりません: {filepath}") # loggerを使用
        raise
    except Exception as e:
        logger.error(f"    CSVファイルの読み込みまたは前処理中にエラーが発生しました ({os.path.basename(filepath)}): {e}") # loggerを使用
        raise