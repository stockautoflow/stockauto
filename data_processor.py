# data_processor.py
import pandas as pd
import logging
from base import load_csv_data # base.py の load_csv_data をインポート

logger = logging.getLogger(__name__)

def resample_ohlcv(df, interval_min, label='right', closed='right'):
    """
    OHLCV DataFrameを指定された時間間隔でリサンプリングする。
    Args:
        df (pd.DataFrame): 'datetime' をインデックスに持つOHLCVデータ。
        interval_min (int): リサンプリングする間隔（分）。
        label (str): 'left' または 'right'。リサンプリングされた期間のどちら側のラベルを使用するか。
        closed (str): 'left' または 'right'。リサンプリングされた期間のどちら側を閉区間とするか。
    Returns:
        pd.DataFrame: リサンプリングされたOHLCVデータ。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("リサンプリングには DatetimeIndex が必要です。")
        return pd.DataFrame()
    if df.empty:
        logger.warning("リサンプリング対象のDataFrameが空です。")
        return pd.DataFrame()

    rule = f'{interval_min}T' # T は分単位のオフセットエイリアス
    
    # 必要なカラムの存在確認
    required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        logger.error(f"リサンプリングに必要なカラムが不足しています: {missing}")
        return pd.DataFrame()

    try:
        resampled_df = df.resample(rule, label=label, closed=closed).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        # リサンプリングによって全てNaNになった行を除外
        resampled_df.dropna(subset=['Open', 'High', 'Low', 'Close'], how='all', inplace=True)
        # Volumeが0になる場合があるが、それは許容する (取引がなかった場合など)
        resampled_df['Volume'] = resampled_df['Volume'].fillna(0)

        logger.debug(f"{interval_min}分足へのリサンプリング完了。 元データ {len(df)}行 -> リサンプル後 {len(resampled_df)}行")
        return resampled_df
    except Exception as e:
        logger.error(f"{interval_min}分足へのリサンプリング中にエラーが発生しました: {e}", exc_info=True)
        return pd.DataFrame()

def load_and_prepare_data_for_strategy(
    filepath,
    strategy_config,
    current_state,
    max_history_bars=500 # 保持する最大履歴バー数 (各足種ごと)
):
    """
    CSVファイルから株価データを読み込み、実行足と環境認識足を生成し、
    状態管理用の履歴データと結合/更新する。
    Args:
        filepath (str): 株価データCSVファイルのパス。
        strategy_config (dict): 戦略設定 (config.yaml の内容)。
        current_state (dict): 現在の銘柄の状態。
        max_history_bars (int): 指標計算用に保持する各足種の最大バー数。
    Returns:
        tuple: (実行足DataFrame, 環境認識足DataFrame, 更新された状態) またはエラー時に (None, None, current_state)
    """
    logger.debug(f"データファイル読み込み開始: {filepath}")
    try:
        # base.py の load_csv_data を使用して1分足データを読み込むと仮定 [cite: 1]
        # この関数は datetime をインデックスにし、タイムゾーンを 'Asia/Tokyo' に設定する
        df_1min = load_csv_data(filepath) # [cite: 1]
        if df_1min is None or df_1min.empty:
            logger.warning(f"ファイル '{filepath}' からデータを読み込めませんでした、またはデータが空です。")
            return None, None, current_state
        logger.info(f"ファイル '{filepath}' から {len(df_1min)} 件の1分足データを読み込みました。")

    except Exception as e:
        logger.error(f"ファイル '{filepath}' の読み込みまたは初期処理中にエラー: {e}", exc_info=True)
        return None, None, current_state

    # strategy_config から実行足と環境認識足のインターバルを取得
    # これらのキー名は strategy.py や config.yaml の定義に依存する
    # ここでは、backtest_cli.py が渡す引数名と、strategy.py の get_merged_param で
    # params_from_backtester から取得されるキーを想定する。
    # base.pyのrun_backtestでは'interval_exec'と'interval_context'が直接渡される。
    # config.yamlには直接これらの設定はないため、strategy_configに直接これらの値が
    # 含まれているか、または別のキーから取得する必要がある。
    # ここでは、仮に strategy_config に直接 exec_interval_min と context_interval_min が
    # 何らかの形で含まれているか、または固定値を使用するとする。
    # より柔軟にするには、realtime_config.yaml にこれらの値を設定する。
    
    # exec_interval_min = strategy_config.get('EXEC_INTERVAL_MIN', 1) # デフォルト1分 (仮)
    # context_interval_min = strategy_config.get('CONTEXT_INTERVAL_MIN', 5) # デフォルト5分 (仮)
    # より良いのは realtime_config.yaml からこれらの値を取得すること。
    # 今回は strategy_config を介して渡される前提で進めるが、
    # 実際のバックテストフレームワークの引数名を参考にする。
    # バックテストのexec_interval, context_intervalはコマンドライン引数で渡され、
    # それがstrategy.pyのparams_from_backtesterに入る想定。
    # ここでは、仮にstrategy_configに以下のキーで設定値があると仮定。
    # なければデフォルト値を使用。
    # 実際には、realtime_config.yaml にこれらの実行インターバルを設定するか、
    # もしくは strategy_config の構造に合わせてキーを調整してください。
    exec_interval_min = strategy_config.get('execution_interval_minutes', 1) # 仮のキー
    context_interval_min = strategy_config.get('context_interval_minutes', 5) # 仮のキー

    logger.info(f"実行足インターバル: {exec_interval_min}分, 環境認識足インターバル: {context_interval_min}分")

    # 実行足と環境認識足をリサンプリングして生成 [cite: 1]
    df_exec = resample_ohlcv(df_1min.copy(), exec_interval_min) # [cite: 1]
    df_context = resample_ohlcv(df_1min.copy(), context_interval_min) # [cite: 1]

    if df_exec.empty:
        logger.warning("実行足の生成に失敗したか、結果が空です。")
        # return None, None, current_state # 環境足だけでも処理を続ける場合あり
    if df_context.empty:
        logger.warning("環境認識足の生成に失敗したか、結果が空です。")
        # return None, None, current_state

    # 状態管理用の履歴データを更新 [cite: 1]
    # ここでは、リサンプリングされた全期間を一旦保持し、strategy.py側で直近N本を使う想定とする
    # または、ここで直近N本に絞っても良い。
    # JSONに保存するため、DataFrameを辞書のリストに変換する
    if not df_exec.empty:
        # indexを 'datetime' カラムとしてリセットし、datetimeをISOフォーマット文字列に変換
        current_state['ohlcv_history_exec'] = df_exec.reset_index().rename(columns={'index': 'datetime'}).to_dict('records')
        # 保持するバー数を制限する場合
        # current_state['ohlcv_history_exec'] = df_exec.tail(max_history_bars).reset_index().rename(columns={'index': 'datetime'}).to_dict('records')
    else:
        current_state['ohlcv_history_exec'] = [] # 空でもキーは保持

    if not df_context.empty:
        current_state['ohlcv_history_context'] = df_context.reset_index().rename(columns={'index': 'datetime'}).to_dict('records')
        # current_state['ohlcv_history_context'] = df_context.tail(max_history_bars).reset_index().rename(columns={'index': 'datetime'}).to_dict('records')
    else:
        current_state['ohlcv_history_context'] = []


    # 最後に処理したバーのタイムスタンプを更新
    # 実行足の最後のバーの時刻を基準とする (もしあれば)
    if not df_exec.empty and isinstance(df_exec.index[-1], pd.Timestamp):
        current_state['last_processed_bar_datetime'] = df_exec.index[-1].isoformat()
    elif not df_1min.empty and isinstance(df_1min.index[-1], pd.Timestamp): # 実行足がない場合は1分足の最後
        current_state['last_processed_bar_datetime'] = df_1min.index[-1].isoformat()


    logger.info(f"実行足 {len(current_state['ohlcv_history_exec'])}本、環境認識足 {len(current_state['ohlcv_history_context'])}本を状態に準備。")

    # 戦略モジュールに渡すためにDataFrameに戻す (calculate_indicators がDataFrameを期待するため)
    # ただし、このDataFrameは履歴全体。strategy.py側で最新のシグナルを計算する際に、
    # どのデータを使うか（最新のバーか、ある程度の期間か）を決める必要がある。
    # ここで返すのは、strategy.py に渡す直前の形式のDataFrame。
    df_exec_for_strategy = pd.DataFrame(current_state['ohlcv_history_exec'])
    if not df_exec_for_strategy.empty and 'datetime' in df_exec_for_strategy.columns:
        df_exec_for_strategy['datetime'] = pd.to_datetime(df_exec_for_strategy['datetime'])
        df_exec_for_strategy = df_exec_for_strategy.set_index('datetime')

    df_context_for_strategy = pd.DataFrame(current_state['ohlcv_history_context'])
    if not df_context_for_strategy.empty and 'datetime' in df_context_for_strategy.columns:
        df_context_for_strategy['datetime'] = pd.to_datetime(df_context_for_strategy['datetime'])
        df_context_for_strategy = df_context_for_strategy.set_index('datetime')

    return df_exec_for_strategy, df_context_for_strategy, current_state