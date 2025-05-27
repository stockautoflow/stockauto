# OutputChart.py (修正版)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import pandas as pd
import numpy as np
import os
import logging
import datetime # datetimeモジュールをインポート

# import matplotlib as mpl # バージョン確認などに使用する場合はコメント解除

# --- 定数の定義 ---
NUM_BARS_TO_DISPLAY = 500 # 表示するローソク足の本数 (0以下で全期間)

# 画像の横幅に関する定数
CHART_BASE_WIDTH_INCHES = 10.0
CHART_WIDTH_PER_BAR_INCHES = 0.08
MIN_CHART_WIDTH_INCHES = 16.0
MAX_CHART_WIDTH_INCHES = 50.0
CHART_HEIGHT_INCHES = 28.0

def plot_chart_for_stock(
    df_context_orig, df_exec_orig, df_with_signals_orig, trade_history_orig,
    stock_code, strategy_params, chart_output_dir, # strategy_params は config.yaml のフラット化された辞書を期待
    base_filename_parts
    ):
    logging.info(f"  銘柄 {stock_code}: チャート生成開始 (期間指定対応, 休場日詰める, 7パネル, constrained_layout)...")
    p = strategy_params # 渡されたパラメータを p として使用

    # --- 期間指定に基づいてデータを絞り込む ---
    df_context, df_exec, df_with_signals, trade_history = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    min_display_time, max_display_time = None, None

    if NUM_BARS_TO_DISPLAY > 0:
        if not df_exec_orig.empty:
            df_exec = df_exec_orig.tail(NUM_BARS_TO_DISPLAY).copy()
            min_display_time = df_exec.index.min()
            max_display_time = df_exec.index.max()
        elif not df_context_orig.empty:
            df_context = df_context_orig.tail(NUM_BARS_TO_DISPLAY).copy()
            min_display_time = df_context.index.min()
            max_display_time = df_context.index.max()

        if min_display_time is not None and max_display_time is not None:
            if not df_exec_orig.empty and df_exec.empty: # tailで空になった場合
                 df_exec = df_exec_orig[(df_exec_orig.index >= min_display_time) & (df_exec_orig.index <= max_display_time)].copy()

            if not df_context_orig.empty:
                df_context = df_context_orig[(df_context_orig.index >= min_display_time) & (df_context_orig.index <= max_display_time)].copy()

            if not df_with_signals_orig.empty:
                df_with_signals = df_with_signals_orig[(df_with_signals_orig.index >= min_display_time) & (df_with_signals_orig.index <= max_display_time)].copy()

            if not trade_history_orig.empty:
                try:
                    th_copy = trade_history_orig.copy()
                    th_copy['entry_date_dt'] = pd.to_datetime(th_copy['entry_date'], errors='coerce')
                    th_copy['exit_date_dt'] = pd.to_datetime(th_copy['exit_date'], errors='coerce')
                    trade_history = th_copy[
                        (th_copy['entry_date_dt'].notna()) & (th_copy['exit_date_dt'].notna()) &
                        (th_copy['entry_date_dt'] <= max_display_time) & # エントリーが期間終了前
                        (th_copy['exit_date_dt'] >= min_display_time)   # エグジットが期間開始後 (少しでも重なれば表示)
                    ].copy()
                    trade_history.drop(columns=['entry_date_dt', 'exit_date_dt'], inplace=True, errors='ignore')
                except Exception as e_th:
                    logging.error(f"  銘柄 {stock_code}: トレード履歴の期間絞り込みエラー: {e_th}")
                    trade_history = pd.DataFrame()
        else: # 期間が決定できない場合 (元データが非常に少ないか空)
            df_context = df_context_orig.tail(NUM_BARS_TO_DISPLAY).copy() if not df_context_orig.empty else pd.DataFrame()
            df_exec = df_exec_orig.tail(NUM_BARS_TO_DISPLAY).copy() if not df_exec_orig.empty else pd.DataFrame()
            df_with_signals = df_with_signals_orig.tail(NUM_BARS_TO_DISPLAY).copy() if not df_with_signals_orig.empty else pd.DataFrame() # 仮
            trade_history = trade_history_orig.copy()


    else: # NUM_BARS_TO_DISPLAY <= 0 (全期間表示)
        df_context = df_context_orig.copy(); df_exec = df_exec_orig.copy()
        df_with_signals = df_with_signals_orig.copy(); trade_history = trade_history_orig.copy()


    if df_context.empty and df_exec.empty:
        logging.warning(f"  銘柄 {stock_code}: 表示対象期間の環境認識足・実行足データが共に空のためスキップ。")
        return

    def _prepare_plot_df(df_input, df_name_for_log=""):
        if df_input.empty: return pd.DataFrame(columns=['datetime', 'x_index', 'Open', 'High', 'Low', 'Close', 'Volume'])

        # インデックスがDatetimeIndexでない場合、'datetime'列を探してインデックスに設定
        if not isinstance(df_input.index, pd.DatetimeIndex):
            if 'datetime' in df_input.columns:
                try:
                    df_input = df_input.set_index('datetime', drop=True)
                    if not isinstance(df_input.index, pd.DatetimeIndex): # 再度確認
                         # datetime列をpd.to_datetimeで変換してみる
                        df_input_reset = df_input.reset_index()
                        df_input_reset['datetime'] = pd.to_datetime(df_input_reset['datetime'], errors='coerce')
                        df_input = df_input_reset.set_index('datetime')
                        if not isinstance(df_input.index, pd.DatetimeIndex):
                            logging.warning(f"  銘柄 {stock_code}: {df_name_for_log} の datetime列を有効なDatetimeIndexに設定できませんでした。")
                            return pd.DataFrame(columns=['datetime', 'x_index', 'Open', 'High', 'Low', 'Close', 'Volume'])
                except Exception as e_set_idx:
                    logging.warning(f"  銘柄 {stock_code}: {df_name_for_log} の 'datetime'列のインデックス設定中にエラー: {e_set_idx}")
                    return pd.DataFrame(columns=['datetime', 'x_index', 'Open', 'High', 'Low', 'Close', 'Volume'])

            elif df_input.index.name == 'datetime' and not isinstance(df_input.index, pd.DatetimeIndex):
                 # インデックス名が 'datetime' だが型が違う場合、変換を試みる
                try:
                    df_input.index = pd.to_datetime(df_input.index, errors='coerce')
                    if not isinstance(df_input.index, pd.DatetimeIndex):
                         logging.warning(f"  銘柄 {stock_code}: {df_name_for_log} の 'datetime'インデックスを有効なDatetimeIndexに変換できませんでした。")
                         return pd.DataFrame(columns=['datetime', 'x_index', 'Open', 'High', 'Low', 'Close', 'Volume'])
                except Exception as e_conv_idx:
                    logging.warning(f"  銘柄 {stock_code}: {df_name_for_log} の 'datetime'インデックスの変換中にエラー: {e_conv_idx}")
                    return pd.DataFrame(columns=['datetime', 'x_index', 'Open', 'High', 'Low', 'Close', 'Volume'])
            else:
                logging.warning(f"  銘柄 {stock_code}: {df_name_for_log} の datetimeインデックスまたは列が見つかりません。")
                return pd.DataFrame(columns=['datetime', 'x_index', 'Open', 'High', 'Low', 'Close', 'Volume'])

        # datetimeインデックスを列に戻す
        current_index_name = df_input.index.name if df_input.index.name is not None else 'datetime'
        df_plot = df_input.reset_index().rename(columns={current_index_name: 'datetime'})

        # datetime列を確実にTimestamp型にし、NaTがあれば警告
        if 'datetime' in df_plot.columns:
            df_plot['datetime'] = pd.to_datetime(df_plot['datetime'], errors='coerce')
            if df_plot['datetime'].isna().any():
                logging.warning(f"  銘柄 {stock_code}: {df_name_for_log} のdatetime列にNaTが含まれています。")
        else: # datetime列が作れなかった場合
             logging.warning(f"  銘柄 {stock_code}: {df_name_for_log} にdatetime列がありません。")
             return pd.DataFrame(columns=['datetime', 'x_index', 'Open', 'High', 'Low', 'Close', 'Volume'])


        if not df_plot.empty: df_plot['x_index'] = np.arange(len(df_plot))
        else: df_plot['x_index'] = pd.Series(dtype='int')

        return df_plot

    df_context_plot = _prepare_plot_df(df_context, "df_context")
    df_exec_plot = _prepare_plot_df(df_exec, "df_exec")

    # 実行足ベースの指標とシグナル (df_with_signals を使用)
    df_with_signals_plot = pd.DataFrame()
    if not df_exec_plot.empty and 'datetime' in df_exec_plot.columns and not df_with_signals.empty:
        df_ws_temp = df_with_signals.copy()
        if not isinstance(df_ws_temp.index, pd.DatetimeIndex):
            if 'datetime' in df_ws_temp.columns: df_ws_temp = df_ws_temp.set_index('datetime')
            elif df_ws_temp.index.name == 'datetime': pass

        if isinstance(df_ws_temp.index, pd.DatetimeIndex):
            df_ws_temp_reset = df_ws_temp.reset_index()
            df_ws_temp_reset['datetime'] = pd.to_datetime(df_ws_temp_reset['datetime'], errors='coerce')

            if pd.api.types.is_datetime64_any_dtype(df_exec_plot['datetime']) and \
               pd.api.types.is_datetime64_any_dtype(df_ws_temp_reset['datetime']):
                required_ohlcv_exec = ['datetime', 'x_index', 'Open', 'High', 'Low', 'Close']
                if 'Volume' in df_exec_plot.columns: required_ohlcv_exec.append('Volume')

                df_with_signals_plot = pd.merge(
                    df_exec_plot[required_ohlcv_exec],
                    df_ws_temp_reset,
                    on='datetime',
                    how='left'
                ).dropna(subset=['x_index']) # x_indexがないものは除外
                if 'x_index' in df_with_signals_plot.columns: df_with_signals_plot['x_index'] = df_with_signals_plot['x_index'].astype(int)
        if df_with_signals_plot.empty and not df_exec_plot.empty : # 実行足はあるのに指標マージで空になった場合
            logging.warning(f"  銘柄 {stock_code}: df_with_signals_plot が空（実行足指標のマージ失敗等）。df_exec_plotを代替使用。")
            df_with_signals_plot = df_exec_plot.copy() # フォールバックとしてexec_plotをコピー


    # 環境認識足ベースの指標 (df_with_signals を使用し、df_context_plotにマージ)
    df_ctx_indicators_plot = pd.DataFrame()
    if not df_context_plot.empty and 'datetime' in df_context_plot.columns and not df_with_signals.empty:
        df_ws_temp_for_ctx = df_with_signals.copy()
        if not isinstance(df_ws_temp_for_ctx.index, pd.DatetimeIndex):
            if 'datetime' in df_ws_temp_for_ctx.columns: df_ws_temp_for_ctx = df_ws_temp_for_ctx.set_index('datetime')
            elif df_ws_temp_for_ctx.index.name == 'datetime': pass

        if isinstance(df_ws_temp_for_ctx.index, pd.DatetimeIndex) and \
           pd.api.types.is_datetime64_any_dtype(df_context_plot['datetime']):

            df_ws_temp_for_ctx = df_ws_temp_for_ctx.sort_index()
            df_context_plot_sorted = df_context_plot.sort_values('datetime')

            # タイムゾーン正規化 (df_ws_temp_for_ctx を df_context_plot_sorted に合わせる)
            ctx_plot_tz = df_context_plot_sorted['datetime'].dt.tz
            ws_tz = df_ws_temp_for_ctx.index.tz

            if ctx_plot_tz is not None and ws_tz is None:
                try: df_ws_temp_for_ctx = df_ws_temp_for_ctx.tz_localize(ctx_plot_tz, ambiguous='infer', nonexistent='NaT').dropna(subset=[df_ws_temp_for_ctx.index.name])
                except TypeError: df_ws_temp_for_ctx = df_ws_temp_for_ctx.tz_convert(ctx_plot_tz) # 既にawareの場合
            elif ctx_plot_tz is None and ws_tz is not None:
                df_ws_temp_for_ctx = df_ws_temp_for_ctx.tz_localize(None)
            elif ctx_plot_tz is not None and ws_tz is not None and ctx_plot_tz != ws_tz:
                df_ws_temp_for_ctx = df_ws_temp_for_ctx.tz_convert(ctx_plot_tz)

            # マージ対象カラムの選定
            ctx_indicator_cols = [col for col in df_ws_temp_for_ctx.columns if '_ctx_ITS' in col or col.endswith('_ctx')]
            # ボリンジャーバンド用のカラム名も確認 (戦略ファイルに依存)
            # 例: 'BB_Middle_ctx', 'BB_Upper_ctx_1dev' など
            bb_cols_ctx = [
                'BB_Middle_ctx_ITS', 'BB_Upper_ctx_ITS', 'BB_Lower_ctx_ITS', # _ITSがつく基本形
                'BB_Middle_ctx', 'BB_Upper_ctx', 'BB_Lower_ctx' # _ITSがつかない形
            ]
            for i in range(1, 4): # 1-3dev
                bb_cols_ctx.append(f'BB_Upper_ctx_{i}dev_ITS')
                bb_cols_ctx.append(f'BB_Lower_ctx_{i}dev_ITS')
                bb_cols_ctx.append(f'BB_Upper_ctx_{i}dev') # _ITSなし
                bb_cols_ctx.append(f'BB_Lower_ctx_{i}dev')  # _ITSなし

            # 存在するカラムのみを抽出
            valid_ctx_indicator_cols = [col for col in ctx_indicator_cols if col in df_ws_temp_for_ctx.columns]
            valid_bb_cols_ctx = [col for col in bb_cols_ctx if col in df_ws_temp_for_ctx.columns]
            all_cols_to_merge_ctx = list(set(valid_ctx_indicator_cols + valid_bb_cols_ctx))


            if not all_cols_to_merge_ctx:
                logging.warning(f"  銘柄 {stock_code}: df_with_signals に環境認識指標カラムが見つかりません (マージ対象なし)。")

            if not df_ws_temp_for_ctx.empty and all_cols_to_merge_ctx:
                df_ctx_indicators_plot = pd.merge_asof(
                    left=df_context_plot_sorted[['datetime', 'x_index']],
                    right=df_ws_temp_for_ctx[all_cols_to_merge_ctx],
                    on='datetime',
                    direction='backward' # df_context_plotの時刻以前で最新のものを採用
                )
            if df_ctx_indicators_plot.empty and not df_context_plot_sorted.empty:
                 logging.warning(f"  銘柄 {stock_code}: 環境認識指標のマージ(merge_asof)結果が空。df_context_plotを代替。")
                 df_ctx_indicators_plot = df_context_plot_sorted[['datetime', 'x_index']].copy() # x_indexとdatetimeだけは保持
        else:
            logging.warning(f"  銘柄 {stock_code}: 環境足指標のマージに必要なdatetime情報が不足。")
    if df_ctx_indicators_plot.empty and not df_context_plot.empty: # さらにフォールバック
        df_ctx_indicators_plot = df_context_plot[['datetime', 'x_index']].copy()


    num_bars_for_width = len(df_exec_plot) if not df_exec_plot.empty else len(df_context_plot)
    if num_bars_for_width > 0 :
        calculated_width = CHART_BASE_WIDTH_INCHES + num_bars_for_width * CHART_WIDTH_PER_BAR_INCHES
        figure_width = max(MIN_CHART_WIDTH_INCHES, min(calculated_width, MAX_CHART_WIDTH_INCHES))
    else: figure_width = MIN_CHART_WIDTH_INCHES

    period_str = f"{NUM_BARS_TO_DISPLAY}bars" if NUM_BARS_TO_DISPLAY > 0 else "All"
    chart_filename = f"Chart_{period_str}_Gapless_{stock_code}_{base_filename_parts[0]}_{base_filename_parts[1]}_{base_filename_parts[2]}_{base_filename_parts[3]}.svg"
    chart_filepath = os.path.join(chart_output_dir, chart_filename)
    os.makedirs(chart_output_dir, exist_ok=True)

    fig = plt.figure(figsize=(figure_width, CHART_HEIGHT_INCHES), constrained_layout=True)
    plt.style.use('seaborn-v0_8-darkgrid')
    gs = fig.add_gridspec(7, 1, height_ratios=[3, 1, 1, 3, 1, 1, 1], hspace=0.05)

    ax1 = fig.add_subplot(gs[0]);
    ax2 = fig.add_subplot(gs[1], sharex=ax1 if not df_context_plot.empty else None);
    ax3 = fig.add_subplot(gs[2], sharex=ax1 if not df_context_plot.empty else None)
    ax4 = fig.add_subplot(gs[3])
    ax5 = fig.add_subplot(gs[4], sharex=ax4 if not df_exec_plot.empty else None);
    ax6 = fig.add_subplot(gs[5], sharex=ax4 if not df_exec_plot.empty else None);
    ax7 = fig.add_subplot(gs[6], sharex=ax4 if not df_exec_plot.empty else None)

    fig.suptitle(f"Chart for {stock_code} ({base_filename_parts[1]}, {period_str}, Gapless) - Strategy: {base_filename_parts[0]} - Data: {base_filename_parts[2]}", fontsize=16, y=0.995)

    candle_colors = {'up': 'red', 'down': 'green'}
    marker_buy_entry = {'marker': '^', 'color': 'orange', 'markersize': 10, 'label': 'Buy Entry'}
    marker_sell_entry = {'marker': 'v', 'color': 'lime', 'markersize': 10, 'label': 'Sell Entry'}
    marker_tp = {'marker': 'o', 'color': 'cyan', 'markersize': 8, 'alpha':0.7, 'label': 'Take Profit'}
    marker_sl = {'marker': 'x', 'color': 'cyan', 'markersize': 10, 'alpha':0.9, 'label': 'Stop Loss', 'markeredgewidth': 2}

    buy_entries = trade_history[trade_history['type'] == 'Long'] if not trade_history.empty else pd.DataFrame()
    sell_entries = trade_history[trade_history['type'] == 'Short'] if not trade_history.empty else pd.DataFrame()
    take_profits = pd.DataFrame(); stop_losses = pd.DataFrame()
    if not trade_history.empty and 'exit_type' in trade_history.columns:
        take_profits = trade_history[trade_history['exit_type'].astype(str).str.contains("TP", case=False, na=False)]
        stop_losses = trade_history[trade_history['exit_type'].astype(str).str.contains("SL", case=False, na=False)]

    def plot_ohlc_gapless(ax, df_ohlc_plot_data, ohlc_title, display_x_labels=False):
        if df_ohlc_plot_data.empty or 'x_index' not in df_ohlc_plot_data.columns or df_ohlc_plot_data['x_index'].isna().all():
            ax.text(0.5, 0.5, f"{ohlc_title}\nNo Data", ha='center', va='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(ohlc_title, fontsize=10); ax.set_xticks([]); return False
        required_ohlc_cols = ['x_index', 'Open', 'High', 'Low', 'Close']
        if not all(col in df_ohlc_plot_data.columns for col in required_ohlc_cols) or \
           df_ohlc_plot_data[required_ohlc_cols].isna().any().any():
            ax.text(0.5, 0.5, f"{ohlc_title}\nData Incomplete", ha='center', va='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(ohlc_title, fontsize=10); ax.set_xticks([]); return False

        if 'datetime' not in df_ohlc_plot_data.columns or not pd.api.types.is_datetime64_any_dtype(df_ohlc_plot_data['datetime']):
            logging.warning(f"  銘柄 {stock_code}: {ohlc_title} のdatetime列が不正または存在しません。OHLCプロットをスキップ。")
            ax.text(0.5, 0.5, f"{ohlc_title}\nDatetime Error", ha='center', va='center', transform=ax.transAxes, fontsize=10, color='red')
            ax.set_title(ohlc_title, fontsize=10); ax.set_xticks([]); return False

        quotes = df_ohlc_plot_data[required_ohlc_cols].values
        current_bar_count = len(quotes)
        width_val = 0.8
        if current_bar_count > 0:
            width_scale_factor = max(1, figure_width / MIN_CHART_WIDTH_INCHES)
            if current_bar_count > 150 * width_scale_factor: width_val = 0.7
            if current_bar_count > 300 * width_scale_factor: width_val = 0.6
            if current_bar_count > 450 * width_scale_factor: width_val = 0.5
        else: width_val = 0.8
        candlestick_ohlc(ax, quotes, width=width_val, colorup=candle_colors['up'], colordown=candle_colors['down'], alpha=0.9)
        ax.set_title(ohlc_title, fontsize=10)

        min_x_val = df_ohlc_plot_data['x_index'].min(); max_x_val = df_ohlc_plot_data['x_index'].max()
        if pd.notna(min_x_val) and pd.notna(max_x_val): ax.set_xlim(min_x_val - width_val, max_x_val + width_val)
        else: ax.set_xlim(-width_val, width_val)

        if display_x_labels:
            set_custom_datetime_x_labels(ax, df_ohlc_plot_data, stock_code)
        else:
            ax.tick_params(labelbottom=False)
            ax.set_xticks([])
        return True


    def plot_markers_gapless(ax, df_ohlc_plot_data, trades_df, marker_config, trade_event_details):
        if trades_df.empty or df_ohlc_plot_data.empty or 'x_index' not in df_ohlc_plot_data.columns or \
           'datetime' not in df_ohlc_plot_data.columns or df_ohlc_plot_data['datetime'].isna().all(): return

        if not pd.api.types.is_datetime64_any_dtype(df_ohlc_plot_data['datetime']):
            logging.warning(f"  銘柄 {stock_code}: plot_markers_gapless のdatetime列が不正です。マーカープロットをスキップ。")
            return

        marker_datetime_col = trade_event_details['datetime_col']; marker_price_col = trade_event_details['price_col']
        for idx, trade_row in trades_df.iterrows():
            trade_event_dt_val = trade_row[marker_datetime_col]
            if pd.isna(trade_event_dt_val): continue
            trade_event_dt = pd.to_datetime(trade_event_dt_val, errors='coerce')
            if pd.isna(trade_event_dt): continue

            plot_ohlc_datetime_col = df_ohlc_plot_data['datetime']
            # タイムゾーン正規化
            ohlc_tz = plot_ohlc_datetime_col.dt.tz
            if ohlc_tz is not None:
                if trade_event_dt.tzinfo is None:
                    try: trade_event_dt = trade_event_dt.tz_localize(ohlc_tz, ambiguous='infer', nonexistent='NaT')
                    except TypeError: # Already localized
                         if trade_event_dt.tzinfo != ohlc_tz: trade_event_dt = trade_event_dt.tz_convert(ohlc_tz)
                elif trade_event_dt.tzinfo != ohlc_tz:
                    trade_event_dt = trade_event_dt.tz_convert(ohlc_tz)
            elif trade_event_dt.tzinfo is not None: # OHLC is naive, trade_event is aware
                 trade_event_dt = trade_event_dt.tz_localize(None)

            if pd.isna(trade_event_dt): continue

            min_ohlc_dt = plot_ohlc_datetime_col.min(); max_ohlc_dt = plot_ohlc_datetime_col.max()
            if pd.isna(min_ohlc_dt) or pd.isna(max_ohlc_dt) or trade_event_dt < min_ohlc_dt or trade_event_dt > max_ohlc_dt: continue

            temp_trade_df = pd.DataFrame({'datetime': [trade_event_dt]})

            df_ohlc_plot_data_sorted = df_ohlc_plot_data.sort_values('datetime')

            merged_df = pd.merge_asof(temp_trade_df.sort_values('datetime'),
                                      df_ohlc_plot_data_sorted,
                                      on='datetime', direction='nearest')
            if merged_df.empty or pd.isna(merged_df['x_index'].iloc[0]):
                continue
            target_row = merged_df.iloc[0]

            if pd.isna(target_row['x_index']) or pd.isna(trade_row[marker_price_col]): continue
            plot_x = target_row['x_index']
            trade_price = trade_row[marker_price_col]
            plot_y = trade_price
            plot_kwargs = {'marker': marker_config['marker'], 'color': marker_config['color'], 'markersize': marker_config.get('markersize', 8), 'alpha': marker_config.get('alpha', 0.9), 'linestyle': 'None'}
            if 'markeredgewidth' in marker_config:
                plot_kwargs['markeredgewidth'] = marker_config['markeredgewidth']
            ax.plot(plot_x, plot_y, **plot_kwargs)

    def set_custom_datetime_x_labels(ax, df_plot_data, stock_code_local):
        if df_plot_data.empty or 'datetime' not in df_plot_data.columns :
            ax.set_xticks([])
            return

        if not pd.api.types.is_datetime64_any_dtype(df_plot_data['datetime']):
            logging.warning(f"  銘柄 {stock_code_local}: datetime列が不正なため、X軸ラベルは表示できません。")
            ax.set_xticks([])
            return

        tick_indices_to_plot = []
        tick_labels_to_plot = []

        time_0900 = datetime.time(9, 0)
        time_1130 = datetime.time(11, 30)

        # NaTを除外してから比較
        df_plot_data_cleaned = df_plot_data.dropna(subset=['datetime'])

        condition_0900 = df_plot_data_cleaned['datetime'].dt.time == time_0900
        target_rows_0900 = df_plot_data_cleaned[condition_0900].copy()
        for _, row in target_rows_0900.iterrows():
            if pd.notna(row['x_index']) and pd.notna(row['datetime']):
                tick_indices_to_plot.append(row['x_index'])
                tick_labels_to_plot.append(row['datetime'].strftime('%y-%m-%d-9'))

        condition_1130 = df_plot_data_cleaned['datetime'].dt.time == time_1130
        target_rows_1130 = df_plot_data_cleaned[condition_1130].copy()
        for _, row in target_rows_1130.iterrows():
            if pd.notna(row['x_index']) and pd.notna(row['datetime']):
                tick_indices_to_plot.append(row['x_index'])
                tick_labels_to_plot.append(row['datetime'].strftime('%y-%m-%d-12'))

        if tick_indices_to_plot:
            unique_ticks = {}
            for x_idx, label in zip(tick_indices_to_plot, tick_labels_to_plot):
                if x_idx not in unique_ticks:
                    unique_ticks[x_idx] = label

            sorted_unique_ticks = sorted(unique_ticks.items())

            final_tick_indices = [item[0] for item in sorted_unique_ticks]
            final_tick_labels = [item[1] for item in sorted_unique_ticks]
            ax.set_xticks(final_tick_indices)
            ax.set_xticklabels(final_tick_labels, rotation=45, ha="right", fontsize=6)
        else:
            ax.set_xticks([])

        min_x = df_plot_data['x_index'].min(); max_x = df_plot_data['x_index'].max()
        if pd.notna(min_x) and pd.notna(max_x): ax.set_xlim(min_x - 0.8, max_x + 0.8)
        else: ax.set_xlim(-0.8, 0.8)
        if not tick_indices_to_plot: ax.set_xticks([])


    # --- 1段目: 環境認識足 (OHLC, MA, BB, VWAP) ---
    ax1.set_ylabel("Context Price", fontsize=9)
    plot_context_ohlc_success = plot_ohlc_gapless(ax1, df_context_plot, "Context Candlestick", display_x_labels=False)
    if plot_context_ohlc_success:
        if not df_ctx_indicators_plot.empty and 'x_index' in df_ctx_indicators_plot.columns:
            # EMA
            ema_short_period_ctx = p.get('EMA_SETTINGS_CONTEXT_PERIOD_SHORT_GC')
            ema_long_period_ctx = p.get('EMA_SETTINGS_CONTEXT_PERIOD_LONG_GC')
            ema_short_col_ctx = f"EMA{ema_short_period_ctx}_ctx_ITS" if ema_short_period_ctx else None
            ema_long_col_ctx = f"EMA{ema_long_period_ctx}_ctx_ITS" if ema_long_period_ctx else None

            if ema_short_col_ctx and ema_short_col_ctx in df_ctx_indicators_plot.columns and df_ctx_indicators_plot[ema_short_col_ctx].notna().any():
                ax1.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot[ema_short_col_ctx], label=f"EMA{ema_short_period_ctx}", c='c', lw=0.8)
            if ema_long_col_ctx and ema_long_col_ctx in df_ctx_indicators_plot.columns and df_ctx_indicators_plot[ema_long_col_ctx].notna().any():
                ax1.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot[ema_long_col_ctx], label=f"EMA{ema_long_period_ctx}", c='m', lw=0.8)

            # SMA
            sma1_period_ctx = p.get('SMA_SETTINGS_CONTEXT_PERIOD_1')
            sma2_period_ctx = p.get('SMA_SETTINGS_CONTEXT_PERIOD_2')
            sma1_col_ctx = f"SMA{sma1_period_ctx}_ctx_ITS" if sma1_period_ctx else None
            sma2_col_ctx = f"SMA{sma2_period_ctx}_ctx_ITS" if sma2_period_ctx else None

            if sma1_col_ctx and sma1_col_ctx in df_ctx_indicators_plot.columns and df_ctx_indicators_plot[sma1_col_ctx].notna().any():
                ax1.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot[sma1_col_ctx], label=f"SMA{sma1_period_ctx}", c='blue', ls='--', lw=0.7)
            if sma2_col_ctx and sma2_col_ctx in df_ctx_indicators_plot.columns and df_ctx_indicators_plot[sma2_col_ctx].notna().any():
                ax1.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot[sma2_col_ctx], label=f"SMA{sma2_period_ctx}", c='purple', ls='--', lw=0.7)

            # VWAP
            if 'VWAP_daily_ctx_ITS' in df_ctx_indicators_plot.columns and df_ctx_indicators_plot['VWAP_daily_ctx_ITS'].notna().any():
                ax1.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot['VWAP_daily_ctx_ITS'], label='VWAP_ctx', c='yellow', lw=1.5, ls='-')

            # 環境認識足のボリンジャーバンド描画
            bb_mid_col_ctx = 'BB_Middle_ctx_ITS'
            if bb_mid_col_ctx not in df_ctx_indicators_plot.columns: bb_mid_col_ctx = 'BB_Middle_ctx'

            if bb_mid_col_ctx in df_ctx_indicators_plot.columns and df_ctx_indicators_plot[bb_mid_col_ctx].notna().any():
                ax1.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot[bb_mid_col_ctx],label='BB_Mid_ctx', c='lime', ls='-', lw=0.9)
                ls_map = {1: ':', 2: '--', 3: '-.'}; alpha_map = {1:0.6, 2:0.7, 3:0.8}
                for i in range(1,4):
                    bb_upper_col_ctx = f'BB_Upper_ctx_{i}dev_ITS'
                    if bb_upper_col_ctx not in df_ctx_indicators_plot.columns: bb_upper_col_ctx = f'BB_Upper_ctx_{i}dev'
                    bb_lower_col_ctx = f'BB_Lower_ctx_{i}dev_ITS'
                    if bb_lower_col_ctx not in df_ctx_indicators_plot.columns: bb_lower_col_ctx = f'BB_Lower_ctx_{i}dev'

                    if bb_upper_col_ctx in df_ctx_indicators_plot.columns and df_ctx_indicators_plot[bb_upper_col_ctx].notna().any():
                        ax1.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot[bb_upper_col_ctx], c='gray', ls=ls_map[i], lw=0.7, alpha=alpha_map[i], label=f'+{i}σ_ctx')
                    if bb_lower_col_ctx in df_ctx_indicators_plot.columns and df_ctx_indicators_plot[bb_lower_col_ctx].notna().any():
                        ax1.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot[bb_lower_col_ctx], c='gray', ls=ls_map[i], lw=0.7, alpha=alpha_map[i], label=f'-{i}σ_ctx')
            elif not df_ctx_indicators_plot.empty:
                 logging.debug(f"銘柄 {stock_code}: 環境足BBミドルカラム ({bb_mid_col_ctx}) が df_ctx_indicators_plot に見つからないか全てNaN。")


        plot_markers_gapless(ax1, df_context_plot, buy_entries, marker_buy_entry, {'datetime_col': 'entry_date', 'price_col': 'entry_price'})
        plot_markers_gapless(ax1, df_context_plot, sell_entries, marker_sell_entry, {'datetime_col': 'entry_date', 'price_col': 'entry_price'})
        plot_markers_gapless(ax1, df_context_plot, take_profits, marker_tp, {'datetime_col': 'exit_date', 'price_col': 'exit_price'})
        plot_markers_gapless(ax1, df_context_plot, stop_losses, marker_sl, {'datetime_col': 'exit_date', 'price_col': 'exit_price'})
        handles, labels = ax1.get_legend_handles_labels()
        if handles: ax1.legend(handles, labels, fontsize='xx-small', loc='upper left', ncol=max(1, len(handles)//4))

    # --- 2段目: 環境認識足のATR & Volume ---
    ax2.set_ylabel("Context ATR/Vol", fontsize=9)
    ax2_plot_success = False
    if not df_ctx_indicators_plot.empty and 'x_index' in df_ctx_indicators_plot.columns:
        atr_p_ctx_chart = p.get('ATR_SETTINGS_PERIOD_CONTEXT') # config.yamlのキー
        atr_col_name_ctx = f'ATR_{atr_p_ctx_chart}_CTX_Chart_ITS' if atr_p_ctx_chart else None
        if not (atr_p_ctx_chart and atr_col_name_ctx and atr_col_name_ctx in df_ctx_indicators_plot.columns):
            atr_col_name_ctx = f'ATR_{atr_p_ctx_chart}_ctx_ITS' if atr_p_ctx_chart else None
        if not (atr_p_ctx_chart and atr_col_name_ctx and atr_col_name_ctx in df_ctx_indicators_plot.columns):
             atr_col_name_ctx = f"ATR_{atr_p_ctx_chart}_ctx" if atr_p_ctx_chart else None

        if atr_p_ctx_chart and atr_col_name_ctx and atr_col_name_ctx in df_ctx_indicators_plot.columns and df_ctx_indicators_plot[atr_col_name_ctx].notna().any():
            ax2.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot[atr_col_name_ctx], label=f'ATR({atr_p_ctx_chart})_ctx', c='b', lw=0.8)
            ax2.tick_params(axis='y', labelcolor='b', labelsize=8)
            ax2_plot_success = True
        else: ax2.text(0.5, 0.25, f"ATR ({atr_col_name_ctx if atr_col_name_ctx else 'N/A'}) Missing", ha='center', va='center', fontsize=8, color='gray', transform=ax2.transAxes)
    else: ax2.text(0.5, 0.5, "Ctx Ind Data Missing", ha='center', va='center', fontsize=9, color='gray')

    ax2_twin = ax2.twinx()
    ax2_twin_plot_success = False
    if not df_context_plot.empty and 'Volume' in df_context_plot.columns and df_context_plot['Volume'].notna().any() and \
       'Open' in df_context_plot.columns and 'Close' in df_context_plot.columns:
        vol_colors_ctx = ['red' if row['Close'] > row['Open'] else 'green' for idx, row in df_context_plot.iterrows()]
        ax2_twin.bar(df_context_plot['x_index'], df_context_plot['Volume'], label='Vol_ctx', color=vol_colors_ctx, alpha=0.7, width=0.8)
        ax2_twin_plot_success = True
    ax2_twin.set_ylabel('Volume_ctx', c='gray', fontsize=9); ax2_twin.tick_params(axis='y', labelcolor='gray', labelsize=8); ax2_twin.set_ylim(bottom=0)
    if ax2_plot_success or ax2_twin_plot_success: ax2.tick_params(labelbottom=False); ax2.set_xticks([])


    # --- 3段目: 環境認識足のADX (X軸ラベル表示) ---
    ax3.set_ylabel("Context ADX", fontsize=9)
    ax3_plot_success = False
    adx_col_ctx = 'ADX_ctx_ITS'
    if adx_col_ctx not in df_ctx_indicators_plot.columns: adx_col_ctx = 'ADX_ctx'

    if not df_ctx_indicators_plot.empty and adx_col_ctx in df_ctx_indicators_plot.columns and 'x_index' in df_ctx_indicators_plot.columns:
        adx_thresh = p.get('ADX_SETTINGS_CONTEXT_THRESHOLD', 18.0) # config.yamlのキー
        if df_ctx_indicators_plot[adx_col_ctx].notna().any():
            ax3.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot[adx_col_ctx], label='ADX_ctx', color='lime', lw=0.8)
            ax3.axhline(adx_thresh, color='r', linestyle='--', linewidth=0.7, label=f'Thresh({adx_thresh})')
            ax3_plot_success = True
            if ax3.has_data(): ax3.legend(fontsize='x-small', loc='upper left'); ax3.tick_params(axis='y', labelsize=8)
        else:
             ax3.text(0.5, 0.5, "Ctx ADX Data (All NaN)", ha='center', va='center', fontsize=9, color='gray')
    else: ax3.text(0.5, 0.5, "Ctx ADX Data Missing", ha='center', va='center', fontsize=9, color='gray')

    if not df_context_plot.empty:
        set_custom_datetime_x_labels(ax3, df_context_plot, stock_code)
    else:
        ax3.set_xticks([])
        min_x_ax3 = -0.8; max_x_ax3 = 0.8
        if not df_ctx_indicators_plot.empty and 'x_index' in df_ctx_indicators_plot.columns:
             if df_ctx_indicators_plot['x_index'].notna().any():
                min_x_val = df_ctx_indicators_plot['x_index'].min(); max_x_val = df_ctx_indicators_plot['x_index'].max()
                if pd.notna(min_x_val) and pd.notna(max_x_val): min_x_ax3=min_x_val; max_x_ax3=max_x_val
        ax3.set_xlim(min_x_ax3 - 0.8, max_x_ax3 + 0.8)


    # --- 4段目: 実行足 (OHLC, MA, BB, VWAP) ---
    ax4.set_ylabel("Exec Price", fontsize=9)
    plot_exec_ohlc_success = plot_ohlc_gapless(ax4, df_exec_plot, "Execution Candlestick", display_x_labels=False)
    if plot_exec_ohlc_success:
        if not df_with_signals_plot.empty and 'x_index' in df_with_signals_plot.columns:
            # EMA
            ema_s_p_exec = p.get('EMA_SETTINGS_SHORT_EXEC_CHART') # config.yamlのキー
            ema_l_p_exec = p.get('EMA_SETTINGS_LONG_EXEC_CHART')   # config.yamlのキー
            ema_s_col_exec = f"EMA{ema_s_p_exec}_exec" if ema_s_p_exec else None
            ema_l_col_exec = f"EMA{ema_l_p_exec}_exec" if ema_l_p_exec else None

            if ema_s_col_exec and ema_s_col_exec in df_with_signals_plot.columns and df_with_signals_plot[ema_s_col_exec].notna().any():
                ax4.plot(df_with_signals_plot['x_index'], df_with_signals_plot[ema_s_col_exec], label=f"EMA{ema_s_p_exec}", c='c', lw=0.8)
            if ema_l_col_exec and ema_l_col_exec in df_with_signals_plot.columns and df_with_signals_plot[ema_l_col_exec].notna().any():
                ax4.plot(df_with_signals_plot['x_index'], df_with_signals_plot[ema_l_col_exec], label=f"EMA{ema_l_p_exec}", c='m', lw=0.8)

            # SMA
            sma1_p_exec = p.get('SMA_SETTINGS_EXEC_PERIOD_1') # config.yamlのキー
            sma2_p_exec = p.get('SMA_SETTINGS_EXEC_PERIOD_2') # config.yamlのキー
            sma1_col_exec = f"SMA{sma1_p_exec}_exec" if sma1_p_exec else None
            sma2_col_exec = f"SMA{sma2_p_exec}_exec" if sma2_p_exec else None

            if sma1_col_exec and sma1_col_exec in df_with_signals_plot.columns and df_with_signals_plot[sma1_col_exec].notna().any():
                ax4.plot(df_with_signals_plot['x_index'], df_with_signals_plot[sma1_col_exec], label=f"SMA{sma1_p_exec}", c='blue', ls='--', lw=0.7)
            if sma2_col_exec and sma2_col_exec in df_with_signals_plot.columns and df_with_signals_plot[sma2_col_exec].notna().any():
                ax4.plot(df_with_signals_plot['x_index'], df_with_signals_plot[sma2_col_exec], label=f"SMA{sma2_p_exec}", c='purple', ls='--', lw=0.7)

            # VWAP
            if 'VWAP_daily_exec' in df_with_signals_plot.columns and df_with_signals_plot['VWAP_daily_exec'].notna().any():
                ax4.plot(df_with_signals_plot['x_index'], df_with_signals_plot['VWAP_daily_exec'], label='VWAP_exec', c='yellow', lw=1.5, ls='-')
            # Bollinger Bands
            if 'BB_Middle_exec' in df_with_signals_plot.columns and df_with_signals_plot['BB_Middle_exec'].notna().any():
                ax4.plot(df_with_signals_plot['x_index'], df_with_signals_plot['BB_Middle_exec'],label='BB_Mid_exec', c='lime', ls='-', lw=0.9)
                ls_map = {1: ':', 2: '--', 3: '-.'}; alpha_map = {1:0.6, 2:0.7, 3:0.8}
                for i in range(1,4):
                    if f'BB_Upper_exec_{i}dev' in df_with_signals_plot.columns and df_with_signals_plot[f'BB_Upper_exec_{i}dev'].notna().any():
                        ax4.plot(df_with_signals_plot['x_index'], df_with_signals_plot[f'BB_Upper_exec_{i}dev'], c='gray', ls=ls_map[i], lw=0.7, alpha=alpha_map[i], label=f'+{i}σ_exec')
                    if f'BB_Lower_exec_{i}dev' in df_with_signals_plot.columns and df_with_signals_plot[f'BB_Lower_exec_{i}dev'].notna().any():
                        ax4.plot(df_with_signals_plot['x_index'], df_with_signals_plot[f'BB_Lower_exec_{i}dev'], c='gray', ls=ls_map[i], lw=0.7, alpha=alpha_map[i], label=f'-{i}σ_exec')
        plot_markers_gapless(ax4, df_exec_plot, buy_entries, marker_buy_entry, {'datetime_col': 'entry_date', 'price_col': 'entry_price'})
        plot_markers_gapless(ax4, df_exec_plot, sell_entries, marker_sell_entry, {'datetime_col': 'entry_date', 'price_col': 'entry_price'})
        plot_markers_gapless(ax4, df_exec_plot, take_profits, marker_tp, {'datetime_col': 'exit_date', 'price_col': 'exit_price'})
        plot_markers_gapless(ax4, df_exec_plot, stop_losses, marker_sl, {'datetime_col': 'exit_date', 'price_col': 'exit_price'})
        handles_ax4, labels_ax4 = ax4.get_legend_handles_labels()
        if handles_ax4: ax4.legend(handles_ax4, labels_ax4, fontsize='xx-small', loc='upper left', ncol=max(1, len(handles_ax4)//4))


    # --- 5段目: 実行足のスローストキャスティクス ---
    ax5.set_ylabel("Exec Stoch", fontsize=9)
    ax5_plot_success = False
    if not df_with_signals_plot.empty and 'STOCH_K_exec' in df_with_signals_plot.columns and 'STOCH_D_exec' in df_with_signals_plot.columns and 'x_index' in df_with_signals_plot.columns:
        if df_with_signals_plot['STOCH_K_exec'].notna().any() or df_with_signals_plot['STOCH_D_exec'].notna().any() :
            ob = p.get('STOCH_SETTINGS_OVERBOUGHT_LEVEL', 70.0) # config.yamlのキー
            os_level = p.get('STOCH_SETTINGS_OVERSOLD_LEVEL', 30.0) # config.yamlのキー
            ax5.plot(df_with_signals_plot['x_index'], df_with_signals_plot['STOCH_K_exec'], label='%K', c='orange', lw=0.8)
            ax5.plot(df_with_signals_plot['x_index'], df_with_signals_plot['STOCH_D_exec'], label='%D', c='dodgerblue', lw=0.8)
            ax5.axhline(ob, c='r', ls='--', lw=0.7, label=f'OB({ob})'); ax5.axhline(os_level, c='g', ls='--', lw=0.7, label=f'OS({os_level})')
            ax5.set_ylim(0, 100);
            ax5_plot_success = True
            if ax5.has_data(): ax5.legend(fontsize='x-small', loc='upper left'); ax5.tick_params(axis='y', labelsize=8)
        else:
            ax5.text(0.5, 0.5, "Exec Stoch (All NaN)", ha='center', va='center', fontsize=9, color='gray')
    else: ax5.text(0.5, 0.5, "Exec Stoch Data Missing", ha='center', va='center', fontsize=9, color='gray')
    if ax5_plot_success: ax5.tick_params(labelbottom=False); ax5.set_xticks([])


    # --- 6段目: 実行足のMACD
    ax6.set_ylabel("Exec MACD", fontsize=9)
    ax6_plot_success = False
    if not df_with_signals_plot.empty and 'MACD_exec' in df_with_signals_plot.columns and \
    'MACDsignal_exec' in df_with_signals_plot.columns and 'MACDhist_exec' in df_with_signals_plot.columns and \
    'x_index' in df_with_signals_plot.columns:
        if df_with_signals_plot['MACD_exec'].notna().any() or df_with_signals_plot['MACDsignal_exec'].notna().any():
            ax6.plot(df_with_signals_plot['x_index'], df_with_signals_plot['MACD_exec'], label='MACD', color='cyan', lw=0.8)
            ax6.plot(df_with_signals_plot['x_index'], df_with_signals_plot['MACDsignal_exec'], label='Signal', c='darkorange', lw=0.8)
            hist_colors = np.where(df_with_signals_plot['MACDhist_exec'] > 0, 'red', 'green')
            ax6.bar(df_with_signals_plot['x_index'], df_with_signals_plot['MACDhist_exec'], label='Hist', color=hist_colors, width=0.8, alpha=0.5)
            ax6.axhline(0, color='grey', linestyle='--', linewidth=0.5);

            # MACDヒストグラムEMAのデータを取得してプロット用に保持
            if 'MACDhist_EMA_exec' in df_with_signals_plot.columns:
                macd_hist_ema_data = df_with_signals_plot['MACDhist_EMA_exec']
            else:
                macd_hist_ema_data = None

            ax6_plot_success = True
            if ax6.has_data(): ax6.legend(fontsize='x-small', loc='upper left'); ax6.tick_params(axis='y', labelsize=8)
        else:
            ax6.text(0.5, 0.5, "Exec MACD (All NaN)", ha='center', va='center', fontsize=9, color='gray')
    else: ax6.text(0.5, 0.5, "Exec MACD Data Missing", ha='center', va='center', fontsize=9, color='gray')
    if ax6_plot_success: ax6.tick_params(labelbottom=False); ax6.set_xticks([])

    # MACDヒストグラムEMAをプロット
    if macd_hist_ema_data is not None:
        ax6.plot(df_with_signals_plot['x_index'], macd_hist_ema_data, label='Hist EMA', color='purple', lw=0.8)
        if ax6.has_data(): ax6.legend(fontsize='x-small', loc='upper left'); ax6.tick_params(axis='y', labelsize=8)


    # --- 7段目: 実行足のATR & Volume (X軸ラベル表示) ---
    ax7.set_ylabel("Exec ATR/Vol", fontsize=9)
    ax7_plot_success = False; ax7_twin_plot_success = False
    if not df_exec_plot.empty:
        atr_p_exec_chart = p.get('ATR_SETTINGS_PERIOD_EXEC') # config.yamlのキー
        atr_col_name_exec = f'ATR_{atr_p_exec_chart}_EXEC_Chart' if atr_p_exec_chart else None

        if not df_with_signals_plot.empty and atr_col_name_exec and atr_col_name_exec in df_with_signals_plot.columns and \
           'x_index' in df_with_signals_plot.columns:
            if df_with_signals_plot[atr_col_name_exec].notna().any():
                ax7.plot(df_with_signals_plot['x_index'], df_with_signals_plot[atr_col_name_exec], label=f'ATR({atr_p_exec_chart})_exec', c='b', lw=0.8)
                ax7.tick_params(axis='y', labelcolor='b', labelsize=8)
                ax7_plot_success = True
            else:
                ax7.text(0.5, 0.25, "ATR (All NaN)", ha='center', va='center', fontsize=8, color='gray', transform=ax7.transAxes)
        elif not df_with_signals_plot.empty :
             ax7.text(0.5, 0.25, f"ATR ({atr_col_name_exec if atr_col_name_exec else 'N/A'}) Missing in Signals DF", ha='center', va='center', fontsize=8, color='gray', transform=ax7.transAxes)
        else:
            ax7.text(0.5, 0.25, "ATR Data (Signals DF) Missing", ha='center', va='center', fontsize=8, color='gray', transform=ax7.transAxes)

        ax7_twin = ax7.twinx()
        if 'Volume' in df_exec_plot.columns and df_exec_plot['Volume'].notna().any() and \
            'Open' in df_exec_plot.columns and 'Close' in df_exec_plot.columns:
            vol_colors_exec = ['red' if row['Close'] > row['Open'] else 'green' for idx, row in df_exec_plot.iterrows()]
            ax7_twin.bar(df_exec_plot['x_index'], df_exec_plot['Volume'], label='Vol_exec', color=vol_colors_exec, alpha=0.7, width=0.8)
            ax7_twin_plot_success = True
        ax7_twin.set_ylabel('Volume_exec', c='gray', fontsize=9); ax7_twin.tick_params(axis='y', labelcolor='gray', labelsize=8); ax7_twin.set_ylim(bottom=0)

        set_custom_datetime_x_labels(ax7, df_exec_plot, stock_code)
    else:
        ax7.text(0.5, 0.5, "Exec Data Missing", ha='center', va='center', fontsize=9, color='gray')
        ax7.set_xticks([])
        ax7.set_xlim(-0.8, 0.8)

    # --- チャート保存 ---
    try:
        plt.savefig(chart_filepath, dpi=120, format='svg')
        logging.info(f"  銘柄 {stock_code}: チャートを保存しました: {chart_filepath}")
    except Exception as e:
        logging.error(f"  銘柄 {stock_code}: チャートの保存中にエラーが発生しました: {e}", exc_info=True)
    finally:
        plt.close(fig)