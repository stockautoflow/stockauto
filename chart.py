# chart.py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import pandas as pd
import numpy as np
import os
import logging
import datetime

# --- 定数の定義 ---
NUM_BARS_TO_DISPLAY = 2000
CHART_BASE_WIDTH_INCHES = 10.0
CHART_WIDTH_PER_BAR_INCHES = 0.08
MIN_CHART_WIDTH_INCHES = 16.0
MAX_CHART_WIDTH_INCHES = 50.0
CHART_HEIGHT_INCHES = 28.0

def plot_chart_for_stock(
    df_context_orig, df_exec_orig, df_with_signals_orig, trade_history_orig,
    stock_code, strategy_params, chart_output_dir,
    base_filename_parts
    ):
    logging.info(f"  銘柄 {stock_code}: チャート生成開始...")
    logging.debug(f"  銘柄 {stock_code}: plot_chart_for_stock CALLED with df_context_orig len={len(df_context_orig) if df_context_orig is not None else 'None'}, df_exec_orig len={len(df_exec_orig) if df_exec_orig is not None else 'None'}, df_with_signals_orig len={len(df_with_signals_orig) if df_with_signals_orig is not None else 'None'}, trade_history_orig len={len(trade_history_orig) if trade_history_orig is not None else 'None'}")
    p = strategy_params

    df_context, df_exec, df_with_signals, trade_history_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    min_display_time, max_display_time = None, None

    base_df_for_period = df_exec_orig if df_exec_orig is not None and not df_exec_orig.empty else (df_context_orig if df_context_orig is not None and not df_context_orig.empty else None)
    logging.debug(f"  銘柄 {stock_code}: base_df_for_period selected, len={len(base_df_for_period) if base_df_for_period is not None else 'None'}")

    if NUM_BARS_TO_DISPLAY > 0 and base_df_for_period is not None and not base_df_for_period.empty: #
        temp_display_df = base_df_for_period.tail(NUM_BARS_TO_DISPLAY).copy()
        if not temp_display_df.empty:
            min_display_time = temp_display_df.index.min()
            max_display_time = temp_display_df.index.max()
            logging.debug(f"  銘柄 {stock_code}: Display period (from base_df_for_period.tail): {min_display_time} to {max_display_time}")
        
        if min_display_time is not None and max_display_time is not None:
            if df_exec_orig is not None and not df_exec_orig.empty:
                df_exec = df_exec_orig[(df_exec_orig.index >= min_display_time) & (df_exec_orig.index <= max_display_time)].copy()
            if df_context_orig is not None and not df_context_orig.empty:
                df_context = df_context_orig[(df_context_orig.index >= min_display_time) & (df_context_orig.index <= max_display_time)].copy()
            if df_with_signals_orig is not None and not df_with_signals_orig.empty:
                df_with_signals = df_with_signals_orig[(df_with_signals_orig.index >= min_display_time) & (df_with_signals_orig.index <= max_display_time)].copy()
            
            if trade_history_orig is not None and not trade_history_orig.empty:
                try:
                    th_copy = trade_history_orig.copy()
                    if 'entry_date' in th_copy.columns:
                        th_copy['entry_date_dt'] = pd.to_datetime(th_copy['entry_date'], errors='coerce')
                    if 'exit_date' in th_copy.columns:
                         th_copy['exit_date_dt'] = pd.to_datetime(th_copy['exit_date'], errors='coerce')
                    
                    entry_date_exists = 'entry_date_dt' in th_copy.columns and th_copy['entry_date_dt'].notna().any()
                    exit_date_exists = 'exit_date_dt' in th_copy.columns and th_copy['exit_date_dt'].notna().any()

                    if entry_date_exists and exit_date_exists:
                        trade_history_df = th_copy[
                            (th_copy['entry_date_dt'] <= max_display_time) &
                            (th_copy['exit_date_dt'] >= min_display_time)
                        ].copy()
                    elif entry_date_exists:
                         trade_history_df = th_copy[th_copy['entry_date_dt'] <= max_display_time].copy()
                    elif exit_date_exists:
                         trade_history_df = th_copy[th_copy['exit_date_dt'] >= min_display_time].copy()
                    else:
                        trade_history_df = th_copy.copy()
                    trade_history_df.drop(columns=['entry_date_dt', 'exit_date_dt'], inplace=True, errors='ignore')
                    logging.debug(f"  銘柄 {stock_code}: trade_history_df filtered, len={len(trade_history_df)}")
                except Exception as e_th:
                    logging.error(f"  銘柄 {stock_code}: トレード履歴の期間絞り込みエラー: {e_th}")
                    trade_history_df = pd.DataFrame()
        else: # min_display_time or max_display_time is None
            df_exec = df_exec_orig.tail(NUM_BARS_TO_DISPLAY).copy() if df_exec_orig is not None and not df_exec_orig.empty else pd.DataFrame()
            df_context = df_context_orig.tail(NUM_BARS_TO_DISPLAY).copy() if df_context_orig is not None and not df_context_orig.empty else pd.DataFrame()
            df_with_signals = df_with_signals_orig.tail(NUM_BARS_TO_DISPLAY).copy() if df_with_signals_orig is not None and not df_with_signals_orig.empty else pd.DataFrame()
            trade_history_df = trade_history_orig.copy() if trade_history_orig is not None else pd.DataFrame()
            logging.debug(f"  銘柄 {stock_code}: Display period could not be determined from base, using tail({NUM_BARS_TO_DISPLAY}) on all DFs.")
    else: # NUM_BARS_TO_DISPLAY <= 0 (全期間表示)
        df_context = df_context_orig.copy() if df_context_orig is not None else pd.DataFrame()
        df_exec = df_exec_orig.copy() if df_exec_orig is not None else pd.DataFrame()
        df_with_signals = df_with_signals_orig.copy() if df_with_signals_orig is not None else pd.DataFrame()
        trade_history_df = trade_history_orig.copy() if trade_history_orig is not None else pd.DataFrame()
        logging.debug(f"  銘柄 {stock_code}: Displaying all bars (NUM_BARS_TO_DISPLAY <= 0).")

    logging.debug(f"  銘柄 {stock_code}: df_context len={len(df_context)}, df_exec len={len(df_exec)}, df_with_signals len={len(df_with_signals)}, trade_history_df len={len(trade_history_df)} after period filtering.")

    if df_context.empty and df_exec.empty:
        logging.warning(f"  銘柄 {stock_code}: 表示対象期間の環境認識足・実行足データが共に空のためスキップ。")
        return

    def _prepare_plot_df(df_input, df_name_for_log=""):
        if df_input is None or df_input.empty:
            logging.debug(f"  銘柄 {stock_code}: _prepare_plot_df for {df_name_for_log} - input df is None or empty. Returning empty DF.")
            return pd.DataFrame(columns=['datetime', 'x_index', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        logging.debug(f"  銘柄 {stock_code}: _prepare_plot_df for {df_name_for_log} - input df shape {df_input.shape}, index type: {type(df_input.index)}")

        if not isinstance(df_input.index, pd.DatetimeIndex):
            if 'datetime' in df_input.columns:
                try:
                    df_input = df_input.set_index('datetime', drop=True)
                    if not isinstance(df_input.index, pd.DatetimeIndex): 
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
        
        current_index_name = df_input.index.name if df_input.index.name is not None else 'datetime'
        df_plot = df_input.reset_index().rename(columns={current_index_name: 'datetime'})
        if 'datetime' in df_plot.columns:
            df_plot['datetime'] = pd.to_datetime(df_plot['datetime'], errors='coerce')
            if df_plot['datetime'].isna().any():
                logging.warning(f"  銘柄 {stock_code}: {df_name_for_log} のdatetime列にNaTが含まれています。")
        else: 
             logging.warning(f"  銘柄 {stock_code}: {df_name_for_log} にdatetime列がありません。")
             return pd.DataFrame(columns=['datetime', 'x_index', 'Open', 'High', 'Low', 'Close', 'Volume'])
        if not df_plot.empty: df_plot['x_index'] = np.arange(len(df_plot))
        else: df_plot['x_index'] = pd.Series(dtype='int')
        logging.debug(f"  銘柄 {stock_code}: _prepare_plot_df for {df_name_for_log} - output df_plot shape {df_plot.shape}")
        return df_plot

    df_context_plot = _prepare_plot_df(df_context, "df_context")
    df_exec_plot = _prepare_plot_df(df_exec, "df_exec")
    logging.debug(f"  銘柄 {stock_code}: df_context_plot len={len(df_context_plot)}, df_exec_plot len={len(df_exec_plot)} after _prepare_plot_df.")

    df_with_signals_plot = pd.DataFrame()
    if df_with_signals is not None and not df_with_signals.empty and \
       df_exec_plot is not None and not df_exec_plot.empty and 'datetime' in df_exec_plot.columns:
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
                    df_ws_temp_reset, on='datetime', how='left'
                ).dropna(subset=['x_index'])
                if 'x_index' in df_with_signals_plot.columns:
                    df_with_signals_plot['x_index'] = df_with_signals_plot['x_index'].astype(int)
            else: logging.warning(f"  銘柄 {stock_code}: df_exec_plotまたはdf_ws_temp_resetのdatetime型が不正です。")
        else: logging.warning(f"  銘柄 {stock_code}: df_ws_temp のインデックスがDatetimeIndexではありません。")
    if df_with_signals_plot.empty and df_exec_plot is not None and not df_exec_plot.empty :
        logging.warning(f"  銘柄 {stock_code}: df_with_signals_plot が空。df_exec_plotを指標なしで使用。")
        df_with_signals_plot = df_exec_plot.copy()
    logging.debug(f"  銘柄 {stock_code}: df_with_signals_plot len={len(df_with_signals_plot)} after merge with exec_plot.")


    df_ctx_indicators_plot = pd.DataFrame()
    if df_with_signals is not None and not df_with_signals.empty and \
        df_context_plot is not None and not df_context_plot.empty and 'datetime' in df_context_plot.columns:
        df_ws_temp_for_ctx = df_with_signals.copy()
        if not isinstance(df_ws_temp_for_ctx.index, pd.DatetimeIndex):
            if 'datetime' in df_ws_temp_for_ctx.columns: df_ws_temp_for_ctx = df_ws_temp_for_ctx.set_index('datetime')
            elif df_ws_temp_for_ctx.index.name == 'datetime': pass
        if isinstance(df_ws_temp_for_ctx.index, pd.DatetimeIndex) and \
           pd.api.types.is_datetime64_any_dtype(df_context_plot['datetime']):
            df_ws_temp_for_ctx = df_ws_temp_for_ctx.sort_index()
            df_context_plot_sorted = df_context_plot.sort_values('datetime')
            ctx_plot_tz = df_context_plot_sorted['datetime'].dt.tz; ws_tz = df_ws_temp_for_ctx.index.tz
            if ctx_plot_tz is not None and ws_tz is None:
                try: df_ws_temp_for_ctx = df_ws_temp_for_ctx.tz_localize(ctx_plot_tz, ambiguous='infer', nonexistent='NaT').dropna(subset=[df_ws_temp_for_ctx.index.name])
                except TypeError: df_ws_temp_for_ctx = df_ws_temp_for_ctx.tz_convert(ctx_plot_tz)
            elif ctx_plot_tz is None and ws_tz is not None: df_ws_temp_for_ctx = df_ws_temp_for_ctx.tz_localize(None)
            elif ctx_plot_tz is not None and ws_tz is not None and str(ctx_plot_tz) != str(ws_tz): df_ws_temp_for_ctx = df_ws_temp_for_ctx.tz_convert(ctx_plot_tz)
            ctx_indicator_cols = [col for col in df_ws_temp_for_ctx.columns if '_ctx_ITS' in col or col.endswith('_ctx')]
            bb_cols_ctx = ['BB_Middle_ctx_ITS', 'BB_Upper_ctx_ITS', 'BB_Lower_ctx_ITS', 'BB_Middle_ctx', 'BB_Upper_ctx', 'BB_Lower_ctx']
            for i in range(1, 4):
                bb_cols_ctx.append(f'BB_Upper_ctx_{i}dev_ITS'); bb_cols_ctx.append(f'BB_Lower_ctx_{i}dev_ITS')
                bb_cols_ctx.append(f'BB_Upper_ctx_{i}dev'); bb_cols_ctx.append(f'BB_Lower_ctx_{i}dev')
            valid_ctx_indicator_cols = [col for col in ctx_indicator_cols if col in df_ws_temp_for_ctx.columns]
            valid_bb_cols_ctx = [col for col in bb_cols_ctx if col in df_ws_temp_for_ctx.columns]
            all_cols_to_merge_ctx = list(set(valid_ctx_indicator_cols + valid_bb_cols_ctx))
            if not all_cols_to_merge_ctx: logging.warning(f"  銘柄 {stock_code}: df_with_signals に環境認識指標カラムが見つかりません。")
            if not df_ws_temp_for_ctx.empty and all_cols_to_merge_ctx:
                df_ctx_indicators_plot = pd.merge_asof(
                    left=df_context_plot_sorted[['datetime', 'x_index']],
                    right=df_ws_temp_for_ctx[all_cols_to_merge_ctx], on='datetime', direction='backward'
                )
            if df_ctx_indicators_plot.empty and not df_context_plot_sorted.empty:
                 logging.warning(f"  銘柄 {stock_code}: 環境認識指標のマージ結果が空。df_context_plotを代替。")
                 df_ctx_indicators_plot = df_context_plot_sorted[['datetime', 'x_index']].copy()
        else: logging.warning(f"  銘柄 {stock_code}: 環境足指標のマージに必要なdatetime情報が不足。")
    if df_ctx_indicators_plot.empty and df_context_plot is not None and not df_context_plot.empty:
        df_ctx_indicators_plot = df_context_plot[['datetime', 'x_index']].copy()
    logging.debug(f"  銘柄 {stock_code}: df_ctx_indicators_plot len={len(df_ctx_indicators_plot)} after merge with context_plot.")


    num_bars_for_width = len(df_exec_plot) if df_exec_plot is not None and not df_exec_plot.empty else (len(df_context_plot) if df_context_plot is not None else 0)
    if num_bars_for_width > 0 :
        calculated_width = CHART_BASE_WIDTH_INCHES + num_bars_for_width * CHART_WIDTH_PER_BAR_INCHES
        figure_width = max(MIN_CHART_WIDTH_INCHES, min(calculated_width, MAX_CHART_WIDTH_INCHES))
    else: figure_width = MIN_CHART_WIDTH_INCHES
    logging.debug(f"  銘柄 {stock_code}: Figure width calculated: {figure_width} inches for {num_bars_for_width} bars.")

    period_str = f"{NUM_BARS_TO_DISPLAY}bars" if NUM_BARS_TO_DISPLAY > 0 else "All"
    chart_filename = f"Chart_{period_str}_Gapless_{base_filename_parts[3]}_{base_filename_parts[0]}_{base_filename_parts[1]}_{base_filename_parts[2]}.svg"
    chart_filepath = os.path.join(chart_output_dir, chart_filename)
    os.makedirs(chart_output_dir, exist_ok=True)

    fig = plt.figure(figsize=(figure_width, CHART_HEIGHT_INCHES), constrained_layout=True)
    plt.style.use('seaborn-v0_8-darkgrid')
    gs = fig.add_gridspec(7, 1, height_ratios=[3, 1, 1, 3, 1, 1, 1], hspace=0.05)

    ax1 = fig.add_subplot(gs[0]);
    ax2 = fig.add_subplot(gs[1], sharex=ax1 if df_context_plot is not None and not df_context_plot.empty else None);
    ax3 = fig.add_subplot(gs[2], sharex=ax1 if df_context_plot is not None and not df_context_plot.empty else None)
    ax4 = fig.add_subplot(gs[3])
    ax5 = fig.add_subplot(gs[4], sharex=ax4 if df_exec_plot is not None and not df_exec_plot.empty else None);
    ax6 = fig.add_subplot(gs[5], sharex=ax4 if df_exec_plot is not None and not df_exec_plot.empty else None);
    ax7 = fig.add_subplot(gs[6], sharex=ax4 if df_exec_plot is not None and not df_exec_plot.empty else None)

    fig.suptitle(f"Chart for {stock_code} ({base_filename_parts[1]}, {period_str}, Gapless) - Strategy: {base_filename_parts[0]} - Data: {base_filename_parts[2]}", fontsize=16, y=0.995)

    candle_colors = {'up': 'red', 'down': 'green'}
    marker_buy_entry = {'marker': '^', 'color': 'orange', 'markersize': 10, 'label': 'Buy Entry'}
    marker_sell_entry = {'marker': 'v', 'color': 'lime', 'markersize': 10, 'label': 'Sell Entry'}
    marker_tp = {'marker': 'o', 'color': 'cyan', 'markersize': 8, 'alpha':0.7, 'label': 'Take Profit'}
    marker_sl = {'marker': 'x', 'color': 'cyan', 'markersize': 10, 'alpha':0.9, 'label': 'Stop Loss', 'markeredgewidth': 2}

    buy_entries_for_plot = pd.DataFrame()
    sell_entries_for_plot = pd.DataFrame()
    take_profits_for_plot = pd.DataFrame()
    stop_losses_for_plot = pd.DataFrame()

    if trade_history_df is not None and not trade_history_df.empty:
        th_copy_for_markers = trade_history_df.copy()
        logging.debug(f"  銘柄 {stock_code}: plot_chart_for_stock - th_copy_for_markers (渡されたtrade_history_df):\n{th_copy_for_markers.to_string() if not th_copy_for_markers.empty else 'Empty'}")

        if 'entry_date' in th_copy_for_markers.columns:
            th_copy_for_markers['entry_date'] = pd.to_datetime(th_copy_for_markers['entry_date'], errors='coerce')
        if 'exit_date' in th_copy_for_markers.columns:
             th_copy_for_markers['exit_date'] = pd.to_datetime(th_copy_for_markers['exit_date'], errors='coerce')

        if 'type' in th_copy_for_markers.columns:
            valid_entry_long_condition = (th_copy_for_markers['type'] == 'Long') & \
                                         th_copy_for_markers['entry_date'].notna() & \
                                         th_copy_for_markers['entry_price'].notna()
            buy_entries_for_plot = th_copy_for_markers[valid_entry_long_condition].copy()

            valid_entry_short_condition = (th_copy_for_markers['type'] == 'Short') & \
                                          th_copy_for_markers['entry_date'].notna() & \
                                          th_copy_for_markers['entry_price'].notna()
            sell_entries_for_plot = th_copy_for_markers[valid_entry_short_condition].copy()
        
        logging.debug(f"  銘柄 {stock_code}: buy_entries_for_plot (after filtering):\n{buy_entries_for_plot.to_string() if not buy_entries_for_plot.empty else 'Empty'}")
        logging.debug(f"  銘柄 {stock_code}: sell_entries_for_plot (after filtering):\n{sell_entries_for_plot.to_string() if not sell_entries_for_plot.empty else 'Empty'}")

        if 'exit_type' in th_copy_for_markers.columns and 'exit_date' in th_copy_for_markers.columns:
            valid_exits = th_copy_for_markers[
                th_copy_for_markers['exit_date'].notna() & th_copy_for_markers['exit_price'].notna()
            ]
            if not valid_exits.empty:
                take_profits_for_plot = valid_exits[valid_exits['exit_type'].astype(str).str.contains("TP", case=False, na=False)].copy()
                stop_losses_for_plot = valid_exits[valid_exits['exit_type'].astype(str).str.contains("SL", case=False, na=False)].copy()
        
        logging.debug(f"  銘柄 {stock_code}: take_profits_for_plot:\n{take_profits_for_plot.to_string() if not take_profits_for_plot.empty else 'Empty'}")
        logging.debug(f"  銘柄 {stock_code}: stop_losses_for_plot:\n{stop_losses_for_plot.to_string() if not stop_losses_for_plot.empty else 'Empty'}")


    def plot_ohlc_gapless(ax, df_ohlc_plot_data, ohlc_title, display_x_labels=False):
        if df_ohlc_plot_data is None or df_ohlc_plot_data.empty or 'x_index' not in df_ohlc_plot_data.columns or df_ohlc_plot_data['x_index'].isna().all():
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
        if display_x_labels: set_custom_datetime_x_labels(ax, df_ohlc_plot_data, stock_code)
        else: ax.tick_params(labelbottom=False); ax.set_xticks([])
        return True

    def plot_markers_gapless(ax, df_ohlc_plot_data, trades_df_for_marker, marker_config, trade_event_details):
        if trades_df_for_marker is None or trades_df_for_marker.empty or \
           df_ohlc_plot_data is None or df_ohlc_plot_data.empty or \
           'x_index' not in df_ohlc_plot_data.columns or \
           'datetime' not in df_ohlc_plot_data.columns or df_ohlc_plot_data['datetime'].isna().all(): return
        if not pd.api.types.is_datetime64_any_dtype(df_ohlc_plot_data['datetime']):
            logging.warning(f"  銘柄 {stock_code}: plot_markers_gapless のdatetime列が不正です。マーカープロットをスキップ。")
            return
        marker_datetime_col = trade_event_details['datetime_col']; marker_price_col = trade_event_details['price_col']
        if trades_df_for_marker.empty: return
        for idx, trade_row in trades_df_for_marker.iterrows():
            trade_event_dt_val = trade_row.get(marker_datetime_col)
            if pd.isna(trade_event_dt_val): continue
            trade_event_dt = pd.to_datetime(trade_event_dt_val, errors='coerce')
            if pd.isna(trade_event_dt): continue
            plot_ohlc_datetime_col = df_ohlc_plot_data['datetime']; ohlc_tz = plot_ohlc_datetime_col.dt.tz
            if ohlc_tz is not None:
                if trade_event_dt.tzinfo is None:
                    try: trade_event_dt = trade_event_dt.tz_localize(ohlc_tz, ambiguous='infer', nonexistent='NaT')
                    except TypeError:
                         if trade_event_dt.tzinfo != ohlc_tz: trade_event_dt = trade_event_dt.tz_convert(ohlc_tz)
                elif str(trade_event_dt.tzinfo) != str(ohlc_tz): trade_event_dt = trade_event_dt.tz_convert(ohlc_tz)
            elif trade_event_dt.tzinfo is not None: trade_event_dt = trade_event_dt.tz_localize(None)
            if pd.isna(trade_event_dt): continue
            if df_ohlc_plot_data.empty: continue
            min_ohlc_dt = plot_ohlc_datetime_col.min(); max_ohlc_dt = plot_ohlc_datetime_col.max()
            if pd.isna(min_ohlc_dt) or pd.isna(max_ohlc_dt) or not (min_ohlc_dt <= trade_event_dt <= max_ohlc_dt): continue
            temp_trade_df = pd.DataFrame({'datetime': [trade_event_dt]})
            df_ohlc_plot_data_sorted = df_ohlc_plot_data.sort_values('datetime')
            merged_df = pd.merge_asof(temp_trade_df.sort_values('datetime'), df_ohlc_plot_data_sorted, on='datetime', direction='nearest')
            if merged_df.empty or pd.isna(merged_df['x_index'].iloc[0]): continue
            target_row = merged_df.iloc[0]
            trade_price = trade_row.get(marker_price_col)
            if pd.isna(target_row['x_index']) or pd.isna(trade_price): continue
            plot_x = target_row['x_index']; plot_y = trade_price
            plot_kwargs = {'marker': marker_config['marker'], 'color': marker_config['color'], 
                           'markersize': marker_config.get('markersize', 8), 
                           'alpha': marker_config.get('alpha', 0.9), 'linestyle': 'None'}
            if 'markeredgewidth' in marker_config: plot_kwargs['markeredgewidth'] = marker_config['markeredgewidth']
            ax.plot(plot_x, plot_y, **plot_kwargs)

    def set_custom_datetime_x_labels(ax, df_plot_data, stock_code_local):
        if df_plot_data is None or df_plot_data.empty or 'datetime' not in df_plot_data.columns :
            ax.set_xticks([]); return
        if not pd.api.types.is_datetime64_any_dtype(df_plot_data['datetime']):
            logging.warning(f"  銘柄 {stock_code_local}: datetime列が不正なため、X軸ラベルは表示できません。")
            ax.set_xticks([]); return
        tick_indices_to_plot = []; tick_labels_to_plot = []
        time_0900 = datetime.time(9, 0); time_1130 = datetime.time(11, 30)
        df_plot_data_cleaned = df_plot_data.dropna(subset=['datetime'])
        condition_0900 = df_plot_data_cleaned['datetime'].dt.time == time_0900
        target_rows_0900 = df_plot_data_cleaned[condition_0900].copy()
        for _, row in target_rows_0900.iterrows():
            if pd.notna(row['x_index']) and pd.notna(row['datetime']):
                tick_indices_to_plot.append(row['x_index'])
                tick_labels_to_plot.append(row['datetime'].strftime('%y-%m-%d_9'))
        condition_1130 = df_plot_data_cleaned['datetime'].dt.time == time_1130
        target_rows_1130 = df_plot_data_cleaned[condition_1130].copy()
        for _, row in target_rows_1130.iterrows():
            if pd.notna(row['x_index']) and pd.notna(row['datetime']):
                tick_indices_to_plot.append(row['x_index'])
                tick_labels_to_plot.append(row['datetime'].strftime('%y-%m-%d_12'))
        if tick_indices_to_plot:
            unique_ticks = {x_idx: label for x_idx, label in zip(tick_indices_to_plot, tick_labels_to_plot)}
            sorted_unique_ticks = sorted(unique_ticks.items())
            final_tick_indices = [item[0] for item in sorted_unique_ticks]
            final_tick_labels = [item[1] for item in sorted_unique_ticks]
            ax.set_xticks(final_tick_indices)
            ax.set_xticklabels(final_tick_labels, rotation=45, ha="right", fontsize=6)
        else: ax.set_xticks([])
        min_x = df_plot_data['x_index'].min(); max_x = df_plot_data['x_index'].max()
        if pd.notna(min_x) and pd.notna(max_x): ax.set_xlim(min_x - 0.8, max_x + 0.8)
        else: ax.set_xlim(-0.8, 0.8)
        if not tick_indices_to_plot: ax.set_xticks([])

    # --- 1段目: 環境認識足 ---
    ax1.set_ylabel("Context Price", fontsize=9)
    plot_context_ohlc_success = plot_ohlc_gapless(ax1, df_context_plot, "Context Candlestick", display_x_labels=False)
    if plot_context_ohlc_success:
        if df_ctx_indicators_plot is not None and not df_ctx_indicators_plot.empty and 'x_index' in df_ctx_indicators_plot.columns:
            ema_short_period_ctx = p.get('EMA_SETTINGS_CONTEXT_PERIOD_SHORT_GC'); ema_long_period_ctx = p.get('EMA_SETTINGS_CONTEXT_PERIOD_LONG_GC')
            ema_short_col_ctx = f"EMA{ema_short_period_ctx}_ctx_ITS" if ema_short_period_ctx else None
            ema_long_col_ctx = f"EMA{ema_long_period_ctx}_ctx_ITS" if ema_long_period_ctx else None
            if ema_short_col_ctx and ema_short_col_ctx in df_ctx_indicators_plot.columns and df_ctx_indicators_plot[ema_short_col_ctx].notna().any():
                ax1.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot[ema_short_col_ctx], label=f"EMA{ema_short_period_ctx}", c='c', lw=0.8)
            if ema_long_col_ctx and ema_long_col_ctx in df_ctx_indicators_plot.columns and df_ctx_indicators_plot[ema_long_col_ctx].notna().any():
                ax1.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot[ema_long_col_ctx], label=f"EMA{ema_long_period_ctx}", c='m', lw=0.8)
            sma1_period_ctx = p.get('SMA_SETTINGS_CONTEXT_PERIOD_1'); sma2_period_ctx = p.get('SMA_SETTINGS_CONTEXT_PERIOD_2')
            sma1_col_ctx = f"SMA{sma1_period_ctx}_ctx_ITS" if sma1_period_ctx else None
            sma2_col_ctx = f"SMA{sma2_period_ctx}_ctx_ITS" if sma2_period_ctx else None
            if sma1_col_ctx and sma1_col_ctx in df_ctx_indicators_plot.columns and df_ctx_indicators_plot[sma1_col_ctx].notna().any():
                ax1.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot[sma1_col_ctx], label=f"SMA{sma1_period_ctx}", c='blue', ls='--', lw=0.7)
            if sma2_col_ctx and sma2_col_ctx in df_ctx_indicators_plot.columns and df_ctx_indicators_plot[sma2_col_ctx].notna().any():
                ax1.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot[sma2_col_ctx], label=f"SMA{sma2_period_ctx}", c='purple', ls='--', lw=0.7)
            if 'VWAP_daily_ctx_ITS' in df_ctx_indicators_plot.columns and df_ctx_indicators_plot['VWAP_daily_ctx_ITS'].notna().any():
                ax1.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot['VWAP_daily_ctx_ITS'], label='VWAP_ctx', c='yellow', lw=1.5, ls='-')
            bb_mid_col_ctx = 'BB_Middle_ctx_ITS'
            if bb_mid_col_ctx not in df_ctx_indicators_plot.columns: bb_mid_col_ctx = 'BB_Middle_ctx'
            if bb_mid_col_ctx in df_ctx_indicators_plot.columns and df_ctx_indicators_plot[bb_mid_col_ctx].notna().any():
                ax1.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot[bb_mid_col_ctx],label='BB_Mid_ctx', c='lime', ls='-', lw=0.9)
                ls_map = {1: ':', 2: '--', 3: '-.'}; alpha_map = {1:0.6, 2:0.7, 3:0.8}
                for i in range(1,4):
                    bb_upper_col_ctx = f'BB_Upper_ctx_{i}dev_ITS'; bb_lower_col_ctx = f'BB_Lower_ctx_{i}dev_ITS'
                    if bb_upper_col_ctx not in df_ctx_indicators_plot.columns: bb_upper_col_ctx = f'BB_Upper_ctx_{i}dev'
                    if bb_lower_col_ctx not in df_ctx_indicators_plot.columns: bb_lower_col_ctx = f'BB_Lower_ctx_{i}dev'
                    if bb_upper_col_ctx in df_ctx_indicators_plot.columns and df_ctx_indicators_plot[bb_upper_col_ctx].notna().any():
                        ax1.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot[bb_upper_col_ctx], c='gray', ls=ls_map[i], lw=0.7, alpha=alpha_map[i], label=f'+{i}σ_ctx')
                    if bb_lower_col_ctx in df_ctx_indicators_plot.columns and df_ctx_indicators_plot[bb_lower_col_ctx].notna().any():
                        ax1.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot[bb_lower_col_ctx], c='gray', ls=ls_map[i], lw=0.7, alpha=alpha_map[i], label=f'-{i}σ_ctx')
            elif df_ctx_indicators_plot is not None and not df_ctx_indicators_plot.empty:
                 logging.debug(f"銘柄 {stock_code}: 環境足BBミドルカラム ({bb_mid_col_ctx}) が df_ctx_indicators_plot に見つからないか全てNaN。")
        plot_markers_gapless(ax1, df_context_plot, buy_entries_for_plot, marker_buy_entry, {'datetime_col': 'entry_date', 'price_col': 'entry_price'})
        plot_markers_gapless(ax1, df_context_plot, sell_entries_for_plot, marker_sell_entry, {'datetime_col': 'entry_date', 'price_col': 'entry_price'})
        plot_markers_gapless(ax1, df_context_plot, take_profits_for_plot, marker_tp, {'datetime_col': 'exit_date', 'price_col': 'exit_price'})
        plot_markers_gapless(ax1, df_context_plot, stop_losses_for_plot, marker_sl, {'datetime_col': 'exit_date', 'price_col': 'exit_price'})
        handles, labels = ax1.get_legend_handles_labels()
        if handles: ax1.legend(handles, labels, fontsize='xx-small', loc='upper left', ncol=max(1, len(handles)//4))

    # --- 2段目: 環境認識足のATR & Volume ---
    ax2.set_ylabel("Context ATR/Vol", fontsize=9); ax2_plot_success = False
    if df_ctx_indicators_plot is not None and not df_ctx_indicators_plot.empty and 'x_index' in df_ctx_indicators_plot.columns:
        atr_p_ctx_chart = p.get('ATR_SETTINGS_PERIOD_CONTEXT'); atr_col_name_ctx = f'ATR_{atr_p_ctx_chart}_CTX_Chart_ITS' if atr_p_ctx_chart else None
        if not (atr_p_ctx_chart and atr_col_name_ctx and atr_col_name_ctx in df_ctx_indicators_plot.columns and df_ctx_indicators_plot[atr_col_name_ctx].notna().any()): 
            atr_col_name_ctx = f'ATR_{atr_p_ctx_chart}_ctx_ITS' if atr_p_ctx_chart else None
        if not (atr_p_ctx_chart and atr_col_name_ctx and atr_col_name_ctx in df_ctx_indicators_plot.columns and df_ctx_indicators_plot[atr_col_name_ctx].notna().any()): 
             atr_col_name_ctx = f"ATR_{atr_p_ctx_chart}_ctx" if atr_p_ctx_chart else None
        if atr_p_ctx_chart and atr_col_name_ctx and atr_col_name_ctx in df_ctx_indicators_plot.columns and df_ctx_indicators_plot[atr_col_name_ctx].notna().any():
            ax2.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot[atr_col_name_ctx], label=f'ATR({atr_p_ctx_chart})_ctx', c='b', lw=0.8)
            ax2.tick_params(axis='y', labelcolor='b', labelsize=8); ax2_plot_success = True
        else: ax2.text(0.5, 0.25, f"ATR ({atr_col_name_ctx if atr_col_name_ctx else 'N/A'}) Missing", ha='center', va='center', fontsize=8, color='gray', transform=ax2.transAxes)
    else: ax2.text(0.5, 0.5, "Ctx Ind Data Missing", ha='center', va='center', fontsize=9, color='gray')
    ax2_twin = ax2.twinx(); ax2_twin_plot_success = False
    if df_context_plot is not None and not df_context_plot.empty and 'Volume' in df_context_plot.columns and df_context_plot['Volume'].notna().any() and \
       'Open' in df_context_plot.columns and 'Close' in df_context_plot.columns:
        vol_colors_ctx = ['red' if row['Close'] > row['Open'] else 'green' for idx, row in df_context_plot.iterrows()]
        ax2_twin.bar(df_context_plot['x_index'], df_context_plot['Volume'], label='Vol_ctx', color=vol_colors_ctx, alpha=0.7, width=0.8)
        ax2_twin_plot_success = True
    ax2_twin.set_ylabel('Volume_ctx', c='gray', fontsize=9); ax2_twin.tick_params(axis='y', labelcolor='gray', labelsize=8); ax2_twin.set_ylim(bottom=0)
    if ax2_plot_success or ax2_twin_plot_success: ax2.tick_params(labelbottom=False); ax2.set_xticks([])

    # --- 3段目: 環境認識足のADX, DI+, DI- (X軸ラベル表示) ---
    ax3.set_ylabel("Context ADX/DMI", fontsize=9)
    ax3_plot_success = False
    adx_col_ctx = 'ADX_ctx_ITS'
    plus_di_col_ctx = 'PLUS_DI_ctx_ITS'
    minus_di_col_ctx = 'MINUS_DI_ctx_ITS'

    if df_ctx_indicators_plot is not None:
        if adx_col_ctx not in df_ctx_indicators_plot.columns: adx_col_ctx = 'ADX_ctx'
        if plus_di_col_ctx not in df_ctx_indicators_plot.columns: plus_di_col_ctx = 'PLUS_DI_ctx'
        if minus_di_col_ctx not in df_ctx_indicators_plot.columns: minus_di_col_ctx = 'MINUS_DI_ctx'

    if df_ctx_indicators_plot is not None and not df_ctx_indicators_plot.empty and 'x_index' in df_ctx_indicators_plot.columns:
        adx_plotted = False; plus_di_plotted = False; minus_di_plotted = False

        # ADXプロット
        if adx_col_ctx in df_ctx_indicators_plot.columns and df_ctx_indicators_plot[adx_col_ctx].notna().any():
            adx_thresh = p.get('ADX_SETTINGS_CONTEXT_THRESHOLD', 18.0)
            ax3.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot[adx_col_ctx], label='ADX_ctx', color='deepskyblue', lw=0.8, linestyle='-') # 水色実線
            ax3.axhline(adx_thresh, color='hotpink', linestyle='--', linewidth=0.7, label=f'ADX Thresh({adx_thresh})') # ピンク破線
            adx_plotted = True; ax3_plot_success = True
        else:
             ax3.text(0.5, 0.7, "Ctx ADX Data Missing or All NaN", ha='center', va='center', fontsize=8, color='gray', transform=ax3.transAxes)

        # DI+ プロット
        if plus_di_col_ctx in df_ctx_indicators_plot.columns and df_ctx_indicators_plot[plus_di_col_ctx].notna().any():
            ax3.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot[plus_di_col_ctx], label='DI+_ctx', color='red', lw=0.8, linestyle='-') # 赤色実線
            plus_di_plotted = True; ax3_plot_success = True
        else:
            ax3.text(0.5, 0.5, "Ctx DI+ Data Missing or All NaN", ha='center', va='center', fontsize=8, color='gray', transform=ax3.transAxes)

        # DI- プロット
        if minus_di_col_ctx in df_ctx_indicators_plot.columns and df_ctx_indicators_plot[minus_di_col_ctx].notna().any():
            ax3.plot(df_ctx_indicators_plot['x_index'], df_ctx_indicators_plot[minus_di_col_ctx], label='DI-_ctx', color='green', lw=0.8, linestyle='-') # 緑色実線
            minus_di_plotted = True; ax3_plot_success = True
        else:
            ax3.text(0.5, 0.3, "Ctx DI- Data Missing or All NaN", ha='center', va='center', fontsize=8, color='gray', transform=ax3.transAxes)
            
        if ax3_plot_success and ax3.has_data():
            ax3.legend(fontsize='x-small', loc='upper left')
            ax3.tick_params(axis='y', labelsize=8)
            ax3.set_ylim(bottom=0)
            
    else: ax3.text(0.5, 0.5, "Ctx ADX/DMI Data Missing", ha='center', va='center', fontsize=9, color='gray')

    if df_context_plot is not None and not df_context_plot.empty:
        set_custom_datetime_x_labels(ax3, df_context_plot, stock_code)
    else:
        ax3.set_xticks([]); min_x_ax3 = -0.8; max_x_ax3 = 0.8
        if df_ctx_indicators_plot is not None and not df_ctx_indicators_plot.empty and 'x_index' in df_ctx_indicators_plot.columns and df_ctx_indicators_plot['x_index'].notna().any():
            min_x_val = df_ctx_indicators_plot['x_index'].min(); max_x_val = df_ctx_indicators_plot['x_index'].max()
            if pd.notna(min_x_val) and pd.notna(max_x_val): min_x_ax3=min_x_val; max_x_ax3=max_x_val
        ax3.set_xlim(min_x_ax3 - 0.8, max_x_ax3 + 0.8)

    # --- 4段目: 実行足 ---
    ax4.set_ylabel("Exec Price", fontsize=9)
    plot_exec_ohlc_success = plot_ohlc_gapless(ax4, df_exec_plot, "Execution Candlestick", display_x_labels=False)
    if plot_exec_ohlc_success:
        if df_with_signals_plot is not None and not df_with_signals_plot.empty and 'x_index' in df_with_signals_plot.columns:
            ema_s_p_exec = p.get('EMA_SETTINGS_SHORT_EXEC_CHART'); ema_l_p_exec = p.get('EMA_SETTINGS_LONG_EXEC_CHART')
            ema_s_col_exec = f"EMA{ema_s_p_exec}_exec" if ema_s_p_exec else None; ema_l_col_exec = f"EMA{ema_l_p_exec}_exec" if ema_l_p_exec else None
            if ema_s_col_exec and ema_s_col_exec in df_with_signals_plot.columns and df_with_signals_plot[ema_s_col_exec].notna().any():
                ax4.plot(df_with_signals_plot['x_index'], df_with_signals_plot[ema_s_col_exec], label=f"EMA{ema_s_p_exec}", c='c', lw=0.8)
            if ema_l_col_exec and ema_l_col_exec in df_with_signals_plot.columns and df_with_signals_plot[ema_l_col_exec].notna().any():
                ax4.plot(df_with_signals_plot['x_index'], df_with_signals_plot[ema_l_col_exec], label=f"EMA{ema_l_p_exec}", c='m', lw=0.8)
            sma1_p_exec = p.get('SMA_SETTINGS_EXEC_PERIOD_1'); sma2_p_exec = p.get('SMA_SETTINGS_EXEC_PERIOD_2')
            sma1_col_exec = f"SMA{sma1_p_exec}_exec" if sma1_p_exec else None; sma2_col_exec = f"SMA{sma2_p_exec}_exec" if sma2_p_exec else None
            if sma1_col_exec and sma1_col_exec in df_with_signals_plot.columns and df_with_signals_plot[sma1_col_exec].notna().any():
                ax4.plot(df_with_signals_plot['x_index'], df_with_signals_plot[sma1_col_exec], label=f"SMA{sma1_p_exec}", c='blue', ls='--', lw=0.7)
            if sma2_col_exec and sma2_col_exec in df_with_signals_plot.columns and df_with_signals_plot[sma2_col_exec].notna().any():
                ax4.plot(df_with_signals_plot['x_index'], df_with_signals_plot[sma2_col_exec], label=f"SMA{sma2_p_exec}", c='purple', ls='--', lw=0.7)
            if 'VWAP_daily_exec' in df_with_signals_plot.columns and df_with_signals_plot['VWAP_daily_exec'].notna().any():
                ax4.plot(df_with_signals_plot['x_index'], df_with_signals_plot['VWAP_daily_exec'], label='VWAP_exec', c='yellow', lw=1.5, ls='-')
            if 'BB_Middle_exec' in df_with_signals_plot.columns and df_with_signals_plot['BB_Middle_exec'].notna().any():
                ax4.plot(df_with_signals_plot['x_index'], df_with_signals_plot['BB_Middle_exec'],label='BB_Mid_exec', c='lime', ls='-', lw=0.9)
                ls_map = {1: ':', 2: '--', 3: '-.'}; alpha_map = {1:0.6, 2:0.7, 3:0.8}
                for i in range(1,4):
                    if f'BB_Upper_exec_{i}dev' in df_with_signals_plot.columns and df_with_signals_plot[f'BB_Upper_exec_{i}dev'].notna().any():
                        ax4.plot(df_with_signals_plot['x_index'], df_with_signals_plot[f'BB_Upper_exec_{i}dev'], c='gray', ls=ls_map[i], lw=0.7, alpha=alpha_map[i], label=f'+{i}σ_exec')
                    if f'BB_Lower_exec_{i}dev' in df_with_signals_plot.columns and df_with_signals_plot[f'BB_Lower_exec_{i}dev'].notna().any():
                        ax4.plot(df_with_signals_plot['x_index'], df_with_signals_plot[f'BB_Lower_exec_{i}dev'], c='gray', ls=ls_map[i], lw=0.7, alpha=alpha_map[i], label=f'-{i}σ_exec')

            # 一目均衡表の描画 (実行足)
            tenkan_col_exec = 'tenkan_sen_exec'
            kijun_col_exec = 'kijun_sen_exec'
            span_a_col_exec = 'senkou_span_a_exec' # 未来の雲
            span_b_col_exec = 'senkou_span_b_exec' # 未来の雲
            current_span_a_col_exec = 'senkou_span_a_raw_exec' # 現在の雲
            current_span_b_col_exec = 'senkou_span_b_raw_exec' # 現在の雲
            chikou_col_exec = 'chikou_span_exec'

            if tenkan_col_exec in df_with_signals_plot.columns and df_with_signals_plot[tenkan_col_exec].notna().any():
                ax4.plot(df_with_signals_plot['x_index'], df_with_signals_plot[tenkan_col_exec], label='Tenkan', color='saddlebrown', lw=0.7, ls='-')
            if kijun_col_exec in df_with_signals_plot.columns and df_with_signals_plot[kijun_col_exec].notna().any():
                ax4.plot(df_with_signals_plot['x_index'], df_with_signals_plot[kijun_col_exec], label='Kijun', color='blueviolet', lw=0.7, ls='-')
            
            if span_a_col_exec in df_with_signals_plot.columns and df_with_signals_plot[span_a_col_exec].notna().any() and \
               span_b_col_exec in df_with_signals_plot.columns and df_with_signals_plot[span_b_col_exec].notna().any():
                ax4.fill_between(df_with_signals_plot['x_index'], 
                                 df_with_signals_plot[span_a_col_exec], 
                                 df_with_signals_plot[span_b_col_exec], 
                                 where=df_with_signals_plot[span_a_col_exec] >= df_with_signals_plot[span_b_col_exec], 
                                 color='lightcoral', alpha=0.2, label='Future Kumo (Up)')
                ax4.fill_between(df_with_signals_plot['x_index'], 
                                 df_with_signals_plot[span_a_col_exec], 
                                 df_with_signals_plot[span_b_col_exec], 
                                 where=df_with_signals_plot[span_a_col_exec] < df_with_signals_plot[span_b_col_exec], 
                                 color='lightgreen', alpha=0.2, label='Future Kumo (Down)') # 雲の色を緑系に変更

            if current_span_a_col_exec in df_with_signals_plot.columns and df_with_signals_plot[current_span_a_col_exec].notna().any():
                 ax4.plot(df_with_signals_plot['x_index'], df_with_signals_plot[current_span_a_col_exec], label='Current SpanA', color='gray', lw=0.5, ls='--')
            if current_span_b_col_exec in df_with_signals_plot.columns and df_with_signals_plot[current_span_b_col_exec].notna().any():
                 ax4.plot(df_with_signals_plot['x_index'], df_with_signals_plot[current_span_b_col_exec], label='Current SpanB', color='darkgray', lw=0.5, ls='--')

            if chikou_col_exec in df_with_signals_plot.columns and df_with_signals_plot[chikou_col_exec].notna().any():
                ax4.plot(df_with_signals_plot['x_index'], df_with_signals_plot[chikou_col_exec], label='Chikou', color='olivedrab', lw=0.7, ls=':')

        plot_markers_gapless(ax4, df_exec_plot, buy_entries_for_plot, marker_buy_entry, {'datetime_col': 'entry_date', 'price_col': 'entry_price'})
        plot_markers_gapless(ax4, df_exec_plot, sell_entries_for_plot, marker_sell_entry, {'datetime_col': 'entry_date', 'price_col': 'entry_price'})
        plot_markers_gapless(ax4, df_exec_plot, take_profits_for_plot, marker_tp, {'datetime_col': 'exit_date', 'price_col': 'exit_price'})
        plot_markers_gapless(ax4, df_exec_plot, stop_losses_for_plot, marker_sl, {'datetime_col': 'exit_date', 'price_col': 'exit_price'})
        handles_ax4, labels_ax4 = ax4.get_legend_handles_labels()
        if handles_ax4: ax4.legend(handles_ax4, labels_ax4, fontsize='xx-small', loc='upper left', ncol=max(1, len(handles_ax4)//4))

    # --- 5段目: 実行足のスローストキャスティクス ---
    ax5.set_ylabel("Exec Stoch", fontsize=9); ax5_plot_success = False
    if df_with_signals_plot is not None and not df_with_signals_plot.empty and 'STOCH_K_exec' in df_with_signals_plot.columns and 'STOCH_D_exec' in df_with_signals_plot.columns and 'x_index' in df_with_signals_plot.columns:
        if df_with_signals_plot['STOCH_K_exec'].notna().any() or df_with_signals_plot['STOCH_D_exec'].notna().any() :
            ob = p.get('STOCH_SETTINGS_OVERBOUGHT_LEVEL', 70.0); os_level = p.get('STOCH_SETTINGS_OVERSOLD_LEVEL', 30.0)
            ax5.plot(df_with_signals_plot['x_index'], df_with_signals_plot['STOCH_K_exec'], label='%K', c='orange', lw=0.8)
            ax5.plot(df_with_signals_plot['x_index'], df_with_signals_plot['STOCH_D_exec'], label='%D', c='dodgerblue', lw=0.8)
            ax5.axhline(ob, c='r', ls='--', lw=0.7, label=f'OB({ob})'); ax5.axhline(os_level, c='g', ls='--', lw=0.7, label=f'OS({os_level})')
            ax5.set_ylim(0, 100); ax5_plot_success = True
            if ax5.has_data(): ax5.legend(fontsize='x-small', loc='upper left'); ax5.tick_params(axis='y', labelsize=8)
        else: ax5.text(0.5, 0.5, "Exec Stoch (All NaN)", ha='center', va='center', fontsize=9, color='gray')
    else: ax5.text(0.5, 0.5, "Exec Stoch Data Missing", ha='center', va='center', fontsize=9, color='gray')
    if ax5_plot_success: ax5.tick_params(labelbottom=False); ax5.set_xticks([])

    # --- 6段目: 実行足のMACD ---
    ax6.set_ylabel("Exec MACD", fontsize=9); ax6_plot_success = False; macd_hist_ema_data = None
    if df_with_signals_plot is not None and not df_with_signals_plot.empty and 'MACD_exec' in df_with_signals_plot.columns and \
    'MACDsignal_exec' in df_with_signals_plot.columns and 'MACDhist_exec' in df_with_signals_plot.columns and \
    'x_index' in df_with_signals_plot.columns:
        if df_with_signals_plot['MACD_exec'].notna().any() or df_with_signals_plot['MACDsignal_exec'].notna().any():
            ax6.plot(df_with_signals_plot['x_index'], df_with_signals_plot['MACD_exec'], label='MACD', color='cyan', lw=0.8)
            ax6.plot(df_with_signals_plot['x_index'], df_with_signals_plot['MACDsignal_exec'], label='Signal', c='darkorange', lw=0.8)
            hist_colors = np.where(df_with_signals_plot['MACDhist_exec'] > 0, 'red', 'green')
            ax6.bar(df_with_signals_plot['x_index'], df_with_signals_plot['MACDhist_exec'], label='Hist', color=hist_colors, width=0.8, alpha=0.5)
            ax6.axhline(0, color='grey', linestyle='--', linewidth=0.5);
            if 'MACDhist_EMA_exec' in df_with_signals_plot.columns:
                macd_hist_ema_data = df_with_signals_plot['MACDhist_EMA_exec']
            ax6_plot_success = True
            if ax6.has_data(): ax6.legend(fontsize='x-small', loc='upper left'); ax6.tick_params(axis='y', labelsize=8)
        else: ax6.text(0.5, 0.5, "Exec MACD (All NaN)", ha='center', va='center', fontsize=9, color='gray')
    else: ax6.text(0.5, 0.5, "Exec MACD Data Missing", ha='center', va='center', fontsize=9, color='gray')
    if macd_hist_ema_data is not None and macd_hist_ema_data.notna().any() and df_with_signals_plot is not None and not df_with_signals_plot.empty and 'x_index' in df_with_signals_plot.columns:
        ax6.plot(df_with_signals_plot['x_index'], macd_hist_ema_data, label='Hist EMA', color='purple', lw=0.8)
        if ax6.has_data(): 
            handles_ax6, labels_ax6 = ax6.get_legend_handles_labels()
            ax6.legend(handles_ax6, labels_ax6, fontsize='x-small', loc='upper left')
            ax6.tick_params(axis='y', labelsize=8)
    if ax6_plot_success: ax6.tick_params(labelbottom=False); ax6.set_xticks([])

    # --- 7段目: 実行足のATR & Volume ---
    ax7.set_ylabel("Exec ATR/Vol", fontsize=9); ax7_plot_success = False; ax7_twin_plot_success = False
    if df_exec_plot is not None and not df_exec_plot.empty:
        atr_p_exec_chart = p.get('ATR_SETTINGS_PERIOD_EXEC'); atr_col_name_exec = f'ATR_{atr_p_exec_chart}_EXEC_Chart' if atr_p_exec_chart else None
        if df_with_signals_plot is not None and not df_with_signals_plot.empty and atr_col_name_exec and atr_col_name_exec in df_with_signals_plot.columns and \
           'x_index' in df_with_signals_plot.columns:
            if df_with_signals_plot[atr_col_name_exec].notna().any():
                ax7.plot(df_with_signals_plot['x_index'], df_with_signals_plot[atr_col_name_exec], label=f'ATR({atr_p_exec_chart})_exec', c='b', lw=0.8)
                ax7.tick_params(axis='y', labelcolor='b', labelsize=8); ax7_plot_success = True
            else: ax7.text(0.5, 0.25, "ATR (All NaN)", ha='center', va='center', fontsize=8, color='gray', transform=ax7.transAxes)
        elif df_with_signals_plot is not None and not df_with_signals_plot.empty :
             ax7.text(0.5, 0.25, f"ATR ({atr_col_name_exec if atr_col_name_exec else 'N/A'}) Missing in Signals DF", ha='center', va='center', fontsize=8, color='gray', transform=ax7.transAxes)
        else: ax7.text(0.5, 0.25, "ATR Data (Signals DF) Missing", ha='center', va='center', fontsize=8, color='gray', transform=ax7.transAxes)
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
        ax7.set_xticks([]); ax7.set_xlim(-0.8, 0.8)

    try:
        plt.savefig(chart_filepath, dpi=120, format='svg')
        logging.info(f"  銘柄 {stock_code}: チャートを保存しました: {chart_filepath}")
    except Exception as e:
        logging.error(f"  銘柄 {stock_code}: チャートの保存中にエラーが発生しました: {e}", exc_info=True)
    finally:
        plt.close(fig)