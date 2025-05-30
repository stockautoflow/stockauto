# converter.py (修正版 - ExcelFileクローズ処理追加)
import pandas as pd
from datetime import datetime, time # datetime.time をインポート
import os
import argparse
import re
import logging

# --- ロギング設定 ---
LOG_DIR = "log"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_filename = datetime.now().strftime("conversion_log_%Y%m%d_%H%M%S.log")
log_filepath = os.path.join(LOG_DIR, log_filename)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(log_filepath, encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
fh.setFormatter(fh_formatter)
logger.addHandler(fh)

sh = logging.StreamHandler()
sh.setLevel(logging.INFO) # コンソールにはINFO以上を表示
sh_formatter = logging.Formatter('%(message)s') # コンソールはメッセージのみ
sh.setFormatter(sh_formatter)
logger.addHandler(sh)
# --------------------

def sanitize_filename(filename):
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
    sanitized = sanitized.strip('. ')
    if not sanitized:
        sanitized = "untitled"
    return sanitized

def create_csv_from_excel(excel_path, output_dir):
    logger.info(f"処理開始: Excelファイル '{excel_path}', 出力先: '{output_dir}'")
    logger.debug(f"入力Excelファイルフルパス: {os.path.abspath(excel_path)}")
    logger.debug(f"出力ディレクトリフルパス: {os.path.abspath(output_dir)}")

    xls = None  # finallyブロックで参照できるように初期化
    try:
        xls = pd.ExcelFile(excel_path, engine='openpyxl') #
        logger.debug(f"Excelファイル '{excel_path}' の読み込みに成功。")
        logger.info(f"Excelファイル内のシート名リスト: {xls.sheet_names}")

        try:
            param_sheet_df = pd.read_excel(xls, sheet_name="param", header=None) #
            if param_sheet_df.shape[0] < 1 or param_sheet_df.shape[1] < 2:
                 logger.error("エラー: 'param'シートが小さすぎるためB1セルにアクセスできません。処理を終了します。")
                 return # xls.close() を finally で行うため、ここで return
            ashi_type = param_sheet_df.iloc[0, 1] #
            if pd.isna(ashi_type): #
                logger.error("エラー: 'param'シートのB1セル（足種）が空です。処理を終了します。")
                return
            ashi_type = str(ashi_type).strip().upper()
            logger.debug(f"'param'シートのB1セルから足種 '{ashi_type}' を取得しました。")
        except KeyError:
            logger.error(f"エラー: 'param'シートがExcelファイル内に見つかりません。処理を終了します。")
            return
        except IndexError:
            logger.error("エラー: 'param'シートのB1セルにアクセスできません。シートの構造を確認してください。処理を終了します。", exc_info=True)
            return
        except Exception as e:
            logger.error(f"エラー: 'param'シートまたは「足種」の読み取りに失敗しました: {e}。処理を終了します。", exc_info=True)
            return

        processing_date_str = datetime.now().strftime("%Y%m%d")
        logger.debug(f"処理日文字列: {processing_date_str}")

        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                logger.info(f"出力フォルダ '{output_dir}' を作成しました。")
            except Exception as e:
                logger.error(f"エラー: 出力フォルダ '{output_dir}' の作成に失敗しました: {e}。処理を終了します。", exc_info=True)
                return

        output_csv_headers = ["datetime", "Open", "High", "Low", "Close", "Volume"]
        processed_sheets_count = 0
        skipped_sheets_count = 0
        
        ashi_types_to_fill_time = ['D', 'W', 'M'] #

        for i in range(1, 51): #
            sheet_name_str = str(i)
            logger.info(f"シート '{sheet_name_str}' の処理開始...")
            meigara_code_processed_str = "N/A"

            try:
                df_sheet_full = pd.read_excel(xls, sheet_name=sheet_name_str, header=None) #
                logger.debug(f"シート '{sheet_name_str}' の読み込み完了。")

                if df_sheet_full.empty or df_sheet_full.shape[0] < 1 or df_sheet_full.shape[1] < 1: #
                    logger.warning(f"シート '{sheet_name_str}' が非常に小さいか空のため、A1セル（銘柄コード）を取得できません。スキップします。")
                    skipped_sheets_count += 1; logger.info(f"シート '{sheet_name_str}' スキップ (理由: シート空または小さすぎる)")
                    continue
                
                meigara_code_raw = df_sheet_full.iloc[0, 0] #
                if pd.isna(meigara_code_raw): #
                    logger.warning(f"シート '{sheet_name_str}' のA1セル（銘柄コード）が空です。スキップします。")
                    skipped_sheets_count += 1; logger.info(f"シート '{sheet_name_str}' スキップ (理由: A1セル空)")
                    continue

                try:
                    if isinstance(meigara_code_raw, float): #
                        meigara_code_processed_str = str(int(meigara_code_raw))
                    elif isinstance(meigara_code_raw, int): #
                        meigara_code_processed_str = str(meigara_code_raw)
                    else: 
                        s_code = str(meigara_code_raw).strip() #
                        if s_code.endswith(".0"):  #
                            meigara_code_processed_str = s_code[:-2]
                        else:
                            meigara_code_processed_str = s_code
                    
                    if not (len(meigara_code_processed_str) == 4 and meigara_code_processed_str.isdigit()): #
                         logger.warning(f"シート '{sheet_name_str}' の銘柄コード '{meigara_code_processed_str}' (元: '{meigara_code_raw}') が4桁の数字ではありません。スキップします。")
                         skipped_sheets_count += 1; logger.info(f"シート '{sheet_name_str}' スキップ (理由: 銘柄コード形式不正)")
                         continue
                except ValueError: 
                    logger.warning(f"シート '{sheet_name_str}' のA1セル（銘柄コード '{meigara_code_raw}'）を期待される形式に変換できませんでした。スキップします。")
                    skipped_sheets_count += 1; logger.info(f"シート '{sheet_name_str}' スキップ (理由: 銘柄コード変換失敗)")
                    continue
                
                meigara_code_sanitized = sanitize_filename(meigara_code_processed_str) #

                if not meigara_code_sanitized or meigara_code_sanitized == "untitled": #
                     logger.warning(f"シート '{sheet_name_str}' のA1セルから有効な銘柄コードを取得できませんでした (raw: {meigara_code_raw}, processed: {meigara_code_processed_str}, sanitized: {meigara_code_sanitized})。スキップします。")
                     skipped_sheets_count += 1; logger.info(f"シート '{sheet_name_str}' スキップ (理由: 無効な銘柄コード)")
                     continue
                logger.debug(f"シート '{sheet_name_str}' A1セルから銘柄コード '{meigara_code_sanitized}' (raw: '{meigara_code_raw}', processed: '{meigara_code_processed_str}') を取得。")

                if df_sheet_full.shape[0] < 3: #
                    logger.warning(f"シート '{sheet_name_str}' (銘柄コード: {meigara_code_sanitized}) は3行未満です（株価データなし）。スキップします。")
                    skipped_sheets_count += 1; logger.info(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) スキップ (理由: データ行なし)")
                    continue
                df_sheet_data_rows = df_sheet_full.iloc[2:] #

                if df_sheet_data_rows.empty: #
                    logger.warning(f"シート '{sheet_name_str}' (銘柄コード: {meigara_code_sanitized}) の3行目以降にデータがありません。スキップします。")
                    skipped_sheets_count += 1; logger.info(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) スキップ (理由: 3行目以降データなし)")
                    continue

                if df_sheet_data_rows.shape[1] <= 10:  #
                    logger.warning(f"シート '{sheet_name_str}' (銘柄コード: {meigara_code_sanitized}) のデータ部分の列数が不足しています (E列からK列まで必要)。スキップします。")
                    skipped_sheets_count += 1; logger.info(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) スキップ (理由: 列数不足)")
                    continue
                df_data_cols = df_sheet_data_rows.iloc[:, 4:11]  #
                logger.debug(f"シート '{sheet_name_str}' (銘柄コード: {meigara_code_sanitized}) からデータ列を抽出。 df_data_cols shape: {df_data_cols.shape}")

                if df_data_cols.empty or df_data_cols.isnull().all().all(): #
                    logger.warning(f"シート '{sheet_name_str}' (銘柄コード: {meigara_code_sanitized}) のE3:K列以降に有効なデータがありません。スキップします。")
                    skipped_sheets_count += 1; logger.info(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) スキップ (理由: E3:K列データなし)")
                    continue

                df_out = pd.DataFrame()

                raw_date_series = df_data_cols.iloc[:, 0] #
                logger.debug(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) - 元の日付データ (先頭5件): {raw_date_series.head().tolist()}")
                first_valid_date_type = 'N/A'
                if not raw_date_series.dropna().empty: first_valid_date_type = type(raw_date_series.dropna().iloc[0])
                logger.debug(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) - 元の日付データ型: {raw_date_series.dtype}, 非NaN最初の要素型: {first_valid_date_type}")
                parsed_dates_fmt = pd.to_datetime(raw_date_series.astype(str), format='%Y/%m/%d', errors='coerce') #
                fallback_mask = parsed_dates_fmt.isna() & raw_date_series.notna() #
                if fallback_mask.any(): #
                    logger.debug(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) - 日付の一部 ({fallback_mask.sum()}件) が '%Y/%m/%d' でパース失敗。自動推測で再試行。")
                    parsed_dates_fmt[fallback_mask] = pd.to_datetime(raw_date_series[fallback_mask], errors='coerce')
                date_part = parsed_dates_fmt.dt.date #
                logger.debug(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) - パース後の日付 date_part (先頭5件): {date_part.head().tolist()}")
                logger.debug(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) - パース後の日付 NaT数: {pd.isna(date_part).sum()} / {len(date_part)}")

                time_series_raw = df_data_cols.iloc[:, 1] #
                logger.debug(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) - 元の時刻データ (先頭5件): {time_series_raw.head().tolist()}")
                first_valid_time_type = 'N/A'
                if not time_series_raw.dropna().empty: first_valid_time_type = type(time_series_raw.dropna().iloc[0])
                logger.debug(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) - 元の時刻データ型: {time_series_raw.dtype}, 非NaN最初の要素型: {first_valid_time_type}")
                def format_time_element_hhmm(t_element): #
                    if pd.isna(t_element): return None
                    if isinstance(t_element, time): return t_element.strftime('%H:%M')
                    if isinstance(t_element, datetime): return t_element.time().strftime('%H:%M')
                    s_element = str(t_element).strip()
                    match_hm = re.fullmatch(r'(\d{1,2}):(\d{2})', s_element)
                    if match_hm: h, m = match_hm.groups(); return f"{h.zfill(2)}:{m.zfill(2)}"
                    match_hms = re.fullmatch(r'(\d{1,2}):(\d{2}):(\d{2})', s_element)
                    if match_hms: h, m, s = match_hms.groups(); return f"{h.zfill(2)}:{m.zfill(2)}"
                    logger.debug(f"時刻要素 '{s_element}' はHH:MM(:SS)形式に変換できませんでした。")
                    return s_element
                time_str_series = time_series_raw.apply(format_time_element_hhmm) #
                logger.debug(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) - 文字列変換後の時刻データ (先頭5件): {time_str_series.head().tolist()}")
                time_part = pd.to_datetime(time_str_series, format='%H:%M', errors='coerce').dt.time #
                logger.debug(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) - パース後の時刻 time_part (先頭5件 до補完): {time_part.head().tolist()}")
                logger.debug(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) - パース後の時刻 NaT数 (補完前): {time_part.isna().sum()} / {len(time_part)}")

                num_na_times = time_part.isna().sum() #
                if num_na_times > 0:
                    if ashi_type in ashi_types_to_fill_time: #
                        default_time = time(9, 0) #
                        time_part = time_part.fillna(default_time) #
                        logger.info(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}, 足種: {ashi_type}) - {num_na_times}件の欠損時刻を {default_time.strftime('%H:%M')} で補完しました。")
                        logger.debug(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) - 補完後の時刻 time_part (先頭5件): {time_part.head().tolist()}")
                        logger.debug(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) - 補完後の時刻 NaT数: {time_part.isna().sum()} / {len(time_part)}")
                    else:
                        logger.warning(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}, 足種: {ashi_type}) - {num_na_times}件の欠損時刻がありますが、補完対象外の足種のためそのままです。")
                
                valid_datetime_mask = date_part.notna() & time_part.notna() #
                logger.debug(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) - valid_datetime_mask trueの数: {valid_datetime_mask.sum()} / {len(valid_datetime_mask)}")

                if not valid_datetime_mask.any(): #
                    logger.warning(f"シート '{sheet_name_str}' (銘柄コード: {meigara_code_sanitized}) で有効な日時データが見つかりません。スキップします。")
                    skipped_sheets_count += 1; logger.info(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) スキップ (理由: 有効日時なし)")
                    continue
                
                date_part_valid = pd.Series(date_part[valid_datetime_mask], dtype='object')  #
                time_part_valid = pd.Series(time_part[valid_datetime_mask], dtype='object')  #
                ohlcv_data_valid = df_data_cols[valid_datetime_mask] #

                if date_part_valid.isna().any() or time_part_valid.isna().any() : #
                     logger.warning(f"シート '{sheet_name_str}' (銘柄コード: {meigara_code_sanitized}) のフィルタリング後データにNaTが含まれています。スキップします。")
                     skipped_sheets_count += 1; logger.info(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) スキップ (理由: フィルタリング後NaT)")
                     continue
                
                ohlcv_data_valid_with_dt = ohlcv_data_valid.copy() #
                datetime_col_values = [] #
                for d_val, t_val in zip(date_part_valid, time_part_valid):
                    if pd.notna(d_val) and pd.notna(t_val):
                        datetime_col_values.append(pd.Timestamp.combine(d_val, t_val))
                    else:
                        datetime_col_values.append(pd.NaT)
                ohlcv_data_valid_with_dt['datetime_col'] = datetime_col_values #
                
                ohlcv_data_valid_with_dt.dropna(subset=['datetime_col'], inplace=True) #
                
                if ohlcv_data_valid_with_dt.empty: #
                     logger.warning(f"シート '{sheet_name_str}' (銘柄コード: {meigara_code_sanitized}) で有効なdatetimeを生成できませんでした(OHLCV結合前)。スキップします。")
                     skipped_sheets_count += 1; logger.info(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) スキップ (理由: datetime生成失敗)")
                     continue

                df_out = pd.DataFrame({ #
                    'datetime': ohlcv_data_valid_with_dt['datetime_col'],
                    'Open':     ohlcv_data_valid_with_dt.iloc[:, 2].values,
                    'High':     ohlcv_data_valid_with_dt.iloc[:, 3].values,
                    'Low':      ohlcv_data_valid_with_dt.iloc[:, 4].values,
                    'Close':    ohlcv_data_valid_with_dt.iloc[:, 5].values,
                    'Volume':   ohlcv_data_valid_with_dt.iloc[:, 6].values 
                })
                logger.debug(f"シート '{sheet_name_str}' (銘柄コード: {meigara_code_sanitized}) - データ整形後 df_out shape: {df_out.shape}")

                if df_out.empty: #
                    logger.warning(f"シート '{sheet_name_str}' (銘柄コード: {meigara_code_sanitized}) は処理後に有効な株価情報を含んでいません。スキップします。")
                    skipped_sheets_count += 1; logger.info(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) スキップ (理由: 処理後データ空)")
                    continue

                output_filename = f"{meigara_code_sanitized}_{ashi_type}_{processing_date_str}.csv" #
                output_filepath = os.path.join(output_dir, output_filename) #

                df_out.to_csv(output_filepath, columns=output_csv_headers, index=False, encoding='utf-8-sig') #
                logger.info(f"シート '{sheet_name_str}' (銘柄: {meigara_code_sanitized}) 完了 -> {output_filename}")
                processed_sheets_count += 1

            except KeyError: 
                logger.info(f"シート '{sheet_name_str}' がExcel内に見つかりませんでした。スキップします。")
                skipped_sheets_count +=1
            except ValueError as e:
                logger.error(f"シート '{sheet_name_str}' (銘柄コード: {meigara_code_processed_str}) のデータ形式に問題。スキップ。エラー: {e}", exc_info=True)
                skipped_sheets_count += 1; logger.info(f"シート '{sheet_name_str}' (銘柄: {meigara_code_processed_str}) エラー (詳細はログ参照)")
            except Exception as e:
                logger.error(f"シート '{sheet_name_str}' (銘柄コード: {meigara_code_processed_str}) の処理中に予期せぬエラー。スキップ。エラー: {e}", exc_info=True)
                skipped_sheets_count += 1; logger.info(f"シート '{sheet_name_str}' (銘柄: {meigara_code_processed_str}) 予期せぬエラー (詳細はログ参照)")
                continue
        
        logger.info(f"\n全シート処理完了。処理済みシート数: {processed_sheets_count}, スキップシート数: {skipped_sheets_count}")
        logger.info(f"詳細ログは {os.path.abspath(log_filepath)} を参照してください。")

    except FileNotFoundError: #
        logger.error(f"エラー: Excelファイル '{excel_path}' が見つかりません。処理を終了します。")
    except Exception as e: #
        logger.error(f"エラー: Excelファイル '{excel_path}' を開けませんでした: {e}", exc_info=True)
    finally:
        if xls is not None:
            try:
                xls.close() # 明示的にファイルを閉じる
                logger.debug(f"Excelファイルオブジェクトをクローズしました: {excel_path}")
            except Exception as e_close:
                logger.error(f"Excelファイルオブジェクトのクローズ中にエラー: {e_close}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Excelの複数シートから株価情報をCSVに変換し出力します。\n詳細ログは 'log' ディレクトリに出力されます。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("excel_file_path", help="入力Excelファイル (.xlsm) のパス") #
    parser.add_argument("output_base_dir", help="CSVファイルを出力するディレクトリのパス") #
    args = parser.parse_args() #
    
    logger.info("======================================================================")
    logger.info(f"           Excel to CSV Converter - 開始 (実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    logger.info("======================================================================")

    create_csv_from_excel(args.excel_file_path, args.output_base_dir) #

    logger.info("======================================================================")
    logger.info(f"           Excel to CSV Converter - 終了 (実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    logger.info("======================================================================")