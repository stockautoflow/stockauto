# backtest_cli.py (修正版 - strategy-moduleの扱い)
import argparse
import os
import sys
import datetime
import re

from base import run_backtest

def main():
    parser = argparse.ArgumentParser(description="株式トレード戦略のバックテストを実行します。")
    parser.add_argument("exec_interval", type=int, help="実行足（分単位）")
    parser.add_argument("context_interval", type=int, help="環境認識足（分単位）")
    parser.add_argument("data_folder_date", type=str, help="株価データ格納フォルダの日付部分 (YYYYMMDD)")
    parser.add_argument("--config", type=str, default="config.yaml", help="設定ファイルへのパス (デフォルト: config.yaml)")
    parser.add_argument(
        "--strategy-module",
        type=str,
        default="strategy",
        help="戦略モジュール名 (例: strategy, my_strategies.custom1)。拡張子 .py は不要です。" # ヘルプを修正
    )
    parser.add_argument(
        "--chart",
        nargs='?',
        const="ALL_STOCKS",
        default=None,
        help="グラフ出力オプション。引数なしで全銘柄、銘柄コード指定(カンマ区切り可)で特定銘柄を出力。例: --chart または --chart 1332,1333"
    )

    args = parser.parse_args()

    exec_interval_min = args.exec_interval
    context_interval_min = args.context_interval
    date_str = args.data_folder_date
    config_filepath = args.config
    
    # strategy_module_name_arg から .py 拡張子とパス区切り文字を処理
    strategy_module_input = args.strategy_module
    # .py 拡張子を除去
    if strategy_module_input.endswith(".py"):
        strategy_module_input = strategy_module_input[:-3]
    # パス区切り文字をドットに置換 (Pythonモジュール形式へ)
    strategy_module_name_arg = strategy_module_input.replace('/', '.').replace('\\', '.')


    output_chart = False
    target_chart_codes_for_run = []
    invalid_codes_detected_by_cli = []

    if args.chart is not None:
        output_chart = True
        if args.chart == "ALL_STOCKS":
            print("グラフ出力オプション (CLI): 全ての銘柄のグラフを出力します。")
        else:
            codes_str = args.chart
            raw_codes = [code.strip() for code in codes_str.split(',')]
            valid_codes_buffer = []
            current_invalid_codes = []
            for code in raw_codes:
                if re.fullmatch(r"\d{4}", code):
                    valid_codes_buffer.append(code)
                else:
                    current_invalid_codes.append(code)
                    invalid_codes_detected_by_cli.append(code)

            if current_invalid_codes:
                print(f"警告 (CLI): 以下の指定された銘柄コードは不正な形式です: {', '.join(current_invalid_codes)}。これらはグラフ出力対象から除外されます。")

            if valid_codes_buffer:
                target_chart_codes_for_run = valid_codes_buffer
                print(f"グラフ出力オプション (CLI): 指定された有効な銘柄のグラフを出力します: {', '.join(target_chart_codes_for_run)}")
            else:
                if current_invalid_codes :
                    print(f"警告 (CLI): 指定された銘柄コードに有効なものがなかったため、特定の銘柄のグラフは出力されません。")
                else:
                    print(f"警告 (CLI): グラフ出力対象の銘柄コードが指定されていません。")
                print(f"情報 (CLI): 有効な対象銘柄がないため、グラフ出力は行われません。全銘柄を出力する場合は --chart のみ指定してください。")
                output_chart = False
    else:
        print("グラフ出力オプション (CLI): グラフは出力されません。")

    if not (1 <= exec_interval_min):
        print(f"エラー (CLI): 実行足インターバル '{exec_interval_min}' は正の整数で指定してください。")
        sys.exit(1)
    if not (1 <= context_interval_min):
        print(f"エラー (CLI): 環境認識足インターバル '{context_interval_min}' は正の整数で指定してください。")
        sys.exit(1)
    if not re.fullmatch(r"\d{8}", date_str):
        print(f"エラー (CLI): 日付はYYYYMMDD形式の8桁の数字で指定してください。入力: '{date_str}'")
        sys.exit(1)
    try:
        datetime.datetime.strptime(date_str, '%Y%m%d')
    except ValueError:
        print(f"エラー (CLI): 日付の形式が正しくありません (例: 20250505)。入力: '{date_str}'")
        sys.exit(1)

    if not os.path.isfile(config_filepath):
        print(f"エラー (CLI): 設定ファイルが見つかりません: {config_filepath}")
        sys.exit(1)

    try:
        run_backtest(
            strategy_module_name=strategy_module_name_arg, # 処理済みのモジュール名を渡す
            config_filepath=config_filepath,
            exec_interval_min=exec_interval_min,
            context_interval_min=context_interval_min,
            date_str=date_str,
            output_chart_flag=output_chart,
            target_chart_codes_list=target_chart_codes_for_run,
            invalid_chart_codes_from_cli=invalid_codes_detected_by_cli
        )
    except Exception as e:
        print(f"バックテスト実行中にエラーが発生しました (CLI): {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()