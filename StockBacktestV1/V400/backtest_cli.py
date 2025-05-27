# backtest_cli.py
import argparse
import os
import sys
import datetime

# base.py から バックテスト実行関数をインポート (仮の関数名)
from base import run_backtest

def main():
    parser = argparse.ArgumentParser(description="株式トレード戦略のバックテストを実行します。")
    parser.add_argument("exec_interval", type=int, help="実行足（分単位）")
    parser.add_argument("context_interval", type=int, help="環境認識足（分単位）")
    parser.add_argument("data_folder_date", type=str, help="株価データ格納フォルダの日付部分 (YYYYMMDD)")
    parser.add_argument("chart_output_flag", type=str, nargs='?', default=None, help="グラフ出力有無 ('chart' または指定なし)")
    parser.add_argument("target_stock_code", type=str, nargs='?', default=None, help="グラフ出力対象の銘柄コード (指定なしで全銘柄)")

    args = parser.parse_args()

    exec_interval_min = args.exec_interval
    context_interval_min = args.context_interval
    date_str = args.data_folder_date
    output_chart = False
    target_chart_codes = [] # 文字列のリストとして扱う

    if args.chart_output_flag is not None:
        if args.chart_output_flag.lower() == 'chart':
            output_chart = True
            print("グラフ出力オプションが指定されました。")
            if args.target_stock_code:
                # 複数の銘柄コードがカンマ区切りで指定される可能性を考慮する場合は、ここで分割処理
                # 例: target_chart_codes = [code.strip() for code in args.target_stock_code.split(',')]
                # 今回の実行例では単一銘柄なのでそのまま
                if args.target_stock_code.isdigit() and len(args.target_stock_code) == 4:
                    target_chart_codes = [args.target_stock_code]
                    print(f"指定された銘柄のグラフのみ出力します: {target_chart_codes}")
                else:
                    # 実行例2のケース (chart のみ指定)
                    if args.target_stock_code.lower() == 'chart': # python backtest_cli.py 1 5 20250505 chart のような場合
                         output_chart = True
                         target_chart_codes = [] # 全銘柄対象
                         print("チャート出力対象の銘柄指定がないため、全銘柄のグラフを出力します。")
                    elif args.target_stock_code.isdigit(): # コードが指定されているが、chartフラグと誤認されている場合
                        print(f"エラー: グラフ出力対象銘柄コードの後に不正な引数 '{args.target_stock_code}' があります。")
                        print(f"グラフ出力対象銘柄を指定する場合は、'chart' の後に銘柄コードを指定してください。")
                        sys.exit(1)
                    else:
                        print(f"警告: グラフ出力対象の銘柄コード '{args.target_stock_code}' が不正です。全銘柄のグラフを出力します。")
                        target_chart_codes = [] # 全銘柄対象

            else: # python backtest_cli.py 1 5 20250505 chart の場合
                print("チャート出力対象の銘柄指定がないため、全銘柄のグラフを出力します。")
                target_chart_codes = [] # 全銘柄対象
        elif args.target_stock_code is None: # python backtest_cli.py 1 5 20250505 chart の場合、chart_output_flag='chart', target_stock_code=None
            output_chart = True
            print("チャート出力対象の銘柄指定がないため、全銘柄のグラフを出力します。")
            target_chart_codes = [] # 全銘柄対象
        else: # chart 以外の文字列が chart_output_flag に入ってきた場合
            # このケースは、引数の順番が重要になるため、argparse の nargs の設定でカバーするか、
            # もしくは、chart_output_flag が 'chart' 以外ならエラーとするか、
            # または、chart_output_flag の次の引数を target_stock_code とみなすか、設計による。
            # ここでは、4番目の引数が 'chart' でなければグラフ出力なしと解釈し、
            # 5番目の引数は存在すれば target_stock_code と解釈するが、
            # chart フラグがなければ target_stock_code は無視される、という argparse の標準的な挙動に任せる。
            # ただし、今回の実行例では4番目が 'chart' でない場合は5番目の引数もないため、
            # この分岐には通常入らないはず。もし入る場合は引数の指定方法が想定と異なる。
            print(f"情報: グラフ出力フラグに 'chart' 以外 ('{args.chart_output_flag}') が指定されたため、グラフは出力しません。")
            output_chart = False
            if args.target_stock_code:
                print(f"情報: グラフ出力が無効なため、指定された銘柄コード '{args.target_stock_code}' は無視されます。")
            target_chart_codes = []


    # --- 引数バリデーション (backtest_fw.py から移植・調整) ---
    if not (1 <= exec_interval_min): # 0以下や大きな値の制限は適宜追加
        print(f"エラー: 実行足インターバル '{exec_interval_min}' は正の整数で指定してください。")
        sys.exit(1)
    if not (1 <= context_interval_min):
        print(f"エラー: 環境認識足インターバル '{context_interval_min}' は正の整数で指定してください。")
        sys.exit(1)
    if len(date_str) != 8 or not date_str.isdigit():
        print(f"エラー: 日付はYYYYMMDD形式の8桁の数字で指定してください (例: 20250505)。入力: '{date_str}'")
        sys.exit(1)
    try:
        datetime.datetime.strptime(date_str, '%Y%m%d')
    except ValueError:
        print(f"エラー: 日付の形式が正しくありません (例: 20250505)。入力: '{date_str}'")
        sys.exit(1)

    # base.py のバックテスト実行関数を呼び出し
    # strategy モジュール名は固定とするか、これも引数で渡すか検討が必要
    # ここでは 'strategy' を固定とする
    strategy_module_name = 'strategy'
    try:
        run_backtest(
            strategy_module_name=strategy_module_name, # strategy.py を想定
            exec_interval_min=exec_interval_min,
            context_interval_min=context_interval_min,
            date_str=date_str,
            output_chart_flag=output_chart,
            target_chart_codes_list=target_chart_codes # 文字列のリスト
        )
    except Exception as e:
        print(f"バックテスト実行中にエラーが発生しました: {e}", file=sys.stderr)
        # 詳細なエラーログは base.py 内のロギングで記録されることを期待
        sys.exit(1)

if __name__ == "__main__":
    main()