# alert_sender.py
import smtplib
import ssl
from email.mime.text import MIMEText
from email.header import Header
import logging
import os
import yaml
import time # timeモジュールをインポート (リトライ用)

logger = logging.getLogger(__name__)

EMAIL_CREDENTIALS_FILE_PATH = "email_credentials.yaml"

# --- リトライ設定 ---
MAX_RETRY_ATTEMPTS = 3  # 最大リトライ回数
RETRY_DELAY_SECONDS = 10 # リトライ間隔（秒）
# --------------------

def _load_email_credentials(filepath=EMAIL_CREDENTIALS_FILE_PATH):
    """指定されたYAMLファイルからメールアカウント情報とSMTPサーバー設定を読み込む。"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                credentials = yaml.safe_load(f)
                if credentials and isinstance(credentials, dict):
                    required_keys = [
                        'sender_email', 'recipients', 'smtp_password',
                        'smtp_server', 'smtp_port'
                    ]
                    missing_keys = [key for key in required_keys if key not in credentials]
                    if missing_keys:
                        logger.error(f"メール情報ファイル '{filepath}' に必須キーがありません: {missing_keys}")
                        return None
                    if not isinstance(credentials.get('recipients'), list) or not credentials.get('recipients'):
                        logger.error(f"メール情報ファイル '{filepath}' の 'recipients' は空でないリストである必要があります。")
                        return None
                    credentials.setdefault('use_tls', False)
                    credentials.setdefault('use_ssl', False)
                    credentials.setdefault('smtp_user', credentials.get('sender_email'))
                    logger.debug(f"メール情報を '{filepath}' からロードしました。")
                    return credentials
                else:
                    logger.warning(f"メール情報ファイル '{filepath}' が空か、有効なYAML辞書を含んでいません。")
                    return None
        else:
            logger.warning(f"メール情報ファイル '{filepath}' が見つかりません。")
            return None
    except yaml.YAMLError as e:
        logger.error(f"メール情報ファイル '{filepath}' のYAML解析エラー: {e}")
        return None
    except Exception as e:
        logger.error(f"メール情報ファイル '{filepath}' の読み込み中にエラー: {e}", exc_info=True)
        return None

def send_email(subject, body):
    """
    指定された設定を使用してメールを送信する（リトライ処理付き）。
    メールに関するすべての設定はYAMLファイルから読み込む。
    """
    credentials = _load_email_credentials()
    if not credentials:
        logger.error("メール情報をファイルから読み込めませんでした。メール送信を中止します。")
        return False

    sender_email = credentials.get('sender_email')
    recipient_emails_list = credentials.get('recipients')
    smtp_password = credentials.get('smtp_password')
    smtp_user = credentials.get('smtp_user')
    smtp_server_host = credentials.get('smtp_server')
    smtp_server_port = credentials.get('smtp_port')
    use_tls = credentials.get('use_tls')
    use_ssl = credentials.get('use_ssl')

    if not all([sender_email, recipient_emails_list, smtp_password, smtp_user, smtp_server_host, smtp_server_port is not None]):
        logger.error("メール送信に必要な情報が不足しています。")
        return False

    msg = MIMEText(body, 'plain', 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = sender_email
    msg['To'] = ", ".join(recipient_emails_list)

    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        logger.info(f"メール送信試行 (ATTEMPT {attempt}/{MAX_RETRY_ATTEMPTS}): To: {recipient_emails_list}, Subject: {subject}")
        server = None # server変数をループの外で初期化
        try:
            if use_ssl:
                context = ssl.create_default_context()
                server = smtplib.SMTP_SSL(smtp_server_host, smtp_server_port, context=context, timeout=10) # タイムアウト追加
            else:
                server = smtplib.SMTP(smtp_server_host, smtp_server_port, timeout=10) # タイムアウト追加
            
            server.set_debuglevel(0)

            if use_tls and not use_ssl:
                server.ehlo()
                server.starttls()
                server.ehlo()
            
            if smtp_user and smtp_password:
                server.login(smtp_user, smtp_password)
            
            server.sendmail(sender_email, recipient_emails_list, msg.as_string())
            logger.info(f"メール送信に成功しました (ATTEMPT {attempt})。")
            return True # 送信成功したらループを抜ける
        
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP認証エラー (ATTEMPT {attempt}): {e}. ユーザー名またはパスワードを確認してください。リトライしません。")
            return False # 認証エラーはリトライしない
        except (smtplib.SMTPServerDisconnected, smtplib.SMTPConnectError, smtplib.SMTPSenderRefused,
                smtplib.SMTPRecipientsRefused, ConnectionRefusedError, TimeoutError) as e: # TimeoutErrorも追加
            logger.warning(f"SMTP接続/送信エラー (ATTEMPT {attempt}): {e}.")
            if attempt < MAX_RETRY_ATTEMPTS:
                logger.info(f"{RETRY_DELAY_SECONDS}秒後にリトライします...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error(f"最大リトライ回数 ({MAX_RETRY_ATTEMPTS}) に達しました。メール送信に失敗しました。")
                return False
        except smtplib.SMTPException as e: # その他のSMTP関連エラー
            logger.error(f"SMTP一般エラー (ATTEMPT {attempt}): {e}", exc_info=True)
            if attempt < MAX_RETRY_ATTEMPTS:
                logger.info(f"{RETRY_DELAY_SECONDS}秒後にリトライします...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error(f"最大リトライ回数 ({MAX_RETRY_ATTEMPTS}) に達しました。メール送信に失敗しました。")
                return False
        except Exception as e: # 予期せぬエラー
            logger.error(f"メール送信中に予期せぬエラー (ATTEMPT {attempt}): {e}", exc_info=True)
            # 予期せぬエラーはリトライしない方が安全な場合もあるが、今回はリトライ対象に含める
            if attempt < MAX_RETRY_ATTEMPTS:
                logger.info(f"{RETRY_DELAY_SECONDS}秒後にリトライします...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error(f"最大リトライ回数 ({MAX_RETRY_ATTEMPTS}) に達しました。メール送信に失敗しました。")
                return False
        finally:
            if server:
                try:
                    server.quit()
                except smtplib.SMTPServerDisconnected:
                    pass
                except Exception as e_quit:
                    logger.warning(f"SMTPサーバーの終了中にエラー: {e_quit}")
    
    return False # ここに到達する場合はリトライ全て失敗

if __name__ == '__main__':
    # --- モジュール単体テスト用の設定 ---
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s',
                            handlers=[logging.StreamHandler()])

    logger.info("alert_sender.py の単体テストを開始します...")

    if not os.path.exists(EMAIL_CREDENTIALS_FILE_PATH):
        logger.warning(f"メール情報ファイル '{EMAIL_CREDENTIALS_FILE_PATH}' が存在しません。")
        # ... (単体テストのメッセージは変更なし) ...
        logger.info("単体テストをスキップします。")
    else:
        creds_for_test = _load_email_credentials()
        # smtp_user のダミー値チェックは、email_credentials.yaml に smtp_user がない場合、
        # sender_email が使われるため、sender_email のダミー値で代用
        if not creds_for_test or creds_for_test.get('sender_email') == "your_actual_sender_email@example.com":
            logger.warning(f"メール情報ファイル '{EMAIL_CREDENTIALS_FILE_PATH}' の sender_email (または smtp_user) がデフォルトのままか、ファイルが正しく読み込めませんでした。")
            logger.info("単体テストをスキップします。")
        else:
            test_subject = "【テストメール・リトライ機能】リアルタイムアラートシステム"
            test_body = (f"これは alert_sender.py からのテストメールです。\n"
                         f"メール送信にリトライ機能が追加されました。\n"
                         f"すべてのメール関連設定は '{EMAIL_CREDENTIALS_FILE_PATH}' から読み込まれました。\n"
                         f"正しく受信できれば、メール送信機能は動作しています。")
            
            success = send_email(test_subject, test_body)
            if success:
                logger.info("テストメールの送信に成功しました（SMTPサーバーが受け付けました）。")
            else:
                logger.error("テストメールの送信に失敗しました。ログを確認してください。")

    logger.info("alert_sender.py の単体テストを終了します。")