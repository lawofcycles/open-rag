# 実行するPythonがあるパス
pythonpath = './'

# ワーカー数
workers = 4

# ワーカーのクラス、*2 にあるようにUvicornWorkerを指定 (Uvicornがインストールされている必要がある)
worker_class = 'uvicorn.workers.UvicornWorker'

# IPアドレスとポート
bind = '127.0.0.1:8080'

# プロセスIDを保存するファイル名
pidfile = 'prod.pid'

# Pythonアプリに渡す環境変数
raw_env = ['MODE=PROD']

# デーモン化する場合はTrue
daemon = True

# エラーログ
errorlog = './logs/error_log.txt'

# プロセスの名前
proc_name = 'e5elyzaapi'

# アクセスログ
accesslog = './logs/access_log.txt'
