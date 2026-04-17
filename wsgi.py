"""
Render 배포용 WSGI 진입점.
파일 이름에 하이픈이 포함되어 있어 직접 import 불가능하므로
importlib 을 통해 동적 로드합니다.

Render 시작 명령: gunicorn wsgi:app
"""
import importlib.util, os, sys

spec = importlib.util.spec_from_file_location(
    "sanbul_pwa_flask",
    os.path.join(os.path.dirname(__file__), "sanbul-pwa-flask.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

app = module.app
