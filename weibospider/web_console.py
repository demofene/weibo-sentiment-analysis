import argparse
import json
import mimetypes
import os
import subprocess
import sys
import threading
import traceback
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

try:
    from pipeline_service import analyze_cleaned_comments, clean_raw_comment_file, list_recent_files
    from project_paths import BASE_DIR, RAW_DATA_DIR, ensure_dir
except ImportError:  # pragma: no cover
    from weibospider.pipeline_service import analyze_cleaned_comments, clean_raw_comment_file, list_recent_files
    from weibospider.project_paths import BASE_DIR, RAW_DATA_DIR, ensure_dir


WEBUI_DIR = os.path.join(BASE_DIR, "webui")
STATIC_DIR = os.path.join(WEBUI_DIR, "static")
INDEX_FILE = os.path.join(WEBUI_DIR, "index.html")


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def normalize_list(values):
    result = []
    for value in values or []:
        text = str(value).strip()
        if text:
            result.append(text)
    return result


def stringify_command(command):
    return " ".join(f'"{part}"' if " " in str(part) else str(part) for part in command)


class JobStore:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()

    def create(self, kind, payload):
        job_id = uuid.uuid4().hex[:10]
        job = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "payload": payload,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "logs": [],
            "result": None,
            "error": None,
        }
        with self.lock:
            self.jobs[job_id] = job
        return self.snapshot(job_id)

    def snapshot(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            return json.loads(json.dumps(job, ensure_ascii=False))

    def update(self, job_id, **changes):
        with self.lock:
            job = self.jobs[job_id]
            job.update(changes)
            job["updated_at"] = now_iso()

    def append_log(self, job_id, message):
        timestamped = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        with self.lock:
            job = self.jobs[job_id]
            job["logs"].append(timestamped)
            job["logs"] = job["logs"][-400:]
            job["updated_at"] = now_iso()


class WebConsoleApp:
    def __init__(self):
        ensure_dir(RAW_DATA_DIR)
        self.jobs = JobStore()

    def get_files_payload(self):
        return list_recent_files()

    def create_background_job(self, kind, payload, runner):
        job = self.jobs.create(kind, payload)
        thread = threading.Thread(
            target=self._run_job,
            args=(job["id"], runner, payload),
            daemon=True,
        )
        thread.start()
        return job

    def _run_job(self, job_id, runner, payload):
        self.jobs.update(job_id, status="running")
        logger = lambda message: self.jobs.append_log(job_id, message)
        try:
            result = runner(payload, logger)
        except Exception as exc:  # pragma: no cover
            logger(str(exc))
            logger(traceback.format_exc())
            self.jobs.update(job_id, status="failed", error=str(exc))
            return

        self.jobs.update(job_id, status="succeeded", result=result)

    def start_crawl_job(self, payload):
        return self.create_background_job("crawl", payload, self._run_crawl_job)

    def start_clean_job(self, payload):
        return self.create_background_job("clean", payload, self._run_clean_job)

    def start_analyze_job(self, payload):
        return self.create_background_job("analyze", payload, self._run_analyze_job)

    def start_full_comment_job(self, payload):
        return self.create_background_job("full-comment", payload, self._run_full_comment_job)

    def start_full_keyword_sentiment_job(self, payload):
        return self.create_background_job("full-keyword-sentiment", payload, self._run_full_keyword_sentiment_job)
    
    def _run_subprocess(self, command, cwd, logger):
        logger(f"Running: {stringify_command(command)}")
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert process.stdout is not None
        for line in process.stdout:
            text = line.rstrip()
            if text:
                logger(text)

        exit_code = process.wait()
        if exit_code != 0:
            raise RuntimeError(f"Command exited with code {exit_code}")

    def _discover_new_raw_file(self, previous_files):
        current_files = []
        for item in os.listdir(RAW_DATA_DIR):
            path = os.path.join(RAW_DATA_DIR, item)
            if os.path.isfile(path):
                current_files.append(path)

        new_files = [path for path in current_files if os.path.basename(path) not in previous_files]
        if not new_files:
            return None

        new_files.sort(key=os.path.getmtime, reverse=True)
        latest = new_files[0]
        return {
            "name": os.path.basename(latest),
            "absolute_path": latest,
            "relative_path": os.path.relpath(latest, BASE_DIR).replace("\\", "/"),
        }

    def _build_crawl_command(self, payload):
        mode = payload.get("mode")
        if mode not in {"comment", "user", "tweet_by_keyword"}:
            raise ValueError("Unsupported crawl mode.")

        command = [sys.executable, "run_spider.py", mode]
        if mode == "comment":
            tweet_ids = normalize_list(payload.get("tweet_ids"))
            if not tweet_ids:
                raise ValueError("At least one tweet id is required for comment crawling.")
            command.extend(["--tweet-ids", *tweet_ids])
            command.extend(["--count", str(int(payload.get("count") or 20))])
        elif mode == "user":
            user_ids = normalize_list(payload.get("user_ids"))
            if not user_ids:
                raise ValueError("At least one user id is required for user crawling.")
            command.extend(["--user-ids", *user_ids])
        elif mode == "tweet_by_keyword":
            keywords = normalize_list(payload.get("keywords"))
            if not keywords:
                raise ValueError("At least one keyword is required for keyword crawling.")
            command.extend(["--keywords", *keywords])
            if payload.get("start_time"):
                command.extend(["--start-time", str(payload["start_time"])])
            if payload.get("end_time"):
                command.extend(["--end-time", str(payload["end_time"])])
            split_by_hour = "true" if payload.get("split_by_hour", True) else "false"
            command.extend(["--split-by-hour", split_by_hour])
        return command

    def _run_crawl_job(self, payload, logger):
        previous_files = set(os.listdir(RAW_DATA_DIR))
        command = self._build_crawl_command(payload)
        self._run_subprocess(command, BASE_DIR, logger)

        latest_raw_file = self._discover_new_raw_file(previous_files)
        if latest_raw_file:
            logger(f"New raw file detected: {latest_raw_file['relative_path']}")
        else:
            logger("No new raw data file was detected after crawl completion.")

        return {
            "mode": payload.get("mode"),
            "command": stringify_command(command),
            "latest_raw_file": latest_raw_file,
        }

    def _run_clean_job(self, payload, logger):
        raw_file = payload.get("raw_file")
        if not raw_file:
            raise ValueError("raw_file is required.")
        return clean_raw_comment_file(raw_file, logger=logger)

    def _run_analyze_job(self, payload, logger):
        cleaned_file = payload.get("cleaned_file")
        if not cleaned_file:
            raise ValueError("cleaned_file is required.")
        model_dir = payload.get("model_dir") or os.path.join(BASE_DIR, "models", "final")
        model_name = payload.get("model_name") 
        batch_size = int(payload.get("batch_size") or 32)
        top_examples_per_label = int(payload.get("top_examples_per_label") or 5)
        return analyze_cleaned_comments(
            cleaned_file,
            model_dir=model_dir,
            model_name=model_name,
            batch_size=batch_size,
            top_examples_per_label=top_examples_per_label,
            logger=logger,
        )

    def _run_full_comment_job(self, payload, logger):
        crawl_payload = {
            "mode": "comment",
            "tweet_ids": payload.get("tweet_ids"),
            "count": payload.get("count") or 20,
        }
        crawl_result = self._run_crawl_job(crawl_payload, logger)
        latest_raw_file = crawl_result.get("latest_raw_file")
        if not latest_raw_file:
            raise RuntimeError("The comment crawl completed but no raw data file was created.")

        clean_result = self._run_clean_job({"raw_file": latest_raw_file["absolute_path"]}, logger)
        analyze_result = self._run_analyze_job(
            {
                "cleaned_file": clean_result["cleaned_file"]["absolute_path"],
                "model_dir": payload.get("model_dir"),
                "model_name": payload.get("model_name"),
                "batch_size": payload.get("batch_size") or 32,
                "top_examples_per_label": payload.get("top_examples_per_label") or 5,
            },
            logger,
        )
        return {
            "crawl": crawl_result,
            "clean": clean_result,
            "analysis": analyze_result,
        }
    
    def _run_full_keyword_sentiment_job(self, payload, logger):
        keyword = payload.get("keyword", "").strip()
        if not keyword:
            raise ValueError("keyword is required.")

        # 1. 抓取关键词微博
        crawl_payload = {
            "mode": "tweet_by_keyword",
            "keywords": [keyword],
            "start_time": payload.get("start_time"),
            "end_time": payload.get("end_time"),
            "split_by_hour": payload.get("split_by_hour", True),
        }
        # 注意：这里强制加上输出文件标识，让爬虫结果落盘
        crawl_result = self._run_crawl_job(crawl_payload, logger)

        # 2. 从爬虫结果中找到新生成的原始文件
        latest_raw_file = crawl_result.get("latest_raw_file")
        if not latest_raw_file:
            raise RuntimeError("关键词抓取已完成，但未检测到新的原始数据文件，请检查爬虫输出。")

        # 3. 调用关键词情感分析（复用已有的函数）
        from pipeline_service import analyze_keyword_tweets
        analysis_result = analyze_keyword_tweets(
            keyword,
            model_dir=payload.get("model_dir") or os.path.join(BASE_DIR, "models", "final"),
            model_name=payload.get("model_name"),
            batch_size=int(payload.get("batch_size") or 32),
            top_examples_per_label=int(payload.get("top_examples_per_label") or 5),
            logger=logger,
            raw_file_path=latest_raw_file["absolute_path"],
        )

        return {
            "crawl": crawl_result,
            "analysis": analysis_result,
        }

class ConsoleRequestHandler(BaseHTTPRequestHandler):
    server_version = "WeiboConsole/1.0"

    @property
    def app(self):
        return self.server.app

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._serve_file(INDEX_FILE, "text/html; charset=utf-8")
            return
        if parsed.path.startswith("/static/"):
            self._serve_static(parsed.path)
            return
        if parsed.path == "/api/files":
            self._write_json(self.app.get_files_payload())
            return
        if parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            job = self.app.jobs.snapshot(job_id)
            if job is None:
                self._write_json({"error": "Job not found."}, status=404)
                return
            self._write_json(job)
            return

        self._write_json({"error": "Not found."}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        payload = self._read_json()
        if parsed.path == "/api/jobs/crawl":
            self._write_json(self.app.start_crawl_job(payload), status=202)
            return
        if parsed.path == "/api/jobs/clean":
            self._write_json(self.app.start_clean_job(payload), status=202)
            return
        if parsed.path == "/api/jobs/analyze":
            self._write_json(self.app.start_analyze_job(payload), status=202)
            return
        if parsed.path == "/api/jobs/full-comment":
            self._write_json(self.app.start_full_comment_job(payload), status=202)
            return
        if parsed.path == "/api/jobs/full-keyword-sentiment":
            self._write_json(self.app.start_full_keyword_sentiment_job(payload), status=202)
            return

        self._write_json({"error": "Not found."}, status=404)

    def _read_json(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw_body = self.rfile.read(content_length)
        if not raw_body:
            return {}
        return json.loads(raw_body.decode("utf-8"))

    def _serve_static(self, request_path):
        relative_path = request_path[len("/static/") :]
        normalized = os.path.normpath(relative_path)
        file_path = os.path.abspath(os.path.join(STATIC_DIR, normalized))
        if not file_path.startswith(os.path.abspath(STATIC_DIR)):
            self._write_json({"error": "Invalid path."}, status=400)
            return
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            self._write_json({"error": "Static file not found."}, status=404)
            return

        content_type, _ = mimetypes.guess_type(file_path)
        self._serve_file(file_path, content_type or "application/octet-stream")

    def _serve_file(self, file_path, content_type):
        with open(file_path, "rb") as fp:
            payload = fp.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _write_json(self, payload, status=200):
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format_string, *args):  # pragma: no cover
        print(f"[web] {self.address_string()} - {format_string % args}")


class AppServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class, app):
        super().__init__(server_address, handler_class)
        self.app = app


def build_parser():
    parser = argparse.ArgumentParser(description="Start the local Weibo workflow console.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind.")
    return parser


def main():
    args = build_parser().parse_args()
    app = WebConsoleApp()
    server = AppServer((args.host, args.port), ConsoleRequestHandler, app)
    print(f"Weibo console running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping console...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
