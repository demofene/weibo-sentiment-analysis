import datetime
import json
import re
from urllib.parse import quote

from scrapy import Request, Spider

try:
    from project_paths import COOKIE_FILE
except ImportError:  # pragma: no cover
    from weibospider.project_paths import COOKIE_FILE
from spiders.common import parse_long_tweet, parse_tweet_info


def _normalize_string_list(value):
    if value is None:
        return []

    raw_items = value if isinstance(value, (list, tuple)) else [value]
    normalized = []
    for item in raw_items:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _parse_datetime(value, default):
    if value in {None, ""}:
        return default

    if isinstance(value, datetime.datetime):
        return value

    text = str(value).strip()
    formats = (
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H",
        "%Y-%m-%d-%H",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
    )
    for fmt in formats:
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unsupported datetime format: {value}")


def _coerce_bool(value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class TweetSpiderByKeyword(Spider):
    """Search Weibo by keyword and collect matched posts (performance‑optimized)."""

    name = "tweet_spider_by_keyword"
    base_url = "https://s.weibo.com/"

    # ---------- 性能优化参数 ----------
    max_pages = 3
    max_tweets = 100
    # ----------------------------------

    default_keywords = [""]
    default_split_by_hour = False        

    def __init__(self, keywords=None, start_time=None, end_time=None, split_by_hour=None, **kwargs):
        super().__init__(**kwargs)

        now = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
        default_end_time = now
        default_start_time = now - datetime.timedelta(days=1)

        self.keywords = _normalize_string_list(keywords) or list(self.default_keywords)
        self.start_time = _parse_datetime(start_time, default_start_time)
        self.end_time = _parse_datetime(end_time, default_end_time)
        self.is_split_by_hour = _coerce_bool(split_by_hour, self.default_split_by_hour)

        # 去重 + 计数
        self.seen_tweet_ids = set()
        self.tweet_count = 0
        self.page_counts = {}          # key: (keyword, start_time, end_time) -> 当前窗口已翻页数

        if self.start_time >= self.end_time:
            raise ValueError("start_time must be earlier than end_time")

    def get_cookie_header(self):
        print(f"Reading cookie file for keyword crawl: {COOKIE_FILE}")
        with open(COOKIE_FILE, "r", encoding="utf-8") as file_obj:
            cookie_str = file_obj.read().strip()
        print(f"Using cookie preview: {cookie_str[:50]}...")
        return cookie_str

    def _build_headers(self, cookie_header):
        return {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Cookie": cookie_header,
        }

    def _iter_time_windows(self):
        if not self.is_split_by_hour:
            yield self.start_time, self.end_time
            return

        time_cur = self.start_time
        while time_cur < self.end_time:
            next_time = min(time_cur + datetime.timedelta(hours=1), self.end_time)
            yield time_cur, next_time
            time_cur = next_time

    def _yield_keyword_request(self, keyword, start_time, end_time, headers):
        start_value = start_time.strftime("%Y-%m-%d-%H")
        end_value = end_time.strftime("%Y-%m-%d-%H")
        url = (
            "https://s.weibo.com/weibo?"
            f"q={quote(keyword)}&timescope=custom%3A{start_value}%3A{end_value}&page=1"
        )
        yield Request(
            url,
            callback=self.parse,
            meta={
                "keyword": keyword,
                "window_start": start_time.strftime("%Y-%m-%d %H:%M"),
                "window_end": end_time.strftime("%Y-%m-%d %H:%M"),
                "page_num": 1,                     # 记录当前页码
            },
            headers=headers,
            priority=10,
        )

    def start_requests(self):
        cookie_header = self.get_cookie_header()
        headers = self._build_headers(cookie_header)

        print(
            "Starting keyword crawl with "
            f"{len(self.keywords)} keyword(s), split_by_hour={self.is_split_by_hour}, "
            f"range={self.start_time} -> {self.end_time}"
        )

        for keyword in self.keywords:
            for start_time, end_time in self._iter_time_windows():
                yield from self._yield_keyword_request(keyword, start_time, end_time, headers)

    def _extract_tweet_ids(self, response):
        tweet_ids = []

        for mid in response.css("div.card-wrap[mid]::attr(mid)").getall():
            text = str(mid).strip()
            if text:
                tweet_ids.append(text)

        href_candidates = response.css("div.card-wrap div.from a::attr(href)").getall()
        for href in href_candidates:
            match = re.search(r"weibo\.com/(?:\d+|[A-Za-z0-9_]+)/([A-Za-z0-9]+)", href)
            if match:
                tweet_ids.append(match.group(1))

        deduped = []
        seen = set()
        for tweet_id in tweet_ids:
            if tweet_id in seen:
                continue
            seen.add(tweet_id)
            deduped.append(tweet_id)
        return deduped

    def parse(self, response, **kwargs):
        # 全局上限检查
        if self.tweet_count >= self.max_tweets:
            self.logger.info("Reached max_tweets limit (%d), stopping.", self.max_tweets)
            return

        print(f"Keyword search status: {response.status}")
        print(f"Keyword search URL: {response.url}")

        if response.status != 200:
            print(f"Response preview: {response.text[:200]}")
            return

        # 提取当前页面上的微博 ID
        tweet_ids = self._extract_tweet_ids(response)
        if not tweet_ids:
            if "没有找到相关结果" in response.text:
                self.logger.info(
                    'Keyword "%s" returned no results in %s -> %s',
                    response.meta["keyword"],
                    response.meta["window_start"],
                    response.meta["window_end"],
                )
            else:
                self.logger.info(
                    'Keyword "%s" produced a page with no parseable tweet ids: %s',
                    response.meta["keyword"],
                    response.url,
                )
            return

        print(f"Found {len(tweet_ids)} tweet id(s) on current page")
        for tweet_id in tweet_ids:
            if self.tweet_count >= self.max_tweets:
                self.logger.info("Reached max_tweets, skip remaining tweet ids on this page.")
                break
            if tweet_id in self.seen_tweet_ids:
                continue
            self.seen_tweet_ids.add(tweet_id)
            self.tweet_count += 1

            url = f"https://weibo.com/ajax/statuses/show?id={tweet_id}"
            print(f"Fetching tweet detail: {url}")
            yield Request(
                url,
                callback=self.parse_tweet,
                meta=response.meta,
                headers=response.request.headers,
                priority=20,
            )

        # ---- 翻页控制 ----
        # 检查当前窗口是否已达翻页上限
        window_key = (
            response.meta["keyword"],
            response.meta["window_start"],
            response.meta["window_end"],
        )
        current_page = response.meta.get("page_num", 1)
        if current_page >= self.max_pages:
            self.logger.info("Reached max_pages (%d) for window %s", self.max_pages, window_key)
            return
        # 全局上限也会阻止翻页
        if self.tweet_count >= self.max_tweets:
            return

        next_page = response.css("a.next::attr(href)").get()
        if next_page:
            url = response.urljoin(next_page)
            print(f"Following next page: {url}")
            yield Request(
                url,
                callback=self.parse,
                meta={
                    **response.meta,
                    "page_num": current_page + 1,
                },
                headers=response.request.headers,
                priority=10,
            )

    @staticmethod
    def parse_tweet(response):
        print(f"Tweet detail status: {response.status}")

        if response.status == 403:
            print(f"Tweet detail forbidden: {response.text[:200]}")
            return

        try:
            data = json.loads(response.text)
        except Exception as exc:
            print(f"Failed to parse JSON: {exc}")
            return

        item = parse_tweet_info(data)
        item["keyword"] = response.meta["keyword"]
        item["search_window_start"] = response.meta.get("window_start", "")
        item["search_window_end"] = response.meta.get("window_end", "")

        if item["isLongText"] and item.get("mblogid"):
            url = f"https://weibo.com/ajax/statuses/longtext?id={item['mblogid']}"
            yield Request(
                url,
                callback=parse_long_tweet,
                meta={"item": item},
                headers=response.request.headers,
                priority=30,
            )
            return

        yield item