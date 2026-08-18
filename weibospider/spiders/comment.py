import json

from scrapy import Spider
from scrapy.http import Request

try:
    from project_paths import COOKIE_FILE
except ImportError:  # pragma: no cover
    from weibospider.project_paths import COOKIE_FILE
from spiders.common import parse_time, parse_user_info, url_to_mid


def _normalize_string_list(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raw_items = [value]

    normalized = []
    for item in raw_items:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


class CommentSpider(Spider):
    """Collect comment data for one or more Weibo posts."""

    name = "comment"
    default_tweet_ids = ["QxaWwkI8f"]
    default_count = 20

    def __init__(self, tweet_ids=None, count=default_count, **kwargs):
        super().__init__(**kwargs)
        self.tweet_ids = _normalize_string_list(tweet_ids) or list(self.default_tweet_ids)
        self.count = max(1, int(count or self.default_count))

    def get_cookies(self):
        print(f"Reading cookie file: {COOKIE_FILE}")
        with open(COOKIE_FILE, "r", encoding="utf-8") as file_obj:
            cookie_str = file_obj.read().strip()

        print(f"Using cookie preview: {cookie_str[:50]}...")
        return cookie_str

    def _resolve_mid(self, tweet_id):
        tweet_id = str(tweet_id).strip()
        if tweet_id.isdigit():
            return tweet_id
        return str(url_to_mid(tweet_id))

    def start_requests(self):
        cookies = self.get_cookies()
        base_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://weibo.com",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }

        print(f"Starting comment crawl for {len(self.tweet_ids)} post(s), page size={self.count}")
        for tweet_id in self.tweet_ids:
            mid = self._resolve_mid(tweet_id)
            url = (
                "https://weibo.com/ajax/statuses/buildComments?"
                f"is_reload=1&id={mid}&is_show_bulletin=2&is_mix=0&count={self.count}"
            )
            headers = dict(base_headers, Referer=f"https://weibo.com/{mid}")
            yield Request(
                url,
                callback=self.parse,
                meta={"source_url": url, "mid": mid},
                cookies=cookies,
                headers=headers,
            )

    def parse(self, response, **kwargs):
        print(f"Comment response status: {response.status}")
        if response.status == 403:
            print(f"Forbidden response preview: {response.text[:200]}")
            return

        try:
            data = json.loads(response.text)
            print(f"Parsed {len(data.get('data', []))} comment item(s)")
        except Exception as exc:
            print(f"Failed to parse JSON: {exc}")
            print(f"Response preview: {response.text[:200]}")
            return

        for comment_info in data.get("data", []):
            yield self.parse_comment(comment_info)

            if "more_info" in comment_info:
                url = (
                    "https://weibo.com/ajax/statuses/buildComments?is_reload=1"
                    f"&id={comment_info['id']}&is_show_bulletin=2&is_mix=1"
                    "&fetch_level=1&max_id=0&count=100"
                )
                yield Request(
                    url,
                    callback=self.parse,
                    priority=20,
                    cookies=response.request.cookies,
                    headers=response.request.headers,
                )

        if data.get("max_id", 0) != 0 and "fetch_level=1" not in response.url:
            url = response.meta["source_url"] + "&max_id=" + str(data["max_id"])
            yield Request(
                url,
                callback=self.parse,
                meta=response.meta,
                cookies=response.request.cookies,
                headers=response.request.headers,
            )

    @staticmethod
    def parse_comment(data):
        item = {
            "created_at": parse_time(data["created_at"]),
            "_id": data["id"],
            "like_counts": data["like_counts"],
            "ip_location": data.get("source", ""),
            "content": data["text_raw"],
            "comment_user": parse_user_info(data["user"]),
        }
        if "reply_comment" in data:
            item["reply_comment"] = {
                "_id": data["reply_comment"]["id"],
                "text": data["reply_comment"]["text"],
                "user": parse_user_info(data["reply_comment"]["user"]),
            }
        return item
