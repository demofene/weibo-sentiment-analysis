import json

from scrapy import Spider
from scrapy.http import Request

try:
    from project_paths import COOKIE_FILE
except ImportError:  # pragma: no cover
    from weibospider.project_paths import COOKIE_FILE
from spiders.common import parse_user_info


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


class UserSpider(Spider):
    """Collect Weibo user profile data."""

    name = "user_spider"
    default_user_ids = ["1749127163"]

    def __init__(self, user_ids=None, **kwargs):
        super().__init__(**kwargs)
        self.user_ids = _normalize_string_list(user_ids) or list(self.default_user_ids)

    def start_requests(self):
        with open(COOKIE_FILE, "r", encoding="utf-8") as file_obj:
            cookie_str = file_obj.read().strip()

        print(f"Reading cookie file: {COOKIE_FILE}")
        print(f"Using cookie preview: {cookie_str[:50]}...")
        cookies = cookie_str

        referer_user = self.user_ids[0]
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": f"https://weibo.com/u/{referer_user}",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }

        print(f"Starting user crawl for {len(self.user_ids)} user id(s)")
        urls = [f"https://weibo.com/ajax/profile/info?uid={user_id}" for user_id in self.user_ids]
        for url in urls:
            yield Request(url, callback=self.parse, cookies=cookies, headers=headers)

    def parse(self, response, **kwargs):
        print(f"User info status: {response.status}")
        print(f"Response preview: {response.text[:200]}")

        data = json.loads(response.text)
        item = parse_user_info(data["data"]["user"])

        url = f"https://weibo.com/ajax/profile/detail?uid={item['_id']}"
        yield Request(
            url,
            callback=self.parse_detail,
            meta={"item": item},
            cookies=response.request.cookies,
            headers=response.request.headers,
        )

    @staticmethod
    def parse_detail(response):
        print(f"User detail status: {response.status}")

        item = response.meta["item"]
        data = json.loads(response.text)["data"]
        item["birthday"] = data.get("birthday", "")
        if "created_at" not in item:
            item["created_at"] = data.get("created_at", "")
        item["desc_text"] = data.get("desc_text", "")
        item["ip_location"] = data.get("ip_location", "")
        item["sunshine_credit"] = data.get("sunshine_credit", {}).get("level", "")
        item["label_desc"] = [label["name"] for label in data.get("label_desc", [])]

        if "company" in data:
            item["company"] = data["company"]
        if "education" in data:
            item["education"] = data["education"]

        yield item
