import json
import re

import dateutil.parser


def base62_decode(string):
    """Decode a base62 string into an integer."""
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    string = str(string)
    num = 0
    for idx, char in enumerate(string):
        power = len(string) - (idx + 1)
        num += alphabet.index(char) * (len(alphabet) ** power)
    return num


def reverse_cut_to_length(content, code_func, cut_num=4, fill_num=7):
    """Split a short Weibo id into chunks and decode each chunk."""
    content = str(content)
    cut_list = [content[i - cut_num if i >= cut_num else 0 : i] for i in range(len(content), 0, -cut_num)]
    cut_list.reverse()

    result = []
    for index, item in enumerate(cut_list):
        decoded = str(code_func(item))
        if index > 0 and len(decoded) < fill_num:
            decoded = (fill_num - len(decoded)) * "0" + decoded
        result.append(decoded)
    return "".join(result)


def url_to_mid(url: str):
    """Convert a short-url style Weibo id to the numeric mid."""
    return int(reverse_cut_to_length(url, base62_decode))


def parse_time(value):
    """Convert Weibo time strings to a stable local datetime string."""
    if not value:
        return ""
    return dateutil.parser.parse(value).strftime("%Y-%m-%d %H:%M:%S")


def _strip_html(text):
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", str(text))


def parse_user_info(data):
    """Parse the user payload returned by Weibo APIs."""
    data = data or {}
    user = {
        "_id": str(data.get("id", "")),
        "avatar_hd": data.get("avatar_hd", ""),
        "nick_name": data.get("screen_name", ""),
        "verified": bool(data.get("verified", False)),
    }

    extra_keys = [
        "description",
        "followers_count",
        "friends_count",
        "statuses_count",
        "gender",
        "location",
        "mbrank",
        "mbtype",
        "credit_score",
    ]
    for key in extra_keys:
        if key in data:
            user[key] = data[key]

    if data.get("created_at"):
        user["created_at"] = parse_time(data["created_at"])

    if user["verified"]:
        user["verified_type"] = data.get("verified_type", "")
        if "verified_reason" in data:
            user["verified_reason"] = data.get("verified_reason", "")

    return user


def parse_tweet_info(data):
    """Parse a single Weibo post payload."""
    data = data or {}
    source = _strip_html(data.get("source", ""))
    user = parse_user_info(data.get("user") or {})

    tweet = {
        "_id": str(data.get("mid", "")),
        "mblogid": data.get("mblogid", ""),
        "created_at": parse_time(data.get("created_at")),
        "geo": data.get("geo"),
        "ip_location": data.get("region_name"),
        "reposts_count": int(data.get("reposts_count", 0) or 0),
        "comments_count": int(data.get("comments_count", 0) or 0),
        "attitudes_count": int(data.get("attitudes_count", 0) or 0),
        "source": source,
        "content": str(data.get("text_raw", "")).replace("\u200b", "").strip(),
        "pic_urls": [f"https://wx1.sinaimg.cn/orj960/{pic_id}" for pic_id in data.get("pic_ids", [])],
        "pic_num": int(data.get("pic_num", 0) or 0),
        "isLongText": bool(data.get("isLongText", False) or data.get("continue_tag")),
        "is_retweet": "retweeted_status" in data,
        "user": user,
    }

    page_info = data.get("page_info") or {}
    if page_info.get("object_type") == "video":
        media_info = page_info.get("media_info")
        if not media_info and page_info.get("cards"):
            media_info = page_info["cards"][0].get("media_info")
        if media_info:
            tweet["video"] = media_info.get("stream_url", "")
            tweet["video_online_numbers"] = media_info.get("online_users_number")

    if user.get("_id") and tweet["mblogid"]:
        tweet["url"] = f"https://weibo.com/{user['_id']}/{tweet['mblogid']}"
    else:
        tweet["url"] = ""

    if "retweeted_status" in data:
        retweeted_status = data.get("retweeted_status") or {}
        tweet["retweet_id"] = str(retweeted_status.get("mid", ""))

    if "reads_count" in data:
        tweet["reads_count"] = data.get("reads_count")

    return tweet


def parse_long_tweet(response):
    """Fill in the long text body for a Weibo post."""
    payload = json.loads(response.text)
    data = payload.get("data") or {}
    item = response.meta["item"]
    long_text = data.get("longTextContent", item.get("content", ""))
    item["content"] = _strip_html(long_text).replace("\u200b", "").strip()
    yield item
