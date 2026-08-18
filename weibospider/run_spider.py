import argparse
import os

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from spiders.comment import CommentSpider
from spiders.tweet_by_keyword import TweetSpiderByKeyword
from spiders.user import UserSpider

# from spiders.tweet_by_user_id import TweetSpiderByUserID
# from spiders.tweet_by_tweet_id import TweetSpiderByTweetID
# from spiders.follower import FollowerSpider
# from spiders.fan import FanSpider
# from spiders.repost import RepostSpider


def build_parser():
    parser = argparse.ArgumentParser(description="Run one of the Weibo spiders.")
    parser.add_argument("mode", choices=["comment", "user", "tweet_by_keyword"], help="Spider mode to run.")
    parser.add_argument(
        "--tweet-ids",
        nargs="+",
        default=None,
        help="One or more Weibo post ids or short ids used by the comment spider.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of comments fetched per page for the comment spider.",
    )
    parser.add_argument(
        "--user-ids",
        nargs="+",
        default=None,
        help="One or more user ids used by the user spider.",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=None,
        help="One or more keywords used by the keyword spider.",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="Keyword crawl start time, for example 2026-01-01 00:00.",
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help="Keyword crawl end time, for example 2026-01-03 23:00.",
    )
    parser.add_argument(
        "--split-by-hour",
        default=None,
        choices=["true", "false"],
        help="Whether keyword crawling should split the requested range by hour.",
    )
    return parser


def build_spider_kwargs(args):
    if args.mode == "comment":
        kwargs = {"count": args.count}
        if args.tweet_ids:
            kwargs["tweet_ids"] = args.tweet_ids
        return kwargs

    if args.mode == "user":
        kwargs = {}
        if args.user_ids:
            kwargs["user_ids"] = args.user_ids
        return kwargs

    if args.mode == "tweet_by_keyword":
        kwargs = {}
        if args.keywords:
            kwargs["keywords"] = args.keywords
        if args.start_time:
            kwargs["start_time"] = args.start_time
        if args.end_time:
            kwargs["end_time"] = args.end_time
        if args.split_by_hour is not None:
            kwargs["split_by_hour"] = args.split_by_hour
        return kwargs

    return {}


def main():
    args = build_parser().parse_args()

    os.environ["SCRAPY_SETTINGS_MODULE"] = "settings"
    settings = get_project_settings()
    process = CrawlerProcess(settings)

    mode_to_spider = {
        "comment": CommentSpider,
        "user": UserSpider,
        "tweet_by_keyword": TweetSpiderByKeyword,
        # 'fan': FanSpider,
        # 'follow': FollowerSpider,
        # 'repost': RepostSpider,
        # 'tweet_by_tweet_id': TweetSpiderByTweetID,
        # 'tweet_by_user_id': TweetSpiderByUserID,
    }

    spider_class = mode_to_spider[args.mode]
    spider_kwargs = build_spider_kwargs(args)
    print(f"Starting spider '{args.mode}' with arguments: {spider_kwargs}")
    process.crawl(spider_class, **spider_kwargs)
    process.start()


if __name__ == "__main__":
    main()
