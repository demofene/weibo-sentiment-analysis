# -*- coding: utf-8 -*-
import datetime
import json
import os.path
import time

try:
    from project_paths import RAW_DATA_DIR, ensure_dir
except ImportError:  # pragma: no cover
    from weibospider.project_paths import RAW_DATA_DIR, ensure_dir


class JsonWriterPipeline(object):
    """
    写入json文件的pipline
    """

    def __init__(self):
        self.file = None
        ensure_dir(RAW_DATA_DIR)

    def process_item(self, item, spider):
        """
        处理item
        """
        if not self.file:
            now = datetime.datetime.now()
            file_name = spider.name + "_" + now.strftime("%Y%m%d%H%M%S") + '.jsonl'
            output_file = os.path.join(RAW_DATA_DIR, file_name)
            self.file = open(output_file, 'wt', encoding='utf-8')
        item['crawl_time'] = int(time.time())
        line = json.dumps(dict(item), ensure_ascii=False) + "\n"
        self.file.write(line)
        self.file.flush()
        return item
