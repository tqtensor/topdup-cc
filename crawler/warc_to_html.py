import glob
import multiprocessing
import os
import time

import pandas as pd
import warc
from tqdm.auto import tqdm


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def warc_to_html(filename):
    try:
        warc_file = warc.open(filename, "rb")
    except:
        print(filename)
        os.remove(filename)
        return
    html_filename = filename.replace("warc.gz", "html")

    for _, record in enumerate(warc_file):
        if not record.url:
            continue
        else:
            payload = record.payload.read()
            url = record.url
            _, html = payload.split(b"\r\n\r\n", maxsplit=1)
            html = html.strip()

            with open(html_filename, "w") as f:
                try:
                    f.write(html.decode("utf-8"))
                except Exception as e:
                    print(url, e)

    os.remove(filename)


if __name__ == "__main__":
    max_attempt = 5
    attempt = 0

    while attempt <= max_attempt:
        all_files = glob.glob("./crawl-data/*/*/*/*/*")
        all_files = [x for x in all_files if "html" not in x]
        if len(all_files) == 0:
            time.sleep(10)
            attempt += 1
        else:
            all_chunks = chunks(all_files, 100)

            for chunk in tqdm(list(all_chunks)):
                with multiprocessing.Pool(multiprocessing.cpu_count() - 2) as pool:
                    _ = pool.map(warc_to_html, chunk)
            time.sleep(120)
