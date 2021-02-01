import base64
import glob
import hashlib
import os
import re
from multiprocessing.dummy import Pool as ThreadPool
from time import time

import boto3
import numpy as np
import pandas as pd
from botocore import UNSIGNED
from botocore.client import Config

# Boto3 anonymous login to common crawl
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))


def url_hash(url):
    return base64.urlsafe_b64encode(hashlib.md5(url.encode("utf-8")).digest())[
        :16
    ].decode("ascii")


def warc_download(row):
    # Ger params from dataframe row
    offset = row["warc_record_offset"]
    length = row["warc_record_length"]
    filename = row["warc_filename"]
    local_filename = filename.replace(".", "").replace("warcgz", ".warc.gz")

    # Count the range
    offset_end = offset + length - 1
    byte_range = "bytes={offset}-{end}".format(offset=offset, end=offset_end)
    gzipped_text = s3.get_object(Bucket="commoncrawl", Key=filename, Range=byte_range)[
        "Body"
    ].read()

    # Create nested directory
    filepath = "/".join(local_filename.split("/")[:-1])
    try:
        if not os.path.exists(filepath):
            os.makedirs(filepath)
    except:
        time.sleep(0.5)

    # Save the requested file in GZIP
    with open(local_filename, "wb") as f:
        f.write(gzipped_text)


if __name__ == "__main__":
    for url_host_name in glob.glob("./gzip/*.gzip"):
        url_host_name = os.path.basename(url_host_name).replace(".parquet.gzip", "")

        regex_dict = {
            "tinhte": r"tinhte.vn/thread",
            "topdev": r"topdev.vn/blog",
            "kipalog": r"kipalog.com/post",
            "vuilaptrinh": r"(/|-)[0-9]{4}(/|-)[0-9]{2}(/|-)[0-9]{2}",
            "toidicodedao": r"(/|-)[0-9]{4}(/|-)[0-9]{2}(/|-)[0-9]{2}",
            "codeaholicguy": r"(/|-)[0-9]{4}(/|-)[0-9]{2}(/|-)[0-9]{2}",
            "angel": r"angel.co/company",
        }

        url_regex = re.compile(r".*")  # default
        for k in regex_dict.keys():
            if k in url_host_name:
                url_regex = re.compile(regex_dict[k])
        print(url_regex)
        # Prepare the list of URLs to crawl
        to_crawl = pd.read_parquet(f"./gzip/{url_host_name}.parquet.gzip")
        samples = to_crawl["url"].values[:10]
        to_crawl["url"] = to_crawl["url"].apply(
            lambda x: x if url_regex.search(x) else np.nan
        )
        to_crawl.dropna(inplace=True)

        print(
            "Going to crawl {0} URLs from domain {1}".format(
                len(to_crawl["url"].values), url_host_name
            )
        )

        # Download function
        test = False
        if test:
            # Speed test with 1000 elements
            ready_to_launch = to_crawl.copy()
            ready_to_launch = ready_to_launch[0:1000]
        else:
            ready_to_launch = to_crawl.copy()

        pool = ThreadPool(32)

        t0 = time()

        results = pool.map(warc_download, ready_to_launch.T.to_dict().values())

        print("Downloading took {0} second".format(round(time() - t0, 2)))
