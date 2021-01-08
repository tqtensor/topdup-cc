import codecs
import glob
import multiprocessing
import os
import re
import time
from multiprocessing.dummy import Pool as ThreadPool
from random import shuffle

import pandas as pd
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

# Define Vietnamese valid regex
chars_vn = [
    "á",
    "à",
    "ả",
    "ã",
    "ạ",
    "â",
    "ấ",
    "ầ",
    "ẩ",
    "ẫ",
    "ậ",
    "ă",
    "ắ",
    "ằ",
    "ẳ",
    "ẵ",
    "ặ",
    "ó",
    "ò",
    "ỏ",
    "õ",
    "ọ",
    "ô",
    "ố",
    "ồ",
    "ổ",
    "ỗ",
    "ộ",
    "ơ",
    "ớ",
    "ờ",
    "ở",
    "ỡ",
    "ợ",
    "é",
    "è",
    "ẻ",
    "ẽ",
    "ẹ",
    "ê",
    "ế",
    "ề",
    "ể",
    "ễ",
    "ệ",
    "ú",
    "ù",
    "ủ",
    "ũ",
    "ụ",
    "ư",
    "ứ",
    "ừ",
    "ử",
    "ữ",
    "ự",
    "í",
    "ì",
    "ỉ",
    "ĩ",
    "ị",
    "ý",
    "ỳ",
    "ỷ",
    "ỹ",
    "ỵ",
    "đ",
]
chars_vn = "".join(chars_vn)
chars_vn = chars_vn + chars_vn.upper()
# regex pattern of invalid character
pattern_unchars_vn = f"[^a-zA-Z{chars_vn} \.]"


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def html_to_text(filename):
    text_filename = filename.replace("crawl-data", "crawl-text").replace("html", "txt")

    try:
        soup = BeautifulSoup(open(filename), "html.parser")
        contents = list()
        for part in soup.find_all("div", {"class": re.compile(r".*content.*")}):
            contents.append(part.get_text())
        contents = " ".join(contents)
    except Exception as e:
        print(e)
        return

    article = re.sub("\.+", ".", contents)  # multi dots to one dot
    article = re.sub("[?!;…]", ".", article)  # alls to one dot
    article = re.sub(r" +", " ", article)  # remove white space
    article = re.sub(pattern_unchars_vn, "", article.strip())  # remove invalid VN words

    if article:
        # Create nested directory
        filepath = "/".join(text_filename.split("/")[:-1])
        try:
            if not os.path.exists(filepath):
                os.makedirs(filepath)

            with open(text_filename, "w") as f:
                f.write(article)
        except:
            print("Something wrong")


def text_reader(filename):
    return [filename, open(filename, "r").read()]


if __name__ == "__main__":
    # Parse html
    all_html_files = glob.glob("./crawl-data/*/*/*/*/*html")
    shuffle(all_html_files)
    all_text_files = glob.glob("./crawl-text/*/*/*/*/*txt")
    all_text_files = [
        x.replace("crawl-text", "crawl-data").replace("txt", "html")
        for x in all_text_files
    ]
    all_files = list(set(all_html_files) - set(all_text_files))
    print("There are {} html files to parse".format(len(all_files)))

    if len(all_files) > 0:
        all_chunks = chunks(all_files, 1000)
        for chunk in tqdm(list(all_chunks)):
            with multiprocessing.Pool(multiprocessing.cpu_count() - 2) as pool:
                _ = pool.map(html_to_text, chunk)

    # Combine text
    all_text_files = glob.glob("./crawl-text/*/*/*/*/*txt")
    print("There are {} text files to combine".format(len(all_text_files)))

    dfs = list()
    all_chunks = chunks(all_text_files, 1000)
    for chunk in tqdm(list(all_chunks)):
        with ThreadPool(64) as pool:
            data = pool.map(text_reader, chunk)
        data = pd.DataFrame(data, columns=["path", "content"])
        dfs.append(data)

    # Save to parquet
    data = pd.concat(dfs)
    print("Saving data to parquet")
    data.to_parquet("topdup.parquet.gzip", compression="gzip")
