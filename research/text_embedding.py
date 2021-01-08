import multiprocessing
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from vncorenlp import VnCoreNLP

# Load rdrsegmenter from VnCoreNLP
rdrsegmenter = VnCoreNLP(
    r"./nlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size="-Xmx500m"
)


def segmentize(row):
    try:
        word_segmented_text = rdrsegmenter.tokenize(row["content"].lower())
        return (
            row["path"],
            " ".join([" ".join(sentence) for sentence in word_segmented_text]),
        )
    except:
        return None


def tf_idf_generator(docs):
    # TF-IDF
    tfidf = TfidfVectorizer(smooth_idf=True, use_idf=True, max_df=0.90, min_df=0.10)
    tfidf.fit(docs)
    pickle.dump(tfidf, open("tfidf.bin", "wb"))


if __name__ == "__main__":
    # Load text data parquet
    df = pd.read_parquet("topdup.parquet.gzip")

    data = list()
    for _, row in df.iterrows():
        data.append(row)

    with multiprocessing.Pool(multiprocessing.cpu_count() - 2) as p:
        p_df = list(tqdm(p.imap(segmentize, data), total=len(data)))

    p_df = pd.DataFrame(p_df, columns=df.columns)

    # Remove empty string content after preprocessing
    p_df = p_df[p_df["content"] != ""]
    p_df.dropna(inplace=True)
    p_df.to_parquet("topdup_seg.parquet.gzip", compression="gzip")
    rdrsegmenter.close()
    print("Completed segmentizing text, {} docs".format(len(p_df["content"].values)))

    # TF-IDF
    tf_idf_generator(p_df["content"].values.tolist())
    print("Completed TF-IDF generation")

    print("Done")
