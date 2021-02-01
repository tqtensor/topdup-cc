import os
import re
import sys
import time

import boto3
import config
import pandas as pd
import s3fs

### Run this command to get AWS session token
"""
aws configure set aws_access_key_id {key_id}
aws configure set aws_secret_access_key {key_secret}
aws sts get-session-token --duration-seconds 129600
"""

key_id = config.key_id
key_secret = config.key_secret

session_key_id = config.session_key_id
session_key_secret = config.session_key_secret
session_token = config.session_token
session = boto3.Session(
    aws_access_key_id=session_key_id,
    aws_secret_access_key=session_key_secret,
    aws_session_token=session_token,
)

gzip_data_folder = config.gzip_data_folder

### Run this query to get the right url_host_name
"""
SELECT COUNT(*)
	,url_host_name
FROM "ccindex"."ccindex"
WHERE regexp_like(crawl, 'CC-MAIN-20[1-2][0-9]-[0-9][0-9]')
	AND subset = 'warc'
	AND url_host_2nd_last_part = 'codeaholicguy'
GROUP BY url_host_name
"""

### Replace the url_host_name
url_host_name = "angel.co"

params = {
    "region": "us-east-1",
    "database": "ccindex",
    "bucket": config.bucket_name,
    "path": "athena/output",
    "query": f"""SELECT url, fetch_time, warc_filename, warc_record_offset, warc_record_length FROM "ccindex"."ccindex" WHERE regexp_like(crawl, 'CC-MAIN-20[1-2][0-9]-[0-9][0-9]') AND subset = 'warc' AND url_host_name = '{url_host_name}'""",
}


def athena_query(client, params):
    response = client.start_query_execution(
        QueryString=params["query"],
        QueryExecutionContext={"Database": params["database"]},
        ResultConfiguration={
            "OutputLocation": "s3://" + params["bucket"] + "/" + params["path"]
        },
    )
    return response


def athena_to_s3(session, params, max_execution=5):
    client = session.client("athena", region_name=params["region"])
    execution = athena_query(client, params)
    execution_id = execution["QueryExecutionId"]
    state = "RUNNING"

    while max_execution > 0 and state in ["RUNNING", "QUEUED"]:
        max_execution = max_execution - 1
        response = client.get_query_execution(QueryExecutionId=execution_id)

        if "QueryExecution" in response and "Status" in response["QueryExecution"]:
            state = response["QueryExecution"]["Status"]["State"]
            if state == "FAILED":
                return state
            elif state == "SUCCEEDED":
                s3_path = response["QueryExecution"]["ResultConfiguration"][
                    "OutputLocation"
                ]
                filename = re.findall(r".*\/(.*)", s3_path)[0]
                return filename
        time.sleep(60)
    return state


def cleanup(session, params):
    # Deletes all files in your path so use carefully!
    s3 = session.resource("s3")
    my_bucket = s3.Bucket(params["bucket"])
    for item in my_bucket.objects.filter(Prefix=params["path"]):
        item.delete()


if __name__ == "__main__":

    if not os.path.exists(gzip_data_folder):
        os.mkdir(gzip_data_folder)

    # Query Athena and get the s3 filename as a result
    s3_filename = athena_to_s3(session, params, 30)
    if s3_filename in ["RUNNING", "FAILED", "QUEUED"]:
        print(s3_filename)
        sys.exit("Failed to query data")
    else:
        print("File name on S3", s3_filename)

    # Read result file into dataframe
    fs = s3fs.S3FileSystem(key=key_id, secret=key_secret)

    fs.invalidate_cache()
    with fs.open(
        "s3://" + params["bucket"] + "/" + params["path"] + "/" + s3_filename, "r"
    ) as f:
        df = pd.read_csv(f)

    df.sort_values(by=["url", "fetch_time"], ascending=False, inplace=True)
    df.drop_duplicates(subset="url", keep="first", inplace=True)

    df.to_parquet(
        "{0}/{1}.parquet.gzip".format(gzip_data_folder, url_host_name.replace(".", "_")),
        compression="gzip",
    )
