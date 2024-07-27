"""
simple rag demo on instagram gcs

prompt -> chromadb query -> groq llm call

noah trupin, 2024
"""

import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
from groq import Groq
import pandas as pd
from typing import Any, List, Dict, Set

GC_DIR = os.environ["MY_GC_DIR"]
GC_NAME = os.environ["MY_GC_NAME"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

client = Groq(
    api_key=GROQ_API_KEY
)

def load_gc(dir: str, name: str = GC_NAME) -> pd.DataFrame:
    """
    load gc from directory, save as pickle with name

    loads gc json files into pandas dataframe, sanitizes, sorts by send time,
    converts timestamps to datetimes, and writes out to pickle.

    if pickle already exists, reads from pickle.
    """

    dbname = f"{name}.pkl"

    if os.path.isfile(dbname):
        df = pd.read_pickle(dbname)
        if isinstance(df, pd.Series):
            raise Exception("found series in archive. expected dataframe")
        return df

    dfs = []
    files: List[str] = [f for f in os.listdir(dir) if ".json" in f]
    for file in files:
        path: str = os.path.join(dir, file)

        with open(path, "r", encoding="utf-8") as fp:
            data: Any = json.load(fp)
            df = pd.json_normalize(data, record_path="messages")
            df = df.dropna(subset=["content"])
            df = df[["sender_name", "timestamp_ms", "content"]]
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("timestamp_ms")
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms")

    df.to_pickle(dbname)

    return df

def make_docs(df: pd.DataFrame) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """
    makes chromadb docs from pandas dataframe. groups by send day, formats
    as line-by-line conversation, returns daily conversation, timestamp
    metadata, and indices for ids.
    """

    df = df.copy()
    df["formatted"] = df.apply(lambda x: f"{x['sender_name']}: {x['content']}", axis=1)
    df["grouping"] = df["timestamp"].dt.floor("d")
    df = df.groupby(df["grouping"], as_index=False).agg({
        "formatted": "\n".join,
        "timestamp": lambda x: x.iloc[0],
    }).reset_index()
    df["timestamp_str"] = df["timestamp"].dt.strftime('%Y-%m-%d')
    meta = [{"timestamp": x} for x in df["timestamp_str"].tolist()]
    ids = [f"{x}" for x in df.index.tolist()]

    return df["formatted"].tolist(), meta, ids

def make_client(name) -> tuple[chromadb.PersistentClient, chromadb.Collection]:
    """
    make chromadb client from given db. inits db with new docs if does not
    exist. returns client and primary collection.
    """

    dbname = f"{name}.chromadb"
    client = chromadb.PersistentClient(path=dbname)
    collection = client.get_or_create_collection(name=f"{name}_collection")
    if collection.count() == 0:
        df = load_gc(GC_DIR, name)
        docs, metas, ids = make_docs(df)
        collection.add(
            documents=docs, metadatas=metas, ids=ids
        )
    return client, collection

def query_collection(collection: chromadb.Collection, prompt: str) -> chromadb.QueryResult:
    """
    query chromadb collection using prompt. returns top 5 documents.
    """

    result = collection.query(
        query_texts=[prompt],
        n_results=5,
        include=["documents"]
    )

    return result

def query_groq(prompt, documents) -> str:
    """
    format prompt and documents and query groq. llama-3.1 needed for context length,
    otherwise we cannot fit docs.
    """

    with open("system_prompt.txt", "r") as fp:
        system_prompt = fp.read()

    prompt = f"""
    Question: {prompt}

    Relevant documents:

    {documents}
    """

    print(len(prompt))

    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return completion.choices[0].message.content

df = load_gc(GC_DIR, GC_NAME)
db, collection = make_client(GC_NAME)

if __name__ == "__main__":
    while True:
        prompt = input("> ")
        if prompt == "exit":
            break

        docs = "\n\n------------------------\n\n".join(
            query_collection(collection, prompt)["documents"][0])
        response = query_groq(prompt, docs)

        print(response)
