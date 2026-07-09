from __future__ import annotations

SOURCES = {
    # general English (fluency). sample-10BT = ~10B tokens.
    "fineweb_edu": dict(
        repo="HuggingFaceFW/fineweb-edu", glob="sample/10BT/*.parquet",
        text_key=["text"], bucket="general", target_tokens=12_000_000_000,
    ),
    # pharma / biomedical (domain): PubMed titles+abstracts.
    "pubmed": dict(
        repo="casinca/PUBMED_title_abstracts_2019_baseline", glob="**/*.parquet",
        text_key=["text", "abstract", "article"], bucket="domain",
        target_tokens=6_000_000_000,
    ),
}


def list_parquet_files(source):
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    spec = SOURCES[source]
    repo = spec["repo"]
    patterns = [
        f"datasets/{repo}/{spec['glob']}",
        f"datasets/{repo}/**/*.parquet",
        # HF auto-converts non-parquet datasets to this hidden branch:
        f"datasets/{repo}@refs/convert/parquet/**/*.parquet",
    ]
    for pat in patterns:
        try:
            paths = fs.glob(pat)
        except Exception:
            paths = []
        if paths:
            return [f"hf://{p}" for p in sorted(paths)]
    return []


def get_text(example, text_key):
    keys = text_key if isinstance(text_key, list) else [text_key]
    for k in keys:
        if k in example and example[k]:
            v = example[k]
            return v if isinstance(v, str) else " ".join(map(str, v))
    for k, v in example.items():
        if isinstance(v, str) and len(v) > 20:
            return v
    return None


def iter_texts_from_files(files, text_key, max_docs=None):
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files=files, split="train", streaming=True)
    n = 0
    for ex in ds:
        txt = get_text(ex, text_key)
        if txt and len(txt.strip()) > 1:
            yield txt
            n += 1
            if max_docs and n >= max_docs:
                return


def iter_texts(name, max_docs=None):
    spec = SOURCES[name]
    files = list_parquet_files(name)
    yield from iter_texts_from_files(files, spec["text_key"], max_docs=max_docs)


def iter_source_shard(name, shard_index, num_shards, max_docs=None):
    from datasets import load_dataset
    spec = SOURCES[name]
    try:
        ds = load_dataset(spec["repo"], split="train", streaming=True)
    except Exception:
        ds = load_dataset(spec["repo"], split="train", streaming=True, trust_remote_code=True)
    ds = ds.shard(num_shards=num_shards, index=shard_index)
    n = 0
    for ex in ds:
        txt = get_text(ex, spec["text_key"])
        if txt and len(txt.strip()) > 1:
            yield txt
            n += 1
            if max_docs and n >= max_docs:
                return
