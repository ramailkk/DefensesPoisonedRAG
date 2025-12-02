from beir import util
import os, json, random, math

datasets = ["nq", "msmarco", "hotpotqa"]

base_out = os.path.join(os.getcwd(), "datasets")
small_out = os.path.join(os.getcwd(), "datasets_small")
os.makedirs(small_out, exist_ok=True)

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def sample_percent(data, pct=0.01):
    k = max(1, math.floor(len(data) * pct))  # ensure at least 1 item
    return random.sample(data, k)

for dataset in datasets:
    print(f"\n=== Processing {dataset} ===")

    # 1. Download & extract dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    if not os.path.exists(os.path.join(base_out, dataset)):
        util.download_and_unzip(url, base_out)

    dataset_path = os.path.join(base_out, dataset)

    # Paths
    corpus_path = os.path.join(dataset_path, "corpus.jsonl")
    queries_path = os.path.join(dataset_path, "queries.jsonl")
    qrels_path   = os.path.join(dataset_path, "qrels", "train.tsv")  # or dev/test depending on your use

    # 2. Load files
    corpus  = read_jsonl(corpus_path)
    queries = read_jsonl(queries_path)

    # qrels is TSV
    qrels = []
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            qid, _, pid, score = line.strip().split("\t")
            qrels.append({"qid": qid, "pid": pid, "score": score})

    # 3. Sample 1%
    corpus_1p  = sample_percent(corpus, pct=0.01)
    queries_1p = sample_percent(queries, pct=0.01)
    qrels_1p   = sample_percent(qrels, pct=0.01)

    # 4. Save to new folder
    out_dir = os.path.join(small_out, dataset)
    os.makedirs(os.path.join(out_dir, "qrels"), exist_ok=True)

    write_jsonl(os.path.join(out_dir, "corpus.jsonl"), corpus_1p)
    write_jsonl(os.path.join(out_dir, "queries.jsonl"), queries_1p)

    with open(os.path.join(out_dir, "qrels", "train.tsv"), "w", encoding="utf-8") as f:
        for item in qrels_1p:
            f.write(f"{item['qid']}\t0\t{item['pid']}\t{item['score']}\n")

    print(f"Saved 1% of {dataset} into {out_dir}")

# Cleanup .zip files
os.system('rm -f datasets/*.zip')
