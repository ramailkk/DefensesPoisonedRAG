from beir import util
import os
import random
import pathlib
from tqdm import tqdm # useful for progress bars

def resize_dataset(source_path, target_path, sample_rate=0.01):
    """
    Reads a BEIR dataset and creates a mini version with {sample_rate} size.
    Ensures consistency so that qrels only reference existing docs/queries.
    """
    print(f"Creating {sample_rate*100}% subset of {os.path.basename(source_path)}...")
    os.makedirs(target_path, exist_ok=True)
    
    # Sets to track IDs that made the cut
    saved_doc_ids = set()
    saved_query_ids = set()

    # 1. Process Corpus (Stream and save 1%)
    source_corpus = os.path.join(source_path, "corpus.jsonl")
    target_corpus = os.path.join(target_path, "corpus.jsonl")
    
    if os.path.exists(source_corpus):
        with open(source_corpus, 'r', encoding='utf-8') as f_in, \
             open(target_corpus, 'w', encoding='utf-8') as f_out:
            
            for line in tqdm(f_in, desc="Resizing Corpus"):
                # Randomly select lines based on sample_rate
                if random.random() < sample_rate:
                    # Parse just the _id to save memory
                    # (BEIR jsonl lines start with {"_id": "..."})
                    doc_id = line.split('"_id": "')[1].split('"')[0]
                    saved_doc_ids.add(doc_id)
                    f_out.write(line)

    # 2. Process Queries (Stream and save 1%)
    source_queries = os.path.join(source_path, "queries.jsonl")
    target_queries = os.path.join(target_path, "queries.jsonl")

    if os.path.exists(source_queries):
        with open(source_queries, 'r', encoding='utf-8') as f_in, \
             open(target_queries, 'w', encoding='utf-8') as f_out:
            
            for line in tqdm(f_in, desc="Resizing Queries"):
                if random.random() < sample_rate:
                    # Parse just the _id
                    query_id = line.split('"_id": "')[1].split('"')[0]
                    saved_query_ids.add(query_id)
                    f_out.write(line)

    # 3. Process Qrels (Filter: Only keep if Doc AND Query exist in sampled sets)
    # BEIR datasets usually have qrels inside a folder named 'qrels'
    source_qrels_dir = os.path.join(source_path, "qrels")
    target_qrels_dir = os.path.join(target_path, "qrels")
    os.makedirs(target_qrels_dir, exist_ok=True)

    if os.path.exists(source_qrels_dir):
        for filename in os.listdir(source_qrels_dir):
            s_qrel = os.path.join(source_qrels_dir, filename)
            t_qrel = os.path.join(target_qrels_dir, filename)
            
            with open(s_qrel, 'r', encoding='utf-8') as f_in, \
                 open(t_qrel, 'w', encoding='utf-8') as f_out:
                
                # Write header
                header = f_in.readline()
                f_out.write(header)
                
                for line in tqdm(f_in, desc=f"Filtering {filename}"):
                    cols = line.strip().split('\t')
                    if len(cols) >= 2:
                        q_id = cols[0]
                        d_id = cols[1]
                        
                        # Only save this qrel if both the query and doc exist in our mini sets
                        if q_id in saved_query_ids and d_id in saved_doc_ids:
                            f_out.write(line)
    
    print(f"Done. Mini dataset saved to: {target_path}")


# --- Main Execution ---

datasets = ['hotpotqa']

# Define where you want the full data and the mini data
out_dir = os.path.join(os.getcwd(), "datasets")

for dataset in datasets:
    # 1. Download Full Dataset
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    data_path = os.path.join(out_dir, dataset)
    
    if not os.path.exists(data_path):
        print(f"Downloading {dataset}...")
        data_path = util.download_and_unzip(url, out_dir)
    
    # 2. Create the 1% Version
    mini_data_path = os.path.join(out_dir, f"{dataset}_mini")
    if not os.path.exists(mini_data_path):
        resize_dataset(data_path, mini_data_path, sample_rate=0.01)

# Cleanup: Remove zip files
os.system('rm datasets/*.zip')

# Cleanup: (Optional) Remove the full datasets if you ONLY want the mini versions
# import shutil
# for dataset in datasets:
#     shutil.rmtree(os.path.join(out_dir, dataset))