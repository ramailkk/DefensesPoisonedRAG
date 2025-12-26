import pickle
import json
import os
import argparse
from loguru import logger

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Saved {path}")

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)
    logger.info(f"Saved {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nq')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--model_code', type=str, default='contriever')
    parser.add_argument('--size', type=int, default=20, help="How many queries to keep")
    args = parser.parse_args()

    new_name = f"{args.dataset}"
    
    # 1. Load the Incorrect Answers (The source of truth for which questions we attack)
    ans_path = f"data_cache/{args.dataset}_answers.pkl"
    logger.info(f"Loading {ans_path}...")
    with open(ans_path, 'rb') as f:
        full_answers = pickle.load(f)

    # 2. Slice the answers
    # Convert dict to list, slice, then back to dict to maintain structure
    full_ans_list = list(full_answers.values())
    mini_ans_list = full_ans_list[:args.size]
    
    # Create the mini dictionary (The code expects a dict with numeric keys 0..N)
    mini_answers = {i: mini_ans_list[i] for i in range(len(mini_ans_list))}
    
    # Get the List of Question IDs we kept
    valid_qids = set([item['id'] for item in mini_ans_list])
    logger.info(f"Kept {len(valid_qids)} unique queries.")

    # 3. Load the Main Dataset (Corpus, Queries, Qrels)
    data_path = f"data_cache/{args.dataset}_{args.split}.pkl"
    logger.info(f"Loading {data_path}...")
    with open(data_path, 'rb') as f:
        corpus, queries, qrels = pickle.load(f)

    # 4. Filter Qrels (Ground Truth)
    mini_qrels = {qid: doc_dict for qid, doc_dict in qrels.items() if qid in valid_qids}
    
    # Check for missing Qrels (This prevents the KEY ERROR you saw earlier)
    missing_ids = valid_qids - set(mini_qrels.keys())
    if missing_ids:
        logger.warning(f"{len(missing_ids)} queries have no qrels! They will be skipped.")
    
    # 5. Filter Queries
    mini_queries = {qid: text for qid, text in queries.items() if qid in valid_qids}

    # 6. Filter BEIR Results (The JSON file)
    res_path = f"results/beir_results/{args.dataset}-{args.model_code}.json"
    logger.info(f"Loading {res_path}...")
    with open(res_path, 'r') as f:
        full_results = json.load(f)
    
    mini_results = {qid: scores for qid, scores in full_results.items() if qid in valid_qids}

    # --- SAVE EVERYTHING AS A NEW DATASET ---
    
    # A. Save the Answers file
    save_pickle(mini_answers, f"data_cache/{new_name}_answers.pkl")

    # B. Save the Data file
    save_pickle((corpus, mini_queries, mini_qrels), f"data_cache/{new_name}_{args.split}.pkl")

    # C. Save the Results file
    # NOTE: We save it with the name '{new_name}-{model_code}.json' because the main code looks for that pattern
    output_res_dir = "results/beir_results"
    os.makedirs(output_res_dir, exist_ok=True)
    save_json(mini_results, f"{output_res_dir}/{new_name}-{args.model_code}.json")

    logger.success(f"Successfully created mini dataset: {new_name}")
    logger.info(f"To run: python main_trustrag.py --eval_dataset {new_name}")

if __name__ == "__main__":
    main()