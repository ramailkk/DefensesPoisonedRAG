from sentence_transformers import SentenceTransformer
import torch
import random
import copy
import torch.multiprocessing as mp
import numpy as np  # <--- Added missing import
from src.utils import load_json, progress_bar
from loguru import logger

class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module.
    """
    def __init__(self, module):
        self._stored_gradient = None
        self.hook_handle = module.register_full_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient
    
    def close(self):
        self.hook_handle.remove()

def get_embeddings(model):
    """Returns the wordpiece embedding module."""
    if isinstance(model, SentenceTransformer):
        embeddings = model[0].auto_model.embeddings.word_embeddings
    else:
        embeddings = model.embeddings.word_embeddings
    return embeddings

def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        if filter is not None:
            gradient_dot_embedding_matrix -= filter
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids

# --- WORKER FUNCTION FOR MULTIPROCESSING ---
def hotflip_worker(rank, subset_queries, model, c_model, tokenizer, args, all_adv_texts, config_dict):
    """
    Worker function to process a chunk of queries on a specific GPU.
    """
    device = f'cuda:{rank}'
    
    # Move models to the specific GPU for this process
    model.to(device)
    model.eval()
    
    c_model.to(device)
    c_model.eval()
    
    # Re-establish helper functions that might depend on device
    # (assuming get_emb was passed as a string or handled inside)
    # Since get_emb is a function, we handle it based on model type inside the loop or pass logic
    # For simplicity, we implement the get_emb logic directly here based on common types
    def local_get_emb(model_instance, input_data):
        if 'contriever' in args.eval_model_code:
            return model_instance(**input_data)
        elif 'ance' in args.eval_model_code:
            input_data.pop('token_type_ids', None)
            return model_instance(input_data)["sentence_embedding"]
        return model_instance(**input_data)

    adv_text_groups = []
    
    # Create progress bar only for rank 0 to avoid messy logs, or use specific description
    iterator = subset_queries
    if rank == 0:
        iterator = progress_bar(subset_queries, desc=f"GPU {rank} Processing")

    for query_score in iterator:
        query = query_score['query']
        top1_score = query_score['top1_score']
        id = query_score['id']
        adv_texts_b = all_adv_texts[id]['adv_texts']

        adv_texts=[]
        for j in range(args.adv_per_query):
            adv_b = adv_texts_b[j]
            adv_b = tokenizer(adv_b, max_length=config_dict['max_seq_length'], truncation=True, padding=False)['input_ids']
            
            if config_dict['gold_init']:
                adv_a = query
                adv_a = tokenizer(adv_a, max_length=config_dict['max_seq_length'], truncation=True, padding=False)['input_ids']
            else: 
                adv_a = [tokenizer.mask_token_id] * config_dict['num_adv_passage_tokens']

            embeddings = get_embeddings(c_model) 
            # Register hook
            embedding_gradient = GradientStorage(embeddings) 
            
            adv_passage = adv_a + adv_b 
            adv_passage_ids = torch.tensor(adv_passage, device=device).unsqueeze(0) 
            adv_passage_attention = torch.ones_like(adv_passage_ids, device=device)     
            adv_passage_token_type = torch.zeros_like(adv_passage_ids, device=device) 

            q_sent = tokenizer(query, max_length=config_dict['max_seq_length'], truncation=True, padding="max_length" if config_dict['pad_to_max_length'] else False, return_tensors="pt")
            q_sent = {key: value.to(device) for key, value in q_sent.items()}
            
            # Detach q_emb to save memory and prevent graph retention
            q_emb = local_get_emb(model, q_sent).detach()            
            
            for it_ in range(config_dict['num_iter']):
                grad = None   
                c_model.zero_grad()

                p_sent = {'input_ids': adv_passage_ids, 
                        'attention_mask': adv_passage_attention, 
                        'token_type_ids': adv_passage_token_type}
                
                # Enable gradient calculation for the input
                p_emb = local_get_emb(c_model, p_sent)  

                if args.score_function == 'dot':
                    sim = torch.mm(p_emb, q_emb.T)
                elif args.score_function == 'cos_sim':
                    sim = torch.cosine_similarity(p_emb, q_emb)
                else: raise KeyError
                
                loss = sim.mean()
                
                # Early stop check
                if config_dict['early_stop'] and sim.item() > top1_score + 0.1: 
                    break
                
                loss.backward()                

                temp_grad = embedding_gradient.get()
                if temp_grad is not None:
                    if grad is None:
                        grad = temp_grad.sum(dim=0)
                    else:
                        grad += temp_grad.sum(dim=0)

                # HotFlip Logic
                token_to_flip = random.randrange(len(adv_a))
                candidates = hotflip_attack(grad[token_to_flip],
                                            embeddings.weight,
                                            increase_loss=True,
                                            num_candidates=config_dict['num_cand'],
                                            filter=None)                
                current_score = 0
                candidate_scores = torch.zeros(config_dict['num_cand'], device=device) 

                temp_score = loss.sum().cpu().item()
                current_score += temp_score

                # Evaluate Candidates
                for i, candidate in enumerate(candidates):
                    temp_adv_passage = adv_passage_ids.clone()
                    temp_adv_passage[:, token_to_flip] = candidate
                    temp_p_sent = {'input_ids': temp_adv_passage, 
                        'attention_mask': adv_passage_attention, 
                        'token_type_ids': adv_passage_token_type}
                    
                    with torch.no_grad():
                        temp_p_emb = local_get_emb(c_model, temp_p_sent)
                        if args.score_function == 'dot':
                            temp_sim = torch.mm(temp_p_emb, q_emb.T)
                        elif args.score_function == 'cos_sim':
                            temp_sim = torch.cosine_similarity(temp_p_emb, q_emb)
                        
                        can_loss = temp_sim.mean()
                        temp_score = can_loss.sum().cpu().item()
                        candidate_scores[i] += temp_score

                # Update best candidate
                if (candidate_scores > current_score).any():
                    best_candidate_idx = candidate_scores.argmax()
                    adv_passage_ids[:, token_to_flip] = candidates[best_candidate_idx]
                else:
                    continue      
            
            # Remove hook to prevent memory leak
            embedding_gradient.close()
            
            adv_text = tokenizer.decode(adv_passage_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            adv_texts.append(adv_text)
        adv_text_groups.append(adv_texts)
    
    return adv_text_groups


class Attacker():
    def __init__(self, args, **kwargs) -> None:
        self.args = args
        self.attack_method = args.attack_method
        self.adv_per_query = args.adv_per_query
        self.model = kwargs.get('model', None)
        self.c_model = kwargs.get('c_model', None)
        self.tokenizer = kwargs.get('tokenizer', None)
        self.get_emb = kwargs.get('get_emb', None)
        
        self.hotflip_config = {}
        if 'hotflip' in args.attack_method:
            self.hotflip_config['max_seq_length'] = kwargs.get('max_seq_length', 128)
            self.hotflip_config['pad_to_max_length'] = kwargs.get('pad_to_max_length', True)
            self.hotflip_config['per_gpu_eval_batch_size'] = kwargs.get('per_gpu_eval_batch_size', 64)
            self.hotflip_config['num_adv_passage_tokens'] = kwargs.get('num_adv_passage_tokens', 30)            
            self.hotflip_config['num_cand'] = kwargs.get('num_cand', 100)
            self.hotflip_config['num_iter'] = kwargs.get('num_iter', 30)
            self.hotflip_config['gold_init'] = kwargs.get('gold_init', True)
            self.hotflip_config['early_stop'] = kwargs.get('early_stop', False)
            
        self.all_adv_texts = load_json(f'results/adv_targeted_results/{args.eval_dataset}.json')
        logger.info(f"Initializing attacker with method: {args.attack_method}")

    def get_attack(self, target_queries) -> list:
        adv_text_groups = [] 
        if self.attack_method == "LM_targeted":
            for i in range(len(target_queries)):
                question = target_queries[i]['query']
                id = target_queries[i]['id']
                adv_texts_b = self.all_adv_texts[id]['adv_texts'][:self.adv_per_query]
                adv_text_a = question + "."
                adv_texts = [adv_text_a + i for i in adv_texts_b]
                adv_text_groups.append(adv_texts)  
        elif self.attack_method == 'hotflip':
            adv_text_groups = self.hotflip_multigpu(target_queries)
        else: 
            raise NotImplementedError
        return adv_text_groups       

    def hotflip_multigpu(self, target_queries) -> list:
        """
        Parallelizes the HotFlip attack across available GPUs.
        """
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            logger.info("Only 1 GPU detected. Running HotFlip sequentially.")
            # Fallback to single process but using the worker function logic locally
            # We need to ensure models are on GPU 0
            self.model.to('cuda:0')
            self.c_model.to('cuda:0')
            return hotflip_worker(0, target_queries, self.model, self.c_model, 
                                  self.tokenizer, self.args, self.all_adv_texts, self.hotflip_config)

        logger.info(f"Parallelizing HotFlip across {gpu_count} GPUs.")
        
        # 1. Move master models to CPU to allow clean copying/spawning
        self.model.cpu()
        self.c_model.cpu()
        
        # 2. Split queries into chunks
        chunk_size = int(np.ceil(len(target_queries) / gpu_count))
        query_chunks = [target_queries[i:i + chunk_size] for i in range(0, len(target_queries), chunk_size)]
        
        # 3. Setup Multiprocessing
        # We use 'spawn' to ensure CUDA context is initialized fresh in each process
        ctx = mp.get_context('spawn')
        
        # Prepare arguments for each process
        process_args = []
        for rank in range(gpu_count):
            if rank < len(query_chunks):
                # Note: Passing the model object here works because it's on CPU.
                # The worker will move its copy to the specific GPU.
                p_args = (
                    rank,
                    query_chunks[rank],
                    self.model,
                    self.c_model,
                    self.tokenizer,
                    self.args,
                    self.all_adv_texts,
                    self.hotflip_config
                )
                process_args.append(p_args)

        # 4. Run Pool
        with ctx.Pool(processes=gpu_count) as pool:
            results = pool.starmap(hotflip_worker, process_args)
            
        # 5. Flatten results (results is a list of lists)
        combined_adv_groups = [item for sublist in results for item in sublist]
        
        # 6. Cleanup: Move models back to GPU 0 if needed later (optional)
        # self.model.to('cuda:0')
        # self.c_model.to('cuda:0')
        
        return combined_adv_groups