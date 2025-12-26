# src/filter_rag_impl.py
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.ensemble import RandomForestClassifier

class FilterRAGDefense:
    def __init__(self, slm_model_name="mistralai/Mistral-7B-Instruct-v0.1", device="cuda"):
        self.device = device
        print(f"Loading FilterRAG components...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        # Initialize SLM
        # Using a smaller model by default for speed if not specified
        self.tokenizer = AutoTokenizer.from_pretrained(slm_model_name)
        self.slm = AutoModelForCausalLM.from_pretrained(
            slm_model_name, 
            torch_dtype=torch.float16, 
            device_map=device
        )
        self.slm.eval()
        self.ml_classifier = None 

    def _generate_slm_response(self, query, document):
        prompt = f"Context: {document}\nQuestion: {query}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.slm.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in generated_text:
            return generated_text.split("Answer:")[-1].strip()
        return generated_text

    def _compute_freq_density(self, query, answer_aj, document, sim_threshold=0.6):
        qa_text = f"{query} {answer_aj}"
        qa_words = list(set(qa_text.lower().split()))
        doc_words = document.lower().split()
        doc_unique_words = list(set(doc_words))
        
        if len(doc_unique_words) == 0: return 0.0

        qa_embeddings = self.embedder.encode(qa_words, convert_to_tensor=True)
        doc_embeddings = self.embedder.encode(doc_unique_words, convert_to_tensor=True)

        cosine_scores = util.cos_sim(doc_embeddings, qa_embeddings)
        max_scores, _ = torch.max(cosine_scores, dim=1)
        
        numerator = 0
        for idx, is_similar in enumerate(max_scores > sim_threshold):
            if is_similar:
                numerator += doc_words.count(doc_unique_words[idx])
        
        return numerator / len(doc_unique_words)

    def filter_rag_threshold(self, top_ks, questions, epsilon=0.2):
        filtered_results = []
        for i, query in enumerate(questions):
            retrieved_items = top_ks[i]
            clean_context_items = []
            
            for doc in retrieved_items:
                aj = self._generate_slm_response(query, doc)
                freq_density = self._compute_freq_density(query, aj, doc)
                
                # If density is LOW (< epsilon), it is CLEAN. 
                # (Adversarial texts have high density of query words)
                if freq_density < epsilon:
                    clean_context_items.append(doc)
            
            # Fallback: if all filtered, keep the one with lowest density or just the top 1
            if not clean_context_items and retrieved_items:
                clean_context_items = [retrieved_items[0]]
                
            filtered_results.append(clean_context_items)
        return filtered_results