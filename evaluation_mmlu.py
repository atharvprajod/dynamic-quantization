"""
Evaluation script for MMLU (using the hendrycks_test dataset) that benchmarks the performance
of the dynamically quantized model. This script demonstrates inference on multiple-choice questions,
computes accuracy using Hugging Face's evaluate library, and also measures token throughput and GPU VRAM usage.
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate

from model_wrapper import apply_dynamic_quant
from dynamic_quant import DynamicQuantConfig
from eval import measure_tokens_per_second, measure_vram_usage

def get_choice_logprob(model, tokenizer, question, choice, device="cuda"):
    """
    Given a question and one candidate answer, constructs a prompt and computes the total log probability
    of the answer using the model. This function assumes a simple prompt structure.
    """
    # Construct a prompt that concatenates the question and the candidate answer.
    text = f"Question: {question}\nAnswer: {choice}"
    tokens = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**tokens)
    
    # We compute log probabilities for each token.
    # Here we simply sum the log likelihoods over the tokens.
    logits = outputs.logits[0, :-1]  # shape: (seq_len-1, vocab_size)
    label_ids = tokens["input_ids"][0, 1:]  # shift by one; assuming teacher forcing style.
    logprobs = torch.log_softmax(logits, dim=-1)
    chosen_logprobs = logprobs[range(logits.size(0)), label_ids]
    return chosen_logprobs.sum().item()

def evaluate_mmlu(model, tokenizer, dataset, device="cuda"):
    """
    Evaluates the model on an MMLU subset.
    The MMLU dataset structure has these fields:
      - "question": The question text
      - "choices": List of possible answers
      - "answer": The correct answer (integer index)
    """
    metric = evaluate.load("accuracy")

    for example in dataset:
        question = example["question"]
        choices = example["choices"]
        correct_answer_idx = example["answer"]
        
        # Compute log probability scores for each choice.
        scores = []
        for choice in choices:
            score = get_choice_logprob(model, tokenizer, question, choice, device=device)
            scores.append(score)
            print(scores)
        
        # Select the answer with the highest log-probability.
        pred_idx = scores.index(max(scores))
        metric.add_batch(predictions=[pred_idx], references=[correct_answer_idx])
        
    return metric.compute()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # (1) Set up the dynamic quantization configuration.
    config = DynamicQuantConfig(bits=4, threshold=0.35, adapt_rank=1, num_power_iters=2, extra_scaling=0.5)
    
    # (2) Load the meta-llama/Llama-3.1-8B model in fp16.
    model_name = "meta-llama/Llama-3.1-8B"  # Ensure you have access to this model.
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    
    # (3) Wrap the model's nn.Linear layers with OutlierAwareLinear.
    model = apply_dynamic_quant(model, config)
    
    # (4) Ensure the full model is in fp16.
    model = model.half()
    model.to(device)
    model.eval()
    
    # (5) Load the corresponding tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # (6) Try loading the dataset with the correct path and error handling
    try:
        dataset = load_dataset("cais/mmlu", "college_physics", split="test")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # (7) Evaluate accuracy on the MMLU subset.
    mmlu_result = evaluate_mmlu(model, tokenizer, dataset, device=device)
    print("MMLU Accuracy:", mmlu_result)
    
    # (8) Also measure token throughput for a couple of prompts.
    prompts = [
        "In a distant future, humanity has reached the stars,",
        "The mysteries of the universe start to unfold as"
    ]
    tps = measure_tokens_per_second(model, tokenizer, prompts, device=device)
    print(f"Tokens per second: {tps:.2f}")
    
    # (9) Report current GPU VRAM usage.
    vram_usage = measure_vram_usage()
    print(f"Current VRAM usage: {vram_usage:.2f} MB")

if __name__ == '__main__':
    main() 
