import os
import torch
from vllm import LLM, SamplingParams

class PPLModel:
    def __init__(self, model_path, tensor_parallel_size=1, use_gpu_ids=[0]):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(use_gpu_ids)
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size
        )
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            prompt_logprobs=0
        )
    
    def calc_ppl(self, texts):
        outputs = self.model.generate(
            texts,
            sampling_params=self.sampling_params
        )
        
        ppls = []
        for output in outputs:
            prompt_logprobs = output.prompt_logprobs
            log_prob_sum = 0.0
            
            # Sum log probabilities for all tokens
            for i, token_logprobs in enumerate(prompt_logprobs):
                token_log_prob = token_logprobs.get(output.prompt_token_ids[i], 0.0)
                log_prob_sum += token_log_prob

            prompt_prob = torch.exp(-log_prob_sum).item()
            ppl = torch.pow(prompt_prob, -1 / len(output.prompt_token_ids))
            ppls.append(ppl)
        
        return ppls