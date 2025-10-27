import os
import math
from vllm import LLM, SamplingParams
from datatrove.utils.logging import logger

class PPLModel:
    def __init__(self, model_path, tensor_parallel_size=1, use_gpu_ids=[0]):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(map(str, use_gpu_ids)))
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.85,
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
            logprob_sum = 0.0
            n = len(prompt_logprobs)
            if n <= 1:
                ppls.append(0)
                continue
            # Sum log probabilities for all tokens
            for i in range(1, n):  # skip the first token
                token_logprobs = prompt_logprobs[i]
                token_logprob = token_logprobs.get(output.prompt_token_ids[i], 0.0)
                logprob_sum += token_logprob.logprob
            avg_logprob = logprob_sum / n
            ppl = math.exp(-avg_logprob)
            ppls.append(ppl)
        return ppls