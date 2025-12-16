import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.utils.logging import logger


class FinewebEduScoreCalculator(PipelineStep):
    def __init__(
        self,
        output_folder: DataFolderLike,
        model_path: str = "HuggingFaceFW/fineweb-edu-classifier",
    ):
        super().__init__()
        self.model_path = model_path
        self.output_folder = get_datafolder(output_folder)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        logger.info(f"loading model from {self.model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        with self.track_time():
            scores = []
            for i, doc in enumerate(data, 1):
                logger.info(f"processing doc {i}")
                text = doc.text
                inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
                outputs = model(**inputs)
                logits = outputs.logits.squeeze(-1).float().detach().numpy()
                score = logits.item()
                scores.append(score)
            with self.output_folder.open(f"{rank:05d}.json", mode="w") as f:
                json.dump(scores, f)
