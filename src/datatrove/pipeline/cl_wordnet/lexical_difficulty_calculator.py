from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.cl_wordnet.sam import SAM
from datatrove.pipeline.cl_wordnet.word_extractor import WordExtractor
from datatrove.utils.logging import logger

class LexicalDifficultyLabeler(PipelineStep):
    name = "🔤 - Lexical difficulty labeler"
    type = "cl"
    
    def __init__(
        self,
        dict_files: dict[str, str],
        language: str="zh",
        use_sam=False,
    ):
        super().__init__()
        self.dict_files = dict_files
        self.language = language
        self.use_sam = use_sam

    def get_zh_chars(self, text: str) -> str:
        return "".join([c if "\u4e00" <= c <= "\u9fff" else " " for c in text])

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        if self.use_sam:
            with self.track_time():
                level_sam_dict = {}
                sam_for_all_words = SAM()
                for level, dict_file in self.dict_files.items():
                    with open(dict_file, "r") as f:
                        words = f.read().splitlines()
                        for word in words:
                            sam_for_all_words.add_string(word)
                        level_sam = SAM()
                        for word in words:
                            level_sam.add_string(word)
                        level_sam_dict[level] = level_sam

            with self.track_time():
                for doc in data:
                    level_word_counter = {}
                    zh_chars = self.get_zh_chars(doc.text)
                    zh_splits = zh_chars.split(" ")
                    
                    def match_exist(sam, s):
                        p = 0
                        cnt = 0
                        for i, c in enumerate(s):
                            p = sam.trans_keep_suffix(p, c)
                            if sam.exist[p]:
                                cnt += 1
                            # logger.info(f"{''.join(s[max(0, i - 2): i + 1])}, p={p}, len={sam.len[p]}")
                        return cnt

                    def match_not_exist(sam, s):
                        p = 0
                        matched_suf_len = []
                        for c in s:
                            p = sam.trans_keep_suffix(p, c)
                            q = p
                            while q > 0 and not sam.exist[q]:
                                q = sam.fa[q]
                            matched_suf_len.append(sam.len[q])
                        cnt = 0
                        for i in range(len(matched_suf_len) - 3):
                            if matched_suf_len[i] >= matched_suf_len[i + 1] >= matched_suf_len[i + 2] >= matched_suf_len[i + 3]:
                                logger.info(f"{s[i:i+4]}, {matched_suf_len[i:i+4]}")
                                cnt += 1
                        return cnt
                        # last_matched_idx = -1
                        # p = 0
                        # cnt = 0
                        # for i, c in enumerate(s):
                        #     p = sam.trans_keep_suffix(p, c)
                        #     if sam.exist[p]:
                        #         if i - last_matched_idx > sam.len[p]:
                        #             cnt += i - sam.len[p] - last_matched_idx
                        #             logger.info(f"{''.join(s[max(0, i - 5): i + 1])}, p={p}, len={sam.len[p]}, last={s[last_matched_idx - sam.len[last_matched_idx] + 1: last_matched_idx + 1] if last_matched_idx != -1 else ''}")
                        #         last_matched_idx = i
                        return cnt

                    for level, level_sam in level_sam_dict.items():
                        level_word_counter[level] = 0
                        for split in zh_splits:
                            level_word_counter[level] += match_exist(level_sam, split)
                    level_word_counter["other"] = 0
                    for split in zh_splits:
                        level_word_counter["other"] += match_not_exist(sam_for_all_words, split)
                    doc.metadata["level_word_counter"] = level_word_counter
                    yield doc
        else:
            level_word_set = {}
            for level, dict_file in self.dict_files.items():
                with open(dict_file, "r") as f:
                    words = f.read().splitlines()
                level_word_set[level] = set(words)
            word_extractor = WordExtractor(self.language)
            for doc in data:
                level_word_counter = {}
                zh_chars = self.get_zh_chars(doc.text)
                words = set(word_extractor.extract_words(zh_chars))
                for level, word_set in level_word_set.items():
                    level_word_counter[level] = len(word_set.intersection(words))
                    # if level != "primary":
                    #     logger.info(f"{level}: {word_set.intersection(words)}")
                level_word_counter["other"] = len(words) - sum(level_word_counter.values())
                doc.metadata["level_word_counter"] = level_word_counter
                yield doc


class LexicalDifficultySorter(PipelineStep):
    name = "🔤 - Lexical difficulty sorter"
    type = "cl"

    def __init__(
        self,
        weights: dict[str, float],
    ):
        super().__init__()
        self.weights = weights

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        with self.track_time():
            """Warning: All data from a file needs to be read in at once. 
            Please ensure each file size is moderate to avoid excessive memory usage."""
            all_data = []
            scores = []
            for doc in data:
                score = 0
                level_word_counter = doc.metadata["level_word_counter"]
                total_word_cnt = sum(level_word_counter.values())
                for level, level_word_cnt in level_word_counter.items():
                    score += level_word_cnt / total_word_cnt * self.weights[level] if total_word_cnt > 0 else 0
                scores.append(score)
                doc.metadata["score"] = score
                all_data.append(doc)
            idxs = list(range(len(all_data)))
            idxs = sorted(idxs, key=lambda x: scores[x])
            for idx in idxs:
                yield all_data[idx]