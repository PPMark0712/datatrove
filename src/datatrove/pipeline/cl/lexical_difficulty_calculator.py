from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.cl.sam import SAM
from datatrove.utils.logging import logger

class LexicalDifficultyLabeler(PipelineStep):
    name = "🔤 - Lexical difficulty labeler"
    type = "cl"
    
    def __init__(
        self,
        dict_files: dict[str, str],
    ):
        super().__init__()
        self.dict_files = dict_files

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        with self.track_time():
            logger.info("Loading dictionaries...")
            level_sam_dict = {}
            sam_for_all_words = SAM()
            for level, dict_file in self.dict_files.items():
                with open(dict_file, "r") as f:
                    words = f.read().splitlines()
                    for word in words:
                        sam_for_all_words.add_string(word)
                    # if level != "stop_words":
                    level_sam = SAM()
                    for word in words:
                        level_sam.add_string(word)
                    level_sam_dict[level] = level_sam

        with self.track_time():
            for doc in data:
                level_word_counter = {}
                def is_zh_char(c):
                    return "\u4e00" <= c <= "\u9fff"
                
                zh_chars = "".join([c if is_zh_char(c) else " " for c in doc.text])
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
                    last_matched_idx = -1
                    p = 0
                    cnt = 0
                    for i, c in enumerate(s):
                        p = sam.trans_keep_suffix(p, c)
                        if sam.exist[p]:
                            if i - last_matched_idx > sam.len[p]:
                                cnt += i - sam.len[p] - last_matched_idx
                            last_matched_idx = i
                        # if not sam.exist[p] and sam.len[p] <= 1 and (sam.len[pre] <= 1 or not sam.exist[pre]):
                        #     cnt += 1
                            # logger.info(f"{''.join(s[max(0, i - 2): i + 1])}, p={p}, len={sam.len[p]}")
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
