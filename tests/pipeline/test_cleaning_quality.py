"""Check language-specific cleaning rules against real word tokenizers."""

import pytest

from datatrove.data import Document
from datatrove.pipeline.filters import GopherQualityFilter
from datatrove.utils.text import split_into_words
from scripts.data_clean.data_cleaning import gopher_quality_options


CHINESE = (
    "城市里的公共图书馆在周末迎来了许多读者。工作人员为孩子们准备了有趣的科学读物，"
    "也向家长介绍了阅读活动的安排。研究人员希望通过观察和访谈了解居民的需求，"
    "并对不同年龄群体的借阅记录进行分析。这个项目的目标是改善公共服务，"
    "让更多的人能够方便地获取知识。学校与社区还会共同组织讨论，"
    "帮助学生理解自然环境和日常生活之间的联系。"
)
ENGLISH = (
    "The public library is a place for people to discover books and share ideas. "
    "Researchers have worked with local schools to understand the needs of young readers. "
    "They collected observations from several communities and studied how families use "
    "these services. Their findings will help the library improve its programs and "
    "provide useful resources for students."
)


def test_chinese_segmentation():
    words = split_into_words(CHINESE, "zh")
    assert "图书馆" in words
    assert "的" in words
    assert len(words) > 50


@pytest.mark.parametrize("language,text", [("zh", CHINESE), ("en", ENGLISH)])
def test_quality_checks_accept_prose_and_reject_bad_words(language, text):
    quality = GopherQualityFilter(language=language, **gopher_quality_options(language))
    assert quality.filter(Document(text=text, id="good")) is True
    # Disable the minimum length only to isolate the other word rules.
    quality.min_doc_words = None
    assert quality.filter(Document(text="abcdefghijklmnop " * 60, id="long")) == (
        False, "gopher_above_avg_threshold"
    )
    assert quality.filter(Document(text="1234 " * 60, id="numbers")) == (
        False, "gopher_below_alpha_threshold"
    )
    assert quality.filter(Document(text="apple banana " * 30, id="no-stop")) == (
        False, "gopher_enough_stop_words"
    )
