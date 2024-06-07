import copy
from typing import List, Tuple, Optional, Callable, Union

import numpy as np
import pytest
from keybert import KeyBERT
from pytest_mock import MockerFixture, MockType

from chunkey_bert.model import (
    _calculate_top_similar_keywords_for_doc,
    ChunkeyBert,
    _extract_chunks_from_docs,
    _get_unique_keywords_by_doc_idx,
)


def get_test_docs(num_docs: int = 3) -> List[str]:
    """
    Creates a list of strings to use as documents in tests.

    Args:
        num_docs: Limits the number of documents returned if smaller than the number of documents available.

    Returns:
        A list of strings to use as documents in tests.
    """
    docs: List[str] = [
        "In Python, Assignment statements do not copy objects, they create bindings between a target and an object. "
        "When we use the = operator, It only creates a new variable that shares the reference of the original object. "
        "In order to create “real copies” or “clones” of these objects, we can use the copy module in Python.",
        "When writing tests it can be easy to write repetitive code. For example, you might find yourself writing the"
        "same snippet of code multiple times to create and use the same object across different tests. Or, writing "
        "multiple tests for the same function in order to test different inputs.",
        "While you can annotate variadic arguments with a type, I don't find it very useful because it assumes that all"
        "arguments are of the same type.",
    ]

    return docs[:num_docs]


def get_test_chunks() -> Tuple[List[str], List[List[Tuple[str, float]]], List[int]]:
    """
    Returns:
        A list of strings to use as document chunks in tests.
    """
    chunks: List[str] = [
        "chunk 1a text",
        "chunk 1b text",
        "chunk 1c text",
        "chunk 1d text",
        "chunk 2a text",
        "chunk 2b text",
        "chunk 2c text",
        "chunk 2d text",
        "chunk 2e text",
        "chunk 2f text",
        "chunk 3a text",
        "chunk 3b text",
        "chunk 4a text",
    ]

    keywords_chunks: List[List[Tuple[str, float]]] = [
        [(f"{c.split(' ')[1]}k{k}", 0.5) for k in range(3)] for c in chunks
    ]

    docs_idx_list: List[int] = [0, 4, 10, 12, 13]
    return copy.deepcopy(chunks), copy.deepcopy(keywords_chunks), copy.deepcopy(docs_idx_list)


def test_chunkey_bert_init_embedding_dim() -> None:
    """Tests the embedding model is initialized properly and the embedding dim is correctly set."""
    target: ChunkeyBert = ChunkeyBert(keybert=KeyBERT(model="all-MiniLM-L6-v2"))
    assert target._embedding_dim == 384


def test_chunkey_bert_extract_keywords_keybert_called_with_correct_parameters(mocker: MockerFixture) -> None:
    """Tests the correct paramaetr values are forwarded to the KeyBERT extract_keywords function."""
    keybert: KeyBERT = KeyBERT(model="all-MiniLM-L6-v2")
    target: ChunkeyBert = ChunkeyBert(keybert=keybert)
    spy: MockType = mocker.spy(obj=keybert, name="extract_keywords")
    docs: List[str] = get_test_docs()
    target.extract_keywords(docs=docs, num_keywords=2)

    spy.assert_called_once_with(
        docs=docs,
        candidates=None,
        keyphrase_ngram_range=(1, 1),
        stop_words="english",
        top_n=3,
        min_df=1,
        use_maxsum=False,
        use_mmr=False,
        diversity=0.5,
        nr_candidates=20,
        vectorizer=None,
        highlight=False,
        seed_keywords=None,
        doc_embeddings=None,
        word_embeddings=None,
        threshold=None,
    )

    keybert = KeyBERT(model="all-MiniLM-L6-v2")
    target = ChunkeyBert(keybert=keybert)
    spy = mocker.spy(obj=keybert, name="extract_keywords")
    docs = get_test_docs()
    target.extract_keywords(
        docs=docs,
        num_keywords=2,
        candidates=["candidate 1", "candidate 2"],
        keyphrase_ngram_range=(2, 2),
        stop_words="french",
        top_n=4,
        min_df=2,
        use_maxsum=True,
        use_mmr=True,
        diversity=0.7,
        nr_candidates=10,
        vectorizer=None,
        highlight=True,
        seed_keywords=["bla1", "bla2"],
        doc_embeddings=None,
        word_embeddings=None,
        threshold=0.7,
    )

    spy.assert_called_once_with(
        docs=docs,
        candidates=["candidate 1", "candidate 2"],
        keyphrase_ngram_range=(2, 2),
        stop_words="french",
        top_n=4,
        min_df=2,
        use_maxsum=True,
        use_mmr=True,
        diversity=0.7,
        nr_candidates=10,
        vectorizer=None,
        highlight=True,
        seed_keywords=["bla1", "bla2"],
        doc_embeddings=None,
        word_embeddings=None,
        threshold=0.7,
    )


def test_chunkey_bert_extract_keywords_returns_none_when_no_keywords_are_returned_from_keybert(
    mocker: MockerFixture,
) -> None:
    """Tests None is returned for documents when KeyBERT has no keywords found."""
    keybert: KeyBERT = KeyBERT(model="all-MiniLM-L6-v2")
    mocker.patch.object(target=keybert, attribute="extract_keywords", new=lambda **kwargs: [])
    target: ChunkeyBert = ChunkeyBert(keybert=keybert)
    docs: List[str] = get_test_docs()
    num_keywords: int = 4
    actual: List[Optional[Union[List[Tuple[str, np.float32]], List[Tuple[str, np.float32, np.ndarray]]]]] = (
        target.extract_keywords(docs=docs, num_keywords=num_keywords)
    )
    assert len(actual) == len(docs)
    assert actual == [None] * 3


def test_chunkey_bert_extract_keywords_expected_outputs() -> None:
    """Tests the expected output of extract_keywords."""
    keybert: KeyBERT = KeyBERT(model="all-MiniLM-L6-v2")
    target: ChunkeyBert = ChunkeyBert(keybert=keybert)
    docs: List[str] = get_test_docs()
    num_keywords: int = 4
    actual: List[Optional[Union[List[Tuple[str, np.float32]], List[Tuple[str, np.float32, np.ndarray]]]]] = (
        target.extract_keywords(docs=docs, num_keywords=num_keywords, top_n=num_keywords)
    )
    assert len(actual) == len(docs)
    doc_keywords_results: Optional[Union[List[Tuple[str, np.float32]], List[Tuple[str, np.float32, np.ndarray]]]]

    for doc_keywords_results in actual:
        assert doc_keywords_results is not None
        assert len(doc_keywords_results) == num_keywords
        for item in doc_keywords_results:
            assert isinstance(item[0], str)
            assert isinstance(item[1], np.float32)

    num_keywords = 2

    actual = target.extract_keywords(docs=docs, num_keywords=num_keywords, return_keywords_embeddings=True)

    assert len(actual) == len(docs)

    for doc_keywords_results in actual:
        assert doc_keywords_results is not None
        assert len(doc_keywords_results) == num_keywords
        for item in doc_keywords_results:
            assert isinstance(item[0], str)
            assert isinstance(item[1], np.float32)
            assert isinstance(item[2], np.ndarray)  # type: ignore [misc]


@pytest.mark.parametrize(
    "docs, chunker, expected_chunks, expected_idx",
    [
        # no chunker empty docs.
        ([], None, [], []),
        # no chunker with docs.
        (get_test_docs(num_docs=1), None, get_test_docs(num_docs=1), [0, 1]),
        (get_test_docs(num_docs=3), None, get_test_docs(num_docs=3), [0, 1, 2, 3]),
        # with a chunker that splits on commas
        (
            get_test_docs(num_docs=1),
            lambda d: [s.strip() for s in d.split(",")],
            [
                "In Python",
                "Assignment statements do not copy objects",
                "they create bindings between a target and an object. When we use the = operator",
                "It only creates a new variable that shares the reference of the original object. In order to create "
                "“real copies” or “clones” of these objects",
                "we can use the copy module in Python.",
            ],
            [0, 5],
        ),
        (
            get_test_docs(num_docs=3),
            lambda d: [s.strip() for s in d.split(",")],
            [
                "In Python",
                "Assignment statements do not copy objects",
                "they create bindings between a target and an object. When we use the = operator",
                "It only creates a new variable that shares the reference of the original object. In order to create "
                "“real copies” or “clones” of these objects",
                "we can use the copy module in Python.",
                "When writing tests it can be easy to write repetitive code. For example",
                "you might find yourself writing thesame snippet of code multiple times to create and use the "
                "same object across different tests. Or",
                "writing multiple tests for the same function in order to test different inputs.",
                "While you can annotate variadic arguments with a type",
                "I don't find it very useful because it assumes that allarguments are of the same type.",
            ],
            [0, 5, 8, 10],
        ),
        # with chunker and empty strings
        (
            [""] + get_test_docs(num_docs=1),
            lambda d: [s.strip() for s in d.split(",")],
            [
                "",
                "In Python",
                "Assignment statements do not copy objects",
                "they create bindings between a target and an object. When we use the = operator",
                "It only creates a new variable that shares the reference of the original object. In order to create "
                "“real copies” or “clones” of these objects",
                "we can use the copy module in Python.",
            ],
            [0, 1, 6],
        ),
        # with chunker that returns an empty list
        (
            [""] + get_test_docs(num_docs=1),
            lambda d: [s.strip() for s in d.split(",") if len(s) > 0],
            [
                "In Python",
                "Assignment statements do not copy objects",
                "they create bindings between a target and an object. When we use the = operator",
                "It only creates a new variable that shares the reference of the original object. In order to create "
                "“real copies” or “clones” of these objects",
                "we can use the copy module in Python.",
            ],
            [0, 0, 5],
        ),
    ],
)
def test_extract_chunks_from_docs_expected_outputs(
    docs: List[str], chunker: Callable[[str], List[str]], expected_chunks: List[str], expected_idx: List[int]
) -> None:
    """Tests _extract_chunks_from_docs returns correct outputs."""
    actual_chunks: List[str]
    actual_idx: List[int]

    actual_chunks, actual_idx = _extract_chunks_from_docs(docs=docs, chunker=chunker)
    assert actual_chunks == expected_chunks
    assert actual_idx == expected_idx


@pytest.mark.parametrize(
    "top_k, counts_doc, expected_idx, expected_score",
    [
        # top_k and weights.
        (None, None, [3, 2, 4, 0, 1], [0.66392495, 0.66392495, 0.57784989, 0.51504675, 0.23495325]),
        # top_k specified, equal weights.
        (3, [1] * 5, [3, 2, 4], [0.66392495, 0.66392495, 0.57784989]),
        # top_k specified and exceeds the number of elements, equal weights.
        (7, None, [3, 2, 4, 0, 1], [0.66392495, 0.66392495, 0.57784989, 0.51504675, 0.23495325]),
        # top_k is zero, equal weights.
        (0, [1] * 5, [], []),
        # top_k unspecified, unequal weights.
        (None, [4, 1, 1, 1, 1], [0, 2, 3, 4, 1], [0.51504675, 0.28593691, 0.28593691, 0.2488664, 0.10118886]),
    ],
)
def test_calculate_top_similar_keywords_for_doc_expected_output(
    top_k: Optional[int], counts_doc: Optional[List[int]], expected_idx: List[int], expected_score: List[float]
) -> None:
    """Tests _calculate_top_similar_keywords_for_doc returns correct outputs."""
    embeddings_doc: np.ndarray = np.asarray(
        [
            [0.1, 0.2, 0.1, 0.5, 0.2],
            [-0.1, -0.2, -0.1, -0.5, -0.2],
            [-0.2, 0.1, 0.1, 0.1, -0.2],
            [-0.2, 0.1, 0.1, 0.1, -0.2],
            [0.1, 0.2, 0.1, 0.3, 0.0],
        ],
        dtype=float,
    )

    embeddings_doc = embeddings_doc / np.linalg.norm(embeddings_doc, axis=1, keepdims=True)
    actual_idx: np.ndarray
    actual_score: np.ndarray

    actual_idx, actual_score = _calculate_top_similar_keywords_for_doc(
        embeddings_doc=embeddings_doc,
        counts_doc=np.asarray(counts_doc, dtype=int) if counts_doc is not None else None,
        top_k=top_k,
    )

    np.testing.assert_equal(actual_idx, expected_idx)
    np.testing.assert_allclose(actual_score, expected_score)


def test_calculate_top_similar_keywords_for_doc_exceptions_raised() -> None:
    """Tests _calculate_top_similar_keywords_for_doc raises expected exceptions."""
    embeddings_doc: np.ndarray = np.asarray(
        [
            [0.1, 0.2, 0.1, 0.5, 0.2],
            [-0.1, -0.2, -0.1, -0.5, -0.2],
            [-0.2, 0.1, 0.1, 0.1, -0.2],
            [-0.2, 0.1, 0.1, 0.1, -0.2],
            [0.1, 0.2, 0.1, 0.3, 0.0],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match="top_k must be greater than or equal to 0, or None."):
        _calculate_top_similar_keywords_for_doc(embeddings_doc=embeddings_doc, counts_doc=None, top_k=-1)


def test_get_unique_keywords_by_doc_idx_expected_values_no_duplicates() -> None:
    """
    Tests _get_unique_keywords_by_doc_idx_expected_values returns expected outputs when no duplicates are present.
    """
    keywords_chunks: List[List[Tuple[str, float]]]
    docs_idx_list: List[int]
    chunks, keywords_chunks, docs_idx_list = get_test_chunks()

    keywords_doc: np.ndarray
    counts_doc: Optional[np.ndarray]

    keywords_doc, counts_doc = _get_unique_keywords_by_doc_idx(
        all_keywords_chunks=keywords_chunks, docs_idx_list=docs_idx_list, doc_idx=0, use_count_weights=False
    )

    np.testing.assert_equal(keywords_doc, [t[0] for lk in keywords_chunks[0:4] for t in lk])
    assert counts_doc is None

    keywords_doc, counts_doc = _get_unique_keywords_by_doc_idx(
        all_keywords_chunks=keywords_chunks, docs_idx_list=docs_idx_list, doc_idx=0, use_count_weights=True
    )

    np.testing.assert_equal(keywords_doc, [t[0] for lk in keywords_chunks[0:4] for t in lk])
    np.testing.assert_equal(counts_doc, [1] * keywords_doc.shape[0])

    keywords_doc, counts_doc = _get_unique_keywords_by_doc_idx(
        all_keywords_chunks=keywords_chunks, docs_idx_list=docs_idx_list, doc_idx=1, use_count_weights=False
    )

    np.testing.assert_equal(keywords_doc, [t[0] for lk in keywords_chunks[4:10] for t in lk])
    assert counts_doc is None

    keywords_doc, counts_doc = _get_unique_keywords_by_doc_idx(
        all_keywords_chunks=keywords_chunks, docs_idx_list=docs_idx_list, doc_idx=3, use_count_weights=True
    )

    np.testing.assert_equal(keywords_doc, ["4ak0", "4ak1", "4ak2"])
    np.testing.assert_equal(counts_doc, [1] * keywords_doc.shape[0])


def test_get_unique_keywords_by_doc_idx_expected_values_with_duplicates() -> None:
    """
    Tests _get_unique_keywords_by_doc_idx_expected_values returns expected outputs when duplicates are present.
    """
    keywords_chunks: List[List[Tuple[str, float]]]
    docs_idx_list: List[int]
    chunks, keywords_chunks, docs_idx_list = get_test_chunks()

    keywords_chunks[0][1] = keywords_chunks[0][0]
    keywords_chunks[0][2] = keywords_chunks[0][0]
    keywords_chunks[1][2] = keywords_chunks[0][0]

    keywords_doc: np.ndarray
    counts_doc: Optional[np.ndarray]

    keywords_doc, counts_doc = _get_unique_keywords_by_doc_idx(
        all_keywords_chunks=keywords_chunks, docs_idx_list=docs_idx_list, doc_idx=0, use_count_weights=False
    )

    np.testing.assert_equal(keywords_doc, ["1ak0", "1bk0", "1bk1", "1ck0", "1ck1", "1ck2", "1dk0", "1dk1", "1dk2"])
    assert counts_doc is None

    keywords_chunks[12][1] = keywords_chunks[12][0]
    keywords_chunks[12][2] = keywords_chunks[12][0]

    keywords_doc, counts_doc = _get_unique_keywords_by_doc_idx(
        all_keywords_chunks=keywords_chunks, docs_idx_list=docs_idx_list, doc_idx=3, use_count_weights=True
    )

    np.testing.assert_equal(keywords_doc, ["4ak0"])
    np.testing.assert_equal(counts_doc, [3])

    keywords_chunks[0][1] = (keywords_chunks[0][0][0].upper(), 0.5)
    keywords_chunks[0][2] = (keywords_chunks[0][0][0].upper(), 0.5)
    keywords_chunks[1][2] = (keywords_chunks[0][0][0].upper(), 0.5)

    keywords_doc, counts_doc = _get_unique_keywords_by_doc_idx(
        all_keywords_chunks=keywords_chunks, docs_idx_list=docs_idx_list, doc_idx=0, use_count_weights=True
    )

    np.testing.assert_equal(
        keywords_doc, ["1AK0", "1ak0", "1bk0", "1bk1", "1ck0", "1ck1", "1ck2", "1dk0", "1dk1", "1dk2"]
    )

    np.testing.assert_equal(counts_doc, [3, 1, 1, 1, 1, 1, 1, 1, 1, 1])


def test_get_unique_keywords_by_doc_idx_expected_values_ignores_empty_and_ws_strings() -> None:
    """Tests _get_unique_keywords_by_doc_idx handles empty and ws strings correctly."""
    keywords_chunks: List[List[Tuple[str, float]]]
    docs_idx_list: List[int]
    chunks, keywords_chunks, docs_idx_list = get_test_chunks()

    keywords_chunks[0][1] = ("", 0.5)
    keywords_chunks[0][2] = ("  ", 0.5)
    keywords_chunks[1][2] = ("\n", 0.5)

    keywords_doc: np.ndarray
    counts_doc: Optional[np.ndarray]

    keywords_doc, counts_doc = _get_unique_keywords_by_doc_idx(
        all_keywords_chunks=keywords_chunks, docs_idx_list=docs_idx_list, doc_idx=0, use_count_weights=False
    )

    np.testing.assert_equal(keywords_doc, ["1ak0", "1bk0", "1bk1", "1ck0", "1ck1", "1ck2", "1dk0", "1dk1", "1dk2"])
