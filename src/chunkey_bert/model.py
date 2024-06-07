import warnings
from typing import Tuple, List, Optional, Union, Callable

import numpy as np
from keybert import KeyBERT
from keybert.backend import BaseEmbedder
from sklearn.feature_extraction.text import CountVectorizer


def _calculate_top_similar_keywords_for_doc(
    embeddings_doc: np.ndarray,
    counts_doc: Optional[np.ndarray],
    top_k: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given embeddings to all keywords extracted from all chunks comprising a document, this method determines the top k
    most similar keywords across all chunks. The method assumes the embeddings are normalized and uses dot product
    similarity (cosine similarity). The score is then normalized to the range [0,1].

    Args:
        embeddings_doc: embeddings of all keywords extracted from a single document.
        counts_doc: If provided, the multiplicity of keywords extracted from the document to use in calculating
            similarity weighting.
        top_k: the number of top keywords to return. If unspecified, returns all keywords sorted by decreasing score.

    Returns:
        The top k most similar keywords across all chunks and their score in [0,1].
    """
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be greater than or equal to 0, or None.")

    sim: np.ndarray = embeddings_doc @ embeddings_doc.T
    np.fill_diagonal(a=sim, val=np.nan)
    sim = np.nanmean(sim, axis=1)
    sim = np.clip(sim, a_min=-1, a_max=1) + 1.0

    if counts_doc is not None:
        weights: np.ndarray = np.log2(counts_doc + 1)
        weights /= np.max(weights)
        sim *= weights

    sim /= 2
    top_idx: np.ndarray = np.argsort(sim)

    if top_k is None:
        top_idx = np.flip(top_idx)
    else:
        top_idx = top_idx[-1 : -top_k - 1 : -1]

    return top_idx, sim[top_idx]


def _get_unique_keywords_by_doc_idx(
    all_keywords_chunks: List[List[Tuple[str, float]]],
    docs_idx_list: List[int],
    doc_idx: int,
    use_count_weights: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract all unique keywords (case sensitive) for a single doc from the list of keywords for all chunks.

    Args:
        all_keywords_chunks: Keywords for all chunks of documents.
        docs_idx_list: Mapping chunks keywords indices to documents.
        doc_idx: The document index to extract keywords for.
        use_count_weights: If True, the number of times a keyword is repeated across chunks in the same document is
            returned. If False, None is returned as counts.

    Returns:
        All unique keywords for a single doc from the list of keywords and optionally their counts.
    """
    if doc_idx < 0 or doc_idx >= len(docs_idx_list):
        raise ValueError("doc_idx out of range")

    chunk_i_keywords: List[List[Tuple[str, float]]] = all_keywords_chunks[
        docs_idx_list[doc_idx] : docs_idx_list[doc_idx + 1]
    ]

    keywords_doc_unique: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]

    keywords_doc_unique = np.unique(
        ar=[t[0] for lk in chunk_i_keywords for t in lk if (len(t[0]) > 0 and not t[0].isspace())],  # type: ignore[call-overload]
        return_counts=use_count_weights,
    )

    return (
        (keywords_doc_unique[0].astype(str), keywords_doc_unique[1])  # type: ignore[return-value]
        if use_count_weights
        else (keywords_doc_unique, None)
    )


def _extract_chunks_from_docs(
    docs: Union[str, List[str]],
    chunker: Optional[Callable[[str], List[str]]],
) -> Tuple[List[str], List[int]]:
    """
    Applies the chunker to the docs to create a flat list of contiguous doc chunks with a list of indices that index the
    beginning and end of each doc's chunks. The chunker is applied to each document independently. These chunks
    represent the document for subsequent processing and each is later provided to KeyBERT to extract keywords. There is
    no need for the chunker to return all the text in the document, and it can apply filtering and sampling to reduce
    downstream processing complexity. If a chunker is not provided then the chunks returned are the input documents.

    Args:
        docs: The documents to chunk.
        chunker: A callable that takes a string and returns a list of strings. This is applied to each document
            independently

    Returns:
        A flat list of contiguous doc chunks with a list of indices that index the beginning and end of each doc's
        chunks.
    """
    if len(docs) == 0:
        return [], []

    if isinstance(docs, str):
        docs = [docs]

    chunks: List[str]
    idx: List[int]

    if chunker is not None:
        chunks = []
        idx = [0] * (len(docs) + 1)
        i: int
        doc: str

        for i, doc in enumerate(docs):
            chunks_doc: List[str] = chunker(doc)
            chunks.extend(chunks_doc)
            idx[i + 1] = len(chunks)
    else:
        chunks = docs
        idx = list(range(len(docs) + 1))

    return chunks, idx


class ChunkeyBert:
    def __init__(
        self,
        keybert: KeyBERT,
    ) -> None:
        self._keybert: KeyBERT = keybert
        self._embedder: BaseEmbedder = keybert.model
        try:
            self._embedding_dim: int = self._embedder.embed(documents=["Determining model embedding dim"]).shape[-1]
        except Exception as exc:
            raise ValueError(f"The provided embedder model is not working as expected. Original exception {exc}")

    def extract_keywords(
        self,
        docs: Union[str, List[str]],
        num_keywords: int,
        chunker: Optional[Callable[[str], List[str]]] = None,
        return_keywords_embeddings: bool = False,
        use_count_weights: bool = True,
        candidates: Optional[List[str]] = None,
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Union[str, List[str]] = "english",
        top_n: int = 3,
        min_df: int = 1,
        use_maxsum: bool = False,
        use_mmr: bool = False,
        diversity: float = 0.5,
        nr_candidates: int = 20,
        vectorizer: Optional[CountVectorizer] = None,
        highlight: bool = False,
        seed_keywords: Optional[Union[List[str], List[List[str]]]] = None,
        doc_embeddings: Optional[np.ndarray] = None,
        word_embeddings: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> List[Optional[Union[List[Tuple[str, np.float32]], List[Tuple[str, np.float32, np.ndarray]]]]]:
        """
        Extract the unique keywords/keyphrases for the provided documents. The method uses the chunker if provided to
        chunk the document and then use the KeyBERT model to extract keywords from each chunk. Finally, it merges all
        the results and finds the most similar keywords across all chunks. If a chunker is not provided the documents
        are provided as inputs to KeyBERT in their entirety and the similarity is calculated for the keywords extracted
        of each complete document.

        Args:
            docs: The documents to extract keywords for.
            num_keywords: The maximum number of keywords to extract.
            chunker: Chunks the documents. The chunker can be any callbale that takes a string and returns a list of
                strings. There are no constraints on the chunks, their length or order, e.g. chunks may be disjoint or
                overlap and can be filtered or even sampled from the document.
            return_keywords_embeddings: True to include the keywords embeddings in the returned list.
            use_count_weights: If True, the number of times a keyword is repeated across chunks in the same document is
                considered in scoring. If False it has no impact. Seems to work best when a small KeyBERT top_n value is
                specified.
            candidates: A KeyBert.extract_keywords parameter. Candidate keywords/keyphrases to use instead of extracting
                them from the document(s). NOTE: This is not used if you passed a `vectorizer`.
            keyphrase_ngram_range: A KeyBert.extract_keywords parameter. Length, in words, of the extracted
                keywords/keyphrases. NOTE: This is not used if you passed a `vectorizer`.
            stop_words: A KeyBert.extract_keywords parameter. Stopwords to remove from the document. NOTE: This is not
                used if you passed a `vectorizer`.
            top_n: A KeyBert.extract_keywords parameter. Return the top n keywords/keyphrases
            min_df: A KeyBert.extract_keywords parameter. Minimum document frequency of a word across all documents if
                keywords for multiple documents need to be extracted. NOTE: This is not used if you passed a
                `vectorizer`.
            use_maxsum: A KeyBert.extract_keywords parameter. Whether to use Max Sum Distance for the selection of
                keywords/keyphrases.
            use_mmr: A KeyBert.extract_keywords parameter. Whether to use Maximal Marginal Relevance (MMR) for the
                selection of keywords/keyphrases.
            diversity: A KeyBert.extract_keywords parameter. The diversity of the results between 0 and 1 if `use_mmr`
                is set to True.
            nr_candidates: A KeyBert.extract_keywords parameter. The number of candidates to consider if `use_maxsum` is
                set to True.
            vectorizer: A KeyBert.extract_keywords parameter. Pass in your own `CountVectorizer` from
                `sklearn.feature_extraction.text.CountVectorizer`
            highlight: A KeyBert.extract_keywords parameter. Whether to print the document and highlight its
                keywords/keyphrases. NOTE: This does not work if multiple documents are passed.
            seed_keywords: A KeyBert.extract_keywords parameter. Seed keywords that may guide the extraction of keywords
                by steering the similarities towards the seeded keywords. NOTE: when multiple documents are passed,
               `seed_keywords`funtions in either of the two ways:
               - globally: when a flat list of str is passed, keywords are shared by all documents,
               - locally: when a nested list of str is passed, keywords differs among documents.
            doc_embeddings: A KeyBert.extract_keywords parameter. The embeddings of each document.
            word_embeddings: A KeyBert.extract_keywords parameter. The embeddings of each potential keyword/keyphrase
                across the vocabulary of the set of input documents. NOTE: The `word_embeddings` should be generated
                through `.extract_embeddings` as the order of these embeddings depend on the vectorizer that was used to
                generate its vocabulary.
            threshold: Used by KeyBERT but is undocumented. Seems to be give to community_detection in
                sentence_transformers.utils to determine clusters.

        Returns:
            The top keywords/keyphrases for each corresponding document and their score or None if no keywords are
            available for a document. Optionally the embeddings for each keyword/keyphrase are returned if specified.
        """
        if chunker is None and num_keywords > top_n:
            warnings.warn(
                message="Setting num_keywords higher than top_n without a chunker will result in at most top_n "
                "keywords/keyphrases returned."
            )

        if isinstance(docs, str):
            docs = [docs]

        chunks: List[str]
        docs_idx_list: List[int]

        chunks, docs_idx_list = _extract_chunks_from_docs(docs=docs, chunker=chunker)

        if len(chunks) == 0:
            return []

        if len(chunks) != docs_idx_list[-1]:
            raise RuntimeError("Indices mapping segments don't map to  documents, this is a likely issue.")

        all_keywords_chunks: List[List[Tuple[str, float]]] = self._keybert.extract_keywords(
            docs=chunks,
            candidates=candidates,
            keyphrase_ngram_range=keyphrase_ngram_range,
            stop_words=stop_words,
            top_n=top_n,
            min_df=min_df,
            use_maxsum=use_maxsum,
            use_mmr=use_mmr,
            diversity=diversity,
            nr_candidates=nr_candidates,
            vectorizer=vectorizer,
            highlight=highlight,
            seed_keywords=seed_keywords,
            doc_embeddings=doc_embeddings,
            word_embeddings=word_embeddings,
            threshold=threshold,
        )

        if len(all_keywords_chunks) > 0 and not isinstance(all_keywords_chunks[0], List):
            raise ValueError("Unexpected type returned by keybert.extract_keywords().")

        keywords: List[Optional[Union[List[Tuple[str, np.float32]], List[Tuple[str, np.float32, np.ndarray]]]]] = []
        doc_idx: int

        for doc_idx in range(len(docs_idx_list) - 1):
            keywords_doc: np.ndarray
            counts_doc: Optional[np.ndarray]

            keywords_doc, counts_doc = _get_unique_keywords_by_doc_idx(
                all_keywords_chunks=all_keywords_chunks,
                docs_idx_list=docs_idx_list,
                doc_idx=doc_idx,
                use_count_weights=use_count_weights,
            )

            if keywords_doc.shape[0] > 0:
                embeddings_doc: np.ndarray = self._embedder.embed(documents=keywords_doc)

                top_idx: np.ndarray
                score: np.ndarray

                top_idx, score = _calculate_top_similar_keywords_for_doc(
                    embeddings_doc=embeddings_doc,
                    counts_doc=counts_doc,
                    top_k=num_keywords,
                )

                if not return_keywords_embeddings:
                    keywords.append(list(zip(keywords_doc[top_idx], score)))
                else:
                    keywords.append(list(zip(keywords_doc[top_idx], score, embeddings_doc[top_idx])))

            else:
                keywords.append(None)

        return keywords
