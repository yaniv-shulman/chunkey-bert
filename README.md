![Tests](https://github.com/yaniv-shulman/chunkey-bert/actions/workflows/linting_and_tests.yml/badge.svg?branch=main)
[![phorm.ai](https://img.shields.io/badge/ask%20phorm.ai-8A2BE2)](https://www.phorm.ai/query?projectId=f7ddaf97-2b90-4515-a364-855258454655)

# ChunkeyBERT #
## Overview ##
ChunkeyBert is a minimal and easy-to-use keyword extraction technique that leverages BERT embeddings for unsupervised 
keyphrase extraction from text documents. ChunkeyBert is a modification of the 
[KeyBERT method](https://towardsdatascience.com/keyword-extraction-with-bert-724efca412ea) to handle documents with 
arbitrary length with better results. ChunkeyBERT works by chunking the documents and uses KeyBERT to extract candidate
keywords/keyphrases from all chunks followed by a similarity based selection stage to produce the final keywords for the
entire document. ChunkeyBert can use any document chunking method as long as it can be wrapped in a simple function, 
however it can also work without a chunker and process the entire document as a single chunk. ChunkeyBert works with any
configuration of KeyBERT and can handle batches of documents. 

## Installation ##
Install from [PyPI](https://pypi.org/project/rsklpr/) using pip (preferred method):
```bash
pip install chunkey-bert
```

## Experimental results ##
Very limited experimental results and demonstration of the library on a small number of documents is available at 
 https://nbviewer.org/github/yaniv-shulman/chunkey-bert/tree/main/src/experiments/.


## Contribution and feedback ##
Contributions and feedback are most welcome. Please see
[CONTRIBUTING.md](https://github.com/yaniv-shulman/chunkey-bert/tree/main/CONTRIBUTING.md) for further details.
