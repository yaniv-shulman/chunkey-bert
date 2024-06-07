![Tests](https://github.com/yaniv-shulman/chunkey-bert/actions/workflows/linting_and_tests.yml/badge.svg?branch=main)
[![phorm.ai](https://img.shields.io/badge/ask%20phorm.ai-8A2BE2)](https://www.phorm.ai/query?projectId=f7ddaf97-2b90-4515-a364-855258454655)
[![Pyversions](https://img.shields.io/pypi/pyversions/chunkey-bert.svg?style=flat-square)](https://pypi.python.org/pypi/chunkey-bert)

Add# ChunkeyBERT - Unsupervised Keyword Extraction from Long Documents #
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
Install from [PyPI](https://pypi.org/project/chunkey-bert/) using pip (preferred method):
```bash
pip install chunkey-bert
```

## Details ##
### How does ChunkeyBERT differs to KeyBERT? ###
ChunkeyBERT differs from KeyBERT primarily in its approach to handling long documents for keyword extraction. While 
KeyBERT directly applies keyword extraction techniques to the entire document, ChunkeyBERT introduces an additional step
of chunking the document into smaller, manageable pieces before applying KeyBERT's keyword extraction methods. This 
modification aims to improve the performance and relevance of the extracted keywords, especially for longer documents 
where directly applying KeyBERT might not yield optimal results due to the complexity and size of the document. Here are
the key differences:

**Document chunking**: ChunkeyBERT uses a chunking method to divide a long document into smaller chunks. This is done through the chunker 
parameter in the extract_keywords method. The chunker can be any callable that takes a string (the document) and returns
a list of strings (the chunks). This allows ChunkeyBERT to process each chunk independently, making it more effective at
handling long documents. A chunker could be as simple as ```python chunker: Callable[[str], List[str]] = lambda text: [t for t in text.split("\n\n") if len(t) > 25] ```
or can wrap more complicated logic such as a Langchain chunker for example.

**Handling of chunks**: After chunking, ChunkeyBERT applies KeyBERT's keyword extraction to each chunk separately.

**Keyword scoring and selection**: ChunkeyBERT introduces additional logic to score and select keywords based on their occurrence across different chunks 
and their similarity. 

### Flexibility in keyword extraction ###
ChunkeyBERT offers flexibility in keyword extraction in a number of ways. It can work with any configuration of KeyBERT
and exposes a superset of KeyBERT's extract_keywords() API, which allow fine-tuning of the keyword extraction process 
based on the characteristics of the chunks and the overall document. It can also work with any chunking method including
semantic chunking, chunk filtering and even sampling from the document to finetune the process. ChunkeyBERT can be 
configured to consider the multiplicity of keywords across chunks to account for repetitions.

### Batching and GPU support ###
ChunkeyBERT works with document batches and attempts to process these batches in parallel on the GPU if possible.

### Compatible with KeyBERT return values ###
ChunkeyBERT returns results in a format similar to KeyBERT but can also optionally return the embeddings for each of the
keywords extracted.

## Usage ##

The following steps describe a basic example on how use ChunkeyBert for keyword extraction:

**Install ChunkeyBert**: First, ensure that ChunkeyBert is installed in your environment. You can install it using pip as
shown below:

```bash
pip install chunkey-bert
```
**Import required libraries**: Import the necessary libraries including ChunkeyBert, KeyBERT, and any other dependencies you
might need for your specific use case.

```python
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from chunkey_bert.model import ChunkeyBert
```

**Initialize KeyBERT**: this could be done for example using a Sentence Transformer model that is used to generate embeddings
for the text. _Note that the quality of extracted keywords depends greatly on how KeyBERT is configured_, so it is 
required to understand how to use KeyBERT effectively.

```python
sentence_model = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2")
keybert = KeyBERT(model=sentence_model)
```

**Define a chunker function (optional)**: If you want to chunk your text into smaller parts (which is the main feature of
ChunkeyBert), define a chunker function. This function takes a string and returns a list of strings (chunks). If you
don't provide a chunker, ChunkeyBert will process the entire document as a single chunk but will still apply a different
keywords selection method to KeyBERT. Here is an example of a very simple chunker:

```python
chunker = lambda text: [t for t in text.split("\n\n") if len(t) > 25]  # Example chunker that splits text into paragraphs
```

**Create a ChunkeyBert instance**: Initialize ChunkeyBert with the KeyBERT instance you created earlier.

```python
chunkey_bert = ChunkeyBert(keybert=keybert)
```
**Extract keywords**: Use the extract_keywords method of ChunkeyBert to extract keywords from your document. You can specify
the number of keywords, whether to use the chunker, and other parameters related to keyword extraction and to 
KeyBERT.extract_keywords.

```python
text = "Your long document text goes here..."
keywords = chunkey_bert.extract_keywords(
    docs=text, 
    num_keywords=10, 
    chunker=chunker,  # Pass your chunker here. If None, the entire document is treated as a single chunk.
    top_n=3,  # Number of keywords to extract from each chunk
    nr_candidates=20,  # Number of candidate keywords/keyphrases to consider from each chunk
)
print(keywords)
```
See a more advanced example in this notebook: https://nbviewer.org/github/yaniv-shulman/chunkey-bert/tree/main/src/experiments/

## Experimental results ##
Very limited experimental results and demonstration of the library on a small number of documents is available at 
 https://nbviewer.org/github/yaniv-shulman/chunkey-bert/tree/main/src/experiments/.

## Contribution and feedback ##
Contributions and feedback are most welcome. Please see
[CONTRIBUTING.md](https://github.com/yaniv-shulman/chunkey-bert/tree/main/CONTRIBUTING.md) for further details.
