# Data schema for the RAG chatbot

## Corpus file

Path: `data/processed/corpus.parquet`

Columns:

- id: unique chunk id
- source: file name or url
- page: page number if pdf
- chunk_id: index of the chunk inside one file or page
- text: text content of the chunk

## Embeddings file

Path: `data/embeddings/embeddings.parquet`

Columns:

- id
- source
- chunk_id
- text
- embedding: list of floats (to be filled later)
