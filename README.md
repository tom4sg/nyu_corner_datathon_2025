# NYU x Corner Datathon 2025 Solution

This notebook implements a RAG recommendation system in Python to match open-ended user queries with NYC
venues using structured tags, reviews, and image data.


#### **Data**
- **Places**: Venue name, neighborhood, tags, description  
- **Reviews**: User-generated content  
- **Media**: Associated image URLs  

---

#### **Processing**
- Aggregates reviews and media per place  
- Merges structured and unstructured data into unified place representations

---

#### **Search System**
- **Dense Embeddings**: `SentenceTransformers` (`MiniLM`) for semantic text  
- **Sparse Embeddings**: `FastEmbed` (`SPLADE`) for keyword relevance  
- **Image Embeddings**: `CLIP` for visual features  
- **Similarity**: FAISS for dense & image search, cosine for sparse  
- **Scoring**: MinMax normalized + configurable hybrid weighting

## Key Functions

### Core Search Functions
- `search_places_dense_metadata()`: Dense text similarity search
- `search_places_sparse_metadata()`: Sparse text similarity search  
- `hybrid_search()`: Combined search with configurable weights

### Query Processing
- `process_user_query()`: Comprehensive query analysis
- `semantic_category_detection()`: Activity type classification
- `expand_query_with_llm()`: Query enhancement

## Usage Examples

```python
# Basic dense search
results = search_places_dense_metadata("where to drink a matcha", top_k=5)

# Sparse search for specific terms
results = search_places_sparse_metadata("dance-y bars that have disco balls", top_k=5)

# Full hybrid search
results = hybrid_search(
    query="what should I do this weekend",
    metadata_index=index_dense_metadata,
    image_index=index_image,
    sparse_embeddings=sparse_embeddings_dense,
    metadata_model=metadata_model,
    sparse_model=sparse_model,
    processor=processor,
    clip_model=clip_model,
    weight_dense=0.4,
    weight_sparse=0.3,
    weight_image=0.3
)
```

## Dependencies

```python
sentence_transformers
fastembed
transformers
faiss-cpu
scikit-learn
pandas
numpy
tqdm
spacy
joblib
pathlib
```

## Output Files

The system generates several precomputed files:
- `merged.csv`: Complete dataset with all embeddings
- `metadata.index`: FAISS index for dense text embeddings
- `image.index`: FAISS index for image embeddings
- `merge_dense_and_sparse_df.joblib`: Pickled dataframe with all embeddings

## Performance

- **Dataset Size**: 1,500+ places, 30,000+ media URLs
- **Embedding Dimensions**: 
  - Dense: 384 (MiniLM)
  - Sparse: 30,315 (Splade)
  - Image: 512 (CLIP)

---
https://github.com/user-attachments/assets/d139b7e4-dce4-49a3-95ab-e7e8c6897689

