# NYU x Corner Datathon 2025 Solution

A comprehensive Retrieval-Augmented Generation (RAG) solution for place/venue search using hybrid embeddings and vector similarity search.

## Overview

This notebook implements a sophisticated search system that combines multiple types of embeddings to provide highly relevant search results for places and venues. The solution uses a hybrid approach combining dense text embeddings, sparse text embeddings, and multimodal image embeddings.

## Features

### 🔍 **Hybrid Search System**
- **Dense Text Embeddings**: Using SentenceTransformers (all-MiniLM-L6-v2) for semantic understanding
- **Sparse Text Embeddings**: Using FastEmbed (Splade_PP_en_v1) for explicit keyword matching
- **Multimodal Image Embeddings**: Using CLIP for visual-semantic understanding
- **Weighted Combination**: Configurable weights for optimal search results

### 📊 **Data Processing**
- Aggregates reviews and media URLs per place
- Combines structured data (name, neighborhood, tags, description) with user reviews
- Handles 1,500+ places with 30,000+ media URLs efficiently

### 🚀 **Vector Search**
- **FAISS**: Fast similarity search for dense and image embeddings
- **Cosine Similarity**: For sparse embeddings
- **Normalized Scoring**: MinMax scaling for fair comparison across different embedding types

### 🎯 **Query Enhancement**
- **NER Detection**: Extracts locations and entities from queries
- **Category Detection**: Identifies activity types (eat, drink, study, dance, etc.)
- **Weather Awareness**: Detects weather-related keywords
- **Query Expansion**: Enhances vague queries with relevant terms

## Data Structure

The system processes three main datasets:
- **Places**: Core venue information (name, neighborhood, tags, description)
- **Reviews**: User-generated reviews for each place
- **Media**: Image URLs associated with each place

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
# Core ML libraries
sentence_transformers
fastembed
transformers
faiss-cpu
scikit-learn

# Data processing
pandas
numpy
tqdm

# NLP
spacy

# Utilities
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
- **Search Speed**: Sub-second response times with FAISS
- **Memory Efficient**: Batch processing for large datasets

## Use Cases

This RAG solution is ideal for:
- **Location-based search applications**
- **Venue recommendation systems**
- **Tourism and hospitality platforms**
- **Event planning and discovery**
- **Local business directories**

## Future Enhancements

- Multi-image embedding per place (currently uses 1 image per place)
- Real-time embedding updates
- User feedback integration for search refinement
- Advanced query understanding with LLM integration
- Geographic clustering and filtering

## Technical Notes

- Uses L2 distance for FAISS indices (suitable for normalized embeddings)
- Implements batch processing for memory efficiency
- Normalizes all similarity scores to [0,1] range for fair comparison
- Configurable weights allow tuning for different use cases
- Handles missing data gracefully with fallback strategies 

---
https://github.com/user-attachments/assets/d139b7e4-dce4-49a3-95ab-e7e8c6897689

