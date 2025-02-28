# Movie Recommendation System with Sentence Transformers (all-MiniLM-L6-v2)

## 📌 Overview

This repository hosts the quantized version of the all-MiniLM-L6-v2 model fine-tuned for movie reccommendation tasks. The model has been trained on the movielens_ratings dataset from Hugging Face. The model is quantized to Float16 (FP16) to optimize inference speed and efficiency while maintaining high performance.

## 🏗 Model Details

- **Model Architecture:** all-MiniLM-L6-v2
- **Task:** Movie Recommendation System  
- **Dataset:** Hugging Face's `movielens_ratings`  
- **Quantization:** Float16 (FP16) for optimized inference  
- **Fine-tuning Framework:** Hugging Face Transformers  

## 🚀 Usage

### Installation

```bash
pip install transformers torch
```

### Loading the Model

```python
from sentence_transformers import SentenceTransformer, models
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "AventIQ-AI/all-MiniLM-L6-v2-movie-recommendation-system"
model = SentenceTransformer(model_name).to(device)
```

### Question Answer Example

```python
def generate_movies(genre, top_k=5):
    genre_embedding = model.encode([genre], convert_to_tensor=True)
    movie_embeddings = model.encode(df['title'].tolist(), convert_to_tensor=True)

    scores = torch.nn.functional.cosine_similarity(genre_embedding, movie_embeddings)
    top_results = torch.argsort(scores, descending=True)

    # Get unique movies while preserving order
    recommended_movies = []
    seen = set()
    for idx in top_results.tolist():
        movie = df.iloc[idx]['title']
        if movie not in seen:
            recommended_movies.append(movie)
            seen.add(movie)
        if len(recommended_movies) == top_k:
            break
    return recommended_movies

print("🎬 Recommended Movies for 'Action':", generate_movies("Action"))
print("🎬 Recommended Movies for 'Comedy':", generate_movies("Comedy"))
print("🎬 Recommended Movies for 'Sci-Fi':", generate_movies("Sci-Fi"))
```

## ⚡ Quantization Details

Post-training quantization was applied using PyTorch's built-in quantization framework. The model was quantized to Float16 (FP16) to reduce model size and improve inference efficiency while balancing accuracy.

## 📂 Repository Structure

```
.
├── model/               # Contains the quantized model files
├── tokenizer_config/    # Tokenizer configuration and vocabulary files
├── model.safetensors/   # Quantized Model
├── README.md            # Model documentation
```

## ⚠️ Limitations

- The model may struggle for out of scope tasks.
- Quantization may lead to slight degradation in accuracy compared to full-precision models.
- Performance may vary across different writing styles and sentence structures.

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.
