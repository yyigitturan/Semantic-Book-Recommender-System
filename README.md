# 📖 Semantic Book Recommender System (AI-Powered)
This project implements an AI-powered book recommendation system that utilizes semantic search and emotional tone filtering to provide highly relevant suggestions. Users can describe a plot, theme, or concept, and the system finds books with descriptions most similar to the query, which can then be refined by genre (category) and emotional tone (e.g., Happy, Sad, Suspenseful).

The application is built around a Retrieval-Augmented Generation (RAG)-like structure, using a vector store for semantic retrieval and a Gradio interface for user interaction.

## ✨ Key Features
Semantic Search (RAG): Uses a pre-trained HuggingFace Sentence Transformer (all-MiniLM-L6-v2) to embed book descriptions and user queries, enabling discovery based on meaning rather than just keywords.

Emotional Filtering: Recommendations can be sorted by an associated emotional score (Joy, Sadness, Fear, Anger, Surprise), allowing for tone-specific book discovery.

Genre Filtering: Standard filtering by book category is integrated to narrow down the results.

Interactive Web UI: A professional and easy-to-use interface is provided via Gradio.

## 🚀 Getting Started
Prerequisites

You need Python 3.9+ and the following files in your project directory (as assumed in the provided script):

books_with_emotions.csv: The main dataset containing book metadata (ISBN, title, author, category, description) and pre-calculated emotion scores (joy, surprise, etc.).

tagged_description.txt: A text file where each book's description is prefixed by its 13-digit ISBN, used for efficient document loading and retrieval.

cover.jpg: A placeholder image for books without a valid thumbnail URL.
