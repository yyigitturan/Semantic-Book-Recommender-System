import pandas as pd 
import numpy as np 
from dotenv import load_dotenv 
from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import CharacterTextSplitter 
from langchain_chroma import Chroma 
import gradio as gr 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document 
import re

# --- Data Loading and Preprocessing ---
# NOTE: These files must exist in the environment where the code is run.
try:
    books = pd.read_csv("books_with_emotions.csv") 
except FileNotFoundError:
    print("Error: 'books_with_emotions.csv' file not found. Please check.")
    # Example empty DataFrame creation or exit
    # books = pd.DataFrame(columns=["isbn13", "thumbnail", "large_thumbnail", "simple_categories", "description", "authors", "title", "joy", "surprise", "anger", "fear", "sadness"])

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where( 
    books["large_thumbnail"].isna(), 
    "cover.jpg", # This file must also exist
    books["large_thumbnail"],
)

try:
    raw_documents = TextLoader("tagged_description.txt").load() 
    full_text_content = raw_documents[0].page_content
except FileNotFoundError:
    print("Error: 'tagged_description.txt' file not found. Please check.")
    full_text_content = "" # Empty content on error

regex = r"(?P<isbn>\d{13})\s*(?P<content>.*?)(?=\d{13}|$)" 

matches = re.finditer(regex, full_text_content, re.DOTALL) 

documents = []
for match in matches:
    isbn = match.group('isbn')
    content = match.group('content').strip()
    
    documents.append(
        Document(
            page_content=f"{isbn} {content}", 
            metadata={"source": "tagged_description.txt"}
        )
    )

# --- Embedding and Vector Database ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# If documents are empty, skip Chroma initialization or use a placeholder
if documents:
    db_books = Chroma.from_documents(
        documents,
        embedding=embeddings
    ) 
else:
    # It might be better to initialize an empty Chroma or use an error flag
    # But the application should continue to run
    db_books = None
    print("Warning: Recommendations function may not work because documents could not be loaded.")

# --- Recommendation Functions (Preserving original logic with minor adjustments for professional use) ---

def retrieve_semantic_recommendations( 
        query: str, 
        category: str = None,
        tone: str = None,  
        initial_top_k: int = 50, 
        final_top_k: int = 16, 
) -> pd.DataFrame: 
    
    if db_books is None:
        return pd.DataFrame() # Return empty if db_books is missing

    recs = db_books.similarity_search(query, k=initial_top_k)

    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)] # Get all results before filtering

    # Category Filter
    if category != "All": 
        book_recs = book_recs[book_recs["simple_categories"] == category] 
    
    # Emotion/Tone Sorting
    if tone == "Happy": 
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)

    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)

    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)

    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False) 

    # Final Top K limit
    return book_recs.head(final_top_k)


def recommend_books( 
        query: str, 
        category: str, 
        tone: str
): 
    # Alert if user query is empty
    if not query.strip():
        return [], "Please enter a description to get recommendations."
    
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = [] 

    if recommendations.empty:
        return [], "No books found matching your selection. Please relax the filters."
    
    for _, row in recommendations.iterrows(): 
        description = row["description"] 
        truncated_desc_split = description.split() 
        # Limit description to 30 words
        truncated_description = " ".join(truncated_desc_split[:30]) + ("..." if len(truncated_desc_split) > 30 else "")

        # Author Formatting
        authors_split = str(row["authors"]).split(";") 
        if len(authors_split) == 2: 
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])} and {authors_split[-1]}"
        else: 
            authors_str = row["authors"] 

        # Creating the Caption
        caption = f"**{row['title']}**\n_Author: {authors_str}_\n\n{truncated_description}"
        # Tuple format suitable for Gallery (URL, Caption)
        results.append((row["large_thumbnail"], caption))

    # Success message, indicating how many books were found
    message = f"📚 {len(results)} book recommendations found."

    return results, message

# --- Options for the Interface ---
categories = ["All"] + sorted(books["simple_categories"].unique().tolist()) 
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"] 

# --- Gradio Interface (Professional Enhancements) ---

# Use a more modern theme and Blocks for better layout
with gr.Blocks(
    theme=gr.themes.Soft(), # Soft and professional theme
    title="Semantic Book Recommender System"
) as dashboard:
    # --- Title Area ---
    gr.Markdown(
        """
        # 📚 Semantic Book Recommender System
        ### A.I. Powered Book Discovery: Define Your Story, Tone, and Genre.
        """
    )
    gr.HTML("<hr style='border-color: #3B82F6;'>") # Professional divider

    # --- Input Controls Area (Horizontal Layout) ---
    with gr.Row(variant="panel"):
        # Query Input (Should occupy more space)
        user_query = gr.Textbox(
            label="📖 Book Description or Plot:", 
            placeholder="e.g., A story about forgiveness, an adventure after a lost treasure...",
            lines=3,
            scale=3 # Occupy 3 times more space than others
        ) 
        
        # Filters (Vertical Layout)
        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(
                choices=categories, 
                label="🏷️ Select a Category:", 
                value="All",
                interactive=True
            ) 
            tone_dropdown = gr.Dropdown(
                choices=tones, 
                label="🎭 Select an Emotional Tone:", 
                value="All",
                interactive=True
            ) 

    # --- Button and Message Area ---
    with gr.Row():
        submit_button = gr.Button(
            "✨ Find Recommendations", 
            variant="primary", # Highlight as the main button
            scale=1
        )
        # Textbox to display the message (Used as output)
        status_message = gr.Textbox(
            label="Status", 
            value="Awaiting your query...",
            interactive=False, # User cannot change this
            scale=3
        )

    gr.HTML("<hr style='border-color: #3B82F6;'>") # Divider

    # --- Output Area (Gallery) ---
    gr.Markdown("## ✨ Recommended Books") 
    output_gallery = gr.Gallery(
        label="Recommendation Results", 
        columns=[4, 8], # 8 columns on large screens, 4 on mobile/small screens
        rows=2,
        height="auto", # Automatic height adjustment
        preview=True, # Enable image preview
        object_fit="contain", # Fit images without cropping
        visible=True # Visible initially
    ) 

    # --- Event Handlers ---
    # The `recommend_books` function now returns two outputs: Gallery content and status message
    submit_button.click(
        fn=recommend_books, 
        inputs=[user_query, category_dropdown, tone_dropdown], 
        outputs=[output_gallery, status_message],
        api_name="get_book_recommendations" # Professional name for API access
    ) 

# --- Launching the Application ---
if __name__ == "__main__":
    # Professionalizing launch settings
    dashboard.launch(
        share=True, # For easy sharing (adjust based on use case)
        inbrowser=True, # Open automatically in browser
        show_api=True # Show API documentation
    )
