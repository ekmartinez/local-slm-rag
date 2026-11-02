"""
rag_helpers.py

Helper functions for the local RAG notebook
(01_local_rag_with_small_language_models.ipynb).

They live here so the notebook can focus on the RAG pipeline itself.
Adapted from Chapter 13 of "Domain-Specific Small Language Models".
"""

from pathlib import Path

import fitz  # PyMuPDF: opens PDFs and extracts their text
import numpy as np
import pandas as pd
from tqdm import tqdm  # progress indicator for the page loop

def text_formatter(text: str) -> str:
    """Clean a chunk of text extracted from a PDF page.

    (docstring unchanged from the previous step)
    """
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    """Read a PDF page by page and collect the text plus basic statistics.

    Each page becomes one dictionary, so the result can be loaded straight
    into a pandas DataFrame for inspection and chunking later on. Because
    the function takes the path as a parameter, it works for any number of
    PDFs: call it once per document.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list with one dictionary per page, containing:
            page_number: Page number, starting at 1.
            page_char_count: Number of characters in the cleaned text.
            page_word_count: Approximate word count (split on spaces).
            page_sentence_count_raw: Rough sentence count (split on ". ").
            page_token_count: Estimated tokens, using the rule of thumb
                of about 4 characters per token.
            text: The cleaned text of the page.
    """
    pages_and_texts = []

    # The `with` block closes the PDF file automatically when we are done.
    with fitz.open(pdf_path) as doc:
        # enumerate() gives us the page index; total=len(doc) lets tqdm show
        # a percentage instead of a plain counter.
        for page_number, page in tqdm(enumerate(doc), total=len(doc)):
            # Extract the raw text of the page, then clean it
            # (newlines become spaces).
            text = page.get_text()
            text = text_formatter(text)

            # Store the text together with simple statistics. They are
            # approximations, but good enough to decide how to chunk the
            # text later.
            pages_and_texts.append({
                "page_number": page_number + 1,  # PDF pages are 1-based for readers
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_sentence_count_raw": len(text.split(". ")),
                "page_token_count": len(text) / 4,  # ~4 characters per token
                "text": text})

    return pages_and_texts

def split_list(input_list: list, slice_size: int) -> list[list]:
    """Split a list into consecutive sublists of a target size.

    Embedding models have a token limit, and the generator model has a
    context window. Splitting the sentences into small groups keeps every
    chunk within those limits.

    Args:
        input_list: The list to split (here, the sentences of a page).
        slice_size: Maximum number of items per sublist.

    Returns:
        A list of sublists. Every sublist has `slice_size` items except
        possibly the last one, which holds whatever is left over.

    Example:
        >>> split_list([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    # range() jumps forward `slice_size` positions at a time, and each slice
    # takes the items from that position up to the next jump.
    return [input_list[i:i + slice_size]
            for i in range(0, len(input_list), slice_size)]

def load_or_build_embeddings_df(csv_path, pages_and_chunks=None):
    """Load the chunks and embeddings from a CSV file, or build the file first.

    Computing embeddings is the slowest step of the pipeline, so the result is
    saved to disk the first time and reused on later runs.

    Args:
        csv_path: Path of the CSV file to read, or to create if it is missing.
        pages_and_chunks: The list of chunk dictionaries (with embeddings)
            built earlier in the notebook. Only needed when the CSV does not
            exist yet.

    Returns:
        A DataFrame with one row per chunk. The `embedding` column holds
        NumPy arrays.

    Raises:
        FileNotFoundError: If the CSV does not exist and no
            `pages_and_chunks` was given to build it from.
    """
    csv_path = Path(csv_path)

    # Reuse the saved file if it exists.
    if csv_path.exists():
        df = pd.read_csv(csv_path)

        # CSV stores each embedding as text, such as "[0.12 -0.03 ...]".
        # Strip the brackets and parse the numbers back into a NumPy array.
        df["embedding"] = df["embedding"].apply(
            lambda x: np.fromstring(x.strip("[]"), sep=" "))
        return df

    # No file yet: we can only build it if the chunks are available.
    if pages_and_chunks is None:
        raise FileNotFoundError(
            f"{csv_path} not found and no pages_and_chunks given to build it. "
            "Run the chunking and embedding cells first.")

    # Build the DataFrame from the chunk dictionaries and save it for next time.
    df = pd.DataFrame(pages_and_chunks)
    df.to_csv(csv_path, index=False)
    return df

def prompt_formatter(query: str, context_items: list[dict]) -> str:
    """Build the prompt that is sent to the language model.

    In RAG, the model doesn't answer from memory alone. The retrieved chunks
    are placed in the prompt as context, next to the user's question, so the
    answer is grounded in the source document.

    Args:
        query: The user's question.
        context_items: The retrieved chunks, as returned by the database
            search. Each dictionary needs a "sentence_chunks" key with the
            chunk text.

    Returns:
        The complete prompt as a single string.
    """
    # Turn the chunk texts into a bulleted list, one chunk per line.
    context = "- " + "\n- ".join([item["sentence_chunks"]
                                  for item in context_items])

    # The prompt template. {context} and {query} are filled in below.
    # The "Relevant passages" line is a cue for the model: it asks it to
    # quote the useful parts of the context before writing its answer.
    base_prompt = """Based on the following context items,
please answer the query.
Make sure your answers are as explanatory as possible.
\nUse the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

    # Insert the context and the question into the template.
    return base_prompt.format(context=context, query=query)