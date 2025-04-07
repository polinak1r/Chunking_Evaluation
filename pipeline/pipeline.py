from chunker.fixed_token_chunker import FixedTokenChunker
import pandas as pd
import numpy as np
import json
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions

from metrics.calculate_metrics import calculate_metrics


def run_retrieval_pipeline(
        corpus_path: str = "data/state_of_the_union.md",
        questions_path: str = "data/questions_state_of_the_union.csv",
        chunk_size: int = 200,
        chunk_overlap: int = 0,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        n_results: int = 5,
        collection_name: str = "my_collection"
):
    # download questions and text
    questions = pd.read_csv(questions_path)
    questions['references'] = questions['references'].apply(json.loads)

    with open(corpus_path, 'r') as f:
        text = f.read()

    # choose chunker and use it
    chunker = FixedTokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, encoding_name="cl100k_base")
    chunked_text = chunker.split_text(text)

    # collecting start_index and end_index for each chunk
    chunked_meta = []
    for chunk in chunked_text:
        if chunk in text:
            start_index = text.find(chunk)
            end_index = start_index + len(chunk)
            chunked_meta.append({"start_index": start_index, "end_index": end_index})
        else:
            chunked_meta.append({"start_index": -1, "end_index": -1})

    client = chromadb.Client()

    my_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=my_embedding_function
    )

    collection.add(
        documents=chunked_text,
        metadatas=chunked_meta,
        ids=[f"chunk_{i}" for i in range(len(chunked_text))]
    )

    model = SentenceTransformer(model_name)
    embeddings = model.encode(questions['question'].tolist(), show_progress_bar=True)
    questions['embedding'] = embeddings.tolist()

    query_embeddings = questions['embedding'].tolist()
    retrievals = collection.query(query_embeddings=query_embeddings, n_results=n_results)

    recall_list, precision_list, iou_list = calculate_metrics(
        questions, retrievals['metadatas'], n_results=n_results
    )

    recall_mean = round(np.mean(recall_list), 2)
    recall_std = round(np.std(recall_list), 2)
    precision_mean = round(np.mean(precision_list), 2)
    precision_std = round(np.std(precision_list), 2)
    iou_mean = round(np.mean(iou_list), 2)
    iou_std = round(np.std(iou_list), 2)

    metrics = {
        "recall": recall_list,
        "precision": precision_list,
        "IoU": iou_list,
        "recall_mean": recall_mean,
        "recall_std": recall_std,
        "precision_mean": precision_mean,
        "precision_std": precision_std,
        "iou_mean": iou_mean,
        "iou_std": iou_std
    }
    return metrics
