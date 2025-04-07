import wandb

from pipeline.pipeline import run_retrieval_pipeline


# This script is designed to be used in a Weights & Biases sweep to run the retrieval pipeline with varying
# hyperparameters and log performance metrics.
def train():
    run = wandb.init()
    config = wandb.config

    run.name = f"ChunkSize_{config.chunk_size}_Overlap_{config.chunk_overlap}_RetrieveResults_{config.n_results}"
    run.save()

    corpus_path = "data/state_of_the_union.md"
    questions_path = "data/questions_state_of_the_union.csv"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    collection_name = "my_collection"

    if config.chunk_overlap > config.chunk_size // 2:
        print(f"[SKIP] overlap={config.chunk_overlap} > half of chunk_size={config.chunk_size // 2}")
        wandb.run.notes = "Invalid parameters: overlap > chunk_size / 2"
        wandb.run.summary["status"] = "skipped"
        wandb.finish()
        return

    metrics = run_retrieval_pipeline(
        corpus_path=corpus_path,
        questions_path=questions_path,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        model_name=model_name,
        n_results=config.n_results,
        collection_name=collection_name
    )

    wandb.log({
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "n_results": config.n_results,
        "recall_mean": metrics["recall_mean"],
        "recall_std": metrics["recall_std"],
        "precision_mean": metrics["precision_mean"],
        "precision_std": metrics["precision_std"],
        "iou_mean": metrics["iou_mean"],
        "iou_std": metrics["iou_std"]
    })

    wandb.run.summary["recall ± std"] = f"{metrics['recall_mean']} ± {metrics['recall_std']}"
    wandb.run.summary["precision ± std"] = f"{metrics['precision_mean']} ± {metrics['precision_std']}"
    wandb.run.summary["IoU ± std"] = f"{metrics['iou_mean']} ± {metrics['iou_std']}"

    run.finish()


if __name__ == "__main__":
    train()
