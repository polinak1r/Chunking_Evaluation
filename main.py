from pipeline.pipeline import run_retrieval_pipeline


def main():
    # here you can change parameters
    corpus_path = "data/state_of_the_union.md"
    questions_path = "data/questions_state_of_the_union.csv"
    chunk_size = 250
    chunk_overlap = 125
    n_results = 5
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # run pipeline
    results = run_retrieval_pipeline(
        corpus_path=corpus_path,
        questions_path=questions_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_name=model_name,
        n_results=n_results,
        collection_name="my_collection"
    )

    print(f"For chunk size {chunk_size}, overlap {chunk_overlap}, and top-{n_results} results:")
    # print("Recall scores: ", results["recall"])
    # print("Precision scores: ", results["precision"])
    print(f"Recall: {results['recall_mean']:.2f} ± {results['recall_std']:.2f}")
    print(f"Precision: {results['precision_mean']:.2f} ± {results['precision_std']:.2f}")
    print(f"IoU: {results['iou_mean']:.2f} ± {results['iou_std']:.2f}")


if __name__ == "__main__":
    main()
