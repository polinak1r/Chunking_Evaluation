from metrics.utils import *


def calculate_metrics(questions, final_metadata, n_results):
    recall_scores = []
    precision_scores = []
    iou_scores = []

    for i, row in questions.iterrows():
        metadatas = final_metadata[i]
        references = row['references']
        intersections = []

        for metadata in metadatas[:n_results]:
            chunk_start, chunk_end = metadata['start_index'], metadata['end_index']
            for ref in references:
                ref_start, ref_end = int(ref['start_index']), int(ref['end_index'])
                local_intersection = intersect_two_ranges((chunk_start, chunk_end), (ref_start, ref_end))
                if local_intersection is not None:
                    intersections = union_ranges([local_intersection] + intersections)
        if intersections:
            true_positive_tokens = sum_of_ranges(intersections)
        else:
            true_positive_tokens = 0

        all_relevant_tokens = sum_of_ranges([(x['start_index'], x['end_index']) for x in references])
        all_retrieved_tokens = sum_of_ranges([(x['start_index'], x['end_index']) for x in metadatas[:n_results]])

        recall_score = true_positive_tokens / all_relevant_tokens
        recall_scores.append(recall_score * 100)

        precision_score = true_positive_tokens / all_retrieved_tokens
        precision_scores.append(precision_score * 100)

        iou_score = true_positive_tokens / (all_relevant_tokens + all_retrieved_tokens - true_positive_tokens)
        iou_scores.append(iou_score * 100)

    return recall_scores, precision_scores, iou_scores
