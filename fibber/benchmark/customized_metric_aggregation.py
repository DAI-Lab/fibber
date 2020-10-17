"""This module defines customized metric aggregation functions."""


def paraphrase_classification_accuracy_agg_fn(use_sim, ppl_score):
    """This function makes a aggregation function for BERT classification metric.

    The aggregation function outputs the after attack accuracy of the BERT classifier.

    Args:
        use_sim (float): the threshold for ``USESemanticSimilarity`` metric. The adversarial
            example should have the USE similarity higher than this threshold.
        ppl_score (float): the threshold for ``GPT2GrammarQuality`` metric. The adversarial example
            should have the perplexity ratio measured by GPT2 lower than this threshold.
    """

    def agg_fn(data_record):
        if data_record["original_text_metrics"]["BertClfPrediction"] != data_record["label"]:
            return 0
        for item in data_record["paraphrase_metrics"]:
            if (item["BertClfPrediction"] != data_record["label"]
                and item["GPT2GrammarQuality"] < ppl_score
                    and item["USESemanticSimilarity"] > use_sim):
                return 0
        return 1
    return agg_fn


customized_metric_aggregation_fn_dict = {
    "3_ParaphraseAcc_use0.90_ppl2":
        paraphrase_classification_accuracy_agg_fn(use_sim=0.90, ppl_score=2),
    "4_ParaphraseAcc_use0.85_ppl5":
        paraphrase_classification_accuracy_agg_fn(use_sim=0.85, ppl_score=5)
}


customized_metric_for_nun_wins = [
    ("3_ParaphraseAcc_use0.90_ppl2", "L"),
    ("4_ParaphraseAcc_use0.85_ppl5", "L")
]
