from fibber.metrics.attack_aggregation_utils import (
    get_best_adv_by_ppl, get_best_adv_by_sim, pairwise_editing_distance_fn,
    paraphrase_classification_accuracy_agg_fn_constructor)


def make_data_record(label, origin_predict, paraphrase_ppl_list,
                     paraphrase_sim_list, paraphrase_pred_list, classifier):
    assert len(paraphrase_ppl_list) == len(paraphrase_sim_list)
    assert len(paraphrase_sim_list) == len(paraphrase_pred_list)

    return {
        "label": label,
        "original_text_metrics": {
            classifier: origin_predict
        },
        "paraphrase_metrics": [
            {
                "GPT2GrammarQualityMetric": ppl,
                "USESemanticSimilarityMetric": sim,
                classifier: pred
            } for ppl, sim, pred in zip(paraphrase_ppl_list,
                                        paraphrase_sim_list,
                                        paraphrase_pred_list)
        ]
    }


def test_paraphrase_classification_accuracy_agg_fn_constructor():
    classifier = "FooClassifier"
    agg_fn = paraphrase_classification_accuracy_agg_fn_constructor(2, 0.9, classifier)

    data_record = make_data_record(label=1, origin_predict=0,
                                   paraphrase_ppl_list=[],
                                   paraphrase_sim_list=[],
                                   paraphrase_pred_list=[],
                                   classifier=classifier)
    assert agg_fn(data_record) == 0

    data_record = make_data_record(label=1, origin_predict=1,
                                   paraphrase_ppl_list=[],
                                   paraphrase_sim_list=[],
                                   paraphrase_pred_list=[],
                                   classifier=classifier)
    assert agg_fn(data_record) == 1

    data_record = make_data_record(label=1, origin_predict=1,
                                   paraphrase_ppl_list=[5, 1.2, 1.1],
                                   paraphrase_sim_list=[0.98, 0.7, 0.95],
                                   paraphrase_pred_list=[2, 3, 1],
                                   classifier=classifier)
    assert agg_fn(data_record) == 1

    data_record = make_data_record(label=1, origin_predict=1,
                                   paraphrase_ppl_list=[5, 1.2, 1.1],
                                   paraphrase_sim_list=[0.98, 0.7, 0.95],
                                   paraphrase_pred_list=[2, 1, 2],
                                   classifier=classifier)
    assert agg_fn(data_record) == 0


def test_pairwise_editing_distance_fn():
    data_record = {
        "text0_paraphrases": [
            "aa bb cc",
            "aa cc ee",
            "aa bb"
        ]
    }
    assert abs(pairwise_editing_distance_fn(data_record) - 5 / 3) < 1e-6


def test_get_best_adv_by_sim():
    classifier = "FooClassifier"
    data_record = make_data_record(label=1, origin_predict=1,
                                   paraphrase_ppl_list=[5.1, 1.2, 1.1, 1.2, 1.3, 1.4],
                                   paraphrase_sim_list=[0.98, 0.7, 0.95, 0.92, 0.93, 0.94],
                                   paraphrase_pred_list=[2, 3, 1, 4, 5, 6],
                                   classifier=classifier)
    best_metric = get_best_adv_by_sim(data_record, classifier, 5, 0.85)
    assert best_metric["GPT2GrammarQualityMetric"] == 1.4
    assert best_metric["USESemanticSimilarityMetric"] == 0.94
    assert best_metric[classifier] == 6


def test_get_best_adv_by_ppl():
    classifier = "FooClassifier"
    data_record = make_data_record(label=1, origin_predict=1,
                                   paraphrase_ppl_list=[5.1, 1.2, 1.1, 1.2, 1.3, 1.4],
                                   paraphrase_sim_list=[0.98, 0.7, 0.95, 0.92, 0.93, 0.94],
                                   paraphrase_pred_list=[2, 3, 1, 4, 5, 6],
                                   classifier=classifier)
    best_metric = get_best_adv_by_ppl(data_record, classifier, 5, 0.85)
    assert best_metric["GPT2GrammarQualityMetric"] == 1.2
    assert best_metric["USESemanticSimilarityMetric"] == 0.92
    assert best_metric[classifier] == 4
