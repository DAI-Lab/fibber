import pandas as pd

from fibber.benchmark.benchmark_utils import load_detailed_result, update_overview_result
from fibber.benchmark.customized_metric_aggregation import customized_metric_for_nun_wins

# Columns where the number of wins should be computed.
# "L" means lower is better.
# "H" means higher is better.
COL_FOR_NUM_WINS = [
    ("USESemanticSimilarity_mean", "H"),
    ("GPT2GrammarQuality_mean", "L")
] + customized_metric_for_nun_wins

DATASET_NAME_COL = "0_dataset_name"
STRATEGY_NAME_COL = "1_paraphrase_strategy_name"


def make_overview():
    """Generate overview table from detailed table."""
    detailed_df = load_detailed_result()

    # verify detailed result
    for group_info, item in detailed_df.groupby([DATASET_NAME_COL, STRATEGY_NAME_COL]):
        assert len(item) == 1, (
            "Detailed results contains multiple runs for %s on %s." % (
                group_info[1], group_info[0]))

    results = {}
    for rid, item in detailed_df.iterrows():
        if item[STRATEGY_NAME_COL] not in results:
            model_name = item[STRATEGY_NAME_COL]
            tmp = dict()

            tmp[STRATEGY_NAME_COL] = item[STRATEGY_NAME_COL]
            for col_name, _ in COL_FOR_NUM_WINS:
                tmp[col_name] = 0

            results[model_name] = tmp

    for group_name, group in detailed_df.groupby(DATASET_NAME_COL):
        for _, r1 in group.iterrows():
            for _, r2 in group.iterrows():
                for column_name, direction in COL_FOR_NUM_WINS:
                    if ((direction == "H" and r1[column_name] > r2[column_name])
                            or (direction == "L" and r1[column_name] < r2[column_name])):
                        results[r1[STRATEGY_NAME_COL]][column_name] += 1

    update_overview_result(pd.DataFrame(list(results.values())))


if __name__ == "__main__":
    make_overview()
