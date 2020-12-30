import pandas as pd

from fibber.benchmark.benchmark_utils import load_detailed_result, update_overview_result
from fibber.metrics.metric_utils import DIRECTION_HIGHER_BETTER, DIRECTION_LOWER_BETTER

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

    col_for_num_win = []

    for col_name in detailed_df.columns.tolist():
        if col_name.endswith(DIRECTION_HIGHER_BETTER):
            col_for_num_win.append((col_name, DIRECTION_HIGHER_BETTER))
        if col_name.endswith(DIRECTION_LOWER_BETTER):
            col_for_num_win.append((col_name, DIRECTION_LOWER_BETTER))

    results = {}
    for rid, item in detailed_df.iterrows():
        if item[STRATEGY_NAME_COL] not in results:
            model_name = item[STRATEGY_NAME_COL]
            tmp = dict()

            tmp[STRATEGY_NAME_COL] = item[STRATEGY_NAME_COL]
            for col_name, _ in col_for_num_win:
                tmp[col_name] = 0

            results[model_name] = tmp

    for group_name, group in detailed_df.groupby(DATASET_NAME_COL):
        for _, r1 in group.iterrows():
            for _, r2 in group.iterrows():
                for column_name, direction in col_for_num_win:
                    if ((direction == DIRECTION_HIGHER_BETTER
                         and r1[column_name] > r2[column_name])
                            or (direction == DIRECTION_LOWER_BETTER
                                and r1[column_name] < r2[column_name])):
                        results[r1[STRATEGY_NAME_COL]][column_name] += 1

    update_overview_result(pd.DataFrame(list(results.values())))


if __name__ == "__main__":
    make_overview()
