import pandas as pd
import json

from ..resource_utils import load_detailed_result
from ..resource_utils import update_overview_result



COL_FOR_NWIN = [
    ("3_ParaphraseAcc_sim0.95_ppl2", "L"),
    ("4_ParaphraseAcc_sim0.90_ppl5", "L"),
    ("USESemanticSimilarity_mean", "H"),
    ("GPT2GrammarQuality_mean", "L")
]

COPY_COL = ["1_model_name"]


def make_overview():
    detailed_df = load_detailed_result()

    # verify detailed result
    for group_info, item in detailed_df.groupby(["0_dataset_name", "1_model_name"]):
        assert len(item) == 1, (
            "Detailed results contains multiple runs for %s on %s." % (
                group_info[1], group_info[0]))

    results = {}
    for rid, item in detailed_df.iterrows():
        if item["1_model_name"] not in results:
            model_name = item["1_model_name"]
            tmp = {}

            for col_name in COPY_COL:
                tmp[col_name] = item[col_name]
            for col_name, _ in COL_FOR_NWIN:
                tmp[col_name] = 0

            results[model_name] = tmp

    for group_name, group in detailed_df.groupby("0_dataset_name"):
        for _, r1 in group.iterrows():
            for _, r2 in group.iterrows():
                for column_name, direction in COL_FOR_NWIN:
                    if ((direction == "H" and r1[column_name] > r2[column_name]) or
                        (direction == "L" and r1[column_name] < r2[column_name])):
                        results[r1["1_model_name"]][column_name] += 1

    update_overview_result(pd.DataFrame(list(results.values())))

if __name__ == "__main__":
    make_overview()
