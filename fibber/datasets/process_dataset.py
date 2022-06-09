import argparse
import json
import os
from collections import namedtuple

import datasets
from sklearn.model_selection import train_test_split

from fibber import get_root_dir

DatasetConfig = namedtuple(
    "DatasetConfig", [
        "dataset", "subset", "splits", "text0", "text1", "label", ])

dataset_configs = {
    "ag_news": DatasetConfig(dataset="ag_news", subset=None, splits=["train", "test"],
                             text0="text", text1=None, label="label"),
    "imdb": DatasetConfig(dataset="imdb", subset=None, splits=["train", "test"],
                          text0="text", text1=None, label="label"),
    "yelp_polarity": DatasetConfig(dataset="yelp_polarity", subset=None, splits=["train", "test"],
                                   text0="text", text1=None, label="label"),
    "sst2": DatasetConfig(dataset="glue", subset="sst2",
                          splits={"train": "train", "test": "validation"},
                          text0="sentence", text1=None, label="label"),
    "trec": DatasetConfig(dataset="trec", subset=None, splits=["train", "test"],
                          text0="text", text1=None, label="label-coarse"),
    "hate_speech_offensive": DatasetConfig(dataset="hate_speech_offensive", subset=None,
                                           splits=["train"], text0="tweet", text1=None,
                                           label="class"),
    "tweets_hate_speech_detection": DatasetConfig(dataset="tweets_hate_speech_detection",
                                                  subset=None, splits=["train"], text0="tweet",
                                                  text1=None, label="label"),
    "snli": DatasetConfig(dataset="snli", subset=None,
                          splits={"train": "train", "test": "validation"},
                          text0="premise", text1="hypothesis", label="label"),
    "mnli_matched": DatasetConfig(dataset="glue", subset="mnli",
                                  splits={"train": "train", "test": "validation_matched"},
                                  text0="premise", text1="hypothesis", label="label"),
    "mnli_mismatched": DatasetConfig(dataset="glue", subset="mnli",
                                     splits={"train": "train", "test": "validation_mismatched"},
                                     text0="premise", text1="hypothesis", label="label"),
}


def convert_to_data_list(dataset_split, config, omit_unlabeled_data):
    data_list = []
    for item in dataset_split:
        if omit_unlabeled_data and item[config.label] == -1:
            continue
        tmp = {
            "label": item[config.label],
            "text0": item[config.text0]
        }
        if config.text1 is not None:
            tmp["text1"]: item[config.text1]

        data_list.append(tmp)
    return data_list


def process_huggingface_dataset(config, output_path, omit_unlabeled_data):
    dataset = datasets.load_dataset(config.dataset, config.subset)
    os.makedirs(output_path, exist_ok=True)

    if len(config.splits) == 2:
        for fibber_split in config.splits:
            if isinstance(config.splits, dict):
                split = config.splits[fibber_split]
            else:
                split = fibber_split
            fibber_dataset = {
                "label_mapping": dataset[split].features[config.label].names,
            }
            data_list = convert_to_data_list(dataset[split], config, omit_unlabeled_data)

            data_list = sorted(data_list, key=lambda x: (x["label"], x["text0"]))
            fibber_dataset["data"] = data_list
            with open(os.path.join(output_path, f"{fibber_split}.json"), "w") as f:
                json.dump(fibber_dataset, f, indent=2)
    else:
        assert len(config.splits) == 1
        split = config.splits[0]
        data_list = convert_to_data_list(dataset[split], config, omit_unlabeled_data)
        train_list, test_list = train_test_split(data_list, test_size=0.2, random_state=42)
        train_list = sorted(train_list, key=lambda x: (x["label"], x["text0"]))
        test_list = sorted(test_list, key=lambda x: (x["label"], x["text0"]))
        fibber_trainset = {
            "label_mapping": dataset[split].features[config.label].names,
            "data": train_list
        }
        fibber_testset = {
            "label_mapping": dataset[split].features[config.label].names,
            "data": test_list
        }
        with open(os.path.join(output_path, "train.json"), "w") as f:
            json.dump(fibber_trainset, f, indent=2)
        with open(os.path.join(output_path, "test.json"), "w") as f:
            json.dump(fibber_testset, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(dataset_configs.keys()), required=True)
    parser.add_argument("--omit_unlabeled_data", choices=["0", "1"], default="1")
    args = parser.parse_args()

    config = dataset_configs[args.dataset]
    output_path = os.path.join(get_root_dir(), "datasets", args.dataset)
    process_huggingface_dataset(config, output_path,
                                omit_unlabeled_data=args.omit_unlabeled_data == "1")


if __name__ == "__main__":
    main()
