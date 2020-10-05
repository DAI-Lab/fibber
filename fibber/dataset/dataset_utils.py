import tqdm


def verify_dataset(data):
    assert "label_mapping" in data
    assert "cased" in data
    assert "paraphrase_field" in data
    assert data["paraphrase_field"] in ["text0", "text1"]

    num_labels = len(data["label_mapping"])
    counter = [0] * num_labels

    for data_record in tqdm.tqdm(data["data"]):
        assert "label" in data_record
        label = data_record["label"]
        assert 0 <= label < num_labels
        counter[label] += 1
        assert "text0" in data_record
        assert data["paraphrase_field"] in data_record

    for item in counter:
        assert item > 0, "empty class"
