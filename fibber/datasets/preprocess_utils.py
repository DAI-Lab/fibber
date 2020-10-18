import os

from fibber.datasets.downloadable_datasets import downloadable_dataset_urls
from fibber.download_utils import download_file, get_root_dir


def download_raw_and_preprocess(dataset_name,
                                download_list,
                                preprocess_fn,
                                preprocess_input_output_list):
    """Download and preprocess raw data into fibber's format.

    Args:
        dataset_name (str): the name of the dataset.
        download_list ([str]): a list of strings indicating which file to download. Each
            element in this list should corresponds to a one key in ``downloadable_dataset_urls``.
        preprocess_fn (fn): a function to preprocess the dataset.
        preprocess_input_output_list ([(str, str), ...]): A list of tuples. Each tuple indicate a
            pair of input and output file or path name.
    """
    root_dir = get_root_dir()
    dataset_dir = "datasets/" + dataset_name

    for item in download_list:
        download_file(**downloadable_dataset_urls[item], subdir=os.path.join(dataset_dir, "raw"))

    for input_name, output_name in preprocess_input_output_list:
        preprocess_fn(os.path.join(root_dir, dataset_dir, input_name),
                      os.path.join(root_dir, dataset_dir, output_name))
