import argparse
import glob
import json
import os

from .. import log
from ..datasets import (
  process_ag, process_imdb, process_mnli, process_mr, process_snli, process_yelp)
from ..datasets.dataset_utils import verify_dataset
from ..download_utils import download_file, get_root_dir
from ..downloadable_resources import resources

logger = log.setup_custom_logger('benchmark')

parser = argparse.ArgumentParser()

parser.add_argument("--process_raw", choices=["0", "1"], default="0",
                    help="Use 1 to download and process raw data on this machine. "
                    "Use 0 (Default) to download processed data.")
parser.add_argument("--verify", choices=["0", "1"], default="1",
                    help="Verify each json in each datasets have proper attributes.")

if __name__ == "__main__":
    FLAGS = parser.parse_args()

    if FLAGS.process_raw == "1":
        datasets_processing_func = {
            "ag": process_ag.download_and_process_ag,
            "imdb": process_imdb.download_and_process_imdb,
            "mnli": process_mnli.download_and_process_mnli,
            "mr": process_mr.download_and_process_mr,
            "snli": process_snli.download_and_process_snli,
            "yelp": process_yelp.download_and_process_yelp
        }

        for name, processing_func in datasets_processing_func.items():
            logger.info("Start download and process %s.", name)
            processing_func()

    else:
        root_dir = get_root_dir()

        download_file(filename=resources["processed-datasets"]["filename"],
                      url=resources["processed-datasets"]["url"],
                      md5_checksum=resources["processed-datasets"]["md5"],
                      subdir="", untar=True)

    if FLAGS.verify == "1":
        datasets_dir = os.path.join(root_dir, "datasets")
        filelist = sorted(glob.glob(datasets_dir + "/*/*.json"))
        for filename in filelist:
            logger.info("Verify %s.", filename)
            with open(filename) as f:
                data = json.load(f)

            verify_dataset(data)
