#
# Based on Data Processors for other GLUE tasks, see
# https://github.com/huggingface/transformers/blob/master/src/transformers/data/processors/glue.py
#

import os
import csv
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from transformers.data.processors.glue import glue_processors, glue_output_modes, glue_tasks_num_labels

def init_ppa():
    glue_processors["ppa"] = PPAProcessor
    glue_output_modes["ppa"] = "classification"
    glue_tasks_num_labels["ppa"] = 2


class PPAProcessor(DataProcessor):
    """Processor for the Prepositional Phrase Attachment data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        raise NotImplementedError()

    def _read_ssv(self, input_file):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter=" "))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_ssv(os.path.join(data_dir, "training")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_ssv(os.path.join(data_dir, "devset")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_ssv(os.path.join(data_dir, "test")), "test")

    def get_labels(self):
        """See base class."""
        return ["N", "V"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []

        for x in lines:
            guid = x[0]
            text_a = " ".join(x[1:5])
            label = x[5]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
