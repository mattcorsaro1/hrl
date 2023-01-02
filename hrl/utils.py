import os
import itertools
import numpy as np
import pickle

def create_log_dir(experiment_name):
    path = os.path.join(os.getcwd(), experiment_name)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    return path


def chunked_inference(states, f, chunk_size=1000):
    """" f must take in np arrays and return np arrays. """
    
    def get_chunks(x, n):
        """ Break x into chunks of size n. """
        for i in range(0, len(x), n):
            yield x[i: i+n]

    state_chunks = get_chunks(states, chunk_size)
    values = np.zeros((len(states),))
    current_idx = 0

    for state_chunk in state_chunks:
        chunk_values = f(state_chunk)
        current_chunk_size = len(state_chunk)
        values[current_idx:current_idx + current_chunk_size] = chunk_values.squeeze()
        current_idx += current_chunk_size

    return values

def flatten(x):
    return list(itertools.chain.from_iterable(x))

class MetaLogger:
    """
    Copied from Rainbow RBFDQN
    """
    def __init__(self, logging_directory) -> None:
        super().__init__()
        self._logging_directory = logging_directory
        os.makedirs(logging_directory, exist_ok=True)
        self._logging_values = {}
        self._filenames = {}

    def add_field(self, field_name, filename):
        assert isinstance(field_name, str)
        assert field_name != ""
        for char in [" ", "/", "\\"]:
            assert char not in field_name

        folder_name = os.path.join(self._logging_directory, field_name)
        os.makedirs(folder_name, exist_ok=True)
        print(f"Successfully created the directory {folder_name}")

        full_path = os.path.join(folder_name, filename)
        self._filenames[field_name] = full_path

        assert field_name not in self._logging_values

        self._logging_values[field_name] = []

    def append_datapoint(self, field_name, datapoint, write=False):
        self._logging_values[field_name].append(datapoint)
        if write:
            self.write_field(field_name)

    def write_field(self, field_name):
        full_path = self._filenames[field_name]
        values = self._logging_values[field_name]
        with open(full_path, "wb+") as f:
            pickle.dump(values, f)

    def write_all_fields(self):
        for field_name in self._filenames.keys():
            self.write_field(field_name)
