import os
from tempfile import TemporaryDirectory
from typing import Union, List, Tuple
import uuid

from pandas import read_csv

from plants_sm.design_patterns.observer import Observer, Subject
from plants_sm.io import CSVWriter, write_csv, CSVReader
from plants_sm.io.h5 import H5Reader, read_h5, H5Writer, write_h5
from plants_sm.io.json import JSONWriter, write_json, read_json
from plants_sm.io.pickle import PickleReader, PickleWriter, write_pickle, read_pickle


class BatchManager(Observer):

    def __init__(self, batch_size: Union[int, None] = None):
        """
        Class that manages the batches of a dataset.

        Parameters
        ----------
        batch_size : int
            the size of the batches
        """
        self._variables_to_save = None
        self._cls = None
        self.batch_size = batch_size
        filename = str(uuid.uuid4().clock_seq)
        folder = TemporaryDirectory(prefix=filename)
        self.temporary_folder = folder
        self.counter = batch_size
        self.batch_i = 0

    def __del__(self):
        """
        Deletes the temporary folder.
        """
        self.temporary_folder.cleanup()

    def end(self):
        """
        Resets the counter.
        """
        self.counter = 0
        self.batch_i = 0

    def update(self, subject: 'Dataset', **kwargs) -> None:
        """
        Updates the batch manager.

        Parameters
        ----------
        subject: Subject
            the subject that is being observed
        """
        if kwargs["function"] == "__next__":
            self.write_intermediate_files()
            self.counter += self.batch_size

        elif kwargs["function"] == "next_batch":
            if self.counter == 0:
                self.counter += self.batch_size
                subject._batch_state = self.read_intermediate_files(subject)
                if subject._folder_to_load_features is not None and subject._batch_state:
                    self._read_features(subject)
                    self.batch_i += 1

            else:
                self.write_intermediate_files()
                self.counter += self.batch_size
                subject._batch_state = self.read_intermediate_files(subject)
                if subject._folder_to_load_features is not None and subject._batch_state:
                    self._read_features(subject)
                    self.batch_i += 1

    def _read_features(self, subject: 'Dataset'):
        """
        Reads the features from the files.
        """
        subject.features = read_pickle(os.path.join(subject._folder_to_load_features,
                                                    f"features_{self.batch_i}.pkl"))

    def register_class(self, cls, variables_to_save: List[Tuple[str, str]] = None):
        """
        Registers the class that will be used to create the batches.

        Parameters
        ----------
        cls: class
            the class that will be used to create the batches
        variables_to_save: List[Tuple[str, str]]
            the variables that will be saved in the intermediate files
        """
        self._cls = cls
        self._variables_to_save = variables_to_save

    def write_intermediate_files(self):
        """
        Creates the intermediate files to be used in the batches.
        """

        os.makedirs(os.path.join(self.temporary_folder.name, str(self.counter)), exist_ok=True)

        for variable_name, variable_format in self._variables_to_save:
            if variable_format in JSONWriter.file_types():
                file_path = os.path.join(self.temporary_folder.name, str(self.counter),
                                         f"{variable_name}.json")

                variable = getattr(self._cls, variable_name)
                if variable is not None:
                    write_json(file_path, variable)

            elif variable_format in CSVWriter.file_types():
                file_path = os.path.join(self.temporary_folder.name, str(self.counter),
                                         f"{variable_name}.csv")

                variable = getattr(self._cls, variable_name)
                if variable is not None:
                    write_csv(file_path, variable, index=False)

            elif variable_format in PickleWriter.file_types():
                file_path = os.path.join(self.temporary_folder.name, str(self.counter),
                                         f"{variable_name}.pkl")

                variable = getattr(self._cls, variable_name)
                if variable is not None:
                    write_pickle(file_path, variable)

            elif variable_format in H5Writer.file_types():
                file_path = os.path.join(self.temporary_folder.name, str(self.counter),
                                         f"{variable_name}.h5")

                variable = getattr(self._cls, variable_name)
                if variable is not None:
                    write_h5(variable, file_path)

    def read_intermediate_files(self, subject: 'Dataset') -> bool:
        """
        Reads the intermediate files to be used in the batches.

        Parameters
        ----------
        subject: Subject
            the subject that is being observed

        Returns
        -------
        bool
            True if the files were read, False otherwise
        """

        if not os.path.exists(os.path.join(self.temporary_folder.name, str(self.counter))):
            return False

        for variable_name, variable_format in self._variables_to_save:
            if variable_format in JSONWriter.file_types():
                file_path = os.path.join(self.temporary_folder.name, str(self.counter),
                                         f"{variable_name}.json")

                variable = getattr(self._cls, variable_name)
                if variable is not None:
                    _variable = read_json(file_path)
                    setattr(subject, variable_name, _variable)
            elif variable_format in CSVReader.file_types():
                file_path = os.path.join(self.temporary_folder.name, str(self.counter),
                                         f"{variable_name}.csv")

                variable = getattr(self._cls, variable_name)
                if variable is not None:
                    _variable = read_csv(file_path)
                    setattr(subject, variable_name, _variable)
            elif variable_format in PickleReader.file_types():
                file_path = os.path.join(self.temporary_folder.name, str(self.counter),
                                         f"{variable_name}.pkl")

                variable = getattr(self._cls, variable_name)
                if variable is not None:
                    _variable = read_pickle(file_path)
                    setattr(subject, variable_name, _variable)
            elif variable_format in H5Reader.file_types():
                file_path = os.path.join(self.temporary_folder.name, str(self.counter),
                                         f"{variable_name}.h5")

                variable = getattr(self._cls, variable_name)
                if variable is not None:
                    _variable = read_h5(file_path)
                    setattr(subject, variable_name, _variable)

        return True
