import os
from tempfile import TemporaryDirectory
from typing import Union, List, Tuple

from pandas import read_csv

from plants_sm.design_patterns.observer import Observer, Subject
from plants_sm.io import CSVWriter, write_csv, CSVReader
from plants_sm.io.json import JSONWriter, write_json, read_json


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
        self.temporary_folder = TemporaryDirectory()
        self.counter = batch_size

    def __del__(self):
        """
        Deletes the temporary folder.
        """
        self.temporary_folder.cleanup()

    def end(self):
        """
        Resets the counter.
        """
        self.counter = self.batch_size

    def update(self, subject: Subject, **kwargs) -> None:
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
            subject._batch_state = self.read_intermediate_files(subject)
            self.counter += self.batch_size

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
                    write_csv(file_path, variable)

    def read_intermediate_files(self, subject: Subject) -> bool:
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

        return True
