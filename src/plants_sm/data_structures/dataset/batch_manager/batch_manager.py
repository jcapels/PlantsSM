import os
from tempfile import TemporaryDirectory
from typing import Union, List, Tuple

from plants_sm.design_patterns.observer import Observer, Subject
from plants_sm.io import CSVWriter, write_csv
from plants_sm.io.json import JSONWriter, write_json


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
        self.counter = 0

    def __del__(self):
        """
        Deletes the temporary folder.
        """
        self.temporary_folder.cleanup()

    def update(self, subject: Subject) -> None:
        """
        Updates the batch manager.

        Parameters
        ----------
        subject: Subject
            the subject that is being observed
        """
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

    def create_intermediate_files(self):
        """
        Creates the intermediate files to be used in the batches.
        """

        for variable_name, variable_format in self._variables_to_save:
            if variable_format in JSONWriter.file_types:
                file_path = os.path.join(self.temporary_folder.name, f"{variable_name}_{self.counter}.json")
                write_json(getattr(self._cls, variable_name), file_path)
            elif variable_format in CSVWriter.file_types:
                file_path = os.path.join(self.temporary_folder.name, f"{variable_name}_{self.counter}.csv")
                write_csv(getattr(self._cls, variable_name), file_path)
