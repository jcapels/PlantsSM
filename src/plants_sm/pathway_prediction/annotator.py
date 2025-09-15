from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Union

from plants_sm.pathway_prediction.entities import BiologicalEntity
from plants_sm.pathway_prediction.solution import Solution

class AnnotatorLinker(ABC):
    """
    Abstract base class for linking annotations.
    """

    def __init__(self, annotators: List["Annotator"] = None, solutions: List[Solution] = None):

        if annotators is None and solutions is None:
            raise ValueError("Either annotators or solutions must be provided")
        
        if annotators == None:
            self.annotators = [] 
        else:
            self.annotators = annotators

        if solutions == None:
            self.solutions = []
        else:
            self.solutions = solutions

        assert len(self.solutions) + len(self.annotators) == 2 

        #create a dictionary where the key is the annotator name and the value is the annotator object or dataframe

    def link_annotations(self, entities_list: List[Union[List[BiologicalEntity], pd.DataFrame]] = []) -> List[Union["Solution", BiologicalEntity]]:
        
        for i, entities in enumerate(entities_list):
            solution = self.annotators[i].annotate(entities)
            self.solutions.append(solution)

        return self._link()
    
    @abstractmethod
    def _link(self):
        pass


class Annotator(ABC):
    """
    Abstract base class for annotating biological entities and predicting interactions.

    Attributes
    ----------
    solution : Solution, optional
        The solution object generated after annotation.
    invalid_entities : Union[List[BiologicalEntity], pd.DataFrame], optional
        Entities that failed validation.
    """

    solution: Solution = None
    invalid_entities: Union[List[BiologicalEntity], pd.DataFrame] = None

    @abstractmethod
    def validate_input(self, entities: Union[List[BiologicalEntity], pd.DataFrame]):
        """
        Validate the input entities.

        Parameters
        ----------
        entities : Union[List[BiologicalEntity], pd.DataFrame]
            Input entities to validate. Can be a list of BiologicalEntity objects or a pandas DataFrame.

        Returns
        -------
        Tuple[Union[List[BiologicalEntity], pd.DataFrame], Union[List[BiologicalEntity], pd.DataFrame]]
            A tuple containing valid entities and invalid entities.

        Raises
        ------
        NotImplementedError
            If not implemented by a subclass.
        """

    @abstractmethod
    def _annotate(self, entities: Union[List[BiologicalEntity], pd.DataFrame]) -> List[Solution]:
         """
        Annotate valid entities and generate solutions.

        Parameters
        ----------
        entities : Union[List[BiologicalEntity], pd.DataFrame]
            Valid entities to annotate.

        Returns
        -------
        List[Solution]
            A list of solution objects generated from the annotation process.

        Raises
        ------
        NotImplementedError
            If not implemented by a subclass.
        """

    def annotate(self, entities: Union[List[BiologicalEntity], pd.DataFrame]) -> List[Solution]:
        """
        Validate input entities and annotate the valid ones.

        Parameters
        ----------
        entities : Union[List[BiologicalEntity], pd.DataFrame]
            Input entities to validate and annotate.

        Returns
        -------
        List[Solution]
            A list of solution objects generated from the annotation process.

        Notes
        -----
        - Calls `validate_input` to separate valid and invalid entities.
        - Calls `_annotate` to generate solutions from valid entities.
        - Stores invalid entities in `self.invalid_entities`.
        - Stores solutions in `self.solution`.
        """
        
        valid_entities, self.invalid_entities = self.validate_input(entities)
        
        self.solution = self._annotate(valid_entities)

        return self.solution
    
    @abstractmethod
    def _convert_to_readable_format(self, file: str, format: str, **kwargs) -> Union[List[BiologicalEntity], pd.DataFrame]:
        """
        Convert an input file to a readable format.

        Parameters
        ----------
        file : str
            Path to the input file.
        format : str
            Format of the input file (e.g., "csv").
        **kwargs
            Additional keyword arguments for format-specific readers.

        Returns
        -------
        Union[List[BiologicalEntity], pd.DataFrame]
            Entities read from the file, either as a list of BiologicalEntity objects or a DataFrame.

        Raises
        ------
        NotImplementedError
            If not implemented by a subclass.
        ValueError
            If the specified format is not supported.
        """


    def annotate_from_file(self, file: str, format: str, **kwargs) -> Solution:
        """
        Annotate entities read from a file.

        Parameters
        ----------
        file : str
            Path to the input file.
        format : str
            Format of the input file (e.g., "csv").
        **kwargs
            Additional keyword arguments for file reading.

        Returns
        -------
        Solution
            The solution object generated from the annotation process.

        Notes
        -----
        - Uses `_convert_to_readable_format` to read the file.
        - Calls `annotate` to validate and annotate the entities.
        """

        entities = self._convert_to_readable_format(file, format, **kwargs)

        self.annotate(entities)

        return self.solution
    
    def _dataframe_from_csv(self, file: str, **kwargs) -> pd.DataFrame:
        """
        Read a CSV file into a pandas DataFrame.

        Parameters
        ----------
        file : str
            Path to the CSV file.
        **kwargs
            Additional keyword arguments for `pd.read_csv`.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the data from the CSV file.
        """

        dataset = pd.read_csv(file, **kwargs)

        return dataset

   

