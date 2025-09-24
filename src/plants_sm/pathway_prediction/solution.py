from abc import abstractmethod
import pandas as pd
from pydantic import BaseModel, validator
from typing import Any, Dict, List, Tuple, Union
from plants_sm.pathway_prediction.ec_numbers_annotator_utils._utils import fasta_to_dataframe
from plants_sm.pathway_prediction.entities import BiologicalEntity, Molecule, Protein, Reaction

def sort_solutions(property_name: str):
    """
    Decorator to sort solution dictionaries by score in descending order.

    Parameters
    ----------
    property_name : str
        The name of the property to sort.

    Returns
    -------
    function
        A decorator function for the Solution class.
    """
    def decorator(cls: 'Solution') -> 'Solution':
        original_init = cls.__init__
        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Sort the dictionary by score in descending order
            solutions = getattr(self, property_name)
            setattr(self, property_name, {k: sorted(v, key=lambda x: x[1], reverse=True) for k, v in solutions.items()})
        cls.__init__ = __init__
        return cls
    return decorator

class Solution(BaseModel):
    """
    Base class for pathway prediction solutions.

    Attributes
    ----------
    score : Any, optional
        The score associated with the solution.
    """
    score: Any = None

    @abstractmethod
    def to_csv(self, path: str):
        """
        Save the solution to a CSV file.

        Attributes
        ----------
        path : str
            The path to save the CSV file.
        """

class ReactionSolution(Solution):
    """
    A solution representing a chemical reaction.

    Attributes
    ----------
    reactants : List[Molecule]
        List of reactant molecules.
    products : List[Molecule]
        List of product molecules.
    reaction : Reaction
        The reaction object.
    ec_numbers : List[str], optional
        List of EC numbers associated with the reaction.
    """

    reactants: List[Molecule]
    products: List[Molecule]
    reaction: Reaction
    ec_numbers: List[str] = None
    enzymes_scores: Dict[str, float] = {}

    def get_reactants_smiles(self) -> List[str]:
        """
        Get the SMILES strings of the reactants.

        Returns
        -------
        List[str]
            A list of SMILES strings for the reactants.
        """
        return [reactant.smiles for reactant in self.reactants]

    def get_products_smiles(self) -> List[str]:
        """
        Get the SMILES strings of the products.

        Returns
        -------
        List[str]
            A list of SMILES strings for the products.
        """
        return [product.smiles for product in self.products]
    
    def to_csv(self, path):
        pass
        


@sort_solutions("substrate_protein_solutions")
class ESISolution(Solution):
    """
    A solution representing enzyme-substrate interactions.

    Attributes
    ----------
    substrate_protein_solutions : Dict[Union[str, int], List[Tuple[Union[str, int], float]]]
        Dictionary mapping substrate IDs to lists of (protein ID, score) tuples, sorted by score.
    dataframe_with_solutions: pd.DataFrame
        keep the dataset with solutions
    """

    substrate_protein_solutions: Dict[Union[str, int], List[Tuple[Union[str, int], float]]]
    dataframe_with_solutions: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True

    @validator('dataframe_with_solutions', pre=True, always=True, allow_reuse=True)
    @classmethod
    def validate_reaction(cls, value):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("dataframe_with_solutions must be a pandas DataFrame")
        return value
    
    def to_csv(self, path):
        self.dataframe_with_solutions.to_csv(path, index=False)

    def get_score(self, compound_id: str, highest: bool = False) -> Union[List[Tuple[str, float]], Tuple[str, float]]:
        """
        Get the score(s) for a specific substrate.

        Parameters
        ----------
        compound_id : str
            The ID of the substrate.
        highest : bool, default=False
            If True, return only the highest score. Otherwise, return all scores.

        Returns
        -------
        Union[List[Tuple[str, float]], Tuple[str, float]]
            The score(s) for the specified substrate ID and respective protein(s).

        Raises
        ------
        ValueError
            If the substrate does not exist in the solution set.
        """
        if compound_id in self.substrate_protein_solutions:
            scores = self.substrate_protein_solutions[compound_id]
            if highest:
                return scores[0]
            return scores
        else:
            raise ValueError("This substrate does not exist in our set of solutions")

class ECSolution(Solution):
    """
    A solution representing EC number annotations for biological entities.

    Attributes
    ----------
    entity_ec_1 : Dict[str, List[Tuple[str, float]]]
        Dictionary mapping entity IDs to lists of (EC1 number, score) tuples.
    entity_ec_2 : Dict[str, List[Tuple[str, float]]]
        Dictionary mapping entity IDs to lists of (EC2 number, score) tuples.
    entity_ec_3 : Dict[str, List[Tuple[str, float]]]
        Dictionary mapping entity IDs to lists of (EC3 number, score) tuples.
    entity_ec_4 : Dict[str, List[Tuple[str, float]]]
        Dictionary mapping entity IDs to lists of (EC4 number, score) tuples.
    entities : Dict[str, BiologicalEntity]
        Dictionary mapping entity IDs to BiologicalEntity objects.
    """

    entity_ec_1: Dict[str, List[Tuple[str, float]]]
    entity_ec_2: Dict[str, List[Tuple[str, float]]]
    entity_ec_3: Dict[str, List[Tuple[str, float]]]
    entity_ec_4: Dict[str, List[Tuple[str, float]]]
    entities: Dict[str, BiologicalEntity]

    @classmethod
    def from_csv_and_fasta(cls, csv_path: str, fasta_file: str, id_field: str):

        def _ec_score_breakdown(ec_column):
            result = []
            if isinstance(ec_column, str):
                different_ecs = ec_column.split(";")
                for ec_ in different_ecs:
                    ec_, score = ec_.split(":")
                    result.append((ec_, score))
                return result
            return []

        dataframe = pd.read_csv(csv_path)

        entity_ec_1 = {}
        entity_ec_2 = {}
        entity_ec_3 = {}
        entity_ec_4 = {}
        entities = {}

        entities_enzymes = fasta_to_dataframe(fasta_file)
        entities_enzymes.columns = [id_field, entities_enzymes.columns[1]]

        dataframe = pd.merge(dataframe, entities_enzymes, on=id_field).drop_duplicates()

        for _, row in dataframe.iterrows():
            id_ = row[id_field]

            EC1 = row["EC1"]
            EC2 = row["EC2"]
            EC3 = row["EC3"]
            EC4 = row["EC4"]
            entity_ec_1[id_] = _ec_score_breakdown(EC1)
            entity_ec_2[id_] = _ec_score_breakdown(EC2)
            entity_ec_3[id_] = _ec_score_breakdown(EC3)
            entity_ec_4[id_] = _ec_score_breakdown(EC4)
            entities[id_] = Protein.from_sequence(row["sequence"])

        return cls(entity_ec_1=entity_ec_1,
                   entity_ec_2=entity_ec_2,
                   entity_ec_3=entity_ec_3,
                   entity_ec_4=entity_ec_4,
                   entities=entities)
    
    def to_csv(self, csv_path: str, id_field: str = "id") -> None:
        """Export entity data, including EC numbers and representations, to a CSV file.

        For each entity, this method collects its EC numbers (EC1, EC2, EC3, EC4) and representation,
        formats them, and writes the result to a CSV file.

        Parameters
        ----------
        csv_path : str
            Path to the output CSV file.
        id_field : str, optional
            Name of the column for entity IDs in the CSV. Default is "id".

        Notes
        -----
        - EC numbers are formatted as "EC:score" and joined by semicolons for each EC class.
        - The representation is taken from the `representation` attribute of each entity.
        - The output CSV will have columns: `id_field`, "representations", "ec1", "ec2", "ec3", "ec4".
        """
        # Prepare data for DataFrame
        ids = []
        representations = []
        ec1_list = []
        ec2_list = []
        ec3_list = []
        ec4_list = []
        for entity_id in self.entities:
            # Get EC breakdowns for each level
            ec1 = self.entity_ec_1.get(entity_id, [])
            ec2 = self.entity_ec_2.get(entity_id, [])
            ec3 = self.entity_ec_3.get(entity_id, [])
            ec4 = self.entity_ec_4.get(entity_id, [])
            # Reconstruct EC strings (e.g., "EC1:score;EC2:score")
            ec1_str = ";".join([f"{ec}:{score}" for ec, score in ec1])
            ec2_str = ";".join([f"{ec}:{score}" for ec, score in ec2])
            ec3_str = ";".join([f"{ec}:{score}" for ec, score in ec3])
            ec4_str = ";".join([f"{ec}:{score}" for ec, score in ec4])
            # Get sequence from the Protein entity
            representation = self.entities[entity_id].representation
            ids.append(entity_id)
            ec1_list.append(ec1_str)
            ec2_list.append(ec2_str)
            ec3_list.append(ec3_str)
            ec4_list.append(ec4_str)
            representations.append(representation)
        pd.DataFrame({
            id_field: ids,
            "representations": representations,
            "ec1": ec1_list,
            "ec2": ec2_list,
            "ec3": ec3_list,
            "ec4": ec4_list,
        }).to_csv(csv_path, index=False)

        
    def get_ecs(self, entity_id: str, ec_number: str) -> List[Union[str, None]]:
        """Retrieve enzyme commission (EC) numbers associated with a given entity and EC class.

        Parameters
        ----------
        entity_id : str
            The identifier of the entity (e.g., molecule, reaction) for which to retrieve EC numbers.
        ec_number : str
            The EC class to query. Must be one of: "EC1", "EC2", "EC3", "EC4".

        Returns
        -------
        List[Union[str, None]]
            A list of EC numbers associated with the entity for the specified EC class.
            Returns an empty list if no EC numbers are found.

        Raises
        ------
        ValueError
            If `ec_number` is not one of the allowed values ("EC1", "EC2", "EC3", "EC4").
        """
        if ec_number == "EC1":
            ecs = self.entity_ec_1.get(entity_id, [])
            return [ec_ for ec_, _ in ecs]
        elif ec_number == "EC2":
            ecs = self.entity_ec_2.get(entity_id, [])
            return [ec_ for ec_, _ in ecs]
        elif ec_number == "EC3":
            ecs = self.entity_ec_3.get(entity_id, [])
            return [ec_ for ec_, _ in ecs]
        elif ec_number == "EC4":
            ecs = self.entity_ec_4.get(entity_id, [])
            return [ec_ for ec_, _ in ecs]
        else:
            raise ValueError(f"Unknown EC number: {ec_number}, available options are EC1, EC2, EC3, EC4.")


    def get_score(self, entity_id: str, ec_number: str) -> List[Tuple[str, float]]:
        """
        Get the score(s) for a specific EC number and entity.

        Parameters
        ----------
        entity_id : str
            The ID of the biological entity.
        ec_number : str
            The EC number level to get the score for (EC1, EC2, EC3, or EC4).

        Returns
        -------
        List[Tuple[str, float]]
            The list of (EC number, score) tuples for the specified entity and EC level.

        Raises
        ------
        ValueError
            If the EC number is not one of EC1, EC2, EC3, or EC4.
        """
        if ec_number == "EC1":
            return self.entity_ec_1.get(entity_id, [])
        elif ec_number == "EC2":
            return self.entity_ec_2.get(entity_id, [])
        elif ec_number == "EC3":
            return self.entity_ec_3.get(entity_id, [])
        elif ec_number == "EC4":
            return self.entity_ec_4.get(entity_id, [])
        else:
            raise ValueError(f"Unknown EC number: {ec_number}, available options are EC1, EC2, EC3, EC4.")
