import os
import pickle
from typing import List, Tuple
import numpy as np
import torch
import logging
from plants_sm.pathway_prediction._mcts_a_utils import (
    MinMaxStats, Node, ValueEnsemble,
    prepare_starting_molecules_natural, time_limit, value_fn
)
from plants_sm.pathway_prediction.entities import Molecule
from plants_sm.pathway_prediction.precursor_prediction import export_precursors
from plants_sm.pathway_prediction.reactor import Reactor
from plants_sm.pathway_prediction.searcher import Searcher, SearcherSolution
from plants_sm.pathway_prediction.solution import ReactionSolution

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(filename='myapp.log', level=logging.INFO)
logger = logging.getLogger(__name__)

class MCTS_A(Searcher):
    """Monte Carlo Tree Search (MCTS) with AlphaZero-style enhancements for retrosynthetic pathway planning.

    This class implements a MCTS-based search algorithm to find synthetic pathways for a target molecule,
    using a value network to guide the search and a set of reactors to expand the search tree.

    Attributes
    ----------
    value_net_path : str
        Path to the pretrained value network model.
    building_blocks_path : str
        Path to the file containing known starting molecules.
    value_model : ValueEnsemble
        The value network used to evaluate states.
    device : torch.device
        The device (CPU/GPU) on which to run the model.
    cpuct : float
        The exploration constant for PUCT.
    reactors : List[Reactor]
        List of reactors used to generate possible reactions.
    times : int
        Maximum number of iterations for the search.
    known_mols : set
        Set of known starting molecules (building blocks).
    visited_policy : dict
        Cache of policies (reaction solutions) for visited molecules.
    visited_state : list
        List of visited states to avoid cycles.
    opening_size : int
        Number of parallel openings to consider during selection.
    iterations : int
        Current number of iterations performed.
    target_mol : str
        SMILES string of the target molecule.
    """

    value_net_path: str = os.path.join(
        BASE_DIR, "pathway_prediction", "mcts_a_utils", "value_pc.pt"
    )
    building_blocks_path: str = os.path.join(
        BASE_DIR, "pathway_prediction", "mcts_a_utils", "building_blocks.txt"
    )
    building_blocks_path_from_predictor: str = os.path.join(
        BASE_DIR, "pathway_prediction", "precursor_prediction", "building_blocks.txt"
    )

    def __init__(
        self,
        reactors: List[Reactor],
        device: torch.device,
        simulations: int = 100,
        cpuct: float = 4.0,
        times: int = 3000,
        predict_precursors: bool = False
    ):
        """Initialize the MCTS_A searcher.

        Parameters
        ----------
        reactors : List[Reactor]
            List of reactors to use for reaction prediction.
        device : torch.device
            Device to run the value model on.
        simulations : int, optional
            Number of parallel openings to consider during selection. Default is 100.
        cpuct : float, optional
            Exploration constant for PUCT. Default is 4.0.
        times : int, optional
            Maximum number of iterations for the search. Default is 3000.
        predict_precursors : bool, optional
            Whether to use the building blocks from the precursor predictor. Default is False.
        """
        self.value_model = ValueEnsemble(2048, 128, 0.1).to(device)
        self.value_model.load_state_dict(
            torch.load(self.value_net_path, map_location=device)
        )
        self.value_model.eval()
        self.device = device
        self.cpuct = cpuct
        self.reactors = reactors
        self.times = times
        self.predict_precursors = predict_precursors
        self.visited_policy = {}
        self.visited_state = []
        self.opening_size = simulations
        self.iterations = 0
        self.target_mol = None

    def search(self, molecule: Molecule) -> SearcherSolution:
        """Search for a synthetic pathway to the target molecule.

        Parameters
        ----------
        molecule : Molecule
            The target molecule to synthesize.

        Returns
        -------
        SearcherSolution
            The result of the search, including the synthetic route, templates, success status, depth, and counts.
        """
        routes = []
        templates = []
        successes = []
        depths = []
        counts = []
        self.target_mol = molecule.smiles

        if self.predict_precursors:
            result = export_precursors(molecule.smiles)
            if result:
                self.known_mols = prepare_starting_molecules_natural(self.building_blocks_path_from_predictor)
            else:
                self.known_mols = prepare_starting_molecules_natural(self.building_blocks_path)
        else:
            self.known_mols = prepare_starting_molecules_natural(self.building_blocks_path)

        with time_limit(600):
            success, node, count = self._search(molecule)
            route, template, reaction_solutions = self.vis_synthetic_path(node)

        routes.append(route)
        templates.append(template)
        successes.append(success)

        if success:
            depths.append(node.depth)
            counts.append(count)
        else:
            depths.append(32)
            counts.append(-1)

        ans = {
            'route': routes,
            'template': templates,
            'success': successes,
            'depth': depths,
            'counts': counts,
            "reaction_solutions": reaction_solutions
        }
        logging.info(
            f"Search completed with success: {success}, depth: {depths[0]}, count: {counts[0]}"
        )
        logging.info(node.state)
        return SearcherSolution(**ans)

    def _search(self, molecule: Molecule) -> Tuple[bool, Node, int]:
        """Internal search loop for MCTS.

        Parameters
        ----------
        molecule : Molecule
            The target molecule to synthesize.

        Returns
        -------
        Tuple[bool, Node, int]
            A tuple containing success status, the terminal node, and the number of iterations.
        """
        root_value = value_fn(self.value_model, [molecule.smiles], self.device)
        logging.info(
            f"Starting MCTS search for molecule: {molecule.smiles} with root value: {root_value}"
        )
        self.root = Node(
            [molecule.smiles],
            root_value,
            prior=1.0,
            cpuct=self.cpuct,
            reaction_solution=None
        )
        self.open = [self.root]
        self.min_max_stats = MinMaxStats()
        self.min_max_stats.update(self.root.f)

        success, node = False, None
        while (
            self.iterations < self.times
            and not success
            and (not np.all(self.root.child_illegal > 0) or len(self.root.child_illegal) == 0)
        ):
            expand_node = self.select()
            if '.'.join(expand_node.state) in self.visited_state:
                expand_node.parent.child_illegal[expand_node.fmove] = 1000
                back_check_node = expand_node.parent
                while (
                    back_check_node is not None
                    and back_check_node.parent is not None
                    and np.all(back_check_node.child_illegal > 0)
                ):
                    back_check_node.parent.child_illegal[back_check_node.fmove] = 1000
                    back_check_node = back_check_node.parent
                continue
            else:
                logging.info(f"State {expand_node.state} is being expanded.")
                self.visited_state.append('.'.join(expand_node.state))
                success, node = self.expand(expand_node)
                self.update(expand_node)
            if self.visited_policy.get(self.target_mol, None) is None:
                return False, None, self.times
        return success, node, self.iterations

    def select_a_leaf(self) -> Node:
        """Select a leaf node for expansion using PUCT.

        Returns
        -------
        Node
            The selected leaf node.
        """
        current = self.root
        while True:
            current.visited_time += 1
            if not current.is_expanded:
                return current
            best_move = current.select_child(self.min_max_stats)
            current = current.children[best_move]

    def select(self) -> Node:
        """Select the most promising node from multiple parallel openings.

        Returns
        -------
        Node
            The selected node.
        """
        openings = [self.select_a_leaf() for _ in range(self.opening_size)]
        stats = [opening.f for opening in openings]
        index = np.argmin(stats)
        return openings[index]

    def run_reactor(self, molecule: Molecule) -> List[ReactionSolution]:
        """Run all reactors on a molecule to generate possible reaction solutions.

        Parameters
        ----------
        molecule : Molecule
            The molecule to react.

        Returns
        -------
        List[ReactionSolution]
            List of reaction solutions.
        """
        solutions = []
        for reactor in self.reactors:
            solutions.extend(reactor.react([molecule.mol]))
        return solutions

    def expand(self, node: Node) -> Tuple[bool, Node]:
        """Expand a node by generating child nodes from possible reactions.

        Parameters
        ----------
        node : Node
            The node to expand.

        Returns
        -------
        Tuple[bool, Node]
            A tuple containing success status and the expanded child node.
        """
        node.is_expanded = True
        expanded_mol_index = 0
        expanded_mol = node.state[expanded_mol_index]

        if expanded_mol in self.visited_policy:
            expanded_policy = self.visited_policy[expanded_mol]
        else:
            expanded_policy = self.run_reactor(Molecule.from_smiles(expanded_mol))
            self.iterations += 1
            if expanded_policy is not None and len(expanded_policy) > 0:
                self.visited_policy[expanded_mol] = expanded_policy.copy()
            else:
                self.visited_policy[expanded_mol] = None

        if expanded_policy is not None and len(expanded_policy) > 0:
            node.child_illegal = np.array([0] * len(expanded_policy))
            for i in range(len(expanded_policy)):
                reactant = [
                    r for r in expanded_policy[i].get_reactants_smiles()
                    if r not in self.known_mols
                ]
                reactant = reactant + node.state[:expanded_mol_index] + node.state[expanded_mol_index + 1:]
                reactant = sorted(list(set(reactant)))
                cost = -np.log(np.clip(expanded_policy[i].score, 1e-3, 1.0))
                template = str(expanded_policy[i].reaction)
                reaction = ".".join(expanded_policy[i].get_reactants_smiles()) + '>>' + expanded_mol
                priors = np.array([1.0 / len(expanded_policy)] * len(expanded_policy))

                if len(reactant) == 0:
                    child = Node(
                        [],
                        0,
                        reaction_solution=expanded_policy[i],
                        cost=cost,
                        prior=priors[i],
                        action_mol=expanded_mol,
                        reaction=reaction,
                        fmove=len(node.children),
                        template=template,
                        parent=node,
                        cpuct=self.cpuct
                    )
                    return True, child
                else:
                    h = value_fn(self.value_model, reactant, self.device)
                    child = Node(
                        reactant,
                        h,
                        reaction_solution=expanded_policy[i],
                        cost=cost,
                        prior=priors[i],
                        action_mol=expanded_mol,
                        reaction=reaction,
                        fmove=len(node.children),
                        template=template,
                        parent=node,
                        cpuct=self.cpuct
                    )
                    if '.'.join(reactant) in self.visited_state:
                        node.child_illegal[child.fmove] = 1000
                        back_check_node = node
                        while (
                            back_check_node.parent is not None
                            and np.all(back_check_node.child_illegal > 0)
                        ):
                            back_check_node.parent.child_illegal[back_check_node.fmove] = 1000
                            back_check_node = back_check_node.parent
        else:
            if node is not None and node.parent is not None:
                node.parent.child_illegal[node.fmove] = 1000
                back_check_node = node.parent
                while (
                    back_check_node is not None
                    and back_check_node.parent is not None
                    and np.all(back_check_node.child_illegal > 0)
                ):
                    back_check_node.parent.child_illegal[back_check_node.fmove] = 1000
                    back_check_node = back_check_node.parent
        return False, None

    def update(self, node: Node) -> None:
        """Update the statistics for the nodes along the path from the leaf to the root.

        Parameters
        ----------
        node : Node
            The leaf node to update from.
        """
        stat = node.f
        self.min_max_stats.update(stat)
        current = node
        while current is not None:
            current.f_mean_path.append(stat)
            current = current.parent

    def vis_synthetic_path(self, node: Node) -> Tuple[List[str], List[str], List[ReactionSolution]]:
        """Extract the synthetic pathway from the terminal node to the root.

        Parameters
        ----------
        node : Node
            The terminal node of the synthetic pathway.

        Returns
        -------
        Tuple[List[str], List[str], List[ReactionSolution]]
            A tuple containing the reaction path, template path, and reaction solutions.
        """
        if node is None:
            return [], [], []

        reaction_path = []
        template_path = []
        reaction_solutions = []
        current = node
        while current is not None:
            reaction_path.append(current.reaction)
            template_path.append(current.template)
            reaction_solutions.append(current.reaction_solution)
            current = current.parent
        return reaction_path[::-1], template_path[::-1], reaction_solutions[::-1]
