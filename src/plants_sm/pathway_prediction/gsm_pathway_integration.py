import os
import re

import urllib

import pandas as pd
from plants_sm.io.pickle import read_pickle
from plants_sm.pathway_prediction.esi_annotator import ProtBertESIAnnotator
from plants_sm.pathway_prediction.pathway_classification_utils._utils import get_ec_numbers_from_ko_pathway, get_reactions_by_ec
from plants_sm.pathway_prediction.solution import ECSolution
from Bio.KEGG import REST as kegg_api

from cobra import Reaction, Metabolite, Model, Gene
import cobra
import requests
from typing import List, Set

from plants_sm.pathway_prediction.pathway_prediction import KEGG_PATHWAYS, PlantPathwayClassifier

class GSMPathwayAnnotation:
    """
    Class for integrating GSM pathway annotations.
    """

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    plantcyc_reactions_dict_path = os.path.join(BASE_DIR,
                                    "pathway_prediction",
                                    "pathway_classification_utils",
                                    "plantcyc_reactions_dict.pkl")
    
    plantcyc_pathways_to_reaction_path = os.path.join(BASE_DIR,
                                    "pathway_prediction",
                                    "pathway_classification_utils",
                                    "plantcyc_pathway_reactions.pkl")
    
    plantcyc_reaction_to_ec_path = os.path.join(BASE_DIR,
                                    "pathway_prediction",
                                    "pathway_classification_utils",
                                    "plantcyc_reactions_ec.pkl")

    plantcyc_compounds_to_pathways_path = os.path.join(BASE_DIR,
                                    "pathway_prediction",
                                    "pathway_classification_utils",
                                    "plantcyc_compounds_to_pathways.pkl")
    
    plantcyc_compounds_information_path = os.path.join(BASE_DIR,
                                    "pathway_prediction",
                                    "pathway_classification_utils",
                                    "plantcyc_compounds_info.pkl")

    def __init__(self, gsm_model: Model, model_type: str = "KEGG", esi_annotator=ProtBertESIAnnotator()):
        self.gsm_model = cobra.io.read_sbml_model(gsm_model)
        self.model_type = model_type
        self.esi_annotator = esi_annotator

    @staticmethod
    def _extract_pathway_maps(kegg_entry: str):
        """
        Extract pathway maps (e.g., 'map00901') from a KEGG compound entry.
        """
        lines = kegg_entry.splitlines()

        pathway_section = []
        inside_pathway = False

        for line in lines:
            # Start of PATHWAY section
            if line.startswith("PATHWAY"):
                inside_pathway = True
                # First line: remove the "PATHWAY" label and capture the rest
                content = line[len("PATHWAY"):].strip()
                if content:
                    pathway_section.append(content)
                continue

            # If inside PATHWAY, capture indented lines until a non-indented, non-empty field appears
            if inside_pathway:
                # PATHWAY continuation lines start with spaces
                if re.match(r'\s{12,}', line):
                    pathway_section.append(line.strip())
                else:
                    break  # End of PATHWAY section
        # Extract the map IDs like map00901, map01063, etc.
        map_ids = []
        for line in pathway_section:
            match = re.match(r'(map\d+)', line)
            if match and match.group(1) not in map_ids:
                map_ids.append(match.group(1))

        return map_ids
    
    def _link_substrate_to_enzymes(self, compound_ids: List[str], compound_smiles: List[str], 
                                   protein_ids: List[str], protein_sequences: List[str]):

        # make combinations of protein ids/sequences and compound ids/smiles and create a dataframe of all
        interaction_protein_ids = []
        interaction_compound_ids = []
        interaction_compound_smiles = []
        interaction_protein_sequences = []
        for p_id, p_seq in zip(protein_ids, protein_sequences):
            for c_id, c_smiles in zip(compound_ids, compound_smiles):
                interaction_protein_ids.append(p_id)
                interaction_protein_sequences.append(p_seq)
                interaction_compound_ids.append(c_id)
                interaction_compound_smiles.append(c_smiles)

        entities = pd.DataFrame({
            'protein_ids': interaction_protein_ids,
            'protein sequence': interaction_protein_sequences,
            'compound_ids': interaction_compound_ids,
            'compound smiles': interaction_compound_smiles,
        })

        results = self.esi_annotator.annotate(entities)
        solutions = results.dataframe_with_solutions
        # Sort by 'proba' in descending order
        solutions = solutions.sort_values(by="proba", ascending=False)

        return solutions
        

    def _get_kegg_compound_pathways(self, compound: str):
        if compound.startswith("C"):
            pathway_data = kegg_api.kegg_get(compound).read()
            pathways = self._extract_pathway_maps(pathway_data)
            for pathway in pathways:
                if pathway not in KEGG_PATHWAYS:
                    pathways.remove(pathway)
        else:
            pathways = PlantPathwayClassifier().predict(input_smiles=compound)

        return pathways
    
    def _get_reactions_in_pathway(self, pathway_id: str):
        ko_pathway_id = "ko" + pathway_id[3:]
        reactions = set()
        try:
            ko_pathway_data = kegg_api.kegg_get(ko_pathway_id).read().split('\n')
            in_orthology_section = False

            ks = set()

            for line in ko_pathway_data:
                if line.startswith('ORTHOLOGY'):
                    in_orthology_section = True
                    continue
                if in_orthology_section and line.startswith('///'):
                    break
                if in_orthology_section and 'K' in line:
                    # Extract all EC numbers from the line
                    pattern = r'\bK\d{5}\b'
                    matches = re.findall(pattern, line)
                    for k in matches:
                        ks.add(k)
            
            for k in ks:
                link_handle = kegg_api.kegg_link("reaction", k)
                pattern = r"R\d+"
                reaction_ids = re.findall(pattern, link_handle.read())
                for reaction in reaction_ids:
                    reactions.add(reaction)
            
        except urllib.error.HTTPError:
            pass
        
        return reactions
    
    def integrate_pathway_annotations(self, compound: str, ec_solution: 'ECSolution', solutions_path: str = None) -> Set[str]:
        """
        Integrates KEGG pathway annotations into a cobrapy model.
        Fetches reactions and metabolites from KEGG and adds them to the model if missing.
        """
        reactions = set()
        reactions_genes = {}
        if solutions_path:
            os.makedirs(solutions_path, exist_ok=True)
        ec_solution.create_ec_to_entities()
        reaction_to_ec = {}
        if self.model_type == "KEGG":
            reactions_in_proteome = set()
            pathways = self._get_kegg_compound_pathways(compound)
            for pathway in pathways:
                pathway_reactions = self._get_reactions_in_pathway(pathway)
                reactions.update(pathway_reactions)

                list_ecs = get_ec_numbers_from_ko_pathway(pathway)
                list_ecs = [ec for ec in list_ecs if ec_solution.get_entities(ec)]
                for ec in list_ecs:
                    kegg_reactions = get_reactions_by_ec(ec)
                    reaction_to_ec.update({r: ec for r in kegg_reactions})
                    reactions_in_proteome.update(kegg_reactions)
                    entities = ec_solution.get_entities(ec)
                    for reaction in kegg_reactions:
                        reactions_genes[reaction] = [entity[0] for entity in entities]

            # Only keep reactions present in both pathway and proteome
            reactions = reactions.intersection(reactions_in_proteome)

            # Add missing reactions and metabolites to the model
            for r_id in reactions:
                if r_id not in self.gsm_model.reactions:
                    kegg_reaction_data = self._fetch_kegg_reaction(r_id)
                    if kegg_reaction_data:
                        genes = reactions_genes.get(r_id, [])
                        ec_numbers = reaction_to_ec.get(r_id, [])
                        level_4 = True
                        for ec in ec_numbers:
                            if re.match(r'^\d+\.\d+\.\d+\.\d+$', ec) is None:
                                level_4 = False
                        self._add_reaction_to_model(r_id, kegg_reaction_data, genes, {}, level4 = level_4,
                                                    ec_solution=ec_solution, solutions_path=solutions_path)

        elif self.model_type == "PlantCyc":
            pathways_to_reaction = read_pickle(self.plantcyc_pathways_to_reaction_path)
            reaction_to_ec = read_pickle(self.plantcyc_reaction_to_ec_path)
            reaction_information = read_pickle(self.plantcyc_reactions_dict_path)
            compounds_to_pathways = read_pickle(self.plantcyc_compounds_to_pathways_path)
            compounds_info = read_pickle(self.plantcyc_compounds_information_path)
            pathways = compounds_to_pathways.get(compound, [])
            if not pathways:
                pathways = PlantPathwayClassifier(classification_type="PlantCyc").predict(input_smiles=compound)
            
            for pathway in pathways:
                reactions_ = pathways_to_reaction[pathway]
                for reaction in reactions_:
                    if reaction in reaction_to_ec:
                        reactions.update(reactions_)

            #write code to add reactions to the model
            for r_id in reactions:
                if r_id not in self.gsm_model.reactions:
                    reaction_data = reaction_information.get(r_id, None)
                    if reaction_data:
                        genes = []
                        ec_numbers = reaction_to_ec.get(r_id, [])
                        level_4 = True
                        for ec in ec_numbers:
                            entities = ec_solution.get_entities(ec)
                            for entity in entities:
                                genes.append(entity[0])
                            if re.match(r'^\d+\.\d+\.\d+\.\d+$', ec) is None:
                                level_4 = False
                        self._add_reaction_to_model(r_id, reaction_data, genes, compounds_info, level4 = level_4,
                                                    ec_solution=ec_solution, solutions_path=solutions_path)
        return reactions

    def _fetch_kegg_reaction(self, reaction_id: str) -> dict:
        """
        Fetches reaction data from KEGG API.
        Returns a dictionary with reaction equation, metabolites, and other metadata.
        """
        url = f"http://rest.kegg.jp/get/{reaction_id}"
        response = requests.get(url)
        if response.status_code == 200:
            return self._parse_kegg_reaction(response.text)
        return None

    def _parse_kegg_reaction(self, kegg_reaction_text: str) -> dict:
        """
        Parses KEGG reaction text into a structured dictionary.
        Example:
            "EQUATION: A + B <=> C + D"
            "NAME: reaction_name"
            "ENZYME: 1.2.3.4"
        """
        data = {}
        for line in kegg_reaction_text.split('\n'):
            if line.startswith('EQUATION'):
                data['equation'] = line.split('EQUATION ')[1].strip()
            elif line.startswith('NAME'):
                line = " ".join(line.split())
                data['name'] = line.split('NAME ')[1].strip()
            elif line.startswith('ENZYME'):
                data['enzyme'] = line.split('ENZYME ')[1].strip()
        return data

    def _add_reaction_to_model(self, reaction_id: str, reaction_data: dict, genes: list, compounds_info: dict, 
                               level4: bool, ec_solution: 'ECSolution' =None, solutions_path: str = None):
        """
        Adds a reaction and its metabolites to the cobrapy model.
        """

        if self.model_type == "KEGG":
            # Parse equation into reactants and products
            equation = reaction_data['equation']
            reactants, products = equation.split('<=>')
            reactants = [m.strip() for m in reactants.split(' + ')]
            products = [m.strip() for m in products.split(' + ') if m.strip()]

            # Create a new cobra.Reaction
            reaction = Reaction(reaction_id, lower_bound=-1000.0, upper_bound=1000.0)
            reaction.name = reaction_data.get('name', reaction_id)
            reaction.subsystem = 'KEGG'

            metabolites_substrates = []

            # Add metabolites and stoichiometry
            for met_str in reactants:
                coeff, met_id, smiles = self._parse_metabolite_string(met_str)
                metabolites_substrates.append((met_id, smiles))
                metabolite = self._get_or_create_metabolite(met_id)
                reaction.add_metabolites({metabolite: coeff})

            for met_str in products:
                coeff, met_id, smiles = self._parse_metabolite_string(met_str, react=False)
                metabolite = self._get_or_create_metabolite(met_id)
                reaction.add_metabolites({metabolite: coeff})

            if not level4 and ec_solution:
                compound_ids = [met[0] for met in metabolites_substrates]
                compound_smiles = [met[1] for met in metabolites_substrates]
                protein_ids = []
                protein_sequences = []

                for gene in genes:
                    entity_info = ec_solution.entities[gene]
                    if entity_info:
                        protein_ids.append(gene)
                        protein_sequences.append(entity_info.sequence)

                if compound_ids and protein_ids:
                    linked_solutions = self._link_substrate_to_enzymes(compound_ids, compound_smiles, protein_ids, protein_sequences)
                    if not linked_solutions.empty:
                        top_solution = linked_solutions.iloc[:, 0]
                        if solutions_path:
                            linked_solutions.to_csv(f"{solutions_path}/solutions_{reaction_id}.csv", index=False)
                        genes = top_solution.unique()[:4].tolist()
                        print(genes)

        elif self.model_type == "PlantCyc":
            # reaction_data is expected to have 'reactants' and 'products' lists
            metabolites = reaction_data.get('metabolites')

            # Create a new cobra.Reaction
            reaction = Reaction(reaction_id, lower_bound=reaction_data['lower_bound'], upper_bound=1000.0)
            reaction.name = reaction_data.get('name', reaction_id)
            reaction.subsystem = 'PlantCyc'

            metabolites_substrates = []

            # Add metabolites and stoichiometry
            for metabolite in metabolites:
                coeff = metabolites[metabolite]

                if coeff < 0:
                    metabolites_substrates.append((metabolite, compounds_info[metabolite]["smiles"]))

                metabolite_obj = Metabolite(
                    id=metabolite,
                    name=f"Metabolite {metabolite}",
                    compartment='c'  
                )
                reaction.add_metabolites({metabolite_obj: coeff})
            
            if not level4 and ec_solution:
                compound_ids = [met[0] for met in metabolites_substrates]
                compound_smiles = [met[1] for met in metabolites_substrates]
                protein_ids = []
                protein_sequences = []

                for gene in genes:
                    entity_info = ec_solution.entities[gene]
                    if entity_info:
                        protein_ids.append(gene)
                        protein_sequences.append(entity_info.sequence)

                if compound_ids and protein_ids:
                    linked_solutions = self._link_substrate_to_enzymes(compound_ids, compound_smiles, protein_ids, protein_sequences)
                    if not linked_solutions.empty:
                        top_solution = linked_solutions.iloc[:, 0]
                        if solutions_path:
                            linked_solutions.to_csv(f"{solutions_path}/solutions_{reaction_id}.csv", index=False)
                        genes = top_solution.unique()[:4].tolist()
                        print(genes)

        for gene in genes:
            cobra_gene = Gene(gene, name=gene)
            if gene not in self.gsm_model.genes:
                self.gsm_model.genes.append(cobra_gene)

        reaction.gene_reaction_rule = " or ".join(genes)

        # Add reaction to model
        self.gsm_model.add_reactions([reaction])

    def _get_kegg_smiles(self, compound_id: str) -> str:
        import requests
        from rdkit import Chem

        # Else retrieve Molfile and convert
        mol_resp = requests.get(f"https://rest.kegg.jp/get/{compound_id}/mol")
        mol_text = mol_resp.text
        mol = Chem.MolFromMolBlock(mol_text)
        if mol:
            return Chem.MolToSmiles(mol)
        return None


    def _parse_metabolite_string(self, met_str: str, react=True) -> tuple:
        """
        Parses metabolite string (e.g., "2 C00001") into coefficient and metabolite ID.
        """
        parts = met_str.split()
        coeff = -1.0 if react else 1.0  # Default for reactants or products
        if len(parts) > 1 and parts[0].isdigit():
            coeff = -float(parts[0]) if react else float(parts[0])
            met_id = parts[1]
            smiles = self._get_kegg_smiles(met_id)
        else:
            met_id = parts[0]
            smiles = self._get_kegg_smiles(met_id)
        return coeff, met_id, smiles

    def _get_or_create_metabolite(self, metabolite_id: str) -> cobra.Metabolite:
        """
        Returns a metabolite from the model or creates a new one if missing.
        """
        for met in self.gsm_model.metabolites:
            if metabolite_id in met.id:
                return met
        
        metabolite = Metabolite(
            id=metabolite_id,
            name=f"Metabolite {metabolite_id}",
            compartment='c'  
        )
        self.gsm_model.add_metabolites([metabolite])
        return metabolite

            

