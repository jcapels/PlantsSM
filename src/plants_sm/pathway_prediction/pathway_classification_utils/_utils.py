from Bio.KEGG import REST as kegg_api

def get_ec_numbers_from_ko_pathway(ko_pathway_id):
    ko_pathway_id = "ko" + ko_pathway_id[3:]
    # Fetch the KO pathway file
    ko_pathway_data = kegg_api.kegg_get(ko_pathway_id).read().split('\n')

    ec_numbers = set()
    in_orthology_section = False

    for line in ko_pathway_data:
        if line.startswith('ORTHOLOGY'):
            in_orthology_section = True
            continue
        if in_orthology_section and line.startswith('///'):
            break
        if in_orthology_section and '[EC:' in line:
            # Extract all EC numbers from the line
            ec_part = line.split('[EC:')[1:]
            for part in ec_part:
                ec = part.split(']')[0]
                ec_numbers.update(ec.split())

    return sorted(ec_numbers)