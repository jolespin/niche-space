# -*- coding: utf-8 -*-
import re
from collections import defaultdict
import pandas as pd
from Bio.KEGG.REST import kegg_get

from pyexeggutor import check_argument_choice

# Parse KEGG response
def _parse_kegg_response(id:str) -> dict:
    """Use REST API to fetch KEGG response and parse into dictionary

    Args:
        id (str): KEGG identifier

    Returns:
        dict: Dictionary of unformatted parsed KEGG responses
    """
    response = kegg_get(id).read()
    
    kegg_info = defaultdict(list)
    current_field = None
    
    for line in response.splitlines():
        if line.startswith('///'):
            continue
        if not line.startswith(" "):
            current_field, *line_parts = line.split(" ", 1)
            line = " ".join(line_parts)
        line = line.strip()
        if line.endswith(";"):
            line = line[:-1]
        kegg_info[current_field].append(line)
    return kegg_info
        
def _fetch_kegg_ortholog_info(id_ko: str, remove_fields=["GENES"], EC_PATTERN=r'\d+\.\d+\.\d+\.\d+|\d+\.\d+\.\d+\.\-'):

    """
    # Example usage:
    _fetch_kegg_ortholog_info("K00001")

    # Example output:
    {'ENTRY': 'K00001',
     'NAME': 'alcohol dehydrogenase [EC:1.1.1.1]',
     'ENZYMES': {'1.1.1.1'},
     'REACTIONS': {'R00623': 'primary_alcohol:NAD+ oxidoreductase',
      'R00754': 'ethanol:NAD+ oxidoreductase',
      'R02124': 'retinol:NAD+ oxidoreductase',
      'R04805': '',
      'R04880': '3,4-dihydroxyphenylethyleneglycol:NAD+ oxidoreductase',
      'R05233': 'trans-3-chloro-2-propene-1-ol:NAD+ oxidoreductase',
      'R05234': 'cis-3-chloro-2-propene-1-ol:NAD+ oxidoreductase',
      'R06917': '1-hydroxymethylnaphthalene:NAD+ oxidoreductase',
      'R06927': '(2-naphthyl)methanol:NAD+ oxidoreductase',
      'R07105': 'trichloroethanol:NAD+ oxidoreductase',
      'R08281': 'alcophosphamide:NAD+ oxidoreductase',
      'R08306': '2-phenyl-1,3-propanediol monocarbamate:NAD+ oxidoreductase',
      'R08310': '4-hydroxy-5-phenyltetrahydro-1,3-oxazin-2-one:NAD+ oxidoreductase'},
     'MODULES': {},
     'PATHWAYS': {'map00010': 'Glycolysis / Gluconeogenesis',
      'map00071': 'Fatty acid degradation',
      'map00350': 'Tyrosine metabolism',
      'map00620': 'Pyruvate metabolism',
      'map00625': 'Chloroalkane and chloroalkene degradation',
      'map00626': 'Naphthalene degradation',
      'map00830': 'Retinol metabolism',
      'map00980': 'Metabolism of xenobiotics by cytochrome P450',
      'map00982': 'Drug metabolism - cytochrome P450',
      'map01100': 'Metabolic pathways',
      'map01110': 'Biosynthesis of secondary metabolites',
      'map01120': 'Microbial metabolism in diverse environments',
      'map01220': 'Degradation of aromatic compounds'},
     'BRITE': ['KEGG Orthology (KO) [BR:ko00001]',
      '09100 Metabolism',
      '09101 Carbohydrate metabolism',
      '00010 Glycolysis / Gluconeogenesis',
      'K00001  E1.1.1.1, adh; alcohol dehydrogenase',
      '00620 Pyruvate metabolism',
      'K00001  E1.1.1.1, adh; alcohol dehydrogenase',
      '09103 Lipid metabolism',
      '00071 Fatty acid degradation',
      'K00001  E1.1.1.1, adh; alcohol dehydrogenase',
      '09105 Amino acid metabolism',
      '00350 Tyrosine metabolism',
      'K00001  E1.1.1.1, adh; alcohol dehydrogenase',
      '09108 Metabolism of cofactors and vitamins',
      '00830 Retinol metabolism',
      'K00001  E1.1.1.1, adh; alcohol dehydrogenase',
      '09111 Xenobiotics biodegradation and metabolism',
      '00625 Chloroalkane and chloroalkene degradation',
      'K00001  E1.1.1.1, adh; alcohol dehydrogenase',
      '00626 Naphthalene degradation',
      'K00001  E1.1.1.1, adh; alcohol dehydrogenase',
      '00980 Metabolism of xenobiotics by cytochrome P450',
      'K00001  E1.1.1.1, adh; alcohol dehydrogenase',
      '00982 Drug metabolism - cytochrome P450',
      'K00001  E1.1.1.1, adh; alcohol dehydrogenase',
      'Enzymes [BR:ko01000]',
      '1. Oxidoreductases',
      '1.1  Acting on the CH-OH group of donors',
      '1.1.1  With NAD+ or NADP+ as acceptor',
      '1.1.1.1  alcohol dehydrogenase',
      'K00001  E1.1.1.1, adh; alcohol dehydrogenase'],
     'SYMBOL': ['E1.1.1.1, adh'],
     'ENTRY_TYPE': 'KO'}

    """
    # Parse KEGG response
    kegg_info = _parse_kegg_response(id_ko)

    # Remove specified fields
    for field in remove_fields:
        if field in kegg_info:
            del kegg_info[field]

    kegg_info_formatted = {
        "ENTRY": "",
        "NAME": "",
        "ENZYMES": set(),
        "REACTIONS": {},
        "MODULES": {},
        "PATHWAYS": {},
        "BRITE": [],
        "SYMBOL": []
    }

    for k, v in kegg_info.items():
        if k == "ENTRY":
            items = list(filter(bool, v[0].split(" ")))
            id_ko, entry_type = items[0], items[-1]
            kegg_info_formatted["ENTRY"] = id_ko
            kegg_info_formatted["ENTRY_TYPE"] = entry_type
        elif k == "NAME":
            name_str = " ".join(v)
            kegg_info_formatted["NAME"] = name_str
            if "[EC:" in name_str:
                ec_numbers = re.findall(EC_PATTERN, name_str)
                kegg_info_formatted["ENZYMES"] = set(ec_numbers)
        elif k == "REACTION":
            kegg_info_formatted["REACTIONS"] = {
                id: " ".join(name_parts).strip()
                for line in v
                for id, *name_parts in [line.split(" ", 1)]
            }
        elif k == "MODULE":
            kegg_info_formatted["MODULES"] = {
                id: " ".join(name_parts).strip()
                for line in v
                for id, *name_parts in [line.split(" ", 1)]
            }
        elif k == "PATHWAY":
            kegg_info_formatted["PATHWAYS"] = {
                id: " ".join(name_parts).strip()
                for line in v
                for id, *name_parts in [line.split(" ", 1)]
            }
        elif k in {"BRITE", "SYMBOL"}:
            kegg_info_formatted[k].extend(v)

    return kegg_info_formatted

def _fetch_kegg_module_info(id_module: str):
    """
    # Example usage
    _fetch_kegg_module_info("M00532")

    # Example output:
    {'ENTRY': 'M00532',
     'NAME': 'Photorespiration',
     'DEFINITION': '(K01601-K01602) K19269 K11517 K03781 K14272 K00600 K00830 (K15893,K15919) K15918 K00281+K00605+K00382+K02437',
     'ORTHOLOGY': defaultdict(list,
                 {'K01601,K01602': ['ribulose-bisphosphate carboxylase [EC:4.1.1.39] [RN:R03140]'],
                  'K19269': ['phosphoglycolate phosphatase [EC:3.1.3.18] [RN:R01334]'],
                  'K11517': ['(S)-2-hydroxy-acid oxidase [EC:1.1.3.15] [RN:R00475]'],
                  'K03781': ['catalase [EC:1.11.1.6] [RN:R00009]'],
                  'K14272': ['glutamate--glyoxylate aminotransferase [EC:2.6.1.4] [RN:R00372]'],
                  'K00600': ['glycine hydroxymethyltransferase [EC:2.1.2.1] [RN:R00945]'],
                  'K00830': ['serine-glyoxylate transaminase [EC:2.6.1.45] [RN:R00588]'],
                  'K15893,K15919': ['hydroxypyruvate reductase [EC:1.1.1.29] [RN:R01388]'],
                  'K15918': ['D-glycerate 3-kinase [EC:2.7.1.31] [RN:R01514]'],
                  'K00281,K00605,K00382,K02437': ['glycine cleavage system [EC:1.4.1.27] [RN:R01221]']}),
     'CLASS': ['Pathway modules; Carbohydrate metabolism; Other carbohydrate metabolism'],
     'PATHWAYS': {'map00630': 'Glyoxylate and dicarboxylate metabolism',
      'map01200': 'Carbon metabolism',
      'map01100': 'Metabolic pathways',
      'map01110': 'Biosynthesis of secondary metabolites'},
     'REACTIONS': {'R03140': 'C01182 -> C00988 + C00197',
      'R01334': 'C00988 -> C00160',
      'R00475': 'C00160 + C00007 -> C00048 + C00027',
      'R00009': 'C00027 -> C00007',
      'R00372': 'C00048 + C00025 -> C00026 + C00037',
      'R00945': 'C00037 -> C00065',
      'R00588': 'C00065 + C00048 -> C00168 + C00037',
      'R01388': 'C00168 -> C00258',
      'R01514': 'C00258 -> C00197',
      'R01221': 'C00037 -> C00014'},
     'COMPOUNDS': {'C01182': 'D-Ribulose 1,5-bisphosphate',
      'C00988': '2-Phosphoglycolate',
      'C00197': '3-Phospho-D-glycerate',
      'C00160': 'Glycolate',
      'C00007': 'Oxygen',
      'C00048': 'Glyoxylate',
      'C00027': 'Hydrogen peroxide',
      'C00025': 'L-Glutamate',
      'C00026': '2-Oxoglutarate',
      'C00037': 'Glycine',
      'C00065': 'L-Serine',
      'C00168': 'Hydroxypyruvate',
      'C00258': 'D-Glycerate',
      'C00014': 'NH3'},
     'REFERENCES': ['PMID:19575589',
      'AUTHORS   Foyer CH, Bloom AJ, Queval G, Noctor G',
      'TITLE     Photorespiratory metabolism: genes, mutants, energetics, and redox signaling.',
      'JOURNAL   Annu Rev Plant Biol 60:455-84 (2009)',
      'DOI:10.1146/annurev.arplant.043008.091948'],
     'ENTRY_TYPE': 'Module'}
    """
    # Parse KEGG response
    kegg_info = _parse_kegg_response(id_module)

    kegg_info_formatted = {
        "ENTRY": "",
        "NAME": "",
        "DEFINITION": "",
        "ORTHOLOGY": defaultdict(list),
        "CLASS": [],
        "PATHWAYS": {},
        "REACTIONS": {},
        "COMPOUNDS": {},
        "REFERENCES": []
    }

    for k, v in kegg_info.items():
        if k == "ENTRY":
            items = list(filter(bool, v[0].split(" ")))
            id_module, entry_type = items[0], items[-1]
            kegg_info_formatted["ENTRY"] = id_module
            kegg_info_formatted["ENTRY_TYPE"] = entry_type
        elif k == "NAME":
            kegg_info_formatted["NAME"] = " ".join(v)
        elif k == "DEFINITION":
            kegg_info_formatted["DEFINITION"] = " ".join(v)
        elif k == "ORTHOLOGY":
            for item in v:
                id, *description_parts = item.split(" ", 1)
                description = " ".join(description_parts).strip()
                kegg_info_formatted["ORTHOLOGY"][id].append(description)
        elif k == "CLASS":
            kegg_info_formatted["CLASS"] = v #" ".join(v)
        elif k == "PATHWAY":
            kegg_info_formatted["PATHWAYS"] = {
                id: " ".join(name_parts).strip()
                for line in v
                for id, *name_parts in [line.split(" ", 1)]
            }
        elif k == "REACTION":
            kegg_info_formatted["REACTIONS"] = {
                id: " ".join(name_parts).strip()
                for line in v
                for id, *name_parts in [line.split(" ", 1)]
            }
        elif k == "COMPOUND":
            kegg_info_formatted["COMPOUNDS"] = {
                id: " ".join(name_parts).strip()
                for line in v
                for id, *name_parts in [line.split(" ", 1)]
            }
        elif k == "REFERENCE":
            kegg_info_formatted["REFERENCES"].extend(v)
    
    return kegg_info_formatted

def _fetch_kegg_pathway_info(id_pathway: str):
    """
    # Example usage: 
    _fetch_kegg_pathway_info("map00190")

    # Example output:
    {'ENTRY': 'map00190',
     'NAME': 'Oxidative phosphorylation',
     'CLASS': ['Metabolism', 'Energy metabolism'],
     'PATHWAY_MAP': {'map00190': 'Oxidative phosphorylation'},
     'MODULE': {'M00142': 'NADH:ubiquinone oxidoreductase, mitochondria [PATH:map00190]',
      'M00143': 'NADH dehydrogenase (ubiquinone) Fe-S protein/flavoprotein complex, mitochondria [PATH:map00190]',
      'M00144': 'NADH:quinone oxidoreductase, prokaryotes [PATH:map00190]',
      'M00145': 'NAD(P)H:quinone oxidoreductase, chloroplasts and cyanobacteria [PATH:map00190]',
      'M00146': 'NADH dehydrogenase (ubiquinone) 1 alpha subcomplex [PATH:map00190]',
      'M00147': 'NADH dehydrogenase (ubiquinone) 1 beta subcomplex [PATH:map00190]',
      'M00148': 'Succinate dehydrogenase (ubiquinone) [PATH:map00190]',
      'M00149': 'Succinate dehydrogenase, prokaryotes [PATH:map00190]',
      'M00150': 'Fumarate reductase, prokaryotes [PATH:map00190]',
      'M00151': 'Cytochrome bc1 complex respiratory unit [PATH:map00190]',
      'M00152': 'Cytochrome bc1 complex [PATH:map00190]',
      'M00153': 'Cytochrome bd ubiquinol oxidase [PATH:map00190]',
      'M00154': 'Cytochrome c oxidase [PATH:map00190]',
      'M00155': 'Cytochrome c oxidase, prokaryotes [PATH:map00190]',
      'M00156': 'Cytochrome c oxidase, cbb3-type [PATH:map00190]',
      'M00157': 'F-type ATPase, prokaryotes and chloroplasts [PATH:map00190]',
      'M00158': 'F-type ATPase, eukaryotes [PATH:map00190]',
      'M00159': 'V/A-type ATPase, prokaryotes [PATH:map00190]',
      'M00160': 'V-type ATPase, eukaryotes [PATH:map00190]',
      'M00416': 'Cytochrome aa3-600 menaquinol oxidase [PATH:map00190]',
      'M00417': 'Cytochrome o ubiquinol oxidase [PATH:map00190]'},
     'DBLINKS': {},
     'REFERENCES': ['PMID:16469879',
      'AUTHORS   Sazanov LA, Hinchliffe P.',
      'TITLE     Structure of the hydrophilic domain of respiratory complex I from Thermus thermophilus.',
      'JOURNAL   Science 311:1430-6 (2006)',
      'DOI:10.1126/science.1123809',
      'PMID:16584177',
      'AUTHORS   Hinchliffe P, Carroll J, Sazanov LA.',
      'TITLE     Identification of a novel subunit of respiratory complex I from Thermus thermophilus.',
      'JOURNAL   Biochemistry 45:4413-20 (2006)',
      'DOI:10.1021/bi0600998'],
     'REL_PATHWAY': {},
     'KO_PATHWAY': 'ko00190'}
    """
    # Parse KEGG response
    kegg_info = _parse_kegg_response(id_pathway)

    kegg_info_formatted = {
        "ENTRY": "",
        "NAME": "",
        "CLASS": [],
        "PATHWAY_MAP": {},
        "MODULE": {},
        "DBLINKS": {},
        "REFERENCES": [],
        "REL_PATHWAY": {},
        "KO_PATHWAY": ""
    }

    for k, v in kegg_info.items():
        if k == "ENTRY":
            items = list(filter(bool, v[0].split(" ")))
            id_pathway = items[0]
            kegg_info_formatted["ENTRY"] = id_pathway
        elif k == "NAME":
            kegg_info_formatted["NAME"] = " ".join(v)
        elif k == "CLASS":
            kegg_info_formatted["CLASS"] = list(filter(bool," ".join(v).split("; "))) #" ".join(v)
        elif k == "PATHWAY_MAP":
            kegg_info_formatted["PATHWAY_MAP"] = {
                id: " ".join(name_parts).strip()
                for line in v
                for id, *name_parts in [line.split(" ", 1)]
            }
        elif k == "MODULE":
            kegg_info_formatted["MODULE"] = {
                id: " ".join(name_parts).strip()
                for line in v
                for id, *name_parts in [line.split(" ", 1)]
            }
        elif k == "DBLINKS":
            kegg_info_formatted["DBLINKS"] = {
                id: " ".join(name_parts).strip()
                for line in v
                for id, *name_parts in [line.split(" ", 1)]
            }
        elif k == "REFERENCE":
            kegg_info_formatted["REFERENCES"].extend(v)
        elif k == "REL_PATHWAY":
            kegg_info_formatted["REL_PATHWAY"] = {
                id: " ".join(name_parts).strip()
                for line in v
                for id, *name_parts in [line.split(" ", 1)]
            }
        elif k == "KO_PATHWAY":
            kegg_info_formatted["KO_PATHWAY"] = v[0]

    return kegg_info_formatted

def _fetch_kegg_enzyme_info(id_enzyme: str):
    """
    # Example usage
    _fetch_kegg_enzyme_info("1.1.1.1")

    # Example output:
    {'ENTRY': '1.1.1.1',
     'NAME': ['alcohol dehydrogenase',
      'aldehyde reductase',
      'ADH',
      'alcohol dehydrogenase (NAD)',
      'aliphatic alcohol dehydrogenase',
      'ethanol dehydrogenase',
      'NAD-dependent alcohol dehydrogenase',
      'NAD-specific aromatic alcohol dehydrogenase',
      'NADH-alcohol dehydrogenase',
      'NADH-aldehyde dehydrogenase',
      'primary alcohol dehydrogenase',
      'yeast alcohol dehydrogenase'],
     'CLASS': ['Oxidoreductases',
      'Acting on the CH-OH group of donors',
      'With NAD+ or NADP+ as acceptor'],
     'SYSNAME': 'alcohol:NAD+ oxidoreductase',
     'REACTION': ['(1) a primary alcohol + NAD+ = an aldehyde + NADH + H+ [RN:R00623]',
      '(2) a secondary alcohol + NAD+ = a ketone + NADH + H+ [RN:R00624]'],
     'ALL_REAC': ['R00623 > R00754 R02124 R02878 R04805 R04880 R05233 R05234 R06917 R06927 R08281 R08306 R08557 R08558 R10783',
      'R00624 > R08310',
      '(other) R02246 R07105'],
     'SUBSTRATE': ['primary alcohol [CPD:C00226]',
      'NAD+ [CPD:C00003]',
      'secondary alcohol [CPD:C01612]'],
     'PRODUCT': ['aldehyde [CPD:C00071]',
      'NADH [CPD:C00004]',
      'H+ [CPD:C00080]',
      'ketone [CPD:C01450]'],
     'COMMENT': 'A zinc protein. Acts on primary or secondary alcohols or hemi-acetals with very broad specificity; however the enzyme oxidizes methanol much more poorly than ethanol. The animal, but not the yeast, enzyme acts also on cyclic secondary alcohols.',
     'HISTORY': 'EC 1.1.1.1 created 1961, modified 2011',
     'REFERENCES': ['1',
      'AUTHORS   Branden, G.-I., Jornvall, H., Eklund, H. and Furugren, B.',
      'TITLE     Alcohol dehydrogenase.',
      'JOURNAL   In: Boyer, P.D. (Ed.), The Enzymes, 3rd ed., vol. 11, Academic Press, New York, 1975, p. 103-190.',
      '2  [PMID:320001]',
      'AUTHORS   Jornvall H.',
      'TITLE     Differences between alcohol dehydrogenases. Structural properties and evolutionary aspects.',
      'JOURNAL   Eur J Biochem 72:443-52 (1977)',
      'DOI:10.1111/j.1432-1033.1977.tb11268.x',
      '3',
      'AUTHORS   Negelein, E. and Wulff, H.-J.',
      'TITLE     Diphosphopyridinproteid, Alkohol, Acetaldehyd.',
      'JOURNAL   Biochem Z 293:351-389 (1937)',
      '4',
      'AUTHORS   Sund, H. and Theorell, H.',
      'TITLE     Alcohol dehydrogenase.',
      'JOURNAL   In: Boyer, P.D., Lardy, H. and Myrback, K. (Eds.), The Enzymes, 2nd ed., vol. 7, Academic Press, New York, 1963, p. 25-83.',
      '5  [PMID:13605979]',
      'AUTHORS   THEORELL H.',
      'TITLE     Kinetics and equilibria in the liver alcohol dehydrogenase system.',
      'JOURNAL   Adv Enzymol Relat Subj Biochem 20:31-49 (1958)',
      'DOI:10.1002/9780470122655.ch2'],
     'PATHWAY': {'ec00010': 'Glycolysis / Gluconeogenesis',
      'ec00071': 'Fatty acid degradation',
      'ec00260': 'Glycine, serine and threonine metabolism',
      'ec00350': 'Tyrosine metabolism',
      'ec00592': 'alpha-Linolenic acid metabolism',
      'ec00620': 'Pyruvate metabolism',
      'ec00625': 'Chloroalkane and chloroalkene degradation',
      'ec00626': 'Naphthalene degradation',
      'ec00830': 'Retinol metabolism',
      'ec00980': 'Metabolism of xenobiotics by cytochrome P450',
      'ec00982': 'Drug metabolism - cytochrome P450',
      'ec01100': 'Metabolic pathways',
      'ec01110': 'Biosynthesis of secondary metabolites',
      'ec01120': 'Microbial metabolism in diverse environments'},
     'MODULE': {},
     'ORTHOLOGY': {'K00001': 'alcohol dehydrogenase',
      'K00121': 'S-(hydroxymethyl)glutathione dehydrogenase / alcohol dehydrogenase',
      'K04072': 'acetaldehyde dehydrogenase / alcohol dehydrogenase',
      'K11440': 'choline dehydrogenase',
      'K13951': 'alcohol dehydrogenase 1/7',
      'K13952': 'alcohol dehydrogenase 6',
      'K13953': 'alcohol dehydrogenase, propanol-preferring',
      'K13954': 'alcohol dehydrogenase',
      'K13980': 'alcohol dehydrogenase 4',
      'K18857': 'alcohol dehydrogenase class-P'},
     'DBLINKS': {'ExplorEnz': '- The Enzyme Database: 1.1.1.1',
      'IUBMB': 'Enzyme Nomenclature: 1.1.1.1',
      'ExPASy': '- ENZYME nomenclature database: 1.1.1.1',
      'UM-BBD': '(Biocatalysis/Biodegradation Database): 1.1.1.1',
      'BRENDA,': 'the Enzyme Database: 1.1.1.1',
      'CAS:': '9031-72-5'}}
    """
    # Parse KEGG response
    kegg_info = _parse_kegg_response(id_enzyme)

    kegg_info_formatted = {
        "ENTRY": "",
        "NAME": [],
        "CLASS": [],
        "SYSNAME": "",
        "REACTION": [],
        "ALL_REAC": [],
        "SUBSTRATE": [],
        "PRODUCT": [],
        "COMMENT": "",
        "HISTORY": "",
        "REFERENCES": [],
        "PATHWAY": {},
        "MODULE": {},    
        "ORTHOLOGY": {},
        "DBLINKS": {}
    }

    for k, v in kegg_info.items():
        if k == "ENTRY":
            items = list(filter(bool, v[0].split(" ")))
            id_enzyme = items[1]
            kegg_info_formatted["ENTRY"] = id_enzyme
        elif k == "NAME":
            kegg_info_formatted["NAME"] = v#" ".join(v)
        elif k == "CLASS":
            kegg_info_formatted["CLASS"] = v #" ".join(v)
        elif k == "SYSNAME":
            kegg_info_formatted["SYSNAME"] = " ".join(v)
        elif k == "REACTION":
            kegg_info_formatted["REACTION"] = v#" ".join(v)
        elif k == "ALL_REAC":
            kegg_info_formatted["ALL_REAC"] = v
        elif k == "SUBSTRATE":
            kegg_info_formatted["SUBSTRATE"] = v
        elif k == "PRODUCT":
            kegg_info_formatted["PRODUCT"] = v
        elif k == "COMMENT":
            kegg_info_formatted["COMMENT"] = " ".join(v)
        elif k == "HISTORY":
            kegg_info_formatted["HISTORY"] = " ".join(v)
        elif k == "REFERENCE":
            kegg_info_formatted["REFERENCES"].extend(v)
        elif k == "PATHWAY":
            kegg_info_formatted["PATHWAY"] = {
                id: " ".join(name_parts).strip()
                for line in v
                for id, *name_parts in [line.split(" ", 1)]
            }
        elif k == "ORTHOLOGY":
            kegg_info_formatted["ORTHOLOGY"] = {
                id: " ".join(name_parts).strip()
                for line in v
                for id, *name_parts in [line.split(" ", 1)]
            }
        elif k == "MODULE":
            kegg_info_formatted["MODULE"] = {
                id: " ".join(name_parts).strip()
                for line in v
                for id, *name_parts in [line.split(" ", 1)]
            }
        elif k == "DBLINKS":
            kegg_info_formatted["DBLINKS"] = {
                id: " ".join(name_parts).strip()
                for line in v
                for id, *name_parts in [line.split(" ", 1)]
            }

    return kegg_info_formatted

def _fetch_kegg_reaction_info(id_reaction: str):
    """
    Fetches and parses KEGG reaction information.
    
    # Example usage
    _fetch_kegg_reaction_info("R00540")

    # Example output:
    {'ENTRY': 'R00540',
     'NAME': 'nitrile aminohydrolase',
     'DEFINITION': 'Nitrile + 2 H2O <=> Carboxylate + Ammonia',
     'EQUATION': 'C00726 + 2 C00001 <=> C00060 + C00014',
     'COMMENT': 'general reaction',
     'RCLASS': {'RC00325': 'C00060_C00726', 'RC02811': 'C00014_C00726'},
     'ENZYME': {'3.5.5.1', '3.5.5.2', '3.5.5.5', '3.5.5.7'},
     'PATHWAY': {'rn00910': 'Nitrogen metabolism',
      'rn01100': 'Metabolic pathways'},
     'BRITE': ['Enzymatic reactions [BR:br08201]',
      '3. Hydrolase reactions',
      '3.5  Acting on carbon-nitrogen bonds, other than peptide bonds',
      '3.5.5  In nitriles',
      '3.5.5.1',
      'R00540  Nitrile + 2 H2O <=> Carboxylate + Ammonia',
      '3.5.5.2',
      'R00540  Nitrile + 2 H2O <=> Carboxylate + Ammonia',
      '3.5.5.5',
      'R00540  Nitrile + 2 H2O <=> Carboxylate + Ammonia',
      '3.5.5.7',
      'R00540  Nitrile + 2 H2O <=> Carboxylate + Ammonia',
      'IUBMB reaction hierarchy [BR:br08202]',
      '3. Hydrolase reactions',
      '3.5.5.1',
      'R00540  Nitrile <=> Carboxylate',
      '3.5.5.7',
      'R00540  Nitrile <=> Carboxylate'],
     'ORTHOLOGY': {'K01501': 'nitrilase [EC:3.5.5.1]'},
     'DBLINKS': {'RHEA:': '21727'},
     'MODULE': {}}
    """
    # Parse KEGG response
    kegg_info = _parse_kegg_response(id_reaction)

    kegg_info_formatted = {
        "ENTRY": "",
        "NAME": "",
        "DEFINITION": "",
        "EQUATION": "",
        "COMMENT": "",
        "RCLASS": {},
        "ENZYME": [],
        "PATHWAY": {},
        "BRITE": {},
        "ORTHOLOGY": {},
        "DBLINKS": {},
        "MODULE": {},
    }

    for k, v in kegg_info.items():
        if k == "ENTRY":
            items = list(filter(bool, v[0].split(" ")))
            id_reaction = items[0]
            kegg_info_formatted["ENTRY"] = id_reaction
        elif k == "NAME":
            kegg_info_formatted["NAME"] = " ".join(v)
        elif k == "DEFINITION":
            kegg_info_formatted["DEFINITION"] = " ".join(v)
        elif k == "EQUATION":
            kegg_info_formatted["EQUATION"] = " ".join(v)
        elif k == "COMMENT":
            kegg_info_formatted["COMMENT"] = " ".join(v)
        elif k == "RCLASS":
            kegg_info_formatted["RCLASS"] = {
                id: " ".join(name_parts).strip()
                for line in v
                for id, *name_parts in [line.split(" ", 1)]
            }
        elif k == "ENZYME":
            kegg_info_formatted["ENZYME"] = set.union(*map(lambda x: set(filter(bool, x.split(" "))), v))
        elif k == "PATHWAY":
            kegg_info_formatted["PATHWAY"] = {
                id: " ".join(name_parts).strip()
                for line in v
                for id, *name_parts in [line.split(" ", 1)]
            }
        elif k == "BRITE":
            kegg_info_formatted["BRITE"] = v
            
        elif k == "ORTHOLOGY":
            kegg_info_formatted["ORTHOLOGY"] = {
                id: " ".join(name_parts).strip()
                for line in v
                for id, *name_parts in [line.split(" ", 1)]
            }
        elif k == "DBLINKS":
            kegg_info_formatted["DBLINKS"] = {
                id: " ".join(name_parts).strip()
                for line in v
                for id, *name_parts in [line.split(" ", 1)]
            }
        elif k == "MODULE":
            kegg_info_formatted["MODULE"] = {
                id: " ".join(name_parts).strip()
                for line in v
                for id, *name_parts in [line.split(" ", 1)]
            }
    return kegg_info_formatted

# Wrapper
def fetch_kegg_info(id: str, ktype:str, into=dict, **kwargs):
    """
    Fetch KEGG info using REST API to return Mappable (dict, pd.Series)

    Args:
        id (str): Identifier for KEGG
        ktype (str): Identifier type.  Choose between {"reaction", "enzyme", "ortholog", "module", "pathway"}
        into (Mappable): [Default: dict]
    """

    ktype_to_function = {
        "reaction":_fetch_kegg_reaction_info,
        "enzyme":_fetch_kegg_enzyme_info,
        "ortholog":_fetch_kegg_ortholog_info,
        "module":_fetch_kegg_module_info,
        "pathway":_fetch_kegg_pathway_info,
    }
    
    check_argument_choice(ktype, ktype_to_function.keys())

    info = ktype_to_function[ktype](id, **kwargs)
    if into == pd.Series:
        return into(info, name=id)
    else:
        return into(info)