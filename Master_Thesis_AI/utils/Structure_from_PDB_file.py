"""
This skripts purpose is to retrieve and save the struktural data from the PDB file and store it as dataset.
Auth: Florian Zwicker
"""
import os
from Bio.PDB import PDBParser


class Structure_dataset_creator():
    def __init__(self, pdb_directory=None, output_directory=None):
        self.pdb_directory = pdb_directory
        self.output_directory = output_directory


    def get_structure_from_PDB_file(self):
        parser = PDBParser()

        for file in os.listdir(self.pdb_directory):
            if file.endswith(".pdb"):
                structure_id = file.split("_")[0]
                file_path = os.path.join(self.pdb_directory, file)

                structure = parser.get_structure(id=structure_id, file=file_path)

                for model in structure:
                    for chain in model:
                        for residue in chain:
                            for atom in residue:
                                print(atom.name, atom.coord)




if __name__=="__main__":
    get_structure = Structure_dataset_creator(pdb_directory=None, output_directory=None)
    get_structure.get_structure_from_PDB_file()