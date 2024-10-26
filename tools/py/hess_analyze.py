"""
read the hessian from ab initial file.
python hess_analyze.py -a <num_atoms> -b <nbeads> -f <hess file>
"""
import numpy as np 
import sys 
import re 
import argparse 

# ---- parse input from command line -------
def parse_input():
    # INPUT
    parser = argparse.ArgumentParser(
        description= """
        File that read the hessian data from the file.
        Need information about natoms (number of atoms) and nbeads (number of beads)
        """
    )

    parser.add_argument(
        "-a",
        "--natoms",
        help= "Number of atoms for the system."
    )

    parser.add_argument(
        "-b",
        "--nbeads",
        help= "Number of beads in hessian file (half ring polymer)"
    )

    parser.add_argument(
        "-f",
        "--file",
        help= "path to the hessian file"
    )
    args = parser.parse_args()
    natoms = int(args.natoms)
    nbeads = int(args.nbeads) 
    hessian_file_path = args.file 

    return natoms, nbeads, hessian_file_path 
    

def extract_number_from_line(line):
    line = re.split(" ", line.strip())
    line = [ele for ele in line if ele != ""]

    return line

def read_hessian_from_file(hessian_file_path):
    """
    Read the hessian from the file.
    """
    with open(hessian_file_path, "r") as f:
        lines = f.readlines() 
        line = extract_number_from_line(lines[0])

        hessian_data = np.array(list(map(float, line)))

        return hessian_data 

def transform_hessian(hessian_data, natoms, nbeads):
    """
    transform the hessian from 1d to shape [3 * natoms, nbeads * 3 * natoms],
    then to [nbeads, 3 * natom, 3* natom]
    """
    assert len(hessian_data) == np.power(3 * natoms, 2) * nbeads, "the length of hessian does not match nbeads and natoms" 
    hessian_data = np.reshape(hessian_data, (3 * natoms, nbeads, 3 * natoms))
    hessian_data = np.transpose(hessian_data, (1, 0, 2))

    return hessian_data 

def analyze_hessian():
    """
    read the parameter from the file and analyze the hessian.
    Use in pdb form we can compare ab initio hessian with the gpr predicted hessian.
    """
    natoms, nbeads, hessian_file_path = parse_input() 
    
    hessian_data_1d = read_hessian_from_file(hessian_file_path)

    hessian_data = transform_hessian(hessian_data_1d, natoms, nbeads)

    pass 

analyze_hessian()