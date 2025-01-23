# QTC-Project
This project aims to attack the qubit mapping problem through the lens of graph theory.

# Terminal Steps

1. Create a folder inside of Data containing the circuit(s) you would like the program to execute.
2. In the command line, navigate to the NAQCT-release-version folder and enter "python DasAtom.py benchmark path/to/folder", where benchmark is a string used to specify and categorise the process being executed and path/to/folder is the directory path from the current folder to the newly created folder. E.g. "python DasAtom.py initial_test Data/qiskit-bench/qft/qft_5"
3. Navigate to the folder with the same name as your benchmark inside of NAQCT-release-version/results, go to the embeddings folder and find the json file whose name contains your benchmark name. Rename it such that it is of the form: string_qubitnum_code_full.json, where string can be any string and qubitnum is the number of qubits in the provided circuit.
4. In the command line, enter ' python Enola/animation.py path/to/json --dir="./results/animations/" ', where path/to/json is the relative path to the renamed json file
5. After execution is complete, a mp4 file should have been created inside of results/animations/