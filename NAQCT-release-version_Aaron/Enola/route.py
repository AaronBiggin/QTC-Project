import json
import math
import copy
from .codegen import CodeGen, global_dict
from networkx import maximal_independent_set, Graph
from typing import Sequence, Mapping, Any

global_dict["full_code"] = True

def compatible_2D(a: list[int], b: list[int]) -> bool:
    """
    Checks if two 2D points are compatible based on specified rules.

    Parameters:
    a (list[int]): A list of four integers representing the first point. The elements are ordered as [x_loc_before, y_loc_before, x_loc_after, y_loc_after].
    b (list[int]): A list of four integers representing the second point. The elements are ordered as [x_loc_before, y_loc_before, x_loc_after, y_loc_after].

    Returns:
    bool: True if the points are compatible, False otherwise.
    """
    assert len(a) == 4 and len(b) == 4, "Both arguments must be lists with exactly four elements."

    # Check compatibility for the first two elements of each point
    if a[0] == b[0] and a[2] != b[2]:
        return False
    if a[2] == b[2] and a[0] != b[0]:
        return False
    if a[0] < b[0] and a[2] >= b[2]:
        return False
    if a[0] > b[0] and a[2] <= b[2]:
        return False

    # Check compatibility for the last two elements of each point
    if a[1] == b[1] and a[3] != b[3]:
        return False
    if a[3] == b[3] and a[1] != b[1]:
        return False
    if a[1] < b[1] and a[3] >= b[3]:
        return False
    if a[1] > b[1] and a[3] <= b[3]:
        return False

    return True 

def maximalis_solve_sort(n: int, edges: list[tuple[int]], nodes: set[int]) -> list[int]:
    """
    Finds a maximal independent set from the given graph nodes using a sorted approach.
    
    Parameters:
    n (int): Number of nodes in the graph. The nodes were expressed by integers from 0 to n-1.
    edges (list[tuple[int]]): list of edges in the graph, where each edge is a tuple of two nodes.
    nodes (set[int]): Set of nodes to consider for the maximal independent set.

    Returns:
    list[int]: list of nodes in the maximal independent set.
    """
    # Initialize conflict status for each node
    is_node_conflict = [False for _ in range(n)]
    
    # Create a dictionary to store neighbors of each node
    node_neighbors = {i: [] for i in range(n)}
    
    # Populate the neighbors dictionary
    for edge in edges:
        node_neighbors[edge[0]].append(edge[1])
        node_neighbors[edge[1]].append(edge[0])
    
    result = []
    for i in nodes:
        if is_node_conflict[i]:
            continue
        else:
            result.append(i)
            for j in node_neighbors[i]:
                is_node_conflict[j] = True
    return result

def maximalis_solve(nodes:list[int], edges:list[tuple[int]])-> list[int]:
    """
    Wrapper function to find a maximal independent set using the Graph class.

    Parameters:
    n (int): Number of nodes in the graph. The nodes were expressed by integers from 0 to n-1.
    edges (list[tuple[int]]): list of edges in the graph.

    Returns:
    list[int]: list of nodes in the maximal independent set.
    """
    G = Graph()
    for i in nodes:
        G.add_node(i)
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    
    # Use a library function to find the maximal independent set

    #find maximal independet set from the conflict graph
    #maximcal independent set--> largest possible set of compaitble moves
    result = maximal_independent_set(G, seed=0) 
    return result

def get_movements(current_map: list, next_map: list, window_size=None) -> map:
    """
    Determines the movements of qubits between two maps.

    Parameters:
    current_map (list): list of current positions of qubits.
    next_map (list): list of next positions of qubits.
    window_size (optional): Size of the window for movement calculations.

    Returns:
    map: A dictionary with qubit movements.
    """
    #Change the end position such that it is the same as the start.
    
    movements = {}
    # Determine movements of qubits
    for qubit, current_position in enumerate(current_map):
        next_position = next_map[qubit]
        if current_position != next_position:
            move_details = current_position + next_position
            movements[qubit] = move_details
    
    return movements

#DO CONFLICT RESOLUTION HERE

def map_to_layer(map: list) -> dict[str, list]:
    """
    Converts a list of qubit positions to a layer dictionary.

    Parameters:
    map (list): list of qubit positions.

    Returns:
    map: Dictionary representing the layer configuration.
    """
    return {
        "qubits": [{
            "id": i,
            "a": 0,
            "x": map[i][0],
            "y": map[i][1],
            "c": map[i][0],
            "r": map[i][1],
        } for i in range(len(map))],
        "gates": []
    }

def gates_in_layer(gate_list:list[list[int]])->list[dict[str, int]]:
    res = []
    for i in range(len(gate_list)-1,-1,-1):
        assert len(gate_list[i]) == 2
        res.append({'id':i,'q0':gate_list[i][0],'q1':gate_list[i][1]})
    return res

class QuantumRouter:
    def __init__(self, num_qubits: int, embeddings: list[list[list[int]]], gate_list: list[list[int]], arch_size: list[int], routing_strategy: str = "maximalis") -> None:
        """
        Initialize the QuantumRouter object with the given parameters.
        
        Parameters:
        num_qubits (int): Number of qubits.
        embeddings (list[list[list[int]]]): Embeddings for the qubits.
        gate_list (list[list[int]]): list of two-qubit gates.
        arch_size (list[int]): Architecture size as [x, y].
        routing_strategy (str): Strategy used for routing.
        """
        self.num_qubits = num_qubits
        self.validate_embeddings(embeddings)
        self.embeddings = embeddings
        
        assert len(embeddings) == len(gate_list), "The number of embeddings should match the number of two-qubit gates in gate_list."
        self.gate_list = gate_list
        
        self.validate_architecture_size(arch_size)
        self.arch_size = arch_size
        self.routing_strategy = routing_strategy
        self.movement_list = []

    def validate_embeddings(self, embeddings: list[list[list[int]]]) -> None:
        """
        Validate the embeddings to ensure they contain locations for all qubits.
        
        Parameters:
        embeddings (list[list[list[int]]]): Embeddings for the qubits.
        """
        for embedding in embeddings:
            assert len(embedding) == self.num_qubits, f"Each embedding must contain locations for all {self.num_qubits} qubits."
            for loc in embedding:
                assert len(loc) == 2, "Each location must be a list containing exactly two coordinates: [x, y]."

    def validate_architecture_size(self, arch_size: list[int]) -> None:
        """
        Validate the architecture size to ensure it can accommodate all qubits.
        
        Parameters:
        arch_size (list[int]): Architecture size as [x, y].
        """
        assert len(arch_size) == 2, "Architecture size should be specified as a list with two elements: [x, y]."
        assert arch_size[0] * arch_size[1] >= self.num_qubits, (
            f"The product of the architecture dimensions x and y must be at least {self.num_qubits} to accommodate all qubits; "
            f"currently, it is {arch_size[0] * arch_size[1]}."
        )

    def initialize_program(self) -> None:
        """
        Initialize the program with the initial layer and gates.
        """
        layers = [map_to_layer(self.embeddings[0])]
        initial_layer = map_to_layer(self.embeddings[0])
        initial_layer["gates"] = gates_in_layer(self.gate_list[0])
        layers.append(initial_layer)
        return self.generate_program(layers)

    def generate_program(self, layers: list[dict[str, Any]]) -> Sequence[Mapping[str, Any]]:
        """
        Generate the program from the given layers.
        
        Parameters:
        layers (list[dict[str, Any]]): list of layers.
        
        Returns:
        str: The generated program.
        """
        data = {
            "no_transfer": False,
            "layers": layers,
            "n_q": self.num_qubits,
            "g_q": self.gate_list,
            "n_x": self.arch_size[0],
            "n_y": self.arch_size[1],
            "n_r": self.arch_size[0],
            "n_c": self.arch_size[1]
        }
        code_gen = CodeGen(data)
        program = code_gen.builder(no_transfer=False)
        return program.emit_full()

    def process_all_embeddings(self) -> None:
        """
        Process all embeddings to resolve movements and update the program.
        """
        # zero_embedding = self.embeddings[0]
        # # Interleave the 0th embedding into the rest of the embeddings
        # interleaved_embeddings = []
        # for embedding in self.embeddings:
        #     interleaved_embeddings.append(embedding)
        #     interleaved_embeddings.append(zero_embedding)

        #     # Update the original self.embeddings with the interleaved version
        # self.embeddings = interleaved_embeddings
        for current_pos in range(len(self.embeddings) - 1):
            movements = self.resolve_movements(current_pos)
            assert len(movements) > 0, "there should be some movements between embeddings"
            self.movement_list.append(movements)

    def solve_violations(self, movements, violations, sorted_keys):
        """
        Resolves violations in qubit movements based on the routing strategy.

        violations (list[tuple[int, int]]): list of violations.
        movements (dict[int, tuple[int, int, int, int]]): Movements between embeddings.
        sorted_movements (list[int]): Sorted list of movements.
        current_pos (int): The current position in the embeddings list.

        Parameters:
        movements (dict): Dictionary of qubit movements.
        violations (list): list of violations to be resolved.
        sorted_keys (list): list of qubit keys sorted based on priority.
        layer (dict): Dictionary representing the current layer configuration.

        Returns:
        tuple: remaining movements, unresolved violations and movement sequence to finish movement this time
        """
        if self.routing_strategy == "maximalis":
            resolution_order = maximalis_solve(sorted_keys, violations)
        else:
            resolution_order = maximalis_solve_sort(self.num_q, violations, sorted_keys)
        # print(f'Resolution Order: {resolution_order}')
        move_sequence =[]
        for qubit in resolution_order:
            sorted_keys.remove(qubit)
            #removing violated qubits from the maximal independent set of
            #maximal indepdent set is a list of qubits that don't interact with eachother
            move = movements[qubit]
            # print(self.momvents)
            move_sequence.append([qubit,(move[0],move[1]),(move[2],move[3])])
            #identify a set of violations, identify a subset of a graph for which there are no violations
            # print(f'Move qubit {qubit} from ({move[0]}, {move[1]}) to ({move[2]}, {move[3]})')
            # Remove resolved violations
            violations = [v for v in violations if qubit not in v]
            del movements[qubit]
        
        return movements, violations, move_sequence

    def resolve_movements(self, current_pos: int) -> list[int, tuple[int, int], tuple[int, int]]:
        """
        Resolve movements between the current and next embeddings.
        
        Parameters:
        current_pos (int): The current position in the embeddings list.
        
        Returns:
        str: The program for the resolved movements.
        """
        next_pos = current_pos + 1
        movements = get_movements(self.embeddings[current_pos], self.embeddings[next_pos])
        #gets movements by adding the coordinates of the next position to the first position,
        #1st posoition->[x,y], 2nd position [x,y,x1,y1]
        sorted_movements = sorted(movements.keys(), key=lambda k: math.dist(movements[k][:2], movements[k][2:]))
        #sorts the order of keys by euclidean disance from the difference between movements.
        violations = self.check_violations(sorted_movements, movements)
        move_sequences = self.handle_violations(violations, movements, sorted_movements, current_pos)
        return move_sequences

    def handle_violations(self, violations: list[tuple[int, int]], remained_mov_map: dict[int, tuple[int, int, int, int]], sorted_movements: list[int], current_pos: int) -> list[int, tuple[int, int], tuple[int, int]]:
        """
        Handle violations and return the movement sequence accordingly.
        
        Parameters:
        violations (list[tuple[int, int]]): list of violations.
        movements (dict[int, tuple[int, int, int, int]]): Movements between embeddings.
        sorted_movements (list[int]): Sorted list of movements.
        current_pos (int): The current position in the embeddings list.
        
        Returns:
        list[int, tuple[int, int], tuple[int, int]]: movement sequences.
        """
        movement_sequence =[]
        while remained_mov_map:
            remained_mov_map, violations, movement = self.solve_violations(remained_mov_map, violations, sorted_movements)
            movement_sequence.append(movement)
        return movement_sequence

    def check_violations(self, sorted_movements: list[int], remained_mov_map: dict[int, tuple[int, int, int, int]]) -> list[tuple[int, int]]:
        """
        Check for violations between movements.
        
        Parameters:
        sorted_movements (list[int]): Sorted list of movements.
        movements (dict[int, tuple[int, int, int, int]]): Movements between embeddings.
        
        Returns:
        list[tuple[int, int]]: list of violations.
        """
        violations = []
        for i in range(len(sorted_movements)):
            for j in range(i + 1, len(sorted_movements)):
                if not compatible_2D(remained_mov_map[sorted_movements[i]], remained_mov_map[sorted_movements[j]]):
                    #checks for colomn and row violations and confirms the rules set our by the ENOLA paper
                    violations.append((sorted_movements[i], sorted_movements[j]))
        return violations

    def update_layer(self, layer, movements):
        new_layer = copy.deepcopy(layer)
        for qubit, current_pos, next_pos in movements:
            assert layer["qubits"][qubit]["id"] == qubit, "some error happen during layer generation"
            assert layer["qubits"][qubit]["x"] == current_pos[0], f"layer have problem with location of qubit {qubit}, in x-axis"
            assert layer["qubits"][qubit]["y"] == current_pos[1], f"layer have problem with location of qubit {qubit}, in y-axis"

            new_layer["qubits"][qubit]["a"] = 1
            layer["qubits"][qubit]["x"] = next_pos[0]
            layer["qubits"][qubit]["y"] = next_pos[1]
            layer["qubits"][qubit]["c"] = next_pos[0]
            layer["qubits"][qubit]["r"] = next_pos[1]
        return new_layer

    def save_program(self, filename: str) -> None:
        """
        Save the generated program to a file.
        Parameters:
        filename (str): The filename to save the program.
        """
        assert filename.endswith('.json'), "program should be saved to a .json file"
        assert len(self.movement_list) == len(self.embeddings)-1, "before generate program, movement should be finished"
        program = self.initialize_program()
        for i,movements in enumerate(self.movement_list):
            layers = []
            layer = map_to_layer(self.embeddings[i])
            for mov in movements:
                layers.append(self.update_layer(layer,mov))
            layers[-1]["gates"] = gates_in_layer(self.gate_list[i+1])
            program += self.generate_program(layers)[2:]
            layers.append(self.generate_last_layers(layers[-1]))    
        print(layers)
        for index, layer in enumerate(layers):
            if (index>0):
                self.layer_correction(layers[index-1], layers[index])
                self.Teleporter_test1(layers[index-1], layers[index], index)
        with open(filename, 'w') as file:
            json.dump(program, file)
     
    def Teleporter_test1(self, layer1, layer2, index):
        for qubit in range(len(layer1["qubits"])):
            assert layer1["qubits"][qubit]["id"] == qubit, "some error happen during layer generation"
            assert layer2["qubits"][qubit]["id"] == qubit, "some error happen during layer generation"
            # Qubits should not move between layers if there is no AOD trap applied in the first layer 
            if(layer1["qubits"][qubit]["a"] == 0):
                # Check for movement
                assert layer1["qubits"][qubit]["x"] == layer2["qubits"][qubit]["x"], f"layer{index} has problem with location of qubit {qubit}, in x-axis--PROBLEM COORDINATE x in layer 1-->{layer1['qubits'][qubit]['x']}, PROBLEM COORDINATE x in layer 2-->{layer2['qubits'][qubit]['x']}"
                assert layer1["qubits"][qubit]["y"] == layer2["qubits"][qubit]["y"], f"layer{index} has problem with location of qubit {qubit}, in y-axis"
                assert layer1["qubits"][qubit]["r"] == layer2["qubits"][qubit]["r"], f"layer{index} has problem with location of qubit {qubit}, in row"
                assert layer1["qubits"][qubit]["c"] == layer2["qubits"][qubit]["c"], f"layer{index} has problem with location of qubit {qubit}, in column"
                    
             
    def generate_last_layers(self, layer1):
        last_layer = {
                "qubits": [{
                    "id": i,
                    "a": 0,
                    "x": layer1["qubits"][i]["x"],
                    "y": layer1["qubits"][i]["y"],
                    "c": layer1["qubits"][i]["c"],
                    "r": layer1["qubits"][i]["r"],
                } for i in range(10)],
                "gates": []
            }
        return last_layer

    def layer_correction(self, layer1, layer2):
        for j in range(10):
            if (layer1["qubits"][j]["a"] == 0):       
                if (layer1["qubits"][j]["x"] != layer2["qubits"][j]["x"]):
                    layer1["qubits"][j]["a"] = 1
                if (layer1["qubits"][j]["y"] != layer2["qubits"][j]["y"]):
                    layer1["qubits"][j]["a"] = 1
        return


    def run(self) -> None:
        """
        Run the QuantumRouter to initialize, process embeddings.
        """
        self.movement_list = []
        self.process_all_embeddings()