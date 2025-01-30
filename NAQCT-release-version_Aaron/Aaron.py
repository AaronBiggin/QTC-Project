import matplotlib.pyplot as plt
import networkx as nx
from DasAtom import CreateCircuitFromQASM, get_2q_gates_list, gates_list_to_QC
from DasAtom_fun import partition_from_DAG, get_embeddings, get_qubits_num,generate_grid_with_Rb
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.visualization import dag_drawer
from qiskit.dagcircuit import DAGDependency
import math
from Enola.route import compatible_2D, maximalis_solve_sort,map_to_layer, gates_in_layer, QuantumRouter
import copy
from Enola_fun.router_mis import route_qubit_mis
import os

#kust having some problems with layer visualisation right now so I will quickly sort this out.
#In the mean time do u want to make a function that visualises movement between layers?
#y
# By adding an arrow between previous and current qubit positions? Yes and highlight them in red if they are in AOD trap and blue in SLM
# I'll start with the arrows and let you know how I go
def visualize_embeddings(embeddings):
    """
    Visualize embeddings on a 3x3 grid with sequential labels for each point (q0, q1, etc.).
    
    Parameters:
        embeddings (list): A list of lists where each inner list contains tuples representing coordinates.
    """
    for idx, embedding in enumerate(embeddings):
        # Create a 3x3 grid
        grid_size = 3
        plt.figure(figsize=(4, 4))
        plt.title(f"Embedding {idx + 1}")
        plt.xticks(range(grid_size))
        plt.yticks(range(grid_size))
        plt.xlim(-0.5, grid_size - 0.5)
        plt.ylim(-0.5, grid_size - 0.5)  
        plt.grid(True)
        
        # Plot embedding points with labels q0, q1, ...
        for i, (x, y) in enumerate(embedding):
            plt.scatter(x, y, color='blue', s=100)
            plt.text(x, y, f"q{i}", color="red", fontsize=10, ha='center', va='center')
        plt.show()

def visualize_partitions(partitions):
    """
    Visualize partitions as graphs.
    
    Parameters:
        partitions (list): A list of lists where each inner list contains edges as tuples.
    """
    for idx, partition in enumerate(partitions):
        # Create a graph for the current partition
        G = nx.Graph()
        G.add_edges_from(partition)
        
        plt.figure(figsize=(6, 6))
        plt.title(f"Partition {idx + 1}")
        pos = nx.spring_layout(G)  # Generate graph layout
        nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=1500, font_size=10)
        plt.show()

def visualize_dag(dag):
    """
    Visualize a Qiskit DAGCircuit as a directed graph using NetworkX.
    
    Parameters:
        dag (DAGCircuit): The Qiskit DAGCircuit object.
    """
    # Create a NetworkX directed graph
    dag_graph = nx.DiGraph()

    # Add nodes to the graph
    seen_qargs = set()

    # Add nodes to the graph
    for node in dag.topological_op_nodes():
        # Convert qargs to a tuple to make it hashable
        qargs_tuple = tuple(node.qargs)

        # Check if the qargs are distinct
        if qargs_tuple not in seen_qargs:
            # Print information about each node when qargs are distinct
            print(f"Node Name: {node.name}")
            print(f"Node Type: {node.op.name}")
            print(f"Node Qargs: {node.qargs}")
            print("-" * 30)  # Separator for better readability
            
            # Add qargs to seen set
            seen_qargs.add(qargs_tuple)

            # Add node to the graph
            dag_graph.add_node(node.name)

    # Add edges to the graph
    for edge in dag.edges():
        source = edge[0]
        dest = edge[1]
        if hasattr(source, "name") and hasattr(dest, "name"):
            dag_graph.add_edge(source.name, dest.name)

    # Visualize the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(dag_graph)  # Use spring layout for clear visualization
    nx.draw(
        dag_graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=2000,
        font_size=10,
        font_weight="bold",
        edge_color="black"
    )
    plt.title("DAG Visualization")
    plt.show(block=True)  # Ensure the plot stays open
    

def generate_dependency_dag(gate_2q_list):
    """
    Generate and visualize a left-to-right layered dependency DAG from a list of 2-qubit gates.
    
    Args:
        gate_2q_list (list of tuples): Each tuple represents a 2-qubit gate 
                                       in the form (qubit1, qubit2).
    """
    # Step 1: Create a directed graph
    G = nx.DiGraph()

    # Step 2: Add nodes and edges based on dependencies
    for i, (q1, q2) in enumerate(gate_2q_list):
        gate_name = f"g{i}"  # Assign unique gate names like g0, g1, ...
        G.add_node(gate_name, qubits=(q1, q2))  # Add the gate as a node

        # Check dependencies with previous gates
        for j in range(i):
            prev_gate_name = f"g{j}"
            prev_q1, prev_q2 = G.nodes[prev_gate_name]["qubits"]

            # If there is an overlap in qubits, add a dependency edge
            if {q1, q2}.intersection({prev_q1, prev_q2}):
                G.add_edge(prev_gate_name, gate_name)

    # Step 3: Assign gates to layers based on the sequence and dependencies
    layers = []
    for node in G.nodes:
        # Find the earliest layer where this node can be placed
        layer = 0
        for pred in G.predecessors(node):
            pred_layer = next((i for i, l in enumerate(layers) if pred in l), -1)
            if pred_layer >= layer:
                layer = pred_layer + 1
        # Add the node to the appropriate layer
        if layer >= len(layers):
            layers.append([])
        layers[layer].append(node)

    # Step 4: Visualize the left-to-right layered DAG
    plt.figure(figsize=(12, 8))
    pos = {}  # Position dictionary for nodes
    layer_width = 0
    for layer_idx, layer in enumerate(layers):
        y_pos = 0
        for node in layer:
            pos[node] = (layer_width, y_pos)  # Position nodes in layers (left-to-right)
            y_pos += 1
        layer_width += 1

    # Draw the graph with layers
    nx.draw(
        G, pos, with_labels=True, node_color="skyblue", edge_color="black",
        node_size=2000, font_size=10, font_weight="bold", arrowsize=20
    )
    plt.title("Left-to-Right Layered Dependency DAG for 2-Qubit Gates")
    plt.show()


def asap(gate_2q_list, n_q):
    # as soon as possible algorithm for self.g_q
    list_scheduling = []
    list_qubit_time = [0 for i in range(n_q)]
    for i, gate in enumerate(gate_2q_list):
        tq0 = list_qubit_time[gate[0]]
        tq1 = list_qubit_time[gate[1]]
        tg = max(tq0, tq1)
        if tg >= len(list_scheduling):
            list_scheduling.append([])
        list_scheduling[tg].append(i)

        tg += 1
        list_qubit_time[gate[0]] = tg
        list_qubit_time[gate[1]] = tg
    return list_scheduling

def get_qubits_num(gate_2q_list):
    """
    Calculate the number of qubits from a list of 2-qubit gates.
    
    Args:
        gate_2q_list (list of tuples): Each tuple represents a 2-qubit gate 
                                       in the form (qubit1, qubit2).
    
    Returns:
        int: The total number of unique qubits used in the gates.
    """
    unique_qubits = set()  # Use a set to store unique qubits
    for gate in gate_2q_list:
        unique_qubits.add(gate[0])  # Add the first qubit
        unique_qubits.add(gate[1])  # Add the second qubit
    return len(unique_qubits)  # Return the number of unique qubits

def convert_partitions_to_gate_notation(partitions, gate_2q_list):
    """
    Convert partitions of 2-qubit gates into gate notation using indices from gate_2q_list.
    Handles duplicate gates by assigning unique indices to each occurrence.
    
    Args:
        partitions (list of lists): Each partition contains lists of 2-qubit gates.
        gate_2q_list (list of tuples): Each tuple represents a 2-qubit gate in the form (qubit1, qubit2).
    
    Returns:
        list of lists: Each partition contains lists of unique gate indices corresponding to gate_2q_list.
    """
    # Create a dictionary to map gate tuples to lists of indices (for duplicate gates)
    gate_to_indices = {}
    for idx, gate in enumerate(gate_2q_list):
        gate_tuple = tuple(gate)
        if gate_tuple not in gate_to_indices:
            gate_to_indices[gate_tuple] = []
        gate_to_indices[gate_tuple].append(idx)
    
    # Convert partitions to gate notation
    partition_gates = []
    for partition in partitions:
        gate_partition = []
        for gate in partition:
            gate_tuple = tuple(gate)
            if gate_tuple in gate_to_indices:
                # Pop the first index for this gate tuple to ensure uniqueness
                if gate_to_indices[gate_tuple]:
                    gate_partition.append(gate_to_indices[gate_tuple].pop(0))
                else:
                    raise ValueError(f"No more unique indices available for gate {gate_tuple}.")
            else:
                raise ValueError(f"Gate {gate_tuple} not found in gate_2q_list.")
        partition_gates.append(gate_partition)
    
    return partition_gates

def process_all_embeddings(num_q, embeddings):
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
    movement_list = []
    for current_pos in range(len(embeddings) - 1):
        movements = resolve_movements(num_q, embeddings, current_pos)
        assert len(movements) > 0, "there should be some movements between embeddings"
        movement_list.append(movements)
        # if (current_pos>0):
        #     print(f"teleporting qubits  in stage{current_pos-1}and{current_pos}")
        #     print(detect_teleporting_between_movements(movement_list[current_pos-1], movement_list[current_pos]))
        # print(f"movement{current_pos}")
        # print(movements)
    #print(movement_list)
    return movement_list

def resolve_movements(num_q, embeddings, current_pos: int) -> list[int, tuple[int, int], tuple[int, int]]:
    """
    Resolve movements between the current and next embeddings.
        
    Parameters:
    curent_pos (int): The current position in the embeddings list.
        
    Returns:
    str: The program for the resolved movements.
    """
    next_pos = current_pos + 1
    movements = get_movements(embeddings[current_pos], embeddings[next_pos])
    #gets movements by adding the coordinates of the next position to the first position,
    sorted_movements = sorted(movements.keys(), key=lambda k: math.dist(movements[k][:2], movements[k][2:]))
        #sorts the order of keys by euclidean disance from the difference between movements.
    violations = check_violations(sorted_movements, movements)
    move_sequences = handle_violations(num_q, violations, movements, sorted_movements, current_pos)
    return move_sequences

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

def check_violations(sorted_movements: list[int], remained_mov_map: dict[int, tuple[int, int, int, int]]) -> list[tuple[int, int]]:
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
    
def handle_violations(num_q, violations: list[tuple[int, int]], remained_mov_map: dict[int, tuple[int, int, int, int]], sorted_movements: list[int], current_pos: int) -> list[int, tuple[int, int], tuple[int, int]]:
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
        remained_mov_map, violations, movement = solve_violations(num_q, remained_mov_map, violations, sorted_movements)
        movement_sequence.append(movement)
    return movement_sequence   

def solve_violations(num_q, movements, violations, sorted_keys):
    
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
    
    resolution_order = maximalis_solve_sort(num_q, violations, sorted_keys)
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

def detect_teleporting_between_movements(movement1, movement2):
    # Track the last known positions of qubits from movement1
    last_positions = {}
    teleporting_qubits = []

    # Populate the last positions from movement1
    for group in movement1:
        for qubit_id, _, end_pos in group:
            last_positions[qubit_id] = end_pos

    # Check the start positions in movement2
    for group in movement2:
        for qubit_id, start_pos, _ in group:
            if qubit_id in last_positions:
                if last_positions[qubit_id] != start_pos:
                    print(f"⚠️ Teleport detected for qubit {qubit_id}! "
                          f"Previous end: {last_positions[qubit_id]}, New start: {start_pos}")
                    teleporting_qubits.append(qubit_id)
    return teleporting_qubits


def layer_creation(movement_list, embeddings, partitions):
    """
    Save the generated program to a file.
    Parameters:
    filename (str): The filename to save the program.
    """
    layers = []
    count =0
    for i,movements in enumerate(movement_list):
        layer = map_to_layer(embeddings[i])
        layer1 = map_to_layer(embeddings[i+1])
        print(f"this is layer group {i}")
        for j, mov in enumerate(movements):
            print(f"this is layer {count}")
            layers.append(update_layer(layer, layer1, mov))
            count+=1
        print("gates in layer")
        print(partitions[i+1])
        layers[-1]["gates"] = gates_in_layer(partitions[i+1])
        print(f"this is layer {count}")
        layers.append(generate_last_layers(layers[-1]))
        count+=1
        print(generate_last_layers(layers[-1]))     
    #print(layers)
        
    for index, layer in enumerate(layers):
        if (index>0):
            layer_correction(layers[index-1], layers[index])
            Teleporter_test1(layers[index-1], layers[index], index)
    return layers
   
def generate_last_layers(layer1):
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

def layer_correction(layer1, layer2):
    for j in range(10):
        if (layer1["qubits"][j]["a"] == 0):       
            if (layer1["qubits"][j]["x"] != layer2["qubits"][j]["x"]):
                layer1["qubits"][j]["a"] = 1
            if (layer1["qubits"][j]["y"] != layer2["qubits"][j]["y"]):
                layer1["qubits"][j]["a"] = 1
    return
               
def update_layer(layer, layer1, movements):
    #print(movements)
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
    for qubit, current_pos, next_pos in movements:
        assert layer1["qubits"][qubit]["id"] == qubit, "some error happen during layer generation"
        assert layer1["qubits"][qubit]["x"] == next_pos[0], f"layer have problem with location of qubit {qubit}, in x-axis"
        assert layer1["qubits"][qubit]["y"] == next_pos[1], f"layer have problem with location of qubit {qubit}, in y-axis"   
    print(new_layer)
    return new_layer
    
def Teleporter_test1(layer1, layer2, index):
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
            
            
        
def visualize_layer_with_traps(layer, arch_size):
    # Create lists for positions, IDs, and trap types
    qubit_positions = []
    qubit_ids = []
    trap_colors = []
   
    #print(layer)
    for i, qubit in layer["qubits"]:
        for j, move in qubit: 
            x, y, q_id, a = move["qubits"][i]["x"], move["qubits"][i]["y"], move["qubits"][i]["id"], layer["qubits"][i]["a"]
            qubit_positions.append((x, y))
            qubit_ids.append(q_id)
            trap_colors.append("red" if a == 1 else "blue")  # Red for AOD, Blue for SLM
    
    
    # Scatter plot for qubit positions
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, arch_size - 0.5)
    ax.set_ylim(-0.5, arch_size - 0.5)
    ax.set_xticks(range(arch_size))
    ax.set_yticks(range(arch_size))
    ax.grid(which="both", linestyle="--", linewidth=0.5, color="gray")
    
    # Plot the qubits
    for (x, y), q_id, color in zip(qubit_positions, qubit_ids, trap_colors):
        ax.scatter(x, y, color=color, s=300, alpha=0.8, edgecolors="black")  # Circle size and border
        ax.text(x, y, str(q_id), ha="center", va="center", fontsize=10, color="white", weight="bold")  # ID on the dot
    
    # Add legend for trap types
    legend_labels = {"red": "AOD Trap", "blue": "SLM Trap"}
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=label, markeredgecolor="black")
               for color, label in legend_labels.items()]
    ax.legend(handles=handles, loc="upper right", title="Trap Type")

    ax.set_title("Layer Visualization with Traps")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.gca().invert_yaxis()  # Invert y-axis to match the grid view
    plt.show()
    
    
    



# Data
partitions1 = [
    [[4, 3], [4, 3], [4, 2], [4, 2], [3, 2], [4, 1], [3, 2], [4, 1], [3, 1], [4, 0], [3, 1], [4, 0], [2, 1], [3, 0], [2, 1], [3, 0]],
    [[2, 0], [2, 0], [1, 0], [1, 0], [1, 3], [0, 4], [3, 1], [4, 0], [1, 3], [0, 4]]
]
embeddings1 = [
    [(0, 2), (1, 0), (0, 0), (1, 1), (0, 1)],
    [(0, 1), (0, 2), (0, 0), (1, 1), (1, 0)]
]





Rb =1
cz_circuit = CreateCircuitFromQASM("qft_10.qasm", "/Users/aaronbiggin/Downloads/NAQCT-release-version/Data/qiskit-bench/qft/qft_10")
gate_2q_list = get_2q_gates_list(cz_circuit)
n_q = get_qubits_num(gate_2q_list)
# print("gate_2q_list")
#print(gate_2q_list)
cir, dag = gates_list_to_QC(gate_2q_list)
#cz_circuit.draw(output="mpl",scale = 1.0)
#cir.draw(output="mpl",scale = 1.0)
plt.show()
gate_num = len(gate_2q_list)
num_q = get_qubits_num(gate_2q_list)
arch_size = math.ceil(math.sqrt(num_q))
coupling_graph = generate_grid_with_Rb(arch_size, arch_size, Rb)
partitions = partition_from_DAG(dag, coupling_graph)
embeddings, extend_pos = get_embeddings(partitions, coupling_graph, num_q, arch_size, Rb)
arch_size = len(extend_pos)
movement_list = process_all_embeddings(num_q, embeddings)
layer_list = layer_creation(movement_list, embeddings, partitions)
print(arch_size)
route = QuantumRouter(num_q, embeddings, partitions, [4, 4])
route.run()
print(route.save_program(os.path.join(embeddings,"hi.json")))



# print(layer_list)
# for i, layer in enumerate(layer_list):
#     print("hi")
#     print(type(layer))
#     #visualize_layer_with_traps(layer_list[i], arch_size)

     

# for index_list_gate in range(len(gate_2q_list)):
#         data = []
#         data, final_mapping, time_placement_tmp = route_qubit_mis((arch_size, arch_size), n_q, index_list_gate, gate_2q_list, list(final_mapping), "maximalis", True, False, False)
# print("DASATOM")
# print(layer_list)
# print("ENOLA")
# print(data["layers"])



#print("movement_list")
#print(movement_list)


#visualize_embeddings(embeddings)
#print(partitions)
# visualize_partitions(partitions)
#print(len(partitions))
#dag_drawer(dag, 0.7, "DAG_QFT-5.png", "color")
# print(gate_2q_list)

# print("ENOLA")
# print(asap(gate_2q_list, n_q))

# # print(partitions)
# print("DASATOM")
# print(convert_partitions_to_gate_notation(partitions, gate_2q_list))









