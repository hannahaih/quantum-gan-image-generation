import pennylane as qml

@qml.template
def _u_ent(wires):
    """
    Quantum circuit for entangling qubits with CZ gates
    Args:
        wires (list of int): list of labeled qubits in the circuit
    Returns:
        none
    """
    for i in range(len(wires)):
        qml.CZ(wires=[wires[i], wires[(i+1) % len(wires)]])


@qml.template
def _circuit_init(wires):
    """
    Apply Hadmard gate on each qubit
    Args:
        wires (int): list of labeled qubits in the circuit
    Returns:
        none
    """
    for wire in wires:
        qml.Hadamard(wires=wire)


@qml.template
def _circuit_body(params, wires):
    """
    Construct the main body of the quantum circuit of the generator
    Args:
        params (flat array of float): variational parameter for the circuit, passed into RY gates
        wires (list of int): list of labeled qubits in the circuit
    Returns:
        none
    """
    for param in params[:-1]:
        for wire in wires:
            qml.RY(param[wire], wires=wire)
        _u_ent(wires)
    for wire in wires:
            qml.RY(params[-1, wire], wires=wire)


def generator_circuit(params, num_qubits):
    """
    Constructs the generator quantum circuit
    Args: 
        params (flat array of float): variational parameter for the circuit, 
        passed into RY gates
        
        num_qubits (int): # of qubits in the circuit
    Returns:
        (list): list of measured qubit states
    """
    wires = list(range(num_qubits))
    _circuit_init(wires)
    _circuit_body(params, wires)
    return [qml.sample(qml.PauliZ(i)) for i in wires]


def generator_prob_circuit(params, num_qubits):
    """
    Constructs the generator quantum circuit
    Args: 
        params (flat array of float): variational parameter for the circuit, 
        passed into RY gates

        num_qubits (int): # of qubits in the circuit
    Returns:
        (flat array of float): array of the probabilities of measuring computational basis states i 
        given the current state
    """
    wires = list(range(num_qubits))
    _circuit_init(wires)
    _circuit_body(params, wires)
    return qml.probs(wires)