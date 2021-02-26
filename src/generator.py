import pennylane as qml

@qml.template
def _u_ent(wires):
    for i in range(len(wires)):
        qml.CZ(wires=[wires[i], wires[(i+1) % len(wires)]])


@qml.template
def _circuit_init(wires):
    for wire in wires:
        qml.Hadamard(wires=wire)


@qml.template
def _circuit_body(params, wires):
    for param in params[:-1]:
        for wire in wires:
            qml.RY(param[wire], wires=wire)
        _u_ent(wires)
    for wire in wires:
            qml.RY(params[-1, wire], wires=wire)


def generator_circuit(params, num_qubits):
    wires = list(range(num_qubits))
    _circuit_init(wires)
    _circuit_body(params, wires)
    return [qml.sample(qml.PauliZ(i)) for i in wires]


def generator_prob_circuit(params, num_qubits):
    wires = list(range(num_qubits))
    _circuit_init(wires)
    _circuit_body(params, wires)
    return qml.probs(wires)