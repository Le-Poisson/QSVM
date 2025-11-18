import math
from itertools import combinations
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

class ZZFeatureMap:
    def __init__(self, feature_dimension, reps):
        self.nqbits = feature_dimension
        self.reps = reps
        self.circuit = QuantumCircuit(self.nqbits)
        
    def construct_circuit(self, X):
        qc = QuantumCircuit(self.nqbits)
        
        qc.h(range(self.nqbits))
        
        for i in range(self.nqbits):
            qc.rz(2 * X[i], i)
        
        for combo in combinations(range(self.nqbits), 2):
            qc.cx(combo[0], combo[1])
            qc.rz(2 * ((math.pi - X[combo[0]]) * (math.pi - X[combo[1]])), combo[1])
            qc.cx(combo[0], combo[1])
            
        for i in range(self.reps):
            self.circuit.compose(qc, inplace=True)
            
def ZZFeatureValue(X1, X2):
    ZZFM1 = ZZFeatureMap(feature_dimension=len(X1), reps=1)
    ZZFM2 = ZZFeatureMap(feature_dimension=len(X2), reps=1)
    ZZFM1.construct_circuit(X1)
    ZZFM2.construct_circuit(X2)
    
    qc = QuantumCircuit(len(X1))
    qc.compose(ZZFM1.circuit, inplace=True)
    qc.compose(ZZFM2.circuit.inverse(), inplace=True)
    
    # 直接计算态矢量，加快计算速度
    state = Statevector.from_instruction(qc)
    return float(abs(state[0]) ** 2)