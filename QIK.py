import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.visualization import plot_histogram
from robot_descriptions.loaders.mujoco import load_robot_description
import matplotlib.pyplot as plt
import time
from inverse_dynamic_func import LevenbegMarquardtIK
import copy


# ================ 1. Forward kinematic for UR10e ================
joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
joint_indices = {}


def ur10e_dh_params():
    """DH parameters for UR10e"""
    return [
        [0, np.pi / 2, 0.1807, 0],  # 基座到肩部
        [-0.6127, 0, 0, 0],  # 肩部到肘部
        [-0.57155, 0, 0, 0],  # 肘部到腕部1
        [0, np.pi / 2, 0.17415, 0],  # 腕部1到腕部2
        [0, -np.pi / 2, 0.11985, 0],  # 腕部2到腕部3
        [0, 0, 0.11655, 0]  # 腕部3到末端执行器
    ]


def analytical_forward_kinematics(joint_angles):
    """使用DH参数计算UR10e前向运动学"""
    dh = ur10e_dh_params()
    T = np.eye(4)

    for i in range(len(joint_angles)):
        a, alpha, d, _ = dh[i]
        theta = joint_angles[i]

        # DH变换矩阵
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)

        Ti = np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])

        T = T @ Ti

    return T


def forward_kinematics(joint_angles, model=None, data=None):
    """Calculating the forward kinematic"""
    if model is not None and data is not None:
        # Calculating FK by mujoco
        data_cp=copy.deepcopy(data)
        for i in range(min(len(joint_angles), model.nv)):
            data_cp.qpos[i] = joint_angles[i]
        mujoco.mj_forward(model, data_cp)
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        position = data_cp.site_xpos[ee_id].copy()
        rotation_mat = data_cp.site_xmat[ee_id].reshape(3, 3).copy()
        return np.vstack([
            np.hstack([rotation_mat, position.reshape(-1, 1)]),
            np.array([[0, 0, 0, 1]])
        ])
    else:
        # Using analytical way
        return analytical_forward_kinematics(joint_angles)

# ================ 2. QAOA-IK Part including constructing the quantum circuit and measurement ================

def angle_to_binary(angle, min_angle, max_angle, n_bits):
    """Encoding degree to binary"""
    norm_angle = (angle - min_angle) / (max_angle - min_angle)
    norm_angle = max(0, min(0.999, norm_angle))
    int_val = int(norm_angle * (2 ** n_bits))
    return format(int_val, f'0{n_bits}b')


def binary_to_angle(binary, min_angle, max_angle):
    """BinaryToDegree"""
    int_val = int(binary, 2)
    norm_angle = int_val / (2 ** len(binary))
    return min_angle + norm_angle * (max_angle - min_angle)


def create_cost_unitary(gamma, target_pose, angle_ranges, bits_per_joint, model, data):
    """Constructing Cost Unitary Operator"""
    n_joints = len(angle_ranges)
    n_qubits = n_joints * bits_per_joint

    qc = QuantumCircuit(n_qubits)

    # Predicting the error
    for i in range(n_qubits):
        # Assigning the weight value
        weight = 2 ** (i % bits_per_joint) * (np.pi / 2) / (2 ** bits_per_joint)
        qc.rz(gamma * weight, i)

    # Add quadratic terms to approximate the nonlinearity of binary encoding
    for joint_idx in range(n_joints):
        start_idx = joint_idx * bits_per_joint
        for i in range(start_idx, start_idx + bits_per_joint):
            for j in range(i + 1, start_idx + bits_per_joint):
                # Interactions between Qubits within the same joint
                qc.cx(i, j)
                qc.rz(gamma * 0.1, j)  # 0.1 is the weight factor
                qc.cx(i, j)

    return qc


def create_mixer_unitary(beta,angle_ranges,bits_per_joint):
    """Constructing Mixer Unitary Operation"""
    n_qubits =len(angle_ranges)*bits_per_joint
    qc = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        qc.rx(2 * beta, i)

    return qc


def create_qaoa_circuit(gamma, beta, target_pose, angle_ranges, bits_per_joint, model, data, p=1):
    """Constructing the QC"""
    n_joints = len(angle_ranges)
    n_qubits = n_joints * bits_per_joint

    qc = QuantumCircuit(n_qubits)

    # Initialize to a uniform superposition state
    qc.h(range(n_qubits))

    # Apply p layers of QAOA
    for layer in range(p):
        # Cost Unitary Operator
        cost_circuit = create_cost_unitary(
            gamma[layer] if isinstance(gamma, list) or isinstance(gamma, np.ndarray) else gamma,
            target_pose, angle_ranges, bits_per_joint, model, data
        )
        qc = qc.compose(cost_circuit)

        # Mixer Unitary Operator
        mixer_circuit = create_mixer_unitary(
            beta[layer],angle_ranges, bits_per_joint
        )
        qc = qc.compose(mixer_circuit)

    # Measurement
    qc.measure_all()

    return qc


def calculate_ik_error(joint_angles, target_pose, model, data):
    """Calculate IK error"""
    current_pose = forward_kinematics(joint_angles, model, data)

    # Position error
    pos_error = np.linalg.norm(current_pose[:3, 3] - target_pose)

    # Orientation error (simplified to Frobenius norm) For simplification, I just ignore the orientation error
    #rot_error = np.linalg.norm(current_pose[:3, :3] - target_pose[:3, :3], 'fro')

    # Total error (position error has a higher weight)
    #total_error = pos_error + 0.2 * rot_error
    total_error = pos_error
    return total_error


def evaluate_qaoa_parameters(parameters, target_pose, angle_ranges, bits_per_joint, model, data, p=1, shots=1024):
    """Evaluate the quality of QAOA parameters"""
    n_params = len(parameters) // 2
    gamma = parameters[:n_params]
    beta = parameters[n_params:]
    # Create QAOA circuit
    qc = create_qaoa_circuit(gamma, beta, target_pose, angle_ranges, bits_per_joint, model, data, p)

    # Execute the circuit
    backend=AerSimulator()
    job= backend.run(qc,shots=shots)
    result = job.result()
    counts = result.get_counts(qc)

    # Calculate the expected error
    total_error = 0
    for bit_string, count in counts.items():
        # Decode to joint angles
        joint_angles = []
        for i in range(len(angle_ranges)):
            start_idx = i * bits_per_joint
            joint_bin = bit_string[start_idx:start_idx + bits_per_joint]
            min_angle, max_angle = angle_ranges[i]
            angle = binary_to_angle(joint_bin, min_angle, max_angle)
            joint_angles.append(angle)

        # Calculate the error
        error = calculate_ik_error(joint_angles, target_pose, model, data)
        total_error += error * count / shots

    return total_error


def quantum_ik_solver(target_pose, initial_guess, search_radius=0.5, bits_per_joint=3, p=1,
                      model=None, data=None, shots=500, verbose=True):
    """Use QAOA to search for an accurate IK solution around the initial guess"""
    # Define search range for each joint
    angle_ranges = []
    for angle in initial_guess:
        min_angle = max(angle - search_radius, -np.pi)
        max_angle = min(angle + search_radius, np.pi)
        angle_ranges.append((min_angle, max_angle))

    # QAOA parameters
    n_joints = len(initial_guess)
    n_qubits = n_joints * bits_per_joint

    # Initialize optimizer parameters
    initial_parameters = []
    for _ in range(p):
        initial_parameters.append(np.pi / 4)  # gamma
    for _ in range(p):
        initial_parameters.append(np.pi / 2)  # beta

    if verbose:
        print(f"Starting QAOA optimization (p={p}, qubits={n_qubits}, shots={shots})")

    # Define the optimizer by using COBYLA
    optimizer = COBYLA(maxiter=100)

    # Optimize QAOA parameters
    start_time = time.time()

    opt_result = optimizer.minimize(
        fun=lambda params: evaluate_qaoa_parameters(
            params, target_pose, angle_ranges, bits_per_joint, model, data, p, shots
        ),
        x0=initial_parameters
    )

    optimal_parameters = opt_result.x

    if verbose:
        print(f"QAOA parameter optimization completed, time taken: {time.time() - start_time:.2f} s")
        print(f"Optimal parameters: {optimal_parameters}")

    # Execute the QAOA circuit with the optimal parameters
    n_params = len(optimal_parameters) // 2
    gamma_opt = optimal_parameters[:n_params]
    beta_opt = optimal_parameters[n_params:]

    qc = create_qaoa_circuit(gamma_opt, beta_opt, target_pose, angle_ranges, bits_per_joint, model, data, p)

    # Run the circuit and get the results
    simulator = AerSimulator()
    transpiled_qc = transpile(qc, simulator)
    job = simulator.run(transpiled_qc, shots=shots)
    result = job.result()
    counts = result.get_counts(qc)

    if verbose:
        print(f"QAOA circuit execution completed, obtained {len(counts)} different measurement results")

    # Find the result with the minimum error
    best_error = float('inf')
    best_joint_angles = None

    for bit_string, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        # Decode to joint angles
        joint_angles = []
        for i in range(len(angle_ranges)):
            start_idx = i * bits_per_joint
            joint_bin = bit_string[start_idx:start_idx + bits_per_joint]
            min_angle, max_angle = angle_ranges[i]
            angle = binary_to_angle(joint_bin, min_angle, max_angle)
            joint_angles.append(angle)

        # Calculate the error of joint degrees
        joint_angles = np.array(joint_angles)
        error = calculate_ik_error(joint_angles, target_pose, model, data)

        if error < best_error:
            best_error = error
            best_joint_angles = joint_angles

            if verbose:
                print(f"Found a better solution: error={error:.6f}, probability={(count / shots) * 100:.2f}%")
                print(f"Joint angles: {np.degrees(joint_angles)}")


    if verbose:
        print(f"Final QAOA solution error={best_error:.6f}")
        print(f"Best_joint_angles={best_joint_angles}")

    return best_joint_angles, best_error


# ================ 3. Classic-Quantum Hybrid IK solver ================

def hybrid_quantum_ik(target_pose,model,data,classic_solution,bits_per_joint=3, p=1, search_radius=0.5, verbose=True):
    if verbose:
        print("\n=== Stage2: QAOA optimization ===")
    quantum_solution, quantum_error = quantum_ik_solver(
        target_pose, classic_solution, search_radius, bits_per_joint, p, model, data, verbose=verbose
    )

    return quantum_solution
