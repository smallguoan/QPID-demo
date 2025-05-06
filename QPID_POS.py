import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import qiskit_aer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from control import tf, feedback, series,tf2ss,forced_response

class QuantumPIDController:
    def __init__(self):
        # 电机参数 Parameters of BLDC Motor
        self.K = 1
        self.J = 0.1
        self.B = 0.75

        # The range of PID
        self.pid_ranges = {
            'p': (0, 10),
            'i': (0, 1),
            'd': (0, 0.1)
        }

        # The range of Dominator
        self.den_ranges = {
            'c1': (0.001, 0.01),  # s^3系数
            'c2': (0.5, 5),  # s^2系数
            'c3': (0.001, 0.01)  # s^1系数
        }

        # Position transformation function
        self.motor_tf = tf([self.K], [self.J, self.B,0])  # K/(Js^2 + Bs)

    def get_pid_tf(self, Kp, Ki, Kd, c1, c2, c3):
        """
        Construct the tf for position control
        """
        num = [Kd, Kp, Ki]  # Kd*s^2 + Kp*s + Ki
        den = [c1, c2, c3]  # c1*s^2 + c2*s + c3
        return tf(num, den)

    def quantum_state_to_pid(self, theta, phi, den_norms):
        """
        Map Quantum state to the parameters of controller
        theta, phi: for numerator
        den_norms: for denominator
        """
        # Map the numerator
        p = (np.cos(theta) + 1) / 2
        i = phi / (2 * np.pi)
        d = np.cos(theta)

        # Map the denominator
        c1_norm, c2_norm, c3_norm = den_norms
        c1 = c1_norm  # s^2系数
        c2 = c2_norm  # s系数
        c3 = c3_norm  # 常数项

        return p, i, d, c1, c2, c3


    def map_to_real_pid(self, p, i, d, c1, c2, c3):
        """Map to real value of the parameter for the controller"""
        # Numerator
        real_p = self.pid_ranges['p'][0] + p * (self.pid_ranges['p'][1] - self.pid_ranges['p'][0])
        real_i = self.pid_ranges['i'][0] + i * (self.pid_ranges['i'][1] - self.pid_ranges['i'][0])
        real_d = self.pid_ranges['d'][0] + d * (self.pid_ranges['d'][1] - self.pid_ranges['d'][0])

        # Denominator
        real_c1 = self.den_ranges['c1'][0] + c1 * (self.den_ranges['c1'][1] - self.den_ranges['c1'][0])
        real_c2 = self.den_ranges['c2'][0] + c2 * (self.den_ranges['c2'][1] - self.den_ranges['c2'][0])
        real_c3 = self.den_ranges['c3'][0] + c3 * (self.den_ranges['c3'][1] - self.den_ranges['c3'][0])

        return real_p, real_i, real_d, real_c1, real_c2, real_c3

    def create_quantum_circuit(self, error):
        """
        Construct the quantum circuit
        """
        # 创建量子比特和经典比特 Construct Qubit
        qr_pid = QuantumRegister(1, 'q_pid')
        cr_pid = ClassicalRegister(1, 'c_pid')
        qr_den = QuantumRegister(2, 'q_den')  # 用于分母系数 For denominator
        cr_den = ClassicalRegister(2, 'c_den')

        # 误差角度映射 Map the error of angles
        error_angle = np.pi * np.clip(error / 180, -1, 1)  # 假设最大误差为180度 Assume that the maximum error is 180 DEG

        # PID参数的测量电路 Measurement part
        # Z基测量 Measure on pauliZ
        qc_z = QuantumCircuit(qr_pid, cr_pid)
        qc_z.rx(error_angle, qr_pid[0])
        qc_z.measure(qr_pid, cr_pid)

        # X基测量 Measure on pauliX
        qc_x = QuantumCircuit(qr_pid, cr_pid)
        qc_x.rx(error_angle, qr_pid[0])
        qc_x.h(qr_pid[0])
        qc_x.measure(qr_pid, cr_pid)

        # Y基测量 Measure on pauliY
        qc_y = QuantumCircuit(qr_pid, cr_pid)
        qc_y.rx(error_angle, qr_pid[0])
        qc_y.ry(error_angle / 2, qr_pid[0])
        qc_y.measure(qr_pid, cr_pid)

        # 分母系数的测量电路 Measure the constant of Denominators
        qc_den = QuantumCircuit(qr_den, cr_den)
        # 第一个量子比特用于C1和C2 Quantum Circuit for C1 and C2
        qc_den.rx(error_angle / 2, qr_den[0])
        qc_den.h(qr_den[0])
        # 第二个量子比特用于C3 Quantum Circuit for C3
        qc_den.ry(error_angle / 3, qr_den[1])
        qc_den.measure(qr_den, cr_den)

        return qc_z, qc_x, qc_y, qc_den

    def get_quantum_measurements(self, circuits):
        """
        获取量子测量结果，现在包括PID参数和分母系数的测量
        Get the result of measurements
        """
        qc_z, qc_x, qc_y, qc_den = circuits
        backend = qiskit_aer.AerSimulator()
        shots = 2000

        # PID参数测量 Measurement of the numerator
        # Z基测量 PauliZ
        job_z = backend.run(qc_z, shots=shots)
        counts_z = job_z.result().get_counts()
        z_exp = (counts_z.get('0', 0) - counts_z.get('1', 0)) / shots

        # X基测量 PauliX
        job_x = backend.run(qc_x, shots=shots)
        counts_x = job_x.result().get_counts()
        x_exp = (counts_x.get('0', 0) - counts_x.get('1', 0)) / shots

        # Y基测量 PauliY
        job_y = backend.run(qc_y, shots=shots)
        counts_y = job_y.result().get_counts()
        y_exp = (counts_y.get('0', 0) - counts_y.get('1', 0)) / shots

        # 分母系数测量 Measurement of the Denominator
        job_den = backend.run(qc_den, shots=shots)
        counts_den = job_den.result().get_counts()

        # 从测量结果计算角度 Calculate the angles
        theta = np.arccos(z_exp)
        phi = np.arctan2(y_exp, x_exp)
        if phi < 0:
            phi += 2 * np.pi

        # 计算分母系数的基础值 Calculate the Denominator
        den_measurements = {
            '00': counts_den.get('00', 0),
            '01': counts_den.get('01', 0),
            '10': counts_den.get('10', 0),
            '11': counts_den.get('11', 0)
        }

        # 调试信息 Final result
        print("\nQuantum Measurements:")
        print(f"Z measurement: {z_exp:.3f}")
        print(f"X measurement: {x_exp:.3f}")
        print(f"Y measurement: {y_exp:.3f}")
        print(f"Theta: {theta:.3f}")
        print(f"Phi: {phi:.3f}")
        print("Denominator measurements:", den_measurements)

        # 计算分母系数的归一化值 Normalization
        c1_norm = den_measurements['00'] / shots
        c2_norm = den_measurements['01'] / shots
        c3_norm = den_measurements['10'] / shots

        return theta, phi, (c1_norm, c2_norm, c3_norm)

    def system_response(self, p, i, d, c1, c2, c3, t, target_position):
        """计算系统位置响应
        Calculate the response of position"""
        # 获取PID传递函数 Get the tf
        pid_tf = self.get_pid_tf(p, i, d, c1, c2, c3)

        # 构建闭环系统 Construct the close loop system
        open_loop = series(pid_tf, self.motor_tf)
        closed_loop = feedback(open_loop, 1)

        # 计算位置响应 Calculate the response
        t, y = forced_response(closed_loop, T=t, U=[target_position] * len(t))
        return t, y, target_position


def simulate_system():
    """模拟系统位置控制响应
    Simulate the system"""
    controller = QuantumPIDController()
    t = np.linspace(0, 50, 1000)
    target_position = np.pi / 2  # 目标位置

    # 获取量子测量结果 Get the Quantum Measurement
    circuits = controller.create_quantum_circuit(target_position)
    theta, phi, den_norms = controller.get_quantum_measurements(circuits)

    # 获取系统参数 Get the parameter of the controller
    p, i, d, c1, c2, c3 = controller.quantum_state_to_pid(theta, phi, den_norms)
    real_p, real_i, real_d, real_c1, real_c2, real_c3 = \
        controller.map_to_real_pid(p, i, d, c1, c2, c3)

    # 系统响应 System response
    t, y, target = controller.system_response(
        real_p, real_i, real_d,
        real_c1, real_c2, real_c3,
        t, target_position
    )

    # 绘图 Plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, y, 'b-', label='Position')
    plt.plot(t, [target_position] * len(t), 'r--', label='Target Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (rad)')
    plt.title('BLDC Motor Position Control with Quantum PID')
    plt.legend()
    plt.grid(True)

    param_text = f'PID: P={real_p:.3f}, I={real_i:.3f}, D={real_d:.3f}\n'
    param_text += f'Den: C1={real_c1:.3f}, C2={real_c2:.3f}, C3={real_c3:.3f}'
    plt.text(0.02, 0.98, param_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.show()

    # Analyze the performance
    settling_idx = np.where(abs(y - target_position) <= 0.02 * target_position)[0]
    if len(settling_idx) > 0:
        settling_time = settling_idx[0] * t[1]
    else:
        settling_time = None
    print(settling_time)
    #settling_time = np.where(abs(y - target_position) <= 0.02 * target_position)[0][0] * t[1]
    overshoot = (max(y) - target_position) / target_position * 100 if max(y) > target_position else 0
    steady_state_error = abs(y[-1] - target_position)

    print("\nSystem Performance:")
    print(f"Settling Time: {settling_time:.3f} s")
    print(f"Overshoot: {overshoot:.2f}%")
    print(f"Steady State Error: {steady_state_error:.6f} rad")
    print(f"\nSystem Parameters:")
    print(f"PID Parameters: P={real_p:.3f}, I={real_i:.3f}, D={real_d:.3f}")
    print(f"Denominator Parameters: C1={real_c1:.3f}, C2={real_c2:.3f}, C3={real_c3:.3f}")

    return real_p, real_i, real_d, real_c1, real_c2, real_c3

if __name__ == "__main__":
    real_p, real_i, real_d, real_c1, real_c2, real_c3= simulate_system()