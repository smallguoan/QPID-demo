import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import qiskit_aer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from control import tf, feedback, series,tf2ss,forced_response  # 使用python-control库


class QuantumPIDController:
    def __init__(self):
        # Motor Parameter
        self.K = 1  # 转矩常数
        self.J = 0.1  # 转动惯量
        self.B = 0.75  # 阻尼系数

        # The ranges of PID
        self.pid_ranges = {
            'p': (0, 10),
            'i': (0, 1),
            'd': (0, 0.1)
        }

        # Construct transfer function
        self.motor_tf = tf([self.K], [self.J, self.B])

    def get_pid_tf(self, Kp, Ki, Kd):
        """
        Construst PID transfer function
        """
        num = [Kd, Kp, Ki]  # Kd*s^2 + Kp*s + Ki
        den = [0.125, 1.5, 0, 0.00005]  # 0.5*s^3+s^2+s
        return tf(num, den)

    def get_closed_loop_tf(self, pid_tf):
        """
        Construct closed loop transfer function
        """
        # 串联PID和电机传递函数
        open_loop = series(pid_tf, self.motor_tf)
        # 构建负反馈闭环
        closed_loop = feedback(open_loop, 1)
        return closed_loop

    def system_response(self, p, i, d, t, signal):
        """
        Caluculate the system response(step signal)
        """
        # get the Kp Ki and Kd
        pid_tf = self.get_pid_tf(p, i, d)
        # get the closed loop function
        #closed_loop = self.get_closed_loop_tf(pid_tf)
        #print(closed_loop)
        # Calculate the response
        #t, y = signal.step(closed_loop, T=t)

        pid_ss = tf2ss(pid_tf)

        motor_ss = tf2ss(self.motor_tf)

        open_loop_ss = series(pid_ss, motor_ss)
        closed_loop_ss = feedback(open_loop_ss, 1)
        s= signal
        t, y= forced_response(closed_loop_ss, T=t, U=s)
        return t, y, s

    def quantum_state_to_pid(self, theta, phi):
        """map theta and phi to PID parameters"""
        p = (np.cos(theta) + 1) / 2
        i = phi / (2 * np.pi)
        d = np.cos(theta)
        return p, i, d

    def map_to_real_pid(self, p, i, d):
        """map PID parameters to real PID parameters"""
        real_p = self.pid_ranges['p'][0] + p * (self.pid_ranges['p'][1] - self.pid_ranges['p'][0])
        real_i = self.pid_ranges['i'][0] + i * (self.pid_ranges['i'][1] - self.pid_ranges['i'][0])
        d_mid = (self.pid_ranges['d'][1] + self.pid_ranges['d'][0]) / 2
        d_range = (self.pid_ranges['d'][1] - self.pid_ranges['d'][0]) / 2
        real_d = d_mid + d * d_range
        return real_p, real_i, real_d

    def create_quantum_circuit(self, error):
        """Create a quantum circuit to measure theta and phi"""
        qr = QuantumRegister(2, 'q')
        cr = ClassicalRegister(3, 'c')
        qc = QuantumCircuit(qr, cr)

        qc.h(qr[0])
        qc.h(qr[1])

        error_angle = np.pi * np.clip(error / 1000, -1, 1)

        qc.rx(error_angle, qr[0])
        qc.rz(error_angle, qr[1])

        qc.measure_all()

        return qc

    def get_quantum_measurements(self, qc):
        """get the result of measurement"""
        backend = qiskit_aer.AerSimulator()
        job = backend.run(qc, shots=1000)
        counts = job.result().get_counts()

        z_measurement = 0
        x_measurement = 0
        total_shots = 1000

        for key, value in counts.items():
            if key[0] == '1':
                z_measurement += value / total_shots
            if key[1] == '1':
                x_measurement += value / total_shots

        theta = np.arccos(2 * z_measurement - 1)
        phi = 2 * np.pi * x_measurement

        return theta, phi


def simulate_system():
    """Simulate the system response"""
    controller = QuantumPIDController()

    # Time Span
    t = np.linspace(0, 50, 1000)
    # print(t)
    qc = controller.create_quantum_circuit(1)  # 假设初始误差为100
    theta, phi = controller.get_quantum_measurements(qc)
    p, i, d = controller.quantum_state_to_pid(theta, phi)
    real_p, real_i, real_d = controller.map_to_real_pid(p, i, d)

    # Response
    t, y, s = controller.system_response(real_p, real_i, real_d, t, signal=1)

    # y, t= multi_iteration(T=t,controller=controller)
    # print(y)
    # Plot the result
    plt.figure(figsize=(10, 6))
    plt.plot(t, y * 1000, 'b-', label='Speed')
    plt.plot(t, [1000] * len(t), 'r--', label='Setpoint')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (rpm)')
    plt.title('BLDC Motor Speed Control with Quantum PID')
    plt.legend()
    plt.grid(True)
    plt.show()

    analysis(t,y)

    # Print the final PID parameter
    print(f'PID Parameters: P={real_p:.3f}, I={real_i:.3f}, D={real_d:.3f}')

def multi_iteration(T,controller):
    scale = np.shape(T)[0]
    qc = controller.create_quantum_circuit(100)
    res=np.zeros(scale)
    y_pre = 0
    for sc in range(scale-2):
        t_slice=T[sc:sc+2]
        theta, phi = controller.get_quantum_measurements(qc)
        p, i, d = controller.quantum_state_to_pid(theta, phi)
        real_p, real_i, real_d = controller.map_to_real_pid(p, i, d)
        t, y, s = controller.system_response(real_p, real_i, real_d,t_slice,signal=1)
        new_error = s-(y[-1]+y_pre)
        qc= controller.create_quantum_circuit(new_error)
        res[sc:sc+1]=y[-1]+y_pre
        y_pre=y[-1]+y_pre
    return res, T

def analysis(time,response):
    max_speed = max(response)
    max_speed_index = list(response).index(max_speed)
    time_max_speed = time[max_speed_index]

    print(f'Maximum Speed: {max_speed:.3f} rpm,at time {time_max_speed:.3f} s')

if __name__ == "__main__":
    simulate_system()
