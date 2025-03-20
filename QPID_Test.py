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
        den = [0.00125, 2, 0.01, 0.005]  # 0.5*s^3+s^2+s
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
        """Create quantum circuits for all measurements"""
        # 创建三个电路：z基，x基，y基测量
        qr_z = QuantumRegister(1, 'q')
        cr_z = ClassicalRegister(1, 'c')
        qc_z = QuantumCircuit(qr_z, cr_z)

        # 准备初始态并应用旋转
        error_angle = np.pi * np.clip(error / 1000, -1, 1)
        qc_z.rx(error_angle, qr_z[0])
        qc_z.measure(qr_z, cr_z)

        # x基测量电路
        qr_x = QuantumRegister(1, 'q')
        cr_x = ClassicalRegister(1, 'c')
        qc_x = QuantumCircuit(qr_x, cr_x)
        qc_x.rx(error_angle, qr_x[0])
        qc_x.h(qr_x[0])  # 转换到x基
        qc_x.measure(qr_x, cr_x)

        # y基测量电路
        qr_y = QuantumRegister(1, 'q')
        cr_y = ClassicalRegister(1, 'c')
        qc_y = QuantumCircuit(qr_y, cr_y)
        qc_y.rx(error_angle, qr_y[0])
        #qc_y.sdg(qr_y[0])  # S†门
        #qc_y.h(qr_y[0])  # Hadamard门
        qc_y.ry(error_angle/2,qr_y[0])
        qc_y.measure(qr_y, cr_y)

        return qc_z, qc_x, qc_y

    def get_quantum_measurements(self, circuits):
        """Get measurements in all bases"""
        qc_z, qc_x, qc_y = circuits
        backend = qiskit_aer.AerSimulator()

        # 执行所有测量
        job_z = backend.run(qc_z, shots=1000)
        job_x = backend.run(qc_x, shots=1000)
        job_y = backend.run(qc_y, shots=1000)

        # 获取结果
        counts_z = job_z.result().get_counts()
        counts_x = job_x.result().get_counts()
        counts_y = job_y.result().get_counts()
        print(counts_z)
        print(counts_x)
        print(counts_y)
        # 计算期望值
        z_exp = (counts_z.get('0', 0) - counts_z.get('1', 0)) / 1000
        x_exp = (counts_x.get('0', 0) - counts_x.get('1', 0)) / 1000
        y_exp = (counts_y.get('0', 0) - counts_y.get('1', 0)) / 1000

        # 计算角度
        theta = np.arccos(z_exp)
        phi = np.arctan2(y_exp, x_exp)
        if phi < 0:
            phi += 2 * np.pi

        print("Debug info:")
        print(f"Z measurement: {z_exp}")
        print(f"X measurement: {x_exp}")
        print(f"Y measurement: {y_exp}")
        print(f"Calculated theta: {theta}")
        print(f"Calculated phi: {phi}")

        return theta, phi


def simulate_system():
    """Simulate the system response"""
    controller = QuantumPIDController()

    # Time Span
    t = np.linspace(0, 50, 1000)
    # print(t)
    qc_z,qc_x,qc_y = controller.create_quantum_circuit(90)  # 假设初始误差为100
    theta, phi = controller.get_quantum_measurements([qc_z,qc_x,qc_y])
    p, i, d = controller.quantum_state_to_pid(theta, phi)
    real_p, real_i, real_d = controller.map_to_real_pid(p, i, d)

    # Response
    t, y, s = controller.system_response(real_p, real_i, real_d, t, signal=90)

    # y, t= multi_iteration(T=t,controller=controller)
    # print(y)
    # Plot the result
    plt.figure(figsize=(10, 6))
    plt.plot(t, y , 'b-', label='Speed')
    plt.plot(t, [90] * len(t), 'r--', label='Setpoint')
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