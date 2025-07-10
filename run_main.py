import mujoco
import numpy as np
import control as ctrl
import time
import os
import json
from openai import OpenAI
import matplotlib.pyplot as plt
from robot_descriptions.loaders.mujoco import load_robot_description
from inverse_dynamic_func import LevenbegMarquardtIK
from QPID_POS import QuantumPIDController,simulate_system
from QIK import hybrid_quantum_ik

# Set api key of deepseek
client = OpenAI(api_key="YOUR KEY",base_url="https://api.deepseek.com")  # 替换为你的API密钥

# Load MuJoCo Model
model = load_robot_description("ur10e_mj_description")
data = mujoco.MjData(model)
data.qpos=[0, -np.pi / 2, np.pi / 2, 3 * np.pi / 2, 3 * np.pi / 2, 0]

#Initialize the joint name and map to id
joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
joint_amp={"shoulder_pan_joint":1,"shoulder_lift_joint":1,"elbow_joint":1,"wrist_1_joint":1,"wrist_2_joint":1,"wrist_3_joint":1}
joint_indices = {}
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name in joint_names:
        joint_indices[name] = i

# Deepseek Prompt
SYSTEM_PROMPT = """You are an industrial robotic arm control expert. Please convert natural language instructions into standardized JSON control commands.

# Command Types
1. Joint Control: { "action": "joint_control", "joints": [{"name":"joint1", "angle":30}, ...] }
2. End Effector Movement: { "action": "move_to", "position": [x,y,z], "speed": 0.1 }
3. Tool Operation: { "action": "gripper", "state": "open/close" }

# Output Requirements
- Angle units are in degrees, coordinate system is centered at the base
- Return only standard JSON, do not include any other text
- When instructions are unclear, return { "action": "error" }

# Example 1
User: Rotate joint 2 to 45 degrees and joint 5 to -30 degrees
Assistant: {"action":"joint_control", "joints":[{"name":"joint2","angle":45},{"name":"joint5","angle":-30}]}

# Example 2
User: Rotate joint 1 to -90 degrees
Assistant: {"action":"joint_control", "joints":[{"name":"joint1","angle":-90}]}

# Example 3
User: Move the end effector 0.1 in the x direction
Assistant: { "action": "move_to", "position": [0.1,0,0], "speed": 0.1 }

# Example 4
User: Move the end effector -0.5 in the y direction
Assistant: { "action": "move_to", "position": [0,-0.5,0], "speed": 0.1 }

# Example 5
User: Move the end effector 0.4 in the z direction
Assistant: { "action": "move_to", "position": [0,0,0.4], "speed": 0.1 }

# Example 6
User: Move the end effector 0.01 in x direction, 0.05 in y direction, and -0.01 in z direction
Assistant: { "action": "move_to", "position": [0.01,0.05,-0.01], "speed": 0.1 }
"""


# Classical PID controller, I didn't use it in this project
class PIDController:
    def __init__(self, kp=27.5, ki=3.5, kd=5.0):
        self.kp = kp  # Kp
        self.ki = ki  # Ki
        self.kd = kd  # Kd

        self.prev_error = 0  # Initialize the error in previous time
        self.integral = 0  # Initialize the integral error

    def compute(self, target, current, dt):
        """Calculate the output"""
        # Calculate the error
        error = target - current

        # Calculate the integral error
        self.integral += error * dt

        # Calculate the derivative error
        derivative = (error - self.prev_error) / dt if dt > 0 else 0

        # Get the whole output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        # preserve the error in this time
        self.prev_error = error

        return output

    def reset(self):
        """Reset the controller"""
        self.prev_error = 0
        self.integral = 0


class AdvancedPIDController:
    def __init__(self, Kp, Ki, Kd, C1, C2, C3, T):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.T = T  # Sample time

        # Get the discrete transformation function
        self.b, self.a = self.create_discrete_pid()

        # Initialize the history
        self.e_history = [0.0] * (len(self.b) - 1)  # 误差历史（e[k-1], e[k-2], ...）
        self.y_history = [0.0] * (len(self.a) - 1)  # 输出历史（y[k-1], y[k-2], ...）

    def create_discrete_pid(self):
        s = ctrl.TransferFunction.s
        num = [self.Kd, self.Kp, self.Ki]  # numerator：s², s, c
        den = [self.C1, self.C2, self.C3]  # Denominator：s²,s, c
        G_continuous = ctrl.TransferFunction(num, den)

        # Use tustin to get the discrete transformation function
        G_discrete = ctrl.c2d(G_continuous, self.T, method='tustin')

        # Get the list of dominator and numerator
        b = G_discrete.num[0][0].tolist()
        a = G_discrete.den[0][0].tolist()

        return b, a

    def compute(self, target, current_value,amp):
        error = target - current_value

        # Calculate the numerator（b[0]*e[k] + b[1]*e[k-1] + ...）
        numerator = sum(b * e for b, e in zip(self.b, [error] + self.e_history))

        # Calculate the dominator（a[1]*y[k-1] + a[2]*y[k-2] + ...）
        denominator = sum(a * y for a, y in zip(self.a[1:], self.y_history))

        # Output：y[k] = (numerator - denominator) / a[0]
        output = (numerator - denominator) / self.a[0]

        # Update the history
        self.e_history = [error] + self.e_history[:-1]
        self.y_history = [output] + self.y_history[:-1]

        return amp*output

    def reset(self):
        self.e_history = [0.0] * (len(self.b) - 1)  # history of error（e[k-1], e[k-2], ...）
        self.y_history = [0.0] * (len(self.a) - 1)  # history of output（y[k-1], y[k-2], ...)

def get_joint_angles(data):
    """Get the current angles of the joints"""
    joint_angles = {}
    for name, idx in joint_indices.items():
        joint_angles[name] = data.qpos[idx]
    return joint_angles

def get_effector_coordinate(model,data):
    """Get the coordinate of end effector"""
    id= mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    coord=data.site_xpos[id]
    return coord

def degrees_to_radians(degrees):
    """Deg2Rad"""
    return degrees * np.pi / 180.0


def radians_to_degrees(radians):
    """Rad2Deg"""
    return radians * 180.0 / np.pi


def parse_instruction(instruction):
    """Get json format from Deepseek"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # 或你使用的模型
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction}
            ],
            temperature=0
        )
        json_str = response.choices[0].message.content.strip()
        return json.loads(json_str)
    except Exception as e:
        print(f"Error occurred: {e}")
        return {"action": "error"}


def apply_joint_torques(data, torques_dict):
    """Apply torques to joints"""
    for joint_name, torque in torques_dict.items():
        if joint_name in joint_indices:
            idx = joint_indices[joint_name]
            data.ctrl[idx] = torque


def simulate_with_pid(data, target_angles,target_point=None ,duration=1.5, timestep=0.001,task=None,p=None,i=None,d=None,c1=None,c2=None,c3=None):
    """Use controller to simulate the motion of the arm"""
    if task=="joint_control":
        pid_controllers = {name: AdvancedPIDController(Kp=p,Ki=i,Kd=d,C1=c1,C2=c2,C3=c3,T=1e-4) for name in joint_indices.keys()}
        renderer = mujoco.Renderer(model)

        # set the steps
        steps = int(duration / timestep)

        # Preserve the initial status to visualize
        renderer.update_scene(data)
        initial_img = renderer.render()

        # Use plt to visualize the simulation
        plt.figure(figsize=(15, 8))
        plt.ion()

        # Two figures：Initial status and current status
        plt.subplot(1, 2, 1)
        plt.imshow(initial_img)
        plt.title("Previous status")
        plt.axis('off')

        # Collect the history of angles of joints
        angle_history = {name: [] for name in joint_indices.keys()}
        time_points = []

        print(f"Target angle: {', '.join([f'{k}: {radians_to_degrees(v):.2f}°' for k, v in target_angles.items()])}")

        # Start to simulation
        for step in range(steps):
            current_time = step * timestep
            time_points.append(current_time)
            current_angles = get_joint_angles(data)

            # Get the outputs of the controller for every joint
            torques = {}
            for joint_name, target in target_angles.items():
                if joint_name in current_angles:
                    current = current_angles[joint_name]
                    torque = pid_controllers[joint_name].compute(target, current, amp=30)
                    torques[joint_name] = torque
                    #pid_controllers[joint_name].reset()
                    # Collect the history of angles
                    angle_history[joint_name].append(radians_to_degrees(current))

            # Apply to joints
            apply_joint_torques(data, torques)

            # Simulate by step
            mujoco.mj_step(model, data)

            # Render every 10 steps
            if step % 10 == 0:
                renderer.update_scene(data)
                current_img = renderer.render()

                plt.subplot(1, 2, 2)
                plt.imshow(current_img)
                plt.title("Current Status({}s)".format(current_time))
                plt.axis('off')

                plt.draw()
                plt.pause(0.001)
        for joint in joint_names:
            pid_controllers[joint].reset()
    if task=='move_to':
        pid_controllers = {name: AdvancedPIDController(Kp=p,Ki=i,Kd=d,C1=c1,C2=c2,C3=c3, T=1e-4) for name in joint_indices.keys()}
        id= mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        renderer = mujoco.Renderer(model)

        # set the steps
        steps = int(duration / timestep)

        # Preserve the initial status to visualize
        renderer.update_scene(data)
        initial_img = renderer.render()

        # Use plt to visualize the simulation
        plt.figure(figsize=(15, 8))
        plt.ion()

        # Two figures：Initial status and current status
        plt.subplot(1, 2, 1)
        plt.imshow(initial_img)
        plt.title("Previous status")
        plt.axis('off')

        # Get the outputs of the controller for every joint
        angle_history = {name: [] for name in joint_indices.keys()}
        time_points = []
        print("\n=== Stage1: Classical IK--LM algorithm ===")
        # Initialize the Inverse Kinematic method
        IK = LevenbegMarquardtIK(model, data, 0.5, 1e-4, 0.5, np.zeros((3, model.nv)), np.zeros((3, model.nv)), 0.15)
        error = np.subtract(target_point, data.site_xpos[id])
        classical_final_qpos=None
        for step in range(steps):
            current_time = step * timestep
            time_points.append(current_time)
            # Get the current angles
            current_angles = get_joint_angles(data)
            # Calculate the target angles
            target_qpos=IK.calculate(init_q=data.qpos,ee_id=id,error=error)
            for i,name in enumerate(joint_names):
                target_angles[name]= target_qpos[i]
            torques = {}
            for joint_name, target in target_angles.items():
                if joint_name in current_angles:
                    current = current_angles[joint_name]
                    torque = pid_controllers[joint_name].compute(target, current, amp=20)
                    torques[joint_name] = torque
                    # pid_controllers[joint_name].reset()
                    # Collect the history of angles
                    angle_history[joint_name].append(radians_to_degrees(current))
            # Apply torque to joints
            apply_joint_torques(data, torques)

            # Simulate by step
            mujoco.mj_step(model, data)
            error=np.subtract(target_point,data.site_xpos[id])
            # Render every 10 times
            if step % 10 == 0:
                renderer.update_scene(data)
                current_img = renderer.render()

                plt.subplot(1, 2, 2)
                plt.imshow(current_img)
                plt.title("Current Status({}s)".format(current_time))
                plt.axis('off')

                plt.draw()
                plt.pause(0.001)
            classical_final_qpos=target_qpos
        print("Classical final position:",data.site_xpos[id])
        print("Classical error:",np.linalg.norm(data.site_xpos[id]-target_point,ord=2))
        print("Finding solution by QAOA...")
        ft_target_angle=hybrid_quantum_ik(target_pose=target_point,model=model,data=data,classic_solution=classical_final_qpos,bits_per_joint=4,p=2,search_radius=0.3)
        initial_img=current_img
        plt.figure(figsize=(15, 8))
        plt.ion()

        # Two figures：Initial status and current status
        plt.subplot(1, 2, 1)
        plt.imshow(initial_img)
        plt.title("Previous status")
        plt.axis('off')

        for step in range(steps):
            current_time = duration+step * timestep
            time_points.append(current_time)
            current_angles = get_joint_angles(data)
            for i, name in enumerate(joint_names):
                target_angles[name] = ft_target_angle[i]
            torques = {}
            for joint_name, target in target_angles.items():
                if joint_name in current_angles:
                    current = current_angles[joint_name]
                    torque = pid_controllers[joint_name].compute(target, current, amp=30)
                    torques[joint_name] = torque
                    # pid_controllers[joint_name].reset()
                    # Collect the history of angles
                    angle_history[joint_name].append(radians_to_degrees(current))
            # Apply torque to joints
            apply_joint_torques(data, torques)

            # Simulate by step
            mujoco.mj_step(model, data)
            if step % 10 == 0:
                renderer.update_scene(data)
                current_img = renderer.render()

                plt.subplot(1, 2, 2)
                plt.imshow(current_img)
                plt.title("Current Status({}s)".format(current_time))
                plt.axis('off')

                plt.draw()
                plt.pause(0.001)
        print("Current effector position: ",data.site_xpos[id])
        print("Target effector position: ",target_point)
        print("Position error:",np.linalg.norm(data.site_xpos[id]-target_point,ord=2))
        for joint in joint_names:
            pid_controllers[joint].reset()

    # plot the response figure
    plt.figure(figsize=(12, 6))
    for joint_name, history in angle_history.items():
        if joint_name in target_angles:
            plt.plot(time_points, history, label=f"{joint_name} Response")
            # Target line
            target_deg = radians_to_degrees(target_angles[joint_name])
            plt.axhline(y=target_deg, color='r', linestyle='--',
                        label=f"{joint_name} Target ({target_deg:.2f}°)" if joint_name in list(target_angles.keys()) else "")

    plt.xlabel('Time(s)')
    plt.ylabel('Angle')
    plt.title('Angle-Time')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

    # Return the final status
    return get_joint_angles(data)


def execute_command(command, model, data,p,i,d,c1,c2,c3):
    """Execute the parsed command"""
    if command["action"] == "error":
        print("Error,please try again")
        return False

    elif command["action"] == "joint_control":
        target_angles = {}
        for joint in command["joints"]:
            joint_name = joint["name"]
            if joint_name.startswith("joint"):
                try:
                    index = int(joint_name[5:]) - 1
                    if 0 <= index < len(joint_names):
                        joint_name = joint_names[index]
                except:
                    pass

            if joint_name in joint_indices:
                target_angles[joint_name] = degrees_to_radians(joint["angle"])
            else:
                print(f"Unknown Joint: {joint_name}")

        if target_angles:
            # Getting the current joint angles
            current_angles = get_joint_angles(data)

            # For unspecified joints, use the current angle as the target
            for name in joint_indices.keys():
                if name not in target_angles and name in current_angles:
                    target_angles[name] = current_angles[name]

            # Using PID controller
            final_angles = simulate_with_pid(data=data,target_angles=target_angles,task='joint_control',p=p,i=i,d=d,c1=c1,c2=c2,c3=c3)
            print("Instruction Finished.")
            print(f"Final angles: {', '.join([f'{k}: {radians_to_degrees(v):.2f}°' for k, v in final_angles.items()])}")
            return True

    elif command["action"] == "move_to":
        coord=get_effector_coordinate(model, data)
        qpos0=data.qpos
        print("Current coordinate of end effector (x,y,z)：",coord)
        target_point= coord+np.array(command["position"])
        print("Movement（x,y,z）:",command["position"])
        print("Target position（x,y,z）:",target_point) # Movement + current position
        target_angles={}
        final_angles = simulate_with_pid(data, target_angles=target_angles,target_point=target_point,task='move_to',p=p,i=i,d=d,c1=c1,c2=c2,c3=c3)
        print("Instruction Finished.")
        print(f"Final angles: {', '.join([f'{k}: {radians_to_degrees(v):.2f}°' for k, v in final_angles.items()])}")
        return True

    elif command["action"] == "gripper":
        print(f"Gripper: {command['state']}")
        return True

    return False


def main(p,i,d,c1,c2,c3):
    # Initialize the model
    mujoco.mj_forward(model, data)

    running = True

    while running:
        # Getting the instruction from the user
        instruction = input("\nPlease input instruction（input 'quit' to quit）: ")

        if instruction.lower() in ['退出', 'exit', 'quit']:
            running = False
            continue

        # Analyse the instruction
        print("Analyse the instruction...")
        command = parse_instruction(instruction)
        print(f"Command: {command}")

        # Execute
        success = execute_command(command, model,data,p,i,d,c1,c2,c3)

        if not success:
            print("Invalid command.")

    print("Quitting...")
    plt.close('all')


if __name__ == "__main__":
    print("Searching Parameters of Controller...")
    real_p, real_i, real_d, real_c1, real_c2, real_c3= simulate_system()
    main(p=real_p,i=real_i,d=real_d,c1=real_c1,c2=real_c2,c3=real_c3)