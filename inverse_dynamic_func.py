import mujoco
import numpy as np
from mujoco import renderer
from robot_descriptions.loaders.mujoco import load_robot_description
import mediapy as media
import matplotlib.pyplot as plt

# Levenberg-Marquardt method
class LevenbegMarquardtIK:
    def __init__(self, model, data, step_size, tol, alpha, jacp, jacr, damping):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp = jacp
        self.jacr = jacr
        self.damping = damping

    def check_joint_limits(self, q):
        """Check if the joints is under or above its limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0],
                       min(q[i], self.model.jnt_range[i][1]))

    # Levenberg-Marquardt pseudocode implementation
    def calculate(self, init_q, ee_id,error):
        """Calculate the desire joints angles for goal"""
        current_pose = self.data.site_xpos[ee_id]
        site_body_id = self.model.site_bodyid[ee_id]
        point_local = self.model.site_pos[ee_id]
        # calculate jacobian
        if (np.linalg.norm(error) >= self.tol):
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, point_local, site_body_id)
            # calculate delta of joint q
            n = self.jacp.shape[1]
            I = np.identity(n)
            product = self.jacp.T @ self.jacp + self.damping * I

            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ self.jacp.T
            else:
                j_inv = np.linalg.inv(product) @ self.jacp.T

            delta_q = j_inv @ error
            # compute next step
            init_q += self.step_size * delta_q
        # check limits
        self.check_joint_limits(init_q)
        # compute forward kinematics
        # calculate new error
        #error = np.subtract(goal, self.data.body(body_id).xpos)
        return init_q

if __name__=='__main__':
    model= load_robot_description("ur10e_mj_description")
    data= mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.distance=1
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.distance = 1
    qpos0=[3 * np.pi / 2, -np.pi / 2, np.pi / 2, 3 * np.pi / 2, 3 * np.pi / 2, 0]
    data.qpos=qpos0
    body_id = model.body('wrist_3_link').id
    jacp = np.zeros((3, model.nv))  # translation jacobian
    jacr = np.zeros((3, model.nv))  # rotational jacobian
    goal = [0.49, 0.13, 0.59]
    step_size = 0.5
    tol = 0.01
    alpha = 0.5
    init_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    damping = 0.15

    ik = LevenbegMarquardtIK(model, data, step_size, tol, alpha, jacp, jacr, damping)

    # Get desire point
    mujoco.mj_resetDataKeyframe(model, data, 1)  # reset qpos to initial value
    ik.calculate(goal, init_q, body_id)  # calculate the qpos

    result = data.qpos.copy()

    # Plot results
    print("Results")
    data.qpos = qpos0
    mujoco.mj_forward(model, data)
    viewer= mujoco.Renderer(model)
    scene_option = mujoco.MjvOption()
    viewer.update_scene(data)
    target_plot = viewer.render()

    data.qpos = result
    mujoco.mj_forward(model, data)
    result_point = data.body('wrist_3_link').xpos
    viewer.update_scene(data)
    result_plot = viewer.render()

    print("testing point =>", data.body('wrist_3_link').xpos)
    print("Levenberg-Marquardt result =>", result_point, "\n")

    # Display images using matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(target_plot)
    axes[0].set_title('Testing point')
    axes[0].axis('off')
    axes[1].imshow(result_plot)
    axes[1].set_title('Levenberg-Marquardt result')
    axes[1].axis('off')
    plt.show()

