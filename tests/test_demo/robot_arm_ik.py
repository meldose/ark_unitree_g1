#!/usr/bin/env python3

import numpy as np
import pinocchio as pin
import meshcat.geometry as mg
import time
from pinocchio.robot_wrapper import RobotWrapper

try:
    from pinocchio.visualize import MeshcatVisualizer
    USE_MESHCAT = True
except ImportError:
    MeshcatVisualizer = None
    USE_MESHCAT = False


# =========================================================
# Utility: Weighted Moving Average Filter
# =========================================================
class WeightedMovingFilter:
    def __init__(self, weights, data_size):
        self.weights = np.asarray(weights)
        assert np.isclose(self.weights.sum(), 1.0)
        self.window = len(weights)
        self.data_size = data_size
        self.queue = []
        self.filtered = np.zeros(data_size)

    def add_data(self, x):
        x = np.asarray(x).copy()
        if len(self.queue) > 0 and np.allclose(x, self.queue[-1]):
            return
        self.queue.append(x)
        if len(self.queue) > self.window:
            self.queue.pop(0)
        self._update()

    def _update(self):
        if len(self.queue) < self.window:
            self.filtered = self.queue[-1]
        else:
            arr = np.vstack(self.queue)
            self.filtered = np.sum(arr * self.weights[:, None], axis=0)

    @property
    def filtered_data(self):
        return self.filtered


# =========================================================
# G1 – 29 DoF Arm IK (NUMERIC, STABLE)
# =========================================================
class G1_29_ArmIK:
    def __init__(self, Unit_Test=False, Visualization=False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization

        self.robot = RobotWrapper.BuildFromURDF(
            '../../unitree_g1/urdf/urdf/g1_description.urdf',
            '../../unitree_g1/urdf/'
        )

        # -----------------------------------------------------
        # Lock lower body + hands
        # -----------------------------------------------------
        self.mixed_jointsToLockIDs = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint",
            "left_hand_thumb_0_joint", "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint", "left_hand_middle_0_joint",
            "left_hand_middle_1_joint", "left_hand_index_0_joint",
            "left_hand_index_1_joint",
            "right_hand_thumb_0_joint", "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint", "right_hand_index_0_joint",
            "right_hand_index_1_joint", "right_hand_middle_0_joint",
            "right_hand_middle_1_joint"
        ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.zeros(self.robot.model.nq)
        )

        # -----------------------------------------------------
        # End-effector frames
        # -----------------------------------------------------
        self.reduced_robot.model.addFrame(
            pin.Frame(
                'L_ee',
                self.reduced_robot.model.getJointId('left_wrist_yaw_joint'),
                pin.SE3(np.eye(3), np.array([0.10, -0.03, 0.0])),
                pin.FrameType.OP_FRAME
            )
        )

        self.reduced_robot.model.addFrame(
            pin.Frame(
                'R_ee',
                self.reduced_robot.model.getJointId('right_wrist_yaw_joint'),
                pin.SE3(np.eye(3), np.array([0.10, 0.03, 0.0])),
                pin.FrameType.OP_FRAME
            )
        )

        # IMPORTANT: recreate Data AFTER modifying model
        self.model = self.reduced_robot.model
        self.data = self.model.createData()

        self.L_id = self.model.getFrameId("L_ee")
        self.R_id = self.model.getFrameId("R_ee")

        assert self.L_id < self.model.nframes
        assert self.R_id < self.model.nframes

        self.q = pin.neutral(self.model)
        self.filter = WeightedMovingFilter([0.4, 0.3, 0.2, 0.1], self.model.nq)

        # -----------------------------------------------------
        # Visualization
        # -----------------------------------------------------
        self.vis = None
        if self.Visualization and USE_MESHCAT:
            self.vis = MeshcatVisualizer(
                self.model,
                self.reduced_robot.collision_model,
                self.reduced_robot.visual_model
            )
            self.vis.initViewer(open=True)
            self.vis.loadViewerModel("pinocchio")
            self.vis.display(self.q)

    # ---------------------------------------------------------
    # Numeric FK
    # ---------------------------------------------------------
    def compute_fk(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.L_id], self.data.oMf[self.R_id]

    # ---------------------------------------------------------
    # IK Solver (Damped Least Squares, WORLD frame)
    # ---------------------------------------------------------
    def solve_ik(self, T_L, T_R, q_init=None):
        q = q_init.copy() if q_init is not None else self.q.copy()

        lam = 1e-4  # damping
        max_step = 0.1

        for _ in range(20):
            oMl, oMr = self.compute_fk(q)

            e_pos = np.hstack([
                oMl.translation - T_L[:3, 3],
                oMr.translation - T_R[:3, 3],
            ])

            e_rot = np.hstack([
                pin.log3(oMl.rotation @ T_L[:3, :3].T),
                pin.log3(oMr.rotation @ T_R[:3, :3].T),
            ])

            err = np.hstack([e_pos, e_rot])
            if np.linalg.norm(err) < 1e-4:
                break

            JL = pin.computeFrameJacobian(
                self.model, self.data, q, self.L_id, pin.WORLD
            )
            JR = pin.computeFrameJacobian(
                self.model, self.data, q, self.R_id, pin.WORLD
            )

            J = np.vstack([JL, JR])

            H = J @ J.T + lam * np.eye(J.shape[0])
            dq = -J.T @ np.linalg.solve(H, err)

            # Step limiting
            norm = np.linalg.norm(dq)
            if norm > max_step:
                dq *= max_step / norm

            q += dq
            q = np.clip(
                q,
                self.model.lowerPositionLimit,
                self.model.upperPositionLimit
            )

        self.filter.add_data(q)
        q = self.filter.filtered_data
        self.q = q

        tau = pin.rnea(
            self.model,
            self.data,
            q,
            np.zeros(self.model.nv),
            np.zeros(self.model.nv),
        )

        if self.vis:
            self.vis.display(q)

        return q, tau


# =========================================================
# Demo
# =========================================================
if __name__ == "__main__":
    arm_ik = G1_29_ArmIK(Unit_Test=True, Visualization=True)

    L_tf = pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([0.30, 0.25, 0.10]))
    R_tf = pin.SE3(pin.Quaternion(1, 0, 0, 0), np.array([0.30, -0.25, 0.10]))

    q = None
    for _ in range(5):
        q, tau = arm_ik.solve_ik(L_tf.homogeneous, R_tf.homogeneous, q)
        time.sleep(0.1)
