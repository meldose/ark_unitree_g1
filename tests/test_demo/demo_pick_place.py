#!/usr/bin/env python3

# ==================================================
# FIX ARK PYTHON PATH (REQUIRED FOR SOURCE-TREE ARK)
# ==================================================
import sys
import os

ARK_ROOT = os.path.expanduser("~/G1/ark_unitree_g1")
if ARK_ROOT not in sys.path:
    sys.path.insert(0, ARK_ROOT)

# ==================================================
# Standard imports
# ==================================================
import time
import numpy as np
from typing import Optional

import pinocchio as pin
from robot_arm_ik import G1_29_ArmIK

# ✅ Correct ARK import for your tree
from ark.client.comm_infrastructure.instance_node import InstanceNode

from arktypes import joint_group_command_t, joint_state_t
from arktypes.utils import pack, unpack


SIM = False


class AllGroupArmsHandsDemo(InstanceNode):
    def __init__(self) -> None:
        super().__init__("G1AllGroupArmsHandsDemo")

        ns = "unitree_g1_sim" if SIM else "unitree_g1"
        self.pub = self.create_publisher(
            f"{ns}/joint_group_command", joint_group_command_t
        )
        self.create_subscriber(
            f"{ns}/joint_states", joint_state_t, self._state_callback
        )

        self.latest_positions: Optional[np.ndarray] = None
        self.num_joints = 43

    # --------------------------------------------------
    # Joint state callback
    # --------------------------------------------------
    def _state_callback(self, t, channel_name, msg):
        try:
            js = unpack.joint_state(msg)
            self.latest_positions = np.array(js.position, dtype=float)
        except Exception:
            pass

    def get_current_positions(self, timeout_s: float = 2.0) -> np.ndarray:
        start = time.time()
        while self.latest_positions is None and time.time() - start < timeout_s:
            time.sleep(0.01)

        if self.latest_positions is None:
            self.latest_positions = np.zeros(self.num_joints)

        return self.latest_positions.copy()

    # --------------------------------------------------
    # ARK command helpers (CORRECT FOR YOUR VERSION)
    # --------------------------------------------------
    def send_arms_q_tau(self, q14: np.ndarray, tau14: np.ndarray):
        assert q14.shape == (14,)
        assert tau14.shape == (14,)

        payload = np.concatenate([q14, tau14]).tolist()

        cmd = joint_group_command_t()
        cmd.position = payload                  # payload set manually
        pack.joint_group_command(cmd, "arms_q_tau")
        self.pub.publish(cmd)

    def send_right_hand(self, hand7: np.ndarray):
        assert hand7.shape == (7,)

        cmd = joint_group_command_t()
        cmd.position = hand7.tolist()
        pack.joint_group_command(cmd, "right_hand")
        self.pub.publish(cmd)

    # ==================================================
    # PICK AND PLACE DEMO
    # ==================================================
    def run_pick_and_place_demo(self):
        print("Starting Right-Arm Pick-and-Place demo (grasp from above)")

        current = self.get_current_positions()

        LEFT_ARM = slice(15, 22)
        RIGHT_ARM = slice(22, 29)

        # Init IK (keep relative URDF paths working)
        base_dir = os.path.dirname(__file__)
        cwd = os.getcwd()
        try:
            os.chdir(base_dir)
            ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)
        finally:
            os.chdir(cwd)

        left_fixed = current[LEFT_ARM].copy()
        right_curr = current[RIGHT_ARM].copy()

        # --------------------------------------------------
        # Helpers
        # --------------------------------------------------
        def se3(px, py, pz, quat):
            return pin.SE3(quat, np.array([px, py, pz]))

        def grasp_quat():
            rx = np.deg2rad(-90)
            ry = np.deg2rad(20)
            cx, sx = np.cos(rx / 2), np.sin(rx / 2)
            cy, sy = np.cos(ry / 2), np.sin(ry / 2)
            return pin.Quaternion(cy * cx, cy * sx, sy * cx, -sy * sx)

        q_id = pin.Quaternion(1, 0, 0, 0)
        q_grasp = grasp_quat()

        x_pick, y_pick = 0.35, -0.15
        z_above, z_grasp = 0.20, 0.02

        def move_right(px, py, pz, quat):
            nonlocal right_curr
            seed = np.concatenate([left_fixed, right_curr])

            L_tf = se3(0.30, 0.25, z_above, q_id)
            R_tf = se3(px, py, pz, quat)

            sol_q, sol_tau = ik.solve_ik(
                L_tf.homogeneous, R_tf.homogeneous, seed
            )

            next_r = sol_q[7:14]

            for a in np.linspace(0, 1, 20):
                q_r = (1 - a) * right_curr + a * next_r
                q14 = np.concatenate([left_fixed, q_r])
                self.send_arms_q_tau(q14, sol_tau)
                time.sleep(0.05)

            right_curr = next_r.copy()

        def set_hand(vals: np.ndarray, seconds: float = 1.0, rate_hz: float = 10.0):
            period = 1.0 / rate_hz
            steps = max(1, int(seconds * rate_hz))

            cmd = joint_group_command_t()
            cmd.position = vals.tolist()
            pack.joint_group_command(cmd, "right_hand")

            for _ in range(steps):
                self.pub.publish(cmd)
                time.sleep(period)

        # --------------------------------------------------
        # Hand presets
        # --------------------------------------------------
        hand_open = np.zeros(7)
        hand_close = np.array([0.0, -0.3, -1.0, 0.8, 1.0, 0.8, 1.0])

        # --------------------------------------------------
        # Sequence
        # --------------------------------------------------
        move_right(x_pick, y_pick, z_above, q_grasp)
        print("1) Above pick")

        move_right(x_pick, y_pick, z_grasp, q_grasp)
        print("2) Down")

        set_hand(hand_close)
        print("3) Close")

        move_right(x_pick, y_pick, z_above, q_grasp)
        print("4) Lift")

        set_hand(hand_open)
        print("5) Release")

        move_right(0.25, -0.25, 0.15, q_id)
        print("6) Home")

        print("Pick-and-Place demo complete")


# ==================================================
# Main
# ==================================================
if __name__ == "__main__":
    demo = AllGroupArmsHandsDemo()
    time.sleep(2.0)
    demo.run_pick_and_place_demo()
