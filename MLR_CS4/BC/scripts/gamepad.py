"""
Driver class for Gamepad (Xbox / PS) controller via pygame.

Requires: pip install pygame

Left stick  -> X/Y translation
Right stick -> yaw / pitch rotation
R2/L2       -> Z up / Z down
L1/R1       -> roll left / roll right
A (btn 0)   -> toggle gripper
B (btn 1)   -> reset (end demo)
"""

import numpy as np

try:
    import pygame
except ImportError:
    raise ImportError("Gamepad device requires pygame. Install with: pip install pygame")

from robosuite.devices import Device
from robosuite.utils.transform_utils import rotation_matrix


class Gamepad(Device):
    """
    A driver class for Xbox / PlayStation game controllers via pygame.

    Args:
        env (RobotEnv): The environment which contains the robot(s) to control
                        using this device.
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of input rotation command scaling
        deadzone (float): Axis values below this threshold are treated as zero
    """

    def __init__(self, env, pos_sensitivity=1.0, rot_sensitivity=1.0, deadzone=0.15):
        super().__init__(env)

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected. Connect an Xbox or PS controller and try again.")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"[Gamepad] Connected: {self.joystick.get_name()}")

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.deadzone = deadzone
        self._pos_step = 0.008

        self._display_controls()
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = False

    @staticmethod
    def _display_controls():
        def print_command(char, info):
            char += " " * (30 - len(char))
            print(f"{char}\t{info}")

        print("")
        print_command("Control", "Command")
        print_command("Left stick", "move arm in x-y plane")
        print_command("R2 / L2", "move arm up / down (z)")
        print_command("Right stick", "rotate arm (yaw / pitch)")
        print_command("L1 / R1", "roll left / right")
        print_command("A / X (btn 0)", "toggle gripper")
        print_command("B / O (btn 1)", "reset / end demo")
        print("")

    def _reset_internal_state(self):
        super()._reset_internal_state()

        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self.raw_drotation = np.zeros(3)
        self.last_drotation = np.zeros(3)
        self.pos = np.zeros(3)
        self.last_pos = np.zeros(3)

    def _apply_deadzone(self, value):
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def start_control(self):
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        pygame.event.pump()

        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    self.grasp_states[self.active_robot][self.active_arm_index] = (
                        not self.grasp_states[self.active_robot][self.active_arm_index]
                    )
                elif event.button == 1:
                    self._reset_state = 1
                    self._enabled = False
                    self._reset_internal_state()

        # Left stick
        lx = self._apply_deadzone(self.joystick.get_axis(0))
        ly = self._apply_deadzone(self.joystick.get_axis(1))

        # Right stick
        rx = self._apply_deadzone(self.joystick.get_axis(3))
        ry = self._apply_deadzone(self.joystick.get_axis(4))

        # Triggers: rest at -1, fully pressed = 1. Normalize to 0-1.
        try:
            l2 = (self.joystick.get_axis(2) + 1.0) / 2.0
            r2 = (self.joystick.get_axis(5) + 1.0) / 2.0
        except pygame.error:
            l2, r2 = 0.0, 0.0

        # Shoulder buttons
        try:
            l1 = self.joystick.get_button(4)
            r1 = self.joystick.get_button(5)
        except pygame.error:
            l1, r1 = 0, 0

        # Translation
        self.pos[0] += ly * self._pos_step * self.pos_sensitivity
        self.pos[1] += lx * self._pos_step * self.pos_sensitivity
        self.pos[2] += (r2 - l2) * self._pos_step * self.pos_sensitivity

        # Rotation
        rot_scale = 0.1 * self.rot_sensitivity

        if abs(rx) > 0:
            drot = rotation_matrix(angle=rx * rot_scale, direction=[0.0, 0.0, 1.0])[:3, :3]
            self.rotation = self.rotation.dot(drot)
            self.raw_drotation[2] += rx * rot_scale

        if abs(ry) > 0:
            drot = rotation_matrix(angle=ry * rot_scale, direction=[1.0, 0.0, 0.0])[:3, :3]
            self.rotation = self.rotation.dot(drot)
            self.raw_drotation[1] -= ry * rot_scale

        if l1:
            drot = rotation_matrix(angle=rot_scale, direction=[0.0, 1.0, 0.0])[:3, :3]
            self.rotation = self.rotation.dot(drot)
            self.raw_drotation[0] += rot_scale
        if r1:
            drot = rotation_matrix(angle=-rot_scale, direction=[0.0, 1.0, 0.0])[:3, :3]
            self.rotation = self.rotation.dot(drot)
            self.raw_drotation[0] -= rot_scale

        dpos = self.pos - self.last_pos
        self.last_pos = np.array(self.pos)
        raw_drotation = self.raw_drotation - self.last_drotation
        self.last_drotation = np.array(self.raw_drotation)

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=raw_drotation,
            grasp=int(self.grasp),
            reset=self._reset_state,
            base_mode=int(self.base_mode),
        )

    def _postprocess_device_outputs(self, dpos, drotation):
        drotation = drotation * 1.5
        dpos = dpos * 75

        dpos = np.clip(dpos, -1, 1)
        drotation = np.clip(drotation, -1, 1)

        return dpos, drotation
