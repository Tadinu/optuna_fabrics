from abc import ABC
from typing import Dict
import os

import numpy as np

from optuna_fabrics.planner.symbolic_planner import SymbolicFabricPlanner
from fabrics.planner.serialized_planner import SerializedFabricPlanner

from urdfenvs.generic_mujoco.generic_mujoco_env import GenericMujocoEnv
from urdfenvs.generic_mujoco.generic_mujoco_robot import GenericMujocoRobot
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from optuna_fabrics.tune.fabrics_trial import FabricsTrial
from mpscenes.goals.goal_composition import GoalComposition

class PandaTrial(FabricsTrial, ABC):

    def __init__(self, weights=None):
        super().__init__(weights=weights)
        self._degrees_of_freedom = 7
        self._q0 = np.array([0.0, -1.0, 0.0, -1.501, 0.0, 1.8675, 0.0])
        self._qdot0 = np.zeros(7)
        self._collision_links = ['panda_link8', 'panda_link4', "panda_link7", "panda_link5", "panda_hand"]
        self._link_sizes = {
            'panda_link8': 0.1,
            'panda_link4': 0.1,
            "panda_link7": 0.08,
            "panda_link5": 0.08,
            "panda_hand": 0.08
        }
        self._ee_link = "panda_hand"
        self._root_link = "panda_link0"
        self._self_collision_pairs = {
            "panda_hand": ['panda_link2', 'panda_link4'], 
        }
        self._absolute_path = os.path.dirname(os.path.abspath(__file__))
        self._urdf_file = self._absolute_path + "/panda.urdf"
        with open(self._urdf_file, "r") as file:
            self._urdf = file.read()
        self._generic_fk = GenericURDFFk(self._urdf, 'panda_link0', ['panda_hand', 'panda_link5_offset'])

    def initialize_environment(self, render: bool=False):
        xml_file = "../../fabrics/examples/panda/xml/panda_without_gripper.xml"
        self._robots = [
            GenericMujocoRobot(xml_file=xml_file, mode="vel"),
        ]
        # The planner hides all the logic behind the function set_components.
        # Definition of the goal.
        goal_dict = {
            "subgoal0": {
                "weight": 1.0,
                "is_primary_goal": False,
                "indices": [0, 1, 2],
                "parent_link": "panda_link0",
                "child_link": "panda_hand",
                "desired_position": [0.5, 0.6, 0.3],
                "epsilon": 0.05,
                "type": "staticSubGoal",
            },
            "subgoal1": {
                "weight": 2.0,
                "is_primary_goal": True,
                "indices": [0, 1, 2],
                "parent_link": "panda_link7",
                "child_link": "panda_hand",
                "desired_position": [0.0, 0.0, -0.1],
                "epsilon": 0.05,
                "type": "staticSubGoal",
            }
        }
        self._goal = GoalComposition(name="goal", content_dict=goal_dict)
        home_config = np.array([0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
        env = GenericMujocoEnv(robots=self._robots, obstacles=[], goals=self._goal.sub_goals(), render=render)
        env.reset(pos=home_config)
        return env

    def set_planner(self, render=True):
        """
        Initializes the fabric planner for the point robot.

        This function defines the forward kinematics for collision avoidance,
        and goal reaching. These components are fed into the fabrics planner.

        In the top section of this function, an example for optional reconfiguration
        can be found. Commented by default.

        """
        serialize_file = None
        serializing = False
        if serializing:
            serialize_file = self._absolute_path + "/../planner/serialized_planners/panda_planner.pkl"
            if os.path.exists(serialize_file):
                planner = SerializedFabricPlanner(serialize_file)
                return planner
        robot_type = "panda"
        collision_geometry: str = "-4.5 / (x ** 1) * (-0.5 * (ca.sign(xdot) - 1)) * xdot ** 2"
        geometry_plane_constraint: str = (
            "-10.0 / (x ** 1) * (-0.5 * (ca.sign(xdot) - 1)) * xdot ** 2"
        )
        finsler_plane_constraint: str = (
            "1.0/(x**1) * xdot**2"
        )
        forward_kinematics = GenericURDFFk(
            self._urdf,
            root_link=self._root_link,
            end_links=[self._ee_link],
        )
        planner = SymbolicFabricPlanner(
            self._degrees_of_freedom,
            robot_type,
            forward_kinematics=forward_kinematics,
            collision_geometry=collision_geometry,
            geometry_plane_constraint=geometry_plane_constraint,
            finsler_plane_constraint=finsler_plane_constraint
        )
        panda_limits = [
                [-2.8973, 2.8973],
                [-1.7628, 1.7628],
                [-2.8974, 2.8973],
                [-3.0718, -0.0698],
                [-2.8973, 2.8973],
                [-0.0175, 3.7525],
                [-2.8973, 2.8973]
            ]
        planner.set_components(
            self._collision_links,
            self._self_collision_pairs,
            goal=self._goal,
            number_obstacles=len(self._collision_links),
            number_dynamic_obstacles=0,
            limits=panda_limits,
        )
        planner.concretize()
        if serialize_file:
            if not (os.path.exists(serialize_file) and os.path.isdir(serialize_file)):
                os.makedirs(os.path.dirname(serialize_file), exist_ok=True)
            with open(serialize_file, "a", encoding="utf-8"):
                pass
            planner.serialize(serialize_file)
        return planner







