import gym
from typing import Dict, Any
import logging
import optuna
import os
import warnings
from abc import abstractmethod

from MotionPlanningGoal.goalComposition import GoalComposition
from MotionPlanningEnv.sphereObstacle import SphereObstacle

import numpy as np
from optuna_fabrics.planner.symbolic_planner import SymbolicFabricPlanner
from fabrics.planner.serialized_planner import SerializedFabricPlanner

import urdfenvs.generic_urdf_reacher

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from optuna_fabrics.tune.fabrics_trial import FabricsTrial
import quaternionic
 


class IiwaTrial(FabricsTrial):

    def __init__(self, weights=None):
        super().__init__(weights=weights)
        self._degrees_of_freedom = 7
        self._q0 = np.array([0, -0.6, 0.0, -1.1, 0.00, 0, 0.0])
        self._qdot0 = np.zeros(self._degrees_of_freedom)
        self._collision_links = ["iiwa_link_3", "iiwa_link_5", "iiwa_link_7", "iiwa_link_ee"]
        self._self_collision_pairs = {
            "iiwa_link_7": ['iiwa_link_3', 'iiwa_link_5']
        }
        self._absolute_path = os.path.dirname(os.path.abspath(__file__))
        self._urdf_file = self._absolute_path + "/iiwa7.urdf"
        with open(self._urdf_file, "r") as file:
            self._urdf = file.read()
        self._generic_fk = GenericURDFFk(self._urdf, 'iiwa_link_0', 'iiwa_link_ee')

    def initialize_environment(self, render=True, shuffle=True):
        """
        Initializes the simulation environment.

        Adds obstacles and goal visualizaion to the environment based and
        env.add_obstacle(obstacles[1])
        steps the simulation once.
        """
        env = gym.make(
            "generic-urdf-reacher-acc-v0", dt=0.05, urdf=self._urdf_file, render=render
        )

        return env


    def set_planner(self):
        """
        Initializes the fabric planner for the point robot.

        This function defines the forward kinematics for collision avoidance,
        and goal reaching. These components are fed into the fabrics planner.

        In the top section of this function, an example for optional reconfiguration
        can be found. Commented by default.

        """
        serialize_file = self._absolute_path + "/../planner/serialized_planners/iiwa_planner.pkl"
        if os.path.exists(serialize_file):
            planner = SerializedFabricPlanner(serialize_file)
            return planner
        robot_type = "panda"

        ## Optional reconfiguration of the planner
        # base_inertia = 0.03
        # attractor_potential = "20 * ca.norm_2(x)**4"
        # damper = {
        #     "alpha_b": 0.5,
        #     "alpha_eta": 0.5,
        #     "alpha_shift": 0.5,
        #     "beta_distant": 0.01,
        #     "beta_close": 6.5,
        #     "radius_shift": 0.1,
        # }
        # planner = ParameterizedFabricPlanner(
        #     degrees_of_freedom,
        #     robot_type,
        #     base_inertia=base_inertia,
        #     attractor_potential=attractor_potential,
        #     damper=damper,
        # )
        planner = SymbolicFabricPlanner(
            self._degrees_of_freedom,
            robot_type,
            urdf=self._urdf,
            root_link='iiwa_link_0',
            end_link=['iiwa_link_ee'],
        )
        iiwa_limits= [
            [-2.96705973, 2.96705973],
            [-2.0943951, 2.0943951],
            [-2.96705973, 2.96705973],
            [-2.0943951, 2.0943951],
            [-2.96705973, 2.96705973],
            [-2.0943951, 2.0943951],
            [-3.05432619, 3.05432619],
        ]

        # The planner hides all the logic behind the function set_components.
        goal = self.dummy_goal()
        planner.set_components(
            self._collision_links,
            self._self_collision_pairs,
            goal,
            number_obstacles=self._number_obstacles,
            limits=iiwa_limits,
        )
        planner.concretize()
        planner.serialize(serialize_file)
        return planner

    @abstractmethod
    def set_goal_arguments(self, q0: np.ndarray, goal: GoalComposition):
        pass


    def run(self, params, planner: SymbolicFabricPlanner, obstacles, ob, goal: GoalComposition, env, n_steps=1000):
        # Start the simulation
        logging.info("Starting simulation")
        q0 = ob['joint_state']['position']
        arguments, initial_distance_to_goal = self.set_goal_arguments(q0, goal)
        # sub_goal_0_position = np.array(goal.subGoals()[0].position())
        objective_value = 0.0
        distance_to_goal = 0.0
        distance_to_obstacle = 0.0
        path_length = 0.0
        x_old = q0
        self.set_parameters(arguments, obstacles, params)
        for _ in range(n_steps):
            action = planner.compute_action(
                q=ob["joint_state"]['position'],
                qdot=ob["joint_state"]['velocity'],
                weight_goal_0=np.array([1.00]),
                weight_goal_1=np.array([5.00]),
                radius_body_iiwa_link_3=np.array([0.10]),
                radius_body_iiwa_link_5=np.array([0.10]),
                radius_body_iiwa_link_7=np.array([0.08]),
                radius_body_iiwa_link_ee=np.array([0.08]),
                **arguments,
            )
            if np.linalg.norm(action) < 1e-5 or np.linalg.norm(action) > 1e3:
                action = np.zeros(7)
            warnings.filterwarnings("error")
            try:
                ob, *_ = env.step(action)
            except Exception as e:
                logging.warning(e)
                return 100
            q = ob['joint_state']['position']
            path_length += np.linalg.norm(q - x_old)
            x_old = q
            distance_to_goal += self.evaluate_distance_to_goal(q)
            distance_to_obstacles = []
            fk = self._generic_fk.fk(q, 'iiwa_link_0', 'iiwa_link_7', positionOnly=True)
            for obst in obstacles:
                distance_to_obstacles.append(np.linalg.norm(np.array(obst.position()) - fk))
            distance_to_obstacle += np.min(distance_to_obstacles)
        costs = {
            "path_length": path_length/initial_distance_to_goal,
            "time_to_goal": distance_to_goal/n_steps,
            "obstacles": 1/distance_to_obstacle/n_steps
        }
        return self.total_costs(costs)


    def q0(self):
        return self._q0



