from abc import abstractmethod
from time import perf_counter
from typing import Dict, Any, Optional
from urdfenvs.generic_mujoco.generic_mujoco_env import GenericMujocoEnv
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from urdfenvs.urdf_common.generic_robot import GenericRobot
from optuna_fabrics.planner.symbolic_planner import SymbolicFabricPlanner
from mpscenes.goals.goal_composition import GoalComposition
import gymnasium as gym
import logging
import time
import warnings
import optuna
import numpy as np
from numpy.random import default_rng
import casadi as ca

from urdfenvs.robots.generic_urdf import GenericUrdfReacher


class FabricsTrial(object):
    def __init__(self, weights: dict = None) -> None:
        self._root_link: str = ""
        self._q0 = []
        self._self_collision_pairs: dict[str, list[str]] = {}
        self._collision_metric: Optional[ca.Function] = None
        self._degrees_of_freedom: int = 0
        self._number_obstacles: int = 0
        self._link_sizes: dict[str, list] = {}
        self._collision_links: list[str] = []
        self._generic_fk: Optional[GenericURDFFk] = None
        self._weights = weights if weights else {"path_length": 0.4, "time_to_goal": 0.4, "obstacles": 0.2}
        self._maximum_seconds = 30
        self._start_time = time.perf_counter()
        self._dt = 0.05
        self._urdf_file: str = ""
        self._search_space: dict = {}
        self._robots: list[GenericRobot] = []
        self._goal: Optional[GoalComposition] = None
        self.set_search_space()

    def initialize_environment(self, render: bool=True):
        """
        Initializes the simulation environment.
        """
        self._robots = [
            GenericUrdfReacher(urdf=self._urdf_file, mode="acc"),
        ]
        env = gym.make(
            "urdf-env-v0",
            dt=self._dt, robots=self._robots, render=render
        )
        return env

    def set_search_space(self) -> None:
        self._search_space = {'exp_geo_obst_leaf': {'low': 1, 'high': 5, 'int': True, 'log': False},
                              'exp_geo_self_leaf': {'low': 1, 'high': 5, 'int': True, 'log': False},
                              'exp_geo_limit_leaf': {'low': 1, 'high': 5, 'int': True, 'log': False},
                              'exp_fin_obst_leaf': {'low': 1, 'high': 5, 'int': True, 'log': False},
                              'exp_fin_self_leaf': {'low': 1, 'high': 5, 'int': True, 'log': False},
                              'exp_fin_limit_leaf': {'low': 1, 'high': 5, 'int': True, 'log': False},
                              'k_geo_obst_leaf': {'low': 0.01, 'high': 1, 'int': False, 'log': True},
                              'k_geo_self_leaf': {'low': 0.01, 'high': 1, 'int': False, 'log': True},
                              'k_geo_limit_leaf': {'low': 0.01, 'high': 1, 'int': False, 'log': True},
                              'k_fin_obst_leaf': {'low': 0.01, 'high': 1, 'int': False, 'log': True},
                              'k_fin_self_leaf': {'low': 0.01, 'high': 1, 'int': False, 'log': True},
                              'k_fin_limit_leaf': {'low': 0.01, 'high': 1, 'int': False, 'log': True},
                              'alpha_b_damper': {'low': 0.0, 'high': 1, 'int': False, 'log': False},
                              'base_inertia': {'low': 0.01, 'high': 1.0, 'int': False, 'log': False},
                              'beta_distant_damper': {'low': 0.0, 'high': 1.0, 'int': False, 'log': False},
                              'beta_close_damper': {'low': 5.0, 'high': 20.0, 'int': False, 'log': False},
                              'radius_shift_damper': {'low': 0.01, 'high': 0.1, 'int': False, 'log': False},
                              'ex_factor': {'low': 1.0, 'high': 30.0, 'int': False, 'log': False}}

    @abstractmethod
    def set_planner(self, render=True):
        pass

    def total_costs(self, costs: dict):
        return sum([self._weights[i] * costs[i] for i in self._weights])

    def random_parameters(self) -> dict:
        parameters = {}
        for name, space in self._search_space.items():
            if space['int']:
                parameters[name] = np.random.randint(space['low'], space['high'])
            else:
                parameters[name] = space['low'] + np.random.random() * (space['high'] - space['low'])
        return parameters

    def manual_parameters(self) -> dict:
        return {
            "exp_geo_obst_leaf": 3,
            "k_geo_obst_leaf": 0.03,
            "exp_fin_obst_leaf": 3,
            "k_fin_obst_leaf": 0.03,
            "exp_geo_self_leaf": 3,
            "k_geo_self_leaf": 0.03,
            "exp_fin_self_leaf": 3,
            "k_fin_self_leaf": 0.03,
            "exp_geo_limit_leaf": 2,
            "k_geo_limit_leaf": 0.3,
            "exp_fin_limit_leaf": 3,
            "k_fin_limit_leaf": 0.05,
            "weight_attractor": 2,
            "base_inertia": 0.20,
            "alpha_b_damper" : 0.5,
            "beta_distant_damper" : 0.01,
            "beta_close_damper" : 6.5,
            "radius_shift_damper" : 0.05,
            "ex_factor": 15.0,
        }

    def caspar_parameters(self) -> dict:
        return {
            "exp_geo_obst_leaf": 2,         #[1, 5]
            "exp_geo_self_leaf": 2,         #[1, 5]
            "exp_geo_limit_leaf": 2,        #[1, 5]
            "exp_fin_obst_leaf": 2,         #[1, 5]
            "exp_fin_self_leaf": 2,         #[1, 1]
            "exp_fin_limit_leaf": 2,        #[1, 5]
            "k_geo_obst_leaf": 0.01,        #[0.01, 1]
            "k_geo_self_leaf": 0.01,        #[0.01, 1]
            "k_geo_limit_leaf": 0.01,       #[0.01, 1]
            "k_fin_self_leaf": 0.01,        #[0.01, 1]
            "k_fin_obst_leaf": 0.01,        #[0.01, 1]
            "k_fin_limit_leaf": 0.01,       #[0.01, 1]
            "base_inertia": 0.50,           #[0, 1]
            "alpha_b_damper" : 0.7,         #[0, 1]
            "beta_distant_damper" : 0.,     #[0, 1]
            "beta_close_damper" : 9,        #[5, 20]
            "radius_shift_damper" : 0.050,  #[0.01, 0.1]
            "ex_factor": 30.0,              #[1, 30]
        }


    def sample_fabrics_params_uniform(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        parameters = {}
        for name, space in self._search_space.items():
            if space['int']:
                parameters[name] = trial.suggest_int(name, space['low'], space['high'], log=space['log'])
            else:
                parameters[name] = trial.suggest_float(name, space['low'], space['high'], log=space['log'])
        return parameters

    def set_collision_arguments(self, arguments: dict):
        for link in self._collision_links:
            arguments[f"radius_body_{link}"] = np.array([self._link_sizes[link]])

    def set_parameters(self, arguments, obstacles, params):
        for j in self._collision_links:
            for i in range(self._number_obstacles):
                arguments[f"x_obst_{i}"] = np.array(obstacles[i].position())
                arguments[f"radius_obst_{i}"] = np.array(obstacles[i].radius())
                arguments[f"exp_geo_obst_{i}_{j}_leaf"] = np.array([params['exp_geo_obst_leaf']])
                arguments[f"k_geo_obst_{i}_{j}_leaf"] = np.array([params['k_geo_obst_leaf']])
                arguments[f"exp_fin_obst_{i}_{j}_leaf"] = np.array([params['exp_fin_obst_leaf']])
                arguments[f"k_fin_obst_{i}_{j}_leaf"] = np.array([params['k_fin_obst_leaf']])
        for j in range(self._degrees_of_freedom):
            for i in range(2):
                arguments[f"exp_limit_fin_limit_joint_{j}_{i}_leaf"] = np.array([params['exp_fin_limit_leaf']])
                arguments[f"exp_limit_geo_limit_joint_{j}_{i}_leaf"] = np.array([params['exp_geo_limit_leaf']])
                arguments[f"k_limit_fin_limit_joint_{j}_{i}_leaf"] = np.array([params['k_fin_limit_leaf']])
                arguments[f"k_limit_geo_limit_joint_{j}_{i}_leaf"] = np.array([params['k_geo_limit_leaf']])
        for link, paired_links in self._self_collision_pairs.items():
            for paired_link in paired_links:
                arguments[f"exp_self_fin_self_collision_{link}_{paired_link}"] = np.array([params['exp_fin_self_leaf']])
                arguments[f"exp_self_geo_self_collision_{link}_{paired_link}"] = np.array([params['exp_geo_self_leaf']])
                arguments[f"k_self_fin_self_collision_{link}_{paired_link}"] = np.array([params['k_fin_self_leaf']])
                arguments[f"k_self_geo_self_collision_{link}_{paired_link}"] = np.array([params['k_geo_self_leaf']])
        # damper arguments
        arguments['alpha_b_damper'] = np.array([params['alpha_b_damper']])
        arguments['beta_close_damper'] = np.array([params['beta_close_damper']])
        arguments['beta_distant_damper'] = np.array([params['beta_distant_damper']])
        arguments['radius_shift_damper'] = np.array([params['radius_shift_damper']])
        arguments['base_inertia'] = np.array([params['base_inertia']])
        arguments['ex_factor_damper'] = np.array([params['ex_factor']])


    @abstractmethod
    def shuffle_env(self, env, shuffle=True, render=False):
        pass

    @abstractmethod
    def set_goal_arguments(self, q0: np.ndarray, goal:GoalComposition, arguments):
        pass

    def extract_joint_states(self, ob: dict):
        if 'joint_state' in ob['robot_0']:
            joint_state = ob['robot_0']['joint_state']
            return joint_state['position'], joint_state['velocity']
        else:
            return ob['x'], ob['xdot']

    def run(self, params, planner: SymbolicFabricPlanner, obstacles, ob, goal: GoalComposition, env):
        # Start the simulation
        logging.info("Starting simulation")
        q0 = self.extract_joint_states(ob)[0]
        arguments = {}
        self.set_collision_arguments(arguments)
        self.set_goal_arguments(q0, goal, arguments)
        self.set_parameters(arguments, obstacles, params)
        initial_distance_to_obstacles = self.evaluate_distance_to_closest_obstacle(obstacles, q0)
        distances_to_goal = []
        distances_to_closest_obstacle = []
        path_length = 0.0
        x_old = q0

        t3 = time.perf_counter()
        while (t3- self._start_time) < self._maximum_seconds:
            t0 = time.perf_counter()
            action = planner.compute_action(
                q=self.extract_joint_states(ob)[0],
                qdot=self.extract_joint_states(ob)[1],
                **arguments,
            )
            t1 = time.perf_counter()
            if np.linalg.norm(action) < 1e-5 or np.linalg.norm(action) > 1e3:
                action = np.zeros(self._degrees_of_freedom)
            warnings.filterwarnings("error")
            try:
                ob, *_ = env.step(action)
            except Exception as e:
                logging.warning(e)
                return {"path_length": 1.0, "time_to_goal": 1.0, "obstacles": 1.0}
            q = self.extract_joint_states(ob)[0]
            t2 = time.perf_counter()
            path_length += np.linalg.norm(q - x_old)
            x_old = q
            distances_to_goal.append(self.evaluate_distance_to_goal(q))
            distances_to_closest_obstacle.append(self.evaluate_distance_to_closest_obstacle(obstacles, q))
            t3 = time.perf_counter()
            logging.debug(f"Compute action {t1-t0}\nStepping environment: {t2-t1}\nCompute metrics{t3-t2}")
        costs = {
            "path_length": path_length/10,
            "time_to_goal": np.mean(np.array(distances_to_goal)),
            "obstacles": 1 - np.min(distances_to_closest_obstacle) / initial_distance_to_obstacles
        }
        return costs

    def create_collision_metric(self, obstacles):
        q = ca.SX.sym("q", self._degrees_of_freedom)
        distance_to_obstacles = 10000
        for link in self._collision_links:
            fk = self._generic_fk.casadi(q, self._root_link, link, position_only=True)
            for obst in obstacles:
                obst_position = np.array(obst.position())
                distance_to_obstacles = ca.fmin(distance_to_obstacles, ca.norm_2(obst_position - fk))
        self._collision_metric = ca.Function("collision_metric", [q], [distance_to_obstacles])

    def evaluate_distance_to_goal(self, q: np.ndarray):
        pass

    def evaluate_distance_to_closest_obstacle(self, obstacles, q: np.ndarray):
        casadi_metric = self._collision_metric(q)
        return casadi_metric

    def q0(self):
        return self._q0

    def objective(self, trial, planner, env, q0, shuffle=True, render=False):
        ob, _ = env.reset()
        env, obstacles, goal = self.shuffle_env(env, shuffle=shuffle, render=render)
        self.create_collision_metric(obstacles)
        params = self.sample_fabrics_params_uniform(trial)
        return self.total_costs(self.run(params, planner, obstacles, ob, goal, env))
