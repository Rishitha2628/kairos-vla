"""Run RoboTwin's scripted expert with mplib instead of CuRobo (which can't be built here).

The expert only needs plan_path / plan_batch from the per-arm planner. CuRobo in RoboTwin plans against
the robot alone (no scene point cloud is ever passed), so this uses mplib without the scene as well:
otherwise the pot held in the gripper would count as a collision and every lift would fail.
`constraint_pose` (keep part of the end-effector orientation fixed) has no mplib equivalent and is ignored.

RoboTwin's aloha SRDF has its adjacent-link exclusions commented out (CuRobo has its own collision model),
so mplib sees the robot as self-colliding even at its home pose and every IK solution is rejected.
The pairs in collision at the home pose (base/wheels, link_i/link_i+1, ...) are therefore allowed.

Usage (after env.setup_demo(...), before env.play_once()):
    enable_mplib_expert(env)
"""
import numpy as np

from envs.robot.planner import MplibPlanner


class MplibAsCurobo:
    """CuroboPlanner's plan_path / plan_batch / plan_grippers interface on top of MplibPlanner."""

    def __init__(self, planner, arm):
        self.planner = planner
        self.arm = arm

    def plan_path(self, curr_joint_pos, target_gripper_pose, constraint_pose=None, arms_tag=None):
        # Straight-line (screw) first: fast and short. RRT only if that is blocked, with fewer retries than
        # MplibPlanner's default 10 x 5 s, which made every unreachable grasp candidate cost ~50 s.
        arm = arms_tag or self.arm
        result = self.planner.plan_screw(curr_joint_pos, target_gripper_pose, arms_tag=arm)
        if result["status"] != "Success":
            result = self.planner.plan_pose(curr_joint_pos, target_gripper_pose, arms_tag=arm, try_times=3, log=False)
        return result

    def plan_batch(self, curr_joint_pos, target_gripper_pose_list, constraint_pose=None, arms_tag=None):
        results = [self.plan_path(curr_joint_pos, pose, arms_tag=arms_tag) for pose in target_gripper_pose_list]
        out = {"status": [r["status"] if r["status"] == "Success" else "Failure" for r in results]}
        if any(s == "Success" for s in out["status"]):
            out["position"] = [r.get("position", []) for r in results]
            out["velocity"] = [r.get("velocity", []) for r in results]
        return out

    def plan_grippers(self, now_val, target_val):
        return self.planner.plan_grippers(now_val, target_val)


def allow_home_pose_collisions(planner, home_qpos):
    acm = planner.planning_world.get_allowed_collision_matrix()
    for c in planner.check_for_self_collision(home_qpos):
        acm.set_entry(c.link_name1, c.link_name2, True)


def enable_mplib_expert(env):
    """Call right after setup_demo, while the robot is still at its home pose."""
    robot = env.robot
    home_qpos = np.array(robot.left_entity.get_qpos())
    for arm in ("left", "right"):
        planner = MplibPlanner(
            getattr(robot, f"{arm}_urdf_path"),
            getattr(robot, f"{arm}_srdf_path"),
            getattr(robot, f"{arm}_move_group"),
            getattr(robot, f"{arm}_entity_origion_pose"),
            getattr(robot, f"{arm}_entity"),
            "mplib_RRT",
            scene=None,
        )
        allow_home_pose_collisions(planner.planner, home_qpos)
        setattr(robot, f"{arm}_planner", MplibAsCurobo(planner, arm))
