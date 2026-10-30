"""Evaluation variants of RoboTwin tasks (Option A: train on the standard demos, test on both).

lift_pot standard: what the demos contain - pot model 0 only, centre +-5 cm, yaw +-22.5 deg.
    (RoboTwin's own lift_pot draws pot 0 or 1; every seed that drew pot 1 was dropped from the demos
    because the expert failed on it, so the standard test pins pot 0.)
lift_pot hard: a pot model the policy never saw (1-6), centre +-10 cm, yaw +-45 deg.
lift_pot position: the training pot (model 0) at poses the demos never had: centre +-10 cm, yaw +-45 deg,
    redrawn until it lies outside the demos' range (|x| > 5 cm, |y| > 5 cm or |yaw| > 22.5 deg).
lift_pot wide: the training pot anywhere in +-10 cm, +-45 deg (demo range included); for play data only.
Other tasks: 'standard' is RoboTwin's task unchanged; no 'hard' variant yet.

Usage: apply(env, variant) once, right after constructing the task object. It replaces load_actors,
so it holds for every later setup_demo(). The success check is RoboTwin's own and is unchanged.
"""
import types

import numpy as np

VARIANTS = {
    ("lift_pot", "standard"): dict(models=[0], xlim=0.05, ylim=0.05, yaw=np.pi / 8),
    ("lift_pot", "hard"): dict(models=[1, 2, 3, 4, 5, 6], xlim=0.10, ylim=0.10, yaw=np.pi / 4),
    ("lift_pot", "wide"): dict(models=[0], xlim=0.10, ylim=0.10, yaw=np.pi / 4),
    ("lift_pot", "position"): dict(models=[0], xlim=0.10, ylim=0.10, yaw=np.pi / 4,
                                   outside=dict(xlim=0.05, ylim=0.05, yaw=np.pi / 8)),
}
BASE_QPOS = [0.704141, 0, 0, 0.71006]


def _lift_pot_load_actors(self):
    from envs.utils import rand_create_sapien_urdf_obj

    v = self._variant
    self.model_name = "060_kitchenpot"
    self.model_id = int(v["models"][np.random.randint(len(v["models"]))])
    if "outside" in v:  # draw the pose ourselves, rejecting poses inside the demos' range
        import transforms3d as t3d

        o = v["outside"]
        while True:
            x, y = np.random.uniform(-v["xlim"], v["xlim"]), np.random.uniform(-v["ylim"], v["ylim"])
            yaw = np.random.uniform(-v["yaw"], v["yaw"])
            if abs(x) > o["xlim"] or abs(y) > o["ylim"] or abs(yaw) > o["yaw"]:
                break
        self.pot = rand_create_sapien_urdf_obj(
            scene=self, modelname=self.model_name, modelid=self.model_id, xlim=[x, x], ylim=[y, y],
            qpos=list(t3d.quaternions.qmult(BASE_QPOS, t3d.euler.euler2quat(0, 0, yaw))),
        )
    else:
        self.pot = rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.model_name,
            modelid=self.model_id,
            xlim=[-v["xlim"], v["xlim"]],
            ylim=[-v["ylim"], v["ylim"]],
            rotate_rand=True,
            rotate_lim=[0, 0, v["yaw"]],
            qpos=BASE_QPOS,
        )
    x, y = self.pot.get_pose().p[0], self.pot.get_pose().p[1]
    self.prohibited_area.append([x - 0.3, y - 0.1, x + 0.3, y + 0.1])
    # self.info doesn't exist yet here; lift_pot.play_once records the pot model.


LOAD_ACTORS = {"lift_pot": _lift_pot_load_actors}


def apply(env, variant):
    task = type(env).__name__
    if (task, variant) in VARIANTS:
        env._variant = VARIANTS[(task, variant)]
        env.load_actors = types.MethodType(LOAD_ACTORS[task], env)
    elif variant != "standard":
        raise ValueError(f"no {variant!r} variant for {task}")
    return env
