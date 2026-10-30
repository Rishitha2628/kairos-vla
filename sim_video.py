"""High-resolution 'showcase' camera for watching RoboTwin runs.

Only used for videos people watch; the policy never sees it.
"""
import os
import shutil
import tempfile

import imageio
import numpy as np
import sapien

OUTPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def look_at(eye, target, up=(0, 0, 1)):
    """SAPIEN camera pose: x = forward, y = left, z = up."""
    forward = np.asarray(target, float) - np.asarray(eye, float)
    forward /= np.linalg.norm(forward)
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    mat = np.eye(4)
    mat[:3, :3] = np.stack([forward, left, np.cross(forward, left)], axis=1)
    mat[:3, 3] = eye
    return sapien.Pose(mat)


class ShowcaseCamera:
    # Front view from across the table: both arms and the workspace, filling the frame.
    def __init__(self, env, width=1920, height=1080, eye=(0.0, 0.45, 1.30), target=(0.0, -0.30, 0.85), fovy_deg=62,
                 fps=30):
        self.env = env
        self.camera = env.scene.add_camera(
            name="showcase_camera", width=width, height=height, fovy=np.deg2rad(fovy_deg), near=0.1, far=100
        )
        self.camera.entity.set_pose(look_at(eye, target))
        # Frames go straight to a temp file: a 400-step 1080p episode held in RAM is ~2.5 GB.
        os.makedirs(OUTPUTS, exist_ok=True)
        fd, self._tmp = tempfile.mkstemp(suffix=".mp4", prefix=".recording_", dir=OUTPUTS)
        os.close(fd)
        self._writer = None
        self.fps = fps
        self.n_frames = 0
        self._render = type(env)._update_render.__get__(env)  # the class method, never an earlier hook

    def record_physics(self, every=8):
        """Capture a frame every `every` physics steps (sim runs at 250 Hz, so 8 -> ~31 fps real time)."""
        count = 0

        def hooked():
            nonlocal count
            self._render()
            count += 1
            if count % every == 0:
                self._capture()

        self.env._update_render = hooked

    def detach(self):
        """Stop recording physics steps (call before the scene is closed)."""
        self.env.__dict__.pop("_update_render", None)

    def capture(self):
        self._render()
        self._capture()

    def _capture(self):
        self.camera.take_picture()
        rgba = self.camera.get_picture("Color")
        if self._writer is None:
            self._writer = imageio.get_writer(self._tmp, fps=self.fps, quality=8, macro_block_size=8)
        self._writer.append_data((np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8))
        self.n_frames += 1

    def save(self, name):
        self.detach()
        if self._writer is None:  # nothing recorded
            os.remove(self._tmp)
            return None
        self._writer.close()
        path = os.path.join(OUTPUTS, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        os.chmod(self._tmp, 0o644)
        shutil.move(self._tmp, path)
        return path
