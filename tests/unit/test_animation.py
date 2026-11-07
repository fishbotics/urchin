"""Exercise animation without importing a native window backend."""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from urchin import URDF
from urchin import urdf as urdf_module


@pytest.fixture
def robot(tmp_path):
    path = tmp_path / "slider.urdf"
    path.write_text(
        """<robot name="slider">
          <link name="base"/>
          <link name="tip">
            <visual><geometry><box size="1 1 1"/></geometry></visual>
            <collision><geometry><box size="2 2 2"/></geometry></collision>
          </link>
          <joint name="slide" type="prismatic">
            <parent link="base"/><child link="tip"/><axis xyz="1 0 0"/>
            <limit lower="0" upper="1" effort="1" velocity="1"/>
          </joint>
        </robot>"""
    )
    return URDF.load(path)


@pytest.fixture
def backend(monkeypatch):
    state = SimpleNamespace(
        callbacks={},
        frames=[],
        callback_threads=[],
        viewer=None,
        start_calls=0,
        start_error=None,
        intervals=[],
    )

    class Scene:
        def __init__(self):
            self.nodes = []

        def add(self, mesh, pose):
            node = SimpleNamespace(mesh=mesh, matrix=pose.copy())
            self.nodes.append(node)
            return node

    class Viewer:
        def __init__(self, scene, **kwargs):
            self.scene = scene
            self.kwargs = kwargs
            self.is_active = True
            self.render_lock = threading.RLock()
            state.viewer = self

        def start(self):
            state.start_calls += 1
            assert self.kwargs["auto_start"] is False
            assert self.kwargs["run_in_thread"] is False
            if state.start_error:
                raise state.start_error
            for _ in range(4):
                for callback, interval in list(state.callbacks.items()):
                    state.callback_threads.append(threading.current_thread())
                    callback(interval)
                record_frame()
            # A callback delivered after close must leave the scene unchanged.
            before = self.scene.nodes[0].matrix.copy()
            for callback, interval in state.callbacks.items():
                callback(interval)
            np.testing.assert_array_equal(self.scene.nodes[0].matrix, before)

    def record_frame():
        state.frames.append(state.viewer.scene.nodes[0].matrix.copy())
        if len(state.frames) == 4:
            state.viewer.is_active = False

    def sleep(interval):
        assert state.viewer.kwargs["run_in_thread"] is True
        assert state.viewer.kwargs["auto_start"] is True
        state.intervals.append(interval)
        record_frame()

    def schedule_interval(callback, interval):
        state.intervals.append(interval)
        state.callbacks[callback] = interval

    def unschedule(callback):
        del state.callbacks[callback]

    pyribbit = ModuleType("pyribbit")
    pyribbit.Scene = Scene
    pyribbit.Viewer = Viewer
    pyribbit.Mesh = SimpleNamespace(from_trimesh=lambda mesh, smooth: mesh)
    pyglet = ModuleType("pyglet")
    pyglet.clock = SimpleNamespace(schedule_interval=schedule_interval, unschedule=unschedule)
    monkeypatch.setitem(sys.modules, "pyribbit", pyribbit)
    monkeypatch.setitem(sys.modules, "pyglet", pyglet)
    monkeypatch.setattr(urdf_module, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(urdf_module, "time", SimpleNamespace(sleep=sleep))
    return state


@pytest.mark.parametrize("use_collision", [False, True])
def test_macos_animation_updates_and_cleans_up(robot, backend, use_collision):
    robot.animate({"slide": [0.0, 1.0]}, loop_time=4.0 / 30.0, use_collision=use_collision)

    assert backend.start_calls == 1
    assert backend.callback_threads == [threading.main_thread()] * 4
    assert backend.callbacks == {}
    assert backend.intervals == [1.0 / 30.0]
    np.testing.assert_allclose([pose[0, 3] for pose in backend.frames], [0.0, 1.0, 1.0, 0.0])
    mesh = backend.viewer.scene.nodes[0].mesh
    np.testing.assert_allclose(mesh.extents, [2.0 if use_collision else 1.0] * 3)


@pytest.mark.parametrize("platform", ["linux", "win32"])
@pytest.mark.parametrize("use_collision", [False, True])
def test_other_platforms_keep_threaded_viewer(robot, backend, monkeypatch, platform, use_collision):
    monkeypatch.setattr(urdf_module.sys, "platform", platform)
    robot.animate({"slide": [0.0, 1.0]}, loop_time=4.0 / 30.0, use_collision=use_collision)

    assert backend.start_calls == 0
    assert backend.callbacks == {}
    assert backend.intervals == [1.0 / 30.0] * 4
    np.testing.assert_allclose([pose[0, 3] for pose in backend.frames], [0.0, 1.0, 1.0, 0.0])


@pytest.mark.parametrize("failure", ["startup", "frame"])
def test_macos_unschedules_updates_on_error(robot, backend, monkeypatch, failure):
    if failure == "startup":
        backend.start_error = RuntimeError("viewer failed")
    else:
        original_fk = robot.visual_trimesh_fk

        def fail_on_frame(cfg=None):
            if cfg is not None:
                raise RuntimeError("frame failed")
            return original_fk()

        monkeypatch.setattr(robot, "visual_trimesh_fk", fail_on_frame)

    with pytest.raises(RuntimeError, match="failed"):
        robot.animate({"slide": [0.0, 1.0]})
    assert backend.callbacks == {}


def test_macos_rejects_worker_before_importing_graphics(robot, backend, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyribbit", None)
    monkeypatch.setitem(sys.modules, "pyglet", None)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(robot.animate)
        with pytest.raises(RuntimeError, match="main thread on macOS"):
            future.result()
    assert backend.viewer is None


@pytest.mark.parametrize("use_collision", [False, True])
def test_empty_trajectory_still_shows_static_model(robot, backend, monkeypatch, use_collision):
    calls = []
    monkeypatch.setattr(robot, "show", lambda **kwargs: calls.append(kwargs))
    robot.animate({}, use_collision=use_collision)
    assert calls == [{"use_collision": use_collision}]
    assert backend.viewer is None
    assert backend.callbacks == {}
