import numpy as np
import pytest

from urchin import Joint, xyz_rpy_to_matrix


@pytest.mark.parametrize("joint_type", ["continuous", "planar", "floating"])
def test_none_configuration_uses_joint_origin(joint_type):
    origin = xyz_rpy_to_matrix([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    joint = Joint("joint", joint_type, "parent", "child", origin=origin)

    assert np.allclose(joint.get_child_pose(None), origin)
    assert np.allclose(joint.get_child_poses(None, 2), np.tile(origin, (2, 1, 1)))


def test_planar_get_child_poses_matches_scalar_poses():
    origin = xyz_rpy_to_matrix([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    joint = Joint("joint", "planar", "parent", "child", origin=origin)
    cfgs = np.array([[0.5, 1.5], [-1.0, 2.0]])

    expected = np.stack([joint.get_child_pose(cfg) for cfg in cfgs])

    assert np.allclose(joint.get_child_poses(cfgs, len(cfgs)), expected)


def test_floating_get_child_poses_matches_scalar_poses():
    origin = xyz_rpy_to_matrix([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    joint = Joint("joint", "floating", "parent", "child", origin=origin)
    cfgs = np.array(
        [
            [0.5, 1.5, 2.5, 0.1, 0.2, 0.3],
            [-1.0, 2.0, 0.0, -0.3, 0.0, 0.2],
        ]
    )
    expected = np.stack([joint.get_child_pose(cfg) for cfg in cfgs])

    assert np.allclose(joint.get_child_poses(cfgs, len(cfgs)), expected)
    assert np.allclose(joint.get_child_poses(xyz_rpy_to_matrix(cfgs), len(cfgs)), expected)
