from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from urchin import (
    URDF,
    Geometry,
    Joint,
    Link,
    Material,
    Mesh,
    Transmission,
    Visual,
    xyz_rpy_to_matrix,
)


def _make_multi_dof_urdf():
    links = [
        Link("base", None, None, None),
        Link("floating_link", None, None, None),
        Link("planar_link", None, None, None),
        Link("tip", None, None, None),
    ]
    joints = [
        Joint("floating_joint", "floating", "base", "floating_link"),
        Joint("planar_joint", "planar", "floating_link", "planar_link"),
        Joint("continuous_joint", "continuous", "planar_link", "tip"),
    ]
    return URDF("multi_dof", links, joints)


class _CustomPathLike:
    def __init__(self, path):
        self.path = str(path)

    def __fspath__(self):
        return self.path


@pytest.mark.parametrize("path_type", [str, Path, _CustomPathLike])
def test_urchin(tmpdir, path_type):
    outfn = path_type(tmpdir.mkdir("urdf").join("ur5.urdf").strpath)

    # Load
    u = URDF.load(path_type("tests/data/ur5/ur5.urdf"))

    assert isinstance(u, URDF)
    for j in u.joints:
        assert isinstance(j, Joint)
    for link in u.links:
        assert isinstance(link, Link)
    for t in u.transmissions:
        assert isinstance(t, Transmission)
    for m in u.materials:
        assert isinstance(m, Material)

    # Test fk
    fk = u.link_fk()
    assert isinstance(fk, dict)
    for link in fk:
        assert isinstance(link, Link)
        assert isinstance(fk[link], np.ndarray)
        assert fk[link].shape == (4, 4)

    fk = u.link_fk({"shoulder_pan_joint": 2.0})
    assert isinstance(fk, dict)
    for link in fk:
        assert isinstance(link, Link)
        assert isinstance(fk[link], np.ndarray)
        assert fk[link].shape == (4, 4)

    fk = u.link_fk(np.zeros(6))
    assert isinstance(fk, dict)
    for link in fk:
        assert isinstance(link, Link)
        assert isinstance(fk[link], np.ndarray)
        assert fk[link].shape == (4, 4)

    fk = u.link_fk(np.zeros(6), link="upper_arm_link")
    assert isinstance(fk, np.ndarray)
    assert fk.shape == (4, 4)

    fk = u.link_fk(links=["shoulder_link", "upper_arm_link"])
    assert isinstance(fk, dict)
    assert len(fk) == 2
    for link in fk:
        assert isinstance(link, Link)
        assert isinstance(fk[link], np.ndarray)
        assert fk[link].shape == (4, 4)

    fk = u.link_fk(links=list(u.links)[:2])
    assert isinstance(fk, dict)
    assert len(fk) == 2
    for link in fk:
        assert isinstance(link, Link)
        assert isinstance(fk[link], np.ndarray)
        assert fk[link].shape == (4, 4)

    cfg = {j.name: 0.5 for j in u.actuated_joints}
    for _ in range(1000):
        fk = u.collision_trimesh_fk(cfg=cfg)
        for key in fk:
            assert isinstance(fk[key], np.ndarray)
            assert fk[key].shape == (4, 4)

    cfg = {j.name: np.random.uniform(size=1000) for j in u.actuated_joints}
    fk = u.link_fk_batch(cfgs=cfg)
    for key in fk:
        assert isinstance(fk[key], np.ndarray)
        assert fk[key].shape == (1000, 4, 4)

    cfg = {j.name: 0.5 for j in u.actuated_joints}
    for _ in range(1000):
        fk = u.collision_trimesh_fk(cfg=cfg)
        for key in fk:
            assert isinstance(key, trimesh.Trimesh)
            assert fk[key].shape == (4, 4)
    cfg = {j.name: np.random.uniform(size=1000) for j in u.actuated_joints}
    fk = u.collision_trimesh_fk_batch(cfgs=cfg)
    for key in fk:
        assert isinstance(key, trimesh.Trimesh)
        assert fk[key].shape == (1000, 4, 4)

    # Test save
    u.save(outfn)

    nu = URDF.load(outfn)
    assert len(u.links) == len(nu.links)
    assert len(u.joints) == len(nu.joints)

    # Test join
    with pytest.raises(ValueError):
        x = u.join(u, link=u.link_map["tool0"])
    x = u.join(u, link=u.link_map["tool0"], name="copy", prefix="prefix")
    assert isinstance(x, URDF)
    assert x.name == "copy"
    assert len(x.joints) == 2 * len(u.joints) + 1
    assert len(x.links) == 2 * len(u.links)

    # Test scale
    x = u.copy(scale=3)
    assert isinstance(x, URDF)
    x = x.copy(scale=[1, 1, 3])
    assert isinstance(x, URDF)


def test_load_pathlike_lazy_meshes():
    robot = URDF.load(Path("tests/data/ur5/ur5.urdf"), lazy_load_meshes=True)

    meshes = robot.visual_trimesh_fk()

    assert meshes
    assert all(isinstance(mesh, trimesh.Trimesh) for mesh in meshes)


def test_urdf_file_object_round_trip(tmp_path):
    with Path("tests/data/ur5/ur5.urdf").open("rb") as source:
        robot = URDF.load(source)
    output = tmp_path / "ur5.urdf"
    with output.open("wb") as destination:
        robot.save(destination)
    with output.open("rb") as source:
        reloaded = URDF.load(source)

    assert list(reloaded.link_map) == list(robot.link_map)
    assert list(reloaded.joint_map) == list(robot.joint_map)
    assert len(reloaded.visual_trimesh_fk()) == len(robot.visual_trimesh_fk())


@pytest.mark.parametrize("textured", [False, True])
@pytest.mark.parametrize("material_color", [[1.0, 0.0, 0.0, 1.0], None])
def test_visual_trimesh_fk_material_colors_and_textures(textured, material_color):
    mesh = trimesh.creation.box()
    mesh.visual.face_colors = [0, 0, 255, 255]
    if textured:
        image = Image.new("RGB", (2, 2), "blue")
        uv = np.zeros((len(mesh.vertices), 2))
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=image)
    original_visual = mesh.visual
    visual = Visual(
        Geometry(mesh=Mesh("box.obj", combine=False, meshes=mesh)),
        material=Material("material", color=material_color),
    )
    robot = URDF("visuals", [Link("base", None, [visual], None)])

    fk = robot.visual_trimesh_fk()

    assert np.allclose(fk[mesh], np.eye(4))
    assert mesh.visual is original_visual
    if textured:
        assert mesh.visual.material.image is image
        np.testing.assert_array_equal(mesh.visual.uv, uv)
    else:
        expected_color = [255, 0, 0, 255] if material_color is not None else [0, 0, 255, 255]
        np.testing.assert_array_equal(
            mesh.visual.face_colors, np.tile(expected_color, (len(mesh.faces), 1))
        )


def test_multi_dof_flat_configurations():
    robot = _make_multi_dof_urdf()
    cfgs = np.array(
        [
            [0.5, 1.5, 2.5, 0.1, 0.2, 0.3, 3.5, 4.5, 0.4],
            [-1.0, 2.0, 0.0, -0.3, 0.0, 0.2, 1.0, -2.0, -0.5],
        ]
    )

    expected = []
    for cfg in cfgs:
        expected.append(
            robot.link_fk(
                {
                    "floating_joint": cfg[:6],
                    "planar_joint": cfg[6:8],
                    "continuous_joint": cfg[8],
                },
                link="tip",
            )
        )
        assert np.allclose(robot.link_fk(cfg.tolist(), link="tip"), expected[-1])
        assert np.allclose(robot.link_fk([cfg[:6], cfg[6:8], cfg[8]], link="tip"), expected[-1])

    expected_batch = np.stack(expected)
    assert np.allclose(robot.link_fk_batch(cfgs, link="tip"), expected_batch)
    assert np.allclose(
        robot.link_fk_batch(
            {
                "floating_joint": xyz_rpy_to_matrix(cfgs[:, :6]),
                "planar_joint": cfgs[:, 6:8],
                "continuous_joint": cfgs[:, 8],
            },
            link="tip",
        ),
        expected_batch,
    )

    with pytest.raises(ValueError, match="degree of freedom"):
        robot.link_fk(np.zeros(3))
    with pytest.raises(ValueError, match="degree of freedom"):
        robot.link_fk_batch(np.zeros((2, 3)))
