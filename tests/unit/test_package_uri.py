import os

import pytest
import trimesh

from urchin import URDF
from urchin.utils import get_filename, resolve_package_uri

URDF_WITH_PACKAGE_MESH = """<robot name="pkg_bot">
  <link name="base">
    <visual>
      <geometry>
        <mesh filename="package://test_pkg/meshes/box.stl"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="package://test_pkg/meshes/box.stl"/>
      </geometry>
    </collision>
  </link>
</robot>
"""


def _make_package_mesh(package_dir):
    mesh_dir = package_dir / "meshes"
    mesh_dir.mkdir(parents=True)
    trimesh.creation.box(extents=[0.1, 0.2, 0.3]).export(str(mesh_dir / "box.stl"))


@pytest.fixture
def ros_package_path(tmp_path, monkeypatch):
    # ROS_PACKAGE_PATH layout: <entry>/<package>/meshes/box.stl
    entry = tmp_path / "ros_ws" / "src"
    _make_package_mesh(entry / "test_pkg")
    monkeypatch.setenv("ROS_PACKAGE_PATH", str(entry))
    monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)
    return entry


@pytest.fixture
def ament_prefix_path(tmp_path, monkeypatch):
    # AMENT_PREFIX_PATH layout: <prefix>/share/<package>/meshes/box.stl
    prefix = tmp_path / "install"
    _make_package_mesh(prefix / "share" / "test_pkg")
    monkeypatch.setenv("AMENT_PREFIX_PATH", str(prefix))
    monkeypatch.delenv("ROS_PACKAGE_PATH", raising=False)
    return prefix


def test_resolve_via_ros_package_path(ros_package_path):
    resolved = resolve_package_uri("package://test_pkg/meshes/box.stl")
    assert resolved == str(ros_package_path / "test_pkg" / "meshes" / "box.stl")
    assert os.path.exists(resolved)


def test_resolve_via_ament_prefix_path(ament_prefix_path):
    resolved = resolve_package_uri("package://test_pkg/meshes/box.stl")
    assert resolved == str(ament_prefix_path / "share" / "test_pkg" / "meshes" / "box.stl")
    assert os.path.exists(resolved)


def test_resolve_searches_all_entries(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    entry = tmp_path / "ros_ws" / "src"
    _make_package_mesh(entry / "test_pkg")
    monkeypatch.setenv("ROS_PACKAGE_PATH", os.pathsep.join([str(empty), str(entry)]))
    monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)
    resolved = resolve_package_uri("package://test_pkg/meshes/box.stl")
    assert resolved == str(entry / "test_pkg" / "meshes" / "box.stl")


def test_resolve_ros_package_path_wins(tmp_path, monkeypatch):
    ros_entry = tmp_path / "ros_ws" / "src"
    _make_package_mesh(ros_entry / "test_pkg")
    prefix = tmp_path / "install"
    _make_package_mesh(prefix / "share" / "test_pkg")
    monkeypatch.setenv("ROS_PACKAGE_PATH", str(ros_entry))
    monkeypatch.setenv("AMENT_PREFIX_PATH", str(prefix))
    resolved = resolve_package_uri("package://test_pkg/meshes/box.stl")
    assert resolved == str(ros_entry / "test_pkg" / "meshes" / "box.stl")


def test_resolve_unknown_package(ros_package_path):
    with pytest.raises(ValueError, match="no_such_pkg"):
        resolve_package_uri("package://no_such_pkg/meshes/box.stl")


def test_resolve_without_search_paths(monkeypatch):
    monkeypatch.delenv("ROS_PACKAGE_PATH", raising=False)
    monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)
    with pytest.raises(ValueError, match="ROS_PACKAGE_PATH"):
        resolve_package_uri("package://test_pkg/meshes/box.stl")


def test_resolve_malformed_uris(ros_package_path):
    for uri in (
        "package://",
        "package://test_pkg",
        "package://test_pkg/",
        "package:///meshes/box.stl",
        "not-a-uri",
    ):
        with pytest.raises(ValueError):
            resolve_package_uri(uri)


def test_resolve_rejects_escaping_paths(ros_package_path):
    with pytest.raises(ValueError, match="escapes"):
        resolve_package_uri("package://test_pkg/../../secret.stl")


def test_get_filename_package_uri(ros_package_path):
    resolved = get_filename("/some/urdf/dir", "package://test_pkg/meshes/box.stl")
    assert resolved == str(ros_package_path / "test_pkg" / "meshes" / "box.stl")


def test_get_filename_unchanged_for_plain_paths():
    assert get_filename("/base", "meshes/box.stl") == "/base/meshes/box.stl"
    assert get_filename("/base", "/abs/box.stl") == "/abs/box.stl"


def _make_package_manifest(package_dir, name="test_pkg"):
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "package.xml").write_text(
        '<package format="2"><name>{}</name></package>'.format(name)
    )


def test_resolve_nested_package(tmp_path, monkeypatch):
    # Package nested inside a repo directory: <entry>/robot_repo/test_pkg
    entry = tmp_path / "ros_ws" / "src"
    package_dir = entry / "robot_repo" / "test_pkg"
    _make_package_mesh(package_dir)
    _make_package_manifest(package_dir)
    monkeypatch.setenv("ROS_PACKAGE_PATH", str(entry))
    monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)
    resolved = resolve_package_uri("package://test_pkg/meshes/box.stl")
    assert resolved == str(package_dir / "meshes" / "box.stl")
    assert os.path.exists(resolved)


def test_resolve_entry_is_package_root(tmp_path, monkeypatch):
    # The search entry points directly at the package root.
    package_dir = tmp_path / "ws" / "src" / "robot_repo" / "test_pkg"
    _make_package_mesh(package_dir)
    _make_package_manifest(package_dir)
    monkeypatch.setenv("ROS_PACKAGE_PATH", str(package_dir))
    monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)
    resolved = resolve_package_uri("package://test_pkg/meshes/box.stl")
    assert resolved == str(package_dir / "meshes" / "box.stl")
    assert os.path.exists(resolved)


def test_resolve_nested_unknown_package(tmp_path, monkeypatch):
    entry = tmp_path / "ros_ws" / "src"
    package_dir = entry / "robot_repo" / "test_pkg"
    _make_package_mesh(package_dir)
    _make_package_manifest(package_dir)
    monkeypatch.setenv("ROS_PACKAGE_PATH", str(entry))
    monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)
    with pytest.raises(ValueError, match="no_such_pkg"):
        resolve_package_uri("package://no_such_pkg/meshes/box.stl")


@pytest.mark.parametrize("ignored_location", ["entry", "direct", "nested"])
def test_resolve_skips_catkin_ignore(tmp_path, monkeypatch, ignored_location):
    ignored_entry = tmp_path / "ignored_overlay"
    ignored_package = ignored_entry / "test_pkg"
    if ignored_location == "nested":
        ignored_package = ignored_entry / "robot_repo" / "test_pkg"
    _make_package_mesh(ignored_package)
    _make_package_manifest(ignored_package)
    ignored_dir = {
        "entry": ignored_entry,
        "direct": ignored_package,
        "nested": ignored_package.parent,
    }[ignored_location]
    (ignored_dir / "CATKIN_IGNORE").touch()

    active_entry = tmp_path / "active_underlay"
    active_package = active_entry / "test_pkg"
    _make_package_mesh(active_package)
    _make_package_manifest(active_package)
    monkeypatch.setenv("ROS_PACKAGE_PATH", os.pathsep.join(map(str, [ignored_entry, active_entry])))
    monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)

    uri = "package://test_pkg/meshes/box.stl"
    assert resolve_package_uri(uri) == str(active_package / "meshes" / "box.stl")

    monkeypatch.setenv("ROS_PACKAGE_PATH", str(ignored_entry))
    with pytest.raises(ValueError, match="test_pkg"):
        resolve_package_uri(uri)


@pytest.mark.parametrize("with_cycle", [False, True])
def test_load_from_symlinked_repository(tmp_path, monkeypatch, with_cycle):
    external_repo = tmp_path / "external_repo"
    package_dir = external_repo / "test_pkg"
    _make_package_mesh(package_dir)
    _make_package_manifest(package_dir)
    if with_cycle:
        (external_repo / "a_loop").symlink_to(external_repo, target_is_directory=True)

    entry = tmp_path / "ros_ws" / "src"
    entry.mkdir(parents=True)
    linked_repo = entry / "robot_repo"
    linked_repo.symlink_to(external_repo, target_is_directory=True)
    monkeypatch.setenv("ROS_PACKAGE_PATH", str(entry))
    monkeypatch.delenv("AMENT_PREFIX_PATH", raising=False)

    assert resolve_package_uri("package://test_pkg/meshes/box.stl") == str(
        linked_repo / "test_pkg" / "meshes" / "box.stl"
    )
    robot = _load_robot(tmp_path)
    assert robot.links[0].visuals[0].geometry.meshes[0].volume == pytest.approx(0.1 * 0.2 * 0.3)
    with pytest.raises(ValueError, match="no_such_pkg"):
        resolve_package_uri("package://no_such_pkg/meshes/box.stl")


def _load_robot(tmp_path, **load_kwargs):
    urdf_path = tmp_path / "pkg_bot.urdf"
    urdf_path.write_text(URDF_WITH_PACKAGE_MESH)
    return URDF.load(str(urdf_path), **load_kwargs)


def test_urdf_load_with_package_meshes(tmp_path, ros_package_path):
    robot = _load_robot(tmp_path)
    (link,) = robot.links
    (visual,) = link.visuals
    (collision,) = link.collisions
    assert len(visual.geometry.meshes) == 1
    assert len(collision.geometry.meshes) == 1
    mesh = visual.geometry.meshes[0]
    assert isinstance(mesh, trimesh.Trimesh)
    assert mesh.volume == pytest.approx(0.1 * 0.2 * 0.3)


def test_urdf_load_with_package_meshes_ament(tmp_path, ament_prefix_path):
    robot = _load_robot(tmp_path)
    (link,) = robot.links
    assert len(link.visuals[0].geometry.meshes) == 1


def test_urdf_load_with_package_meshes_lazy(tmp_path, ros_package_path):
    robot = _load_robot(tmp_path, lazy_load_meshes=True)
    (link,) = robot.links
    # lazy meshes are only loaded (and the URI resolved) on first access
    assert len(link.visuals[0].geometry.meshes) == 1


def test_urdf_load_with_unknown_package(tmp_path, ros_package_path):
    urdf_path = tmp_path / "bad.urdf"
    urdf_path.write_text(URDF_WITH_PACKAGE_MESH.replace("test_pkg", "no_such_pkg"))
    with pytest.raises(ValueError, match="no_such_pkg"):
        URDF.load(str(urdf_path))


URDF_WITH_PACKAGE_TEXTURE = """<robot name="pkg_bot">
  <material name="checkered">
    <texture filename="package://test_pkg/textures/checker.png"/>
  </material>
  <link name="base">
    <visual>
      <geometry>
        <mesh filename="package://test_pkg/meshes/box.stl"/>
      </geometry>
      <material name="checkered"/>
    </visual>
    <collision>
      <geometry>
        <mesh filename="package://test_pkg/meshes/box.stl"/>
      </geometry>
    </collision>
  </link>
</robot>
"""


def _snapshot_tree(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_save_does_not_overwrite_package_assets(tmp_path, ros_package_path):
    from PIL import Image

    package_dir = ros_package_path / "test_pkg"
    texture_dir = package_dir / "textures"
    texture_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4), (200, 30, 30)).save(str(texture_dir / "checker.png"))
    before = _snapshot_tree(package_dir)

    urdf_path = tmp_path / "pkg_bot.urdf"
    urdf_path.write_text(URDF_WITH_PACKAGE_TEXTURE)
    robot = URDF.load(str(urdf_path))

    # Simulate in-memory edits: saving must still not touch the package.
    for link in robot.links:
        for element in list(link.visuals) + list(link.collisions):
            for mesh in element.geometry.meshes:
                mesh.apply_translation([1.0, 0.0, 0.0])
    robot.materials[0].texture.image.putpixel((0, 0), (10, 220, 10))

    out_dir = tmp_path / "saved"
    out_dir.mkdir()
    out_path = out_dir / "pkg_bot.urdf"
    robot.save(str(out_path))

    # The original package files are byte-for-byte unchanged, and the
    # saved URDF still references the assets by their package URIs.
    assert _snapshot_tree(package_dir) == before
    saved_xml = out_path.read_text()
    assert "package://test_pkg/meshes/box.stl" in saved_xml
    assert "package://test_pkg/textures/checker.png" in saved_xml

    # The saved URDF round-trips: its package URIs still resolve.
    reloaded = URDF.load(str(out_path))
    assert len(reloaded.links[0].visuals[0].geometry.meshes) == 1
