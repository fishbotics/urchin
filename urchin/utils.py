"""Utilities for URDF parsing."""

from __future__ import annotations

import os
from typing import Sequence, Union

import numpy as np
import numpy.typing as npt
import trimesh
from lxml import etree as ET


def rpy_to_matrix(coords: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert roll-pitch-yaw coordinates to a 3x3 homogenous rotation matrix.

    The roll-pitch-yaw axes in a typical URDF are defined as a
    rotation of ``r`` radians around the x-axis followed by a rotation of
    ``p`` radians around the y-axis followed by a rotation of ``y`` radians
    around the z-axis. These are the Z1-Y2-X3 Tait-Bryan angles. See
    Wikipedia_ for more information.

    .. _Wikipedia: https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix

    Parameters
    ----------
    coords : (...,3) float
        The roll-pitch-yaw coordinates in order (x-rot, y-rot, z-rot).

    Returns
    -------
    R : (...,3,3) float
        The corresponding homogenous 3x3 rotation matrix.
    """
    coords = np.asanyarray(coords, dtype=np.float64)
    if coords.shape[-1:] != (3,):
        raise ValueError("Roll-pitch-yaw coordinates must have shape (...,3)")

    coords_cos = np.cos(coords)
    coords_sin = np.sin(coords)
    c1 = coords_cos[..., 2]
    c2 = coords_cos[..., 1]
    c3 = coords_cos[..., 0]
    s1 = coords_sin[..., 2]
    s2 = coords_sin[..., 1]
    s3 = coords_sin[..., 0]

    return np.stack(
        [
            np.stack(
                [c1 * c2, (c1 * s2 * s3) - (c3 * s1), (s1 * s3) + (c1 * c3 * s2)],
                axis=-1,
            ),
            np.stack(
                [c2 * s1, (c1 * c3) + (s1 * s2 * s3), (c3 * s1 * s2) - (c1 * s3)],
                axis=-1,
            ),
            np.stack([-s2, c2 * s3, c2 * c3], axis=-1),
        ],
        axis=-2,
    )


def matrix_to_rpy(R: npt.ArrayLike, solution: int = 1) -> npt.NDArray[np.float64]:
    """Convert a 3x3 transform matrix to roll-pitch-yaw coordinates.

    The roll-pitchRyaw axes in a typical URDF are defined as a
    rotation of ``r`` radians around the x-axis followed by a rotation of
    ``p`` radians around the y-axis followed by a rotation of ``y`` radians
    around the z-axis. These are the Z1-Y2-X3 Tait-Bryan angles. See
    Wikipedia_ for more information.

    .. _Wikipedia: https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix

    There are typically two possible roll-pitch-yaw coordinates that could have
    created a given rotation matrix. Specify ``solution=1`` for the first one
    and ``solution=2`` for the second one.

    Parameters
    ----------
    R : (3,3) float
        A 3x3 homogenous rotation matrix.
    solution : int
        Either 1 or 2, indicating which solution to return.

    Returns
    -------
    coords : (3,) float
        The roll-pitch-yaw coordinates in order (x-rot, y-rot, z-rot).
    """
    R = np.asanyarray(R, dtype=np.float64)
    r = 0.0
    p = 0.0
    y = 0.0

    if np.abs(R[2, 0]) >= 1.0 - 1e-12:
        y = 0.0
        if R[2, 0] < 0:
            p = np.pi / 2
            r = np.arctan2(R[0, 1], R[0, 2])
        else:
            p = -np.pi / 2
            r = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        if solution == 1:
            p = -np.arcsin(R[2, 0])
        else:
            p = np.pi + np.arcsin(R[2, 0])
        r = np.arctan2(R[2, 1] / np.cos(p), R[2, 2] / np.cos(p))
        y = np.arctan2(R[1, 0] / np.cos(p), R[0, 0] / np.cos(p))

    return np.array([r, p, y], dtype=np.float64)


def matrix_to_xyz_rpy(matrix: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert a 4x4 homogenous matrix to xyzrpy coordinates.

    Parameters
    ----------
    matrix : (4,4) float
        The homogenous transform matrix.

    Returns
    -------
    xyz_rpy : (6,) float
        The xyz_rpy vector.
    """
    M = np.asanyarray(matrix, dtype=np.float64)
    xyz = M[:3, 3]
    rpy = matrix_to_rpy(M[:3, :3])
    return np.hstack((xyz, rpy))


def xyz_rpy_to_matrix(xyz_rpy: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert xyz_rpy coordinates to a 4x4 homogenous matrix.

    Parameters
    ----------
    xyz_rpy : (...,6) float
        The xyz_rpy vector.

    Returns
    -------
    matrix : (...,4,4) float
        The homogenous transform matrix.
    """
    arr = np.asanyarray(xyz_rpy, dtype=np.float64)
    if arr.shape[-1:] != (6,):
        raise ValueError("XYZ-RPY coordinates must have shape (...,6)")

    matrix = np.zeros(arr.shape[:-1] + (4, 4), dtype=np.float64)
    matrix[..., 3, 3] = 1.0
    matrix[..., :3, 3] = arr[..., :3]
    matrix[..., :3, :3] = rpy_to_matrix(arr[..., 3:])
    return matrix


def parse_origin(node: ET._Element) -> npt.NDArray[np.float64]:
    """Find the ``origin`` subelement of an XML node and convert it
    into a 4x4 homogenous transformation matrix.

    Parameters
    ----------
    node : :class`lxml.etree.Element`
        An XML node which (optionally) has a child node with the ``origin``
        tag.

    Returns
    -------
    matrix : (4,4) float
        The 4x4 homogneous transform matrix that corresponds to this node's
        ``origin`` child. Defaults to the identity matrix if no ``origin``
        child was found.
    """
    matrix = np.eye(4, dtype=np.float64)
    origin_node = node.find("origin")
    if origin_node is not None:
        if "xyz" in origin_node.attrib:
            matrix[:3, 3] = np.fromstring(origin_node.attrib["xyz"], sep=" ")
        if "rpy" in origin_node.attrib:
            rpy = np.fromstring(origin_node.attrib["rpy"], sep=" ")
            matrix[:3, :3] = rpy_to_matrix(rpy)
    return matrix


def unparse_origin(matrix: npt.ArrayLike) -> ET._Element:
    """Turn a 4x4 homogenous matrix into an ``origin`` XML node.

    Parameters
    ----------
    matrix : (4,4) float
        The 4x4 homogneous transform matrix to convert into an ``origin``
        XML node.

    Returns
    -------
    node : :class`lxml.etree.Element`
        An XML node whose tag is ``origin``. The node has two attributes:

        - ``xyz`` - A string with three space-delimited floats representing
          the translation of the origin.
        - ``rpy`` - A string with three space-delimited floats representing
          the rotation of the origin.
    """
    node = ET.Element("origin")
    M = np.asanyarray(matrix, dtype=np.float64)
    node.attrib["xyz"] = "{} {} {}".format(*M[:3, 3])
    node.attrib["rpy"] = "{} {} {}".format(*matrix_to_rpy(M[:3, :3]))
    return node


def _package_dir_name(package_dir: str) -> Union[str, None]:
    """Return the package name for a directory, or ``None`` if the
    directory is not a ROS package.

    A directory counts as a package when it contains a ``package.xml``
    (catkin/ament) or ``manifest.xml`` (rosbuild) manifest, matching how
    ``rospkg`` identifies packages. The name comes from the manifest's
    ``<name>`` element when present, and otherwise from the directory
    name (the rosbuild convention).
    """
    for manifest in ("package.xml", "manifest.xml"):
        manifest_path = os.path.join(package_dir, manifest)
        if os.path.isfile(manifest_path):
            try:
                name = ET.parse(manifest_path).getroot().findtext("name")
            except Exception:
                name = None
            if name and name.strip():
                return name.strip()
            return os.path.basename(os.path.normpath(package_dir))
    return None


def _find_package_in_entry(entry: str, package_name: str) -> Union[str, None]:
    """Find ``package_name`` under one ``ROS_PACKAGE_PATH`` entry.

    Mirrors ``rospkg`` discovery: the package may be a direct child of
    the entry, the entry itself may be the package root, or the package
    may be nested deeper in the tree (e.g. ``<entry>/robot_repo/pkg``),
    in which case it is identified by its manifest. Symlinked directories
    are followed, while directories containing ``CATKIN_IGNORE`` and
    their descendants are skipped.
    """
    if not os.path.isdir(entry) or os.path.isfile(os.path.join(entry, "CATKIN_IGNORE")):
        return None
    direct = os.path.join(entry, package_name)
    if os.path.isdir(direct) and not os.path.isfile(os.path.join(direct, "CATKIN_IGNORE")):
        return direct

    visited: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(entry, followlinks=True):
        real_path = os.path.realpath(dirpath)
        if "CATKIN_IGNORE" in filenames or real_path in visited:
            dirnames[:] = []
            continue
        # Following symlinks must not revisit a directory through a cycle.
        visited.add(real_path)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        if _package_dir_name(dirpath) == package_name:
            return dirpath
    return None


def resolve_package_uri(uri: str) -> str:
    """Resolve a ``package://`` URI to a filesystem path.

    URDF files in the ROS ecosystem typically reference meshes and
    textures as ``package://<package>/<relative path>``. Without a ROS
    installation, the package is located through the same environment
    variables ROS tools use:

    - ``ROS_PACKAGE_PATH``: a list of directories that are searched the
      way ``rospkg`` searches them -- a package may be a direct child of
      an entry, the entry itself may be the package root, or the
      package may be nested deeper in the entry's tree, where it is
      identified by its ``package.xml``/``manifest.xml`` manifest.
    - ``AMENT_PREFIX_PATH``: a list of install prefixes, where the
      package lives under ``share/<package>``.

    The first search path (in the order above) that contains the package
    wins, matching ``rospkg`` semantics.

    Parameters
    ----------
    uri : str
        The ``package://`` URI to resolve.

    Returns
    -------
    resolved : str
        The filesystem path the URI points to. The file itself is not
        required to exist; only the package directory must be found.

    Raises
    ------
    ValueError
        If the URI is malformed, escapes the package directory, or the
        package cannot be found in either environment variable.
    """
    if not uri.startswith("package://"):
        raise ValueError("Not a package URI: {!r}".format(uri))
    package_name, _, relative_path = uri[len("package://") :].partition("/")
    if not package_name or not relative_path:
        raise ValueError(
            "Invalid package URI {!r}: expected 'package://<package>/<path>'".format(uri)
        )

    package_dirs = []
    for entry in os.environ.get("ROS_PACKAGE_PATH", "").split(os.pathsep):
        if entry:
            found = _find_package_in_entry(entry, package_name)
            if found is not None:
                package_dirs.append(found)
    for entry in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if entry:
            package_dirs.append(os.path.join(entry, "share", package_name))

    for package_dir in package_dirs:
        if os.path.isdir(package_dir):
            resolved = os.path.normpath(os.path.join(package_dir, relative_path))
            if os.path.commonpath([package_dir, resolved]) != os.path.normpath(package_dir):
                raise ValueError(
                    "Invalid package URI {!r}: path escapes the package directory".format(uri)
                )
            return resolved

    raise ValueError(
        "Could not resolve {!r}: package '{}' was not found in "
        "ROS_PACKAGE_PATH or AMENT_PREFIX_PATH".format(uri, package_name)
    )


def get_filename(base_path: str, file_path: str, makedirs: bool = False) -> str:
    """Formats a file path correctly for URDF loading.

    Parameters
    ----------
    base_path : str
        The base path to the URDF's folder.
    file_path : str
        The path to the file. This may also be a ``package://`` URI, in
        which case it is resolved with :func:`resolve_package_uri`.
    makedirs : bool, optional
        If ``True``, the directories leading to the file will be created
        if needed.

    Returns
    -------
    resolved : str
        The resolved filepath -- just the normal ``file_path`` if it was an
        absolute path, otherwise that path joined to ``base_path``.
    """
    fn = file_path
    if file_path.startswith("package://"):
        fn = resolve_package_uri(file_path)
    elif not os.path.isabs(file_path):
        fn = os.path.join(base_path, file_path)
    if makedirs:
        d, _ = os.path.split(fn)
        if not os.path.exists(d):
            os.makedirs(d)
    return fn


def load_meshes(filename: str) -> list[trimesh.Trimesh]:
    """Loads triangular meshes from a file.

    Parameters
    ----------
    filename : str
        Path to the mesh file.

    Returns
    -------
    meshes : list of :class:`~trimesh.base.Trimesh`
        The meshes loaded from the file.
    """
    meshes_obj: trimesh.Geometry = trimesh.load(filename)

    # If we got a scene, dump the meshes
    if isinstance(meshes_obj, trimesh.Scene):
        dumped = list(meshes_obj.dump())
        meshes: list[trimesh.Trimesh] = [g for g in dumped if isinstance(g, trimesh.Trimesh)]
    elif isinstance(meshes_obj, trimesh.Trimesh):
        meshes = [meshes_obj]
    elif isinstance(meshes_obj, (list, tuple, set)):
        meshes = list(meshes_obj)
        if len(meshes) == 0:
            raise ValueError("At least one mesh must be pmeshesent in file")
        for r in meshes:
            if not isinstance(r, trimesh.Trimesh):
                raise TypeError("Could not load meshes from file")
    else:
        raise ValueError("Unable to load mesh from file")

    return meshes


def configure_origin(
    value: Union[None, Sequence[float], npt.ArrayLike],
) -> npt.NDArray[np.float64]:
    """Convert a value into a 4x4 transform matrix.

    Parameters
    ----------
    value : None, (...,6) float, or (...,4,4) float
        The value to turn into the matrix.
        If (...,6), interpreted as xyzrpy coordinates.

    Returns
    -------
    matrix : (...,4,4) float
        The created matrix. If ``value`` is ``None``, returns the identity.
    """
    if value is None:
        value = np.eye(4, dtype=np.float64)
    elif isinstance(value, (list, tuple, np.ndarray)):
        value = np.asanyarray(value, dtype=np.float64)
        if value.shape[-1:] == (6,):
            value = xyz_rpy_to_matrix(value)
        elif value.shape[-2:] != (4, 4):
            raise ValueError(
                "Origin must be specified as an (...,6) vector or (...,4,4) "
                "homogenous transformation matrix"
            )
    else:
        raise TypeError("Invalid type for origin, expect 4x4 matrix")
    return value
