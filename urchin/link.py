from __future__ import annotations

import os
from typing import Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import trimesh
from lxml import etree as ET

from urchin.base import URDFType, URDFTypeWithMesh
from urchin.material import Material
from urchin.utils import (
    configure_origin,
    get_filename,
    load_meshes,
    parse_origin,
    unparse_origin,
)


class Box(URDFType):
    _meshes: list[trimesh.Trimesh]
    """A rectangular prism whose center is at the local origin.

    Parameters
    ----------
    size : (3,) float
        The length, width, and height of the box in meters.
    """

    _ATTRIBS = {"size": (np.ndarray, True)}
    _TAG = "box"

    def __init__(self, size: npt.ArrayLike):
        self.size = size
        self._meshes = []

    @property
    def size(self) -> np.ndarray:
        """(3,) float : The length, width, and height of the box in meters."""
        return self._size

    @size.setter
    def size(self, value: npt.ArrayLike) -> None:
        self._size = np.asanyarray(value).astype(np.float64)
        self._meshes = []

    @property
    def meshes(self) -> list[trimesh.Trimesh]:
        """list of :class:`~trimesh.base.Trimesh` : The triangular meshes
        that represent this object.
        """
        if len(self._meshes) == 0:
            self._meshes = [trimesh.creation.box(extents=self.size)]
        return self._meshes

    def copy(self, prefix: str = "", scale: Union[float, Sequence[float], None] = None) -> "Box":
        """Create a deep copy with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all names.
        scale : float or (3,) float, optional
            Uniform or per-axis scale to apply to the box.

        Returns
        -------
        :class:`.Box`
            A deep copy.
        """
        if scale is None:
            scale = 1.0
        b = self.__class__(
            size=self.size.copy() * scale,
        )
        return b


class Cylinder(URDFType):
    _meshes: list[trimesh.Trimesh]
    """A cylinder whose center is at the local origin.

    Parameters
    ----------
    radius : float
        The radius of the cylinder in meters.
    length : float
        The length of the cylinder in meters.
    """

    _ATTRIBS = {
        "radius": (float, True),
        "length": (float, True),
    }
    _TAG = "cylinder"

    def __init__(self, radius: float, length: float):
        self.radius = radius
        self.length = length
        self._meshes = []

    @property
    def radius(self) -> float:
        """float : The radius of the cylinder in meters."""
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        self._radius = float(value)
        self._meshes = []

    @property
    def length(self) -> float:
        """float : The length of the cylinder in meters."""
        return self._length

    @length.setter
    def length(self, value: float) -> None:
        self._length = float(value)
        self._meshes = []

    @property
    def meshes(self) -> list[trimesh.Trimesh]:
        """list of :class:`~trimesh.base.Trimesh` : The triangular meshes
        that represent this object.
        """
        if len(self._meshes) == 0:
            self._meshes = [trimesh.creation.cylinder(radius=self.radius, height=self.length)]
        return self._meshes

    def copy(
        self, prefix: str = "", scale: Union[float, Sequence[float], None] = None
    ) -> "Cylinder":
        """Create a deep copy with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all names.
        scale : float or (3,) float, optional
            Uniform or per-axis scale to apply. Per-axis must have equal
            X/Y values for cylinders.

        Returns
        -------
        :class:`.Cylinder`
            A deep copy.
        """
        if scale is None:
            scale = 1.0
        if isinstance(scale, (list, tuple, np.ndarray)):
            s = list(scale)
            if s[0] != s[1]:
                raise ValueError("Cannot rescale cylinder geometry with asymmetry in x/y")
            c = self.__class__(
                radius=self.radius * s[0],
                length=self.length * s[2],
            )
        else:
            from typing import cast as _cast

            s_val: float = float(_cast(float, scale))
            c = self.__class__(
                radius=self.radius * s_val,
                length=self.length * s_val,
            )
        return c


class Sphere(URDFType):
    _meshes: list[trimesh.Trimesh]
    """A sphere whose center is at the local origin.

    Parameters
    ----------
    radius : float
        The radius of the sphere in meters.
    """

    _ATTRIBS = {
        "radius": (float, True),
    }
    _TAG = "sphere"

    def __init__(self, radius: float):
        self.radius = radius
        self._meshes = []

    @property
    def radius(self) -> float:
        """float : The radius of the sphere in meters."""
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        self._radius = float(value)
        self._meshes = []

    @property
    def meshes(self) -> list[trimesh.Trimesh]:
        """list of :class:`~trimesh.base.Trimesh` : The triangular meshes
        that represent this object.
        """
        if len(self._meshes) == 0:
            if self.radius == 0:
                print("[urchin]: radius equal to 0 is not supported, using 1e-5.")
                self.radius = 1e-5
            self._meshes = [trimesh.creation.icosphere(radius=self.radius)]
        return self._meshes

    def copy(self, prefix: str = "", scale: Union[float, Sequence[float], None] = None) -> "Sphere":
        """Create a deep copy with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all names.
        scale : float or (3,) float, optional
            Uniform scale only. Non-uniform scales are rejected.

        Returns
        -------
        :class:`.Sphere`
            A deep copy.
        """
        if scale is None:
            scale = 1.0
        if isinstance(scale, (list, tuple, np.ndarray)):
            scale_list = list(scale)
            if scale_list[0] != scale_list[1] or scale_list[0] != scale_list[2]:
                raise ValueError("Spheres do not support non-uniform scaling!")
            scale = scale_list[0]
        from typing import cast as _cast

        sf: float = float(_cast(float, scale))
        result = self.__class__(
            radius=self.radius * sf,
        )
        return result


class Mesh(URDFTypeWithMesh):
    """A triangular mesh object.

    Parameters
    ----------
    filename : str
        The path to the mesh that contains this object. This can be
        relative to the top-level URDF or an absolute path.
    combine : bool
        If ``True``, combine geometries into a single mesh (used for
        collision geometry). Visual meshes are typically kept separate to
        preserve colors and textures.
    scale : (3,) float, optional
        The scaling value for the mesh along the XYZ axes.
        If ``None``, assumes no scale is applied.
    meshes : list of :class:`~trimesh.base.Trimesh` or :class:`~trimesh.base.Trimesh` or ``str``
        A list of meshes or a single mesh that composes this mesh. If a
        ``str`` is provided, the mesh is loaded from disk.
        The list of meshes is useful for visual geometries that
        might be composed of separate trimesh objects.
        If not specified, the mesh is loaded from the file using trimesh.
    """

    _ATTRIBS = {"filename": (str, True), "scale": (np.ndarray, False)}
    _TAG = "mesh"

    def __init__(
        self,
        filename: str,
        combine: bool,
        scale: Optional[npt.ArrayLike] = None,
        meshes: Union[list[trimesh.Trimesh], trimesh.Trimesh, str, None] = None,
        lazy_filename: Optional[str] = None,
    ):
        if meshes is None:
            if lazy_filename is None:
                meshes = load_meshes(filename)
            else:
                meshes = None
        self.filename = filename
        self.scale = scale
        self.lazy_filename = lazy_filename
        self.combine = combine
        self.meshes = meshes

    @property
    def filename(self) -> str:
        """str : The path to the mesh file for this object."""
        return self._filename

    @filename.setter
    def filename(self, value: str) -> None:
        self._filename = value

    @property
    def scale(self) -> Optional[np.ndarray]:
        """(3,) float : A scaling for the mesh along its local XYZ axes."""
        return self._scale

    @scale.setter
    def scale(self, value: Optional[npt.ArrayLike]) -> None:
        if value is not None:
            value = np.asanyarray(value).astype(np.float64)
        self._scale = value

    @property
    def meshes(self) -> list[trimesh.Trimesh]:
        """list of :class:`~trimesh.base.Trimesh` : The triangular meshes
        that represent this object.
        """
        if self.lazy_filename is not None and self._meshes is None:
            self.meshes = self._load_and_combine_meshes(self.lazy_filename, self.combine)
        # At this point meshes should be loaded or assigned
        return self._meshes or []

    @meshes.setter
    def meshes(
        self,
        value: Union[list[trimesh.Trimesh], trimesh.Trimesh, str, None],
    ) -> None:
        if self.lazy_filename is not None and value is None:
            self._meshes = None
        elif isinstance(value, str):
            value = load_meshes(value)
        elif isinstance(value, (list, tuple, set, np.ndarray)):
            value = list(value)
            if len(value) == 0:
                raise ValueError("Mesh must have at least one trimesh.Trimesh")
            for m in value:
                if not isinstance(m, trimesh.Trimesh):
                    raise TypeError("Mesh requires a trimesh.Trimesh or a list of them")
        elif isinstance(value, trimesh.Trimesh):
            value = [value]
        else:
            raise TypeError("Mesh requires a trimesh.Trimesh")
        self._meshes = value

    @classmethod
    def _load_and_combine_meshes(cls, fn: str, combine: bool) -> list[trimesh.Trimesh]:
        meshes = load_meshes(fn)
        if combine:
            # Delete visuals for simplicity
            for m in meshes:
                m.visual = trimesh.visual.ColorVisuals(mesh=m)
            merged = meshes[0]
            for extra in meshes[1:]:
                merged = merged + extra
            return [merged]
        return meshes

    @classmethod
    def _from_xml(cls, node: ET._Element, path: str, lazy_load_meshes: Optional[bool] = None):
        # Explicit parse for filename and optional scale
        filename_attr = str(node.attrib["filename"]) if "filename" in node.attrib else ""
        scale_attr = node.attrib.get("scale")
        scale_val = np.fromstring(scale_attr, sep=" ", dtype=np.float64) if scale_attr else None

        # Resolve actual file for loading
        fn = get_filename(path, filename_attr)
        combine = node.getparent().getparent().tag == Collision._TAG
        if not lazy_load_meshes:
            meshes = cls._load_and_combine_meshes(fn, combine)
            lazy_filename = None
        else:
            meshes = None
            lazy_filename = fn

        return cls(
            filename=filename_attr,
            combine=combine,
            scale=scale_val,
            meshes=meshes,
            lazy_filename=lazy_filename,
        )

    def _to_xml(self, parent: Optional[ET._Element], path: str) -> ET._Element:
        # Get the filename
        fn = get_filename(path, self.filename, makedirs=True)

        # Export the meshes as a single file
        if self._meshes is not None:
            meshes_list = self.meshes or []
            export_obj: Union[trimesh.Trimesh, trimesh.Scene, list[trimesh.Trimesh]]
            if len(meshes_list) == 1:
                export_obj = meshes_list[0]
            elif os.path.splitext(fn)[1] == ".glb":
                export_obj = trimesh.scene.Scene(geometry=meshes_list)
            else:
                export_obj = meshes_list
            trimesh.exchange.export.export_mesh(export_obj, fn)

        # Unparse the node
        node = self._unparse(path)
        return node

    def copy(self, prefix: str = "", scale: Union[float, Sequence[float], None] = None) -> "Mesh":
        """Create a deep copy with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all names.
        scale : float or (3,) float, optional
            Uniform or per-axis scale applied via a transform.

        Returns
        -------
        :class:`.Mesh`
            A deep copy.
        """
        meshes = [mesh_i.copy() for mesh_i in self.meshes]
        if scale is not None:
            sm = np.eye(4)
            if isinstance(scale, (list, tuple, np.ndarray)):
                sm[:3, :3] = np.diag(np.asanyarray(scale, dtype=float))
            else:
                from typing import cast as _cast

                sm[:3, :3] = np.diag(np.repeat(_cast(float, scale), 3))
            for mesh_i in meshes:
                mesh_i.apply_transform(sm)
        base, fn = os.path.split(self.filename)
        fn = "{}{}".format(prefix, self.filename)
        new_mesh = self.__class__(
            filename=os.path.join(base, fn),
            combine=self.combine,
            scale=(self.scale.copy() if self.scale is not None else None),
            meshes=meshes,
            lazy_filename=self.lazy_filename,
        )
        return new_mesh


class Geometry(URDFTypeWithMesh):
    """A wrapper for all geometry types.

    Only one of the following values can be set, all others should be set
    to ``None``.

    Parameters
    ----------
    box : :class:`.Box`, optional
        Box geometry.
    cylinder : :class:`.Cylinder`
        Cylindrical geometry.
    sphere : :class:`.Sphere`
        Spherical geometry.
    mesh : :class:`.Mesh`
        Mesh geometry.
    """

    _ELEMENTS = {
        "box": (Box, False, False),
        "cylinder": (Cylinder, False, False),
        "sphere": (Sphere, False, False),
        "mesh": (Mesh, False, False),
    }
    _TAG = "geometry"

    def __init__(
        self,
        box: Optional[Box] = None,
        cylinder: Optional[Cylinder] = None,
        sphere: Optional[Sphere] = None,
        mesh: Optional[Mesh] = None,
    ):
        if box is None and cylinder is None and sphere is None and mesh is None:
            raise ValueError("At least one geometry element must be set")
        self.box = box
        self.cylinder = cylinder
        self.sphere = sphere
        self.mesh = mesh

    @property
    def box(self) -> Optional[Box]:
        """:class:`.Box` : Box geometry."""
        return self._box

    @box.setter
    def box(self, value: Optional[Box]) -> None:
        if value is not None and not isinstance(value, Box):
            raise TypeError("Expected Box type")
        self._box = value

    @property
    def cylinder(self) -> Optional[Cylinder]:
        """:class:`.Cylinder` : Cylinder geometry."""
        return self._cylinder

    @cylinder.setter
    def cylinder(self, value: Optional[Cylinder]) -> None:
        if value is not None and not isinstance(value, Cylinder):
            raise TypeError("Expected Cylinder type")
        self._cylinder = value

    @property
    def sphere(self) -> Optional[Sphere]:
        """:class:`.Sphere` : Spherical geometry."""
        return self._sphere

    @sphere.setter
    def sphere(self, value: Optional[Sphere]) -> None:
        if value is not None and not isinstance(value, Sphere):
            raise TypeError("Expected Sphere type")
        self._sphere = value

    @property
    def mesh(self) -> Optional[Mesh]:
        """:class:`.Mesh` : Mesh geometry."""
        return self._mesh

    @mesh.setter
    def mesh(self, value: Optional[Mesh]) -> None:
        if value is not None and not isinstance(value, Mesh):
            raise TypeError("Expected Mesh type")
        self._mesh = value

    @property
    def geometry(self) -> Union[Box, Cylinder, Sphere, Mesh, None]:
        """:class:`.Box`, :class:`.Cylinder`, :class:`.Sphere`, or
        :class:`.Mesh` : The valid geometry element.
        """
        if self.box is not None:
            return self.box
        if self.cylinder is not None:
            return self.cylinder
        if self.sphere is not None:
            return self.sphere
        if self.mesh is not None:
            return self.mesh
        return None

    @property
    def meshes(self) -> list[trimesh.Trimesh]:
        """list of :class:`~trimesh.base.Trimesh` : The geometry's triangular
        mesh representation(s).
        """
        assert self.geometry is not None
        return self.geometry.meshes

    def copy(
        self, prefix: str = "", scale: Union[float, Sequence[float], None] = None
    ) -> "Geometry":
        """Create a deep copy with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all names.
        scale : float or (3,) float, optional
            Uniform or per-axis scale to apply to the underlying geometry.

        Returns
        -------
        :class:`.Geometry`
            A deep copy.
        """
        v = self.__class__(
            box=(self.box.copy(prefix=prefix, scale=scale) if self.box else None),
            cylinder=(self.cylinder.copy(prefix=prefix, scale=scale) if self.cylinder else None),
            sphere=(self.sphere.copy(prefix=prefix, scale=scale) if self.sphere else None),
            mesh=(self.mesh.copy(prefix=prefix, scale=scale) if self.mesh else None),
        )
        return v


class Collision(URDFTypeWithMesh):
    """Collision properties of a link.

    Parameters
    ----------
    geometry : :class:`.Geometry`
        The geometry of the element
    name : str, optional
        The name of the collision geometry.
    origin : (4,4) float, optional
        The pose of the collision element relative to the link frame.
        Defaults to identity.
    """

    _ATTRIBS = {"name": (str, False)}
    _ELEMENTS = {
        "geometry": (Geometry, True, False),
    }
    _TAG = "collision"

    def __init__(self, name: Optional[str], origin: Optional[npt.ArrayLike], geometry: Geometry):
        self.geometry = geometry
        self.name = name
        self.origin = origin

    @property
    def geometry(self) -> Geometry:
        """:class:`.Geometry` : The geometry of this element."""
        return self._geometry

    @geometry.setter
    def geometry(self, value: Geometry) -> None:
        if not isinstance(value, Geometry):
            raise TypeError("Must set geometry with Geometry object")
        self._geometry = value

    @property
    def name(self) -> Optional[str]:
        """str : The name of this collision element."""
        return self._name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        if value is not None:
            value = str(value)
        self._name = value

    @property
    def origin(self) -> np.ndarray:
        """(4,4) float : The pose of this element relative to the link frame."""
        return self._origin

    @origin.setter
    def origin(self, value: Optional[npt.ArrayLike]) -> None:
        self._origin = configure_origin(value)

    @classmethod
    def _from_xml(cls, node: ET._Element, path: str, lazy_load_meshes: Optional[bool] = None):
        name = node.attrib.get("name")
        geom_node = node.find("geometry")
        if geom_node is None:
            raise ValueError("Collision element missing geometry")
        geometry = Geometry._from_xml(geom_node, path, lazy_load_meshes)
        origin = parse_origin(node)
        return cls(name=name, origin=origin, geometry=geometry)

    def _to_xml(self, parent: Optional[ET._Element], path: str) -> ET._Element:
        node = self._unparse(path)
        node.append(unparse_origin(self.origin))
        return node

    def copy(
        self, prefix: str = "", scale: Union[float, Sequence[float], None] = None
    ) -> "Collision":
        """Create a deep copy of the visual with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all joint and link names.
        scale : float or (3,) float, optional
            Uniform or per-axis scale applied to the position offset.

        Returns
        -------
        :class:`.Visual`
            A deep copy of the visual.
        """
        origin = self.origin.copy()
        if scale is not None:
            if not isinstance(scale, (list, tuple, np.ndarray)):
                from typing import cast as _cast

                scale_arr = np.repeat(_cast(float, scale), 3)
            else:
                scale_arr = np.asanyarray(scale, dtype=float)
            origin[:3, 3] *= scale_arr
        return self.__class__(
            name="{}{}".format(prefix, self.name),
            origin=origin,
            geometry=self.geometry.copy(prefix=prefix, scale=scale),
        )


class Visual(URDFTypeWithMesh):
    """Visual properties of a link.

    Parameters
    ----------
    geometry : :class:`.Geometry`
        The geometry of the element
    name : str, optional
        The name of the visual geometry.
    origin : (4,4) float, optional
        The pose of the visual element relative to the link frame.
        Defaults to identity.
    material : :class:`.Material`, optional
        The material of the element.
    """

    _ATTRIBS = {"name": (str, False)}
    _ELEMENTS = {
        "geometry": (Geometry, True, False),
        "material": (Material, False, False),
    }
    _TAG = "visual"

    def __init__(
        self,
        geometry: Geometry,
        name: Optional[str] = None,
        origin: Optional[npt.ArrayLike] = None,
        material: Optional[Material] = None,
    ):
        self.geometry = geometry
        self.name = name
        self.origin = origin
        self.material = material

    @property
    def geometry(self) -> Geometry:
        """:class:`.Geometry` : The geometry of this element."""
        return self._geometry

    @geometry.setter
    def geometry(self, value: Geometry) -> None:
        if not isinstance(value, Geometry):
            raise TypeError("Must set geometry with Geometry object")
        self._geometry = value

    @property
    def name(self) -> Optional[str]:
        """str : The name of this visual element."""
        return self._name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        if value is not None:
            value = str(value)
        self._name = value

    @property
    def origin(self) -> np.ndarray:
        """(4,4) float : The pose of this element relative to the link frame."""
        return self._origin

    @origin.setter
    def origin(self, value: Optional[npt.ArrayLike]) -> None:
        self._origin = configure_origin(value)

    @property
    def material(self) -> Optional[Material]:
        """:class:`.Material` : The material for this element."""
        return self._material

    @material.setter
    def material(self, value: Optional[Material]) -> None:
        if value is not None:
            if not isinstance(value, Material):
                raise TypeError("Must set material with Material object")
        self._material = value

    @classmethod
    def _from_xml(cls, node: ET._Element, path: str, lazy_load_meshes: Optional[bool] = None):
        geom_node = node.find("geometry")
        if geom_node is None:
            raise ValueError("Visual element missing geometry")
        geometry = Geometry._from_xml(geom_node, path, lazy_load_meshes)
        name = node.attrib.get("name")
        origin = parse_origin(node)
        mat_node = node.find("material")
        material = Material._from_xml(mat_node, path) if mat_node is not None else None
        return cls(geometry=geometry, name=name, origin=origin, material=material)

    def _to_xml(self, parent: Optional[ET._Element], path: str) -> ET._Element:
        node = self._unparse(path)
        node.append(unparse_origin(self.origin))
        return node

    def copy(self, prefix: str = "", scale: Union[float, Sequence[float], None] = None) -> "Visual":
        """Create a deep copy of the visual with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all joint and link names.
        scale : float or (3,) float, optional
            Uniform or per-axis scale applied to the position offset.

        Returns
        -------
        :class:`.Visual`
            A deep copy of the visual.
        """
        origin = self.origin.copy()
        if scale is not None:
            if not isinstance(scale, (list, tuple, np.ndarray)):
                from typing import cast as _cast

                scale_arr = np.repeat(_cast(float, scale), 3)
            else:
                scale_arr = np.asanyarray(scale, dtype=float)
            origin[:3, 3] *= scale_arr
        return self.__class__(
            geometry=self.geometry.copy(prefix=prefix, scale=scale),
            name="{}{}".format(prefix, self.name),
            origin=origin,
            material=(self.material.copy(prefix=prefix) if self.material else None),
        )


class Inertial(URDFType):
    """The inertial properties of a link.

    Parameters
    ----------
    mass : float
        The mass of the link in kilograms.
    inertia : (3,3) float
        The 3x3 symmetric rotational inertia matrix.
    origin : (4,4) float, optional
        The pose of the inertials relative to the link frame.
        Defaults to identity if not specified.
    """

    _TAG = "inertial"

    def __init__(self, mass: float, inertia: npt.ArrayLike, origin: Optional[npt.ArrayLike] = None):
        self.mass = mass
        self.inertia = inertia
        self.origin = origin

    @property
    def mass(self) -> float:
        """float : The mass of the link in kilograms."""
        return self._mass

    @mass.setter
    def mass(self, value: float) -> None:
        self._mass = float(value)

    @property
    def inertia(self) -> np.ndarray:
        """(3,3) float : The 3x3 symmetric rotational inertia matrix."""
        return self._inertia

    @inertia.setter
    def inertia(self, value: npt.ArrayLike) -> None:
        value = np.asanyarray(value).astype(np.float64)
        if not np.allclose(value, value.T):
            raise ValueError("Inertia must be a symmetric matrix")
        self._inertia = value

    @property
    def origin(self) -> np.ndarray:
        """(4,4) float : The pose of the inertials relative to the link frame."""
        return self._origin

    @origin.setter
    def origin(self, value: Optional[npt.ArrayLike]) -> None:
        self._origin = configure_origin(value)

    @classmethod
    def _from_xml(cls, node: ET._Element, path: str, lazy_load_meshes: Optional[bool] = None):
        origin = parse_origin(node)
        mass = float(node.find("mass").attrib["value"])
        n = node.find("inertia")
        xx = float(n.attrib["ixx"])
        xy = float(n.attrib["ixy"])
        xz = float(n.attrib["ixz"])
        yy = float(n.attrib["iyy"])
        yz = float(n.attrib["iyz"])
        zz = float(n.attrib["izz"])
        inertia = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=np.float64)
        return cls(mass=mass, inertia=inertia, origin=origin)

    def _to_xml(self, parent: Optional[ET._Element], path: str) -> ET._Element:
        node = ET.Element("inertial")
        node.append(unparse_origin(self.origin))
        mass = ET.Element("mass")
        mass.attrib["value"] = str(self.mass)
        node.append(mass)
        inertia = ET.Element("inertia")
        inertia.attrib["ixx"] = str(self.inertia[0, 0])
        inertia.attrib["ixy"] = str(self.inertia[0, 1])
        inertia.attrib["ixz"] = str(self.inertia[0, 2])
        inertia.attrib["iyy"] = str(self.inertia[1, 1])
        inertia.attrib["iyz"] = str(self.inertia[1, 2])
        inertia.attrib["izz"] = str(self.inertia[2, 2])
        node.append(inertia)
        return node

    def copy(
        self,
        prefix: str = "",
        mass: Optional[float] = None,
        origin: Optional[np.ndarray] = None,
        inertia: Optional[np.ndarray] = None,
    ) -> "Inertial":
        """Create a deep copy of the visual with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all joint and link names.

        Returns
        -------
        :class:`.Inertial`
            A deep copy of the visual.
        """
        if mass is None:
            mass = self.mass
        if origin is None:
            origin = self.origin.copy()
        if inertia is None:
            inertia = self.inertia.copy()
        return self.__class__(
            mass=mass,
            inertia=inertia,
            origin=origin,
        )


class Link(URDFTypeWithMesh):
    """A link of a rigid object.

    Parameters
    ----------
    name : str
        The name of the link.
    inertial : :class:`.Inertial`, optional
        The inertial properties of the link.
    visuals : list of :class:`.Visual`, optional
        The visual properties of the link.
    collsions : list of :class:`.Collision`, optional
        The collision properties of the link.
    """

    _ATTRIBS = {
        "name": (str, True),
    }
    _ELEMENTS = {
        "inertial": (Inertial, False, False),
        "visuals": (Visual, False, True),
        "collisions": (Collision, False, True),
    }
    _TAG = "link"

    def __init__(
        self,
        name: str,
        inertial: Optional[Inertial],
        visuals: Optional[Sequence[Visual]],
        collisions: Optional[Sequence[Collision]],
    ):
        self.name = name
        self.inertial = inertial
        self.visuals = visuals
        self.collisions = collisions

        self._collision_mesh: Optional[trimesh.Trimesh] = None

    @property
    def name(self) -> str:
        """str : The name of this link."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)

    @property
    def inertial(self) -> Inertial:
        """:class:`.Inertial` : Inertial properties of the link."""
        return self._inertial

    @inertial.setter
    def inertial(self, value: Optional[Inertial]) -> None:
        if value is not None and not isinstance(value, Inertial):
            raise TypeError("Expected Inertial object")
        # Set default inertial
        if value is None:
            value = Inertial(mass=1.0, inertia=np.eye(3))
        self._inertial = value

    @property
    def visuals(self) -> list[Visual]:
        """list of :class:`.Visual` : The visual properties of this link."""
        return self._visuals

    @visuals.setter
    def visuals(self, value: Optional[Sequence[Visual]]) -> None:
        if value is None:
            value = []
        else:
            value = list(value)
            for v in value:
                if not isinstance(v, Visual):
                    raise ValueError("Expected list of Visual objects")
        self._visuals = value

    @property
    def collisions(self) -> list[Collision]:
        """list of :class:`.Collision` : The collision properties of this link."""
        return self._collisions

    @collisions.setter
    def collisions(self, value: Optional[Sequence[Collision]]) -> None:
        if value is None:
            value = []
        else:
            value = list(value)
            for v in value:
                if not isinstance(v, Collision):
                    raise ValueError("Expected list of Collision objects")
        self._collisions = value

    @property
    def collision_mesh(self) -> Optional[trimesh.Trimesh]:
        """:class:`~trimesh.base.Trimesh` : A single collision mesh for
        the link, specified in the link frame, or None if there isn't one.
        """
        if len(self.collisions) == 0:
            return None
        if self._collision_mesh is None:
            meshes = []
            for c in self.collisions:
                for m in c.geometry.meshes:
                    m = m.copy()
                    pose = c.origin
                    if c.geometry.mesh is not None:
                        if c.geometry.mesh.scale is not None:
                            S = np.eye(4)
                            S[:3, :3] = np.diag(c.geometry.mesh.scale)
                            pose = pose.dot(S)
                    m.apply_transform(pose)
                    meshes.append(m)
            if len(meshes) == 0:
                return None
            merged = meshes[0]
            for extra in meshes[1:]:
                merged = merged + extra
            self._collision_mesh = merged
        return self._collision_mesh

    def copy(
        self,
        prefix: str = "",
        scale: Union[float, Sequence[float], None] = None,
        collision_only: bool = False,
    ) -> "Link":
        """Create a deep copy of the link.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all joint and link names.
        scale : float or (3,) float, optional
            Uniform or per-axis scale applied to meshes and inertial.
        collision_only : bool, optional
            If True, only collision geometry is preserved in the copy.

        Returns
        -------
        link : :class:`.Link`
            A deep copy of the Link.
        """
        inertial = self.inertial.copy() if self.inertial is not None else None
        cm = self._collision_mesh
        if scale is not None:
            if self.collision_mesh is not None and self.inertial is not None:
                sm = np.eye(4)
                if not isinstance(scale, (list, tuple, np.ndarray)):
                    from typing import cast as _cast

                    scale_arr = np.repeat(_cast(float, scale), 3)
                else:
                    scale_arr = np.asanyarray(scale, dtype=float)
                sm[:3, :3] = np.diag(scale_arr)
                cm = self.collision_mesh.copy()
                cm.density = self.inertial.mass / cm.volume
                cm.apply_transform(sm)
                cmm = np.eye(4)
                cmm[:3, 3] = cm.center_mass
                inertial = Inertial(mass=float(cm.mass), inertia=cm.moment_inertia, origin=cmm)

        visuals = None
        if not collision_only:
            visuals = [v.copy(prefix=prefix, scale=scale) for v in self.visuals]

        cpy = self.__class__(
            name="{}{}".format(prefix, self.name),
            inertial=inertial,
            visuals=visuals,
            collisions=[v.copy(prefix=prefix, scale=scale) for v in self.collisions],
        )
        cpy._collision_mesh = cm
        return cpy
