import os

import numpy as np
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
    """A rectangular prism whose center is at the local origin.

    Parameters
    ----------
    size : (3,) float
        The length, width, and height of the box in meters.
    """

    _ATTRIBS = {"size": (np.ndarray, True)}
    _TAG = "box"

    def __init__(self, size):
        self.size = size
        self._meshes = []

    @property
    def size(self):
        """(3,) float : The length, width, and height of the box in meters."""
        return self._size

    @size.setter
    def size(self, value):
        self._size = np.asanyarray(value).astype(np.float64)
        self._meshes = []

    @property
    def meshes(self):
        """list of :class:`~trimesh.base.Trimesh` : The triangular meshes
        that represent this object.
        """
        if len(self._meshes) == 0:
            self._meshes = [trimesh.creation.box(extents=self.size)]
        return self._meshes

    def copy(self, prefix="", scale=None):
        """Create a deep copy with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all names.

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

    def __init__(self, radius, length):
        self.radius = radius
        self.length = length
        self._meshes = []

    @property
    def radius(self):
        """float : The radius of the cylinder in meters."""
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = float(value)
        self._meshes = []

    @property
    def length(self):
        """float : The length of the cylinder in meters."""
        return self._length

    @length.setter
    def length(self, value):
        self._length = float(value)
        self._meshes = []

    @property
    def meshes(self):
        """list of :class:`~trimesh.base.Trimesh` : The triangular meshes
        that represent this object.
        """
        if len(self._meshes) == 0:
            self._meshes = [
                trimesh.creation.cylinder(radius=self.radius, height=self.length)
            ]
        return self._meshes

    def copy(self, prefix="", scale=None):
        """Create a deep copy with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all names.

        Returns
        -------
        :class:`.Cylinder`
            A deep copy.
        """
        if scale is None:
            scale = 1.0
        if isinstance(scale, (list, np.ndarray)):
            if scale[0] != scale[1]:
                raise ValueError(
                    "Cannot rescale cylinder geometry with asymmetry in x/y"
                )
            c = self.__class__(
                radius=self.radius * scale[0],
                length=self.length * scale[2],
            )
        else:
            c = self.__class__(
                radius=self.radius * scale,
                length=self.length * scale,
            )
        return c


class Sphere(URDFType):
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

    def __init__(self, radius):
        self.radius = radius
        self._meshes = []

    @property
    def radius(self):
        """float : The radius of the sphere in meters."""
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = float(value)
        self._meshes = []

    @property
    def meshes(self):
        """list of :class:`~trimesh.base.Trimesh` : The triangular meshes
        that represent this object.
        """
        if len(self._meshes) == 0:
            if self.radius == 0:
                print("[urchin]: radius equal to 0 is not supported, using 1e-5.")
                self.radius = 1e-5
            self._meshes = [trimesh.creation.icosphere(radius=self.radius)]
        return self._meshes

    def copy(self, prefix="", scale=None):
        """Create a deep copy with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all names.

        Returns
        -------
        :class:`.Sphere`
            A deep copy.
        """
        if scale is None:
            scale = 1.0
        if isinstance(scale, (list, np.ndarray)):
            if scale[0] != scale[1] or scale[0] != scale[2]:
                raise ValueError("Spheres do not support non-uniform scaling!")
            scale = scale[0]
        s = self.__class__(
            radius=self.radius * scale,
        )
        return s


class Mesh(URDFTypeWithMesh):
    """A triangular mesh object.

    Parameters
    ----------
    filename : str
        The path to the mesh that contains this object. This can be
        relative to the top-level URDF or an absolute path.
    scale : (3,) float, optional
        The scaling value for the mesh along the XYZ axes.
        If ``None``, assumes no scale is applied.
    meshes : list of :class:`~trimesh.base.Trimesh`
        A list of meshes that compose this mesh.
        The list of meshes is useful for visual geometries that
        might be composed of separate trimesh objects.
        If not specified, the mesh is loaded from the file using trimesh.
    """

    _ATTRIBS = {"filename": (str, True), "scale": (np.ndarray, False)}
    _TAG = "mesh"

    def __init__(self, filename, combine, scale=None, meshes=None, lazy_filename=None):
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
    def filename(self):
        """str : The path to the mesh file for this object."""
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = value

    @property
    def scale(self):
        """(3,) float : A scaling for the mesh along its local XYZ axes."""
        return self._scale

    @scale.setter
    def scale(self, value):
        if value is not None:
            value = np.asanyarray(value).astype(np.float64)
        self._scale = value

    @property
    def meshes(self):
        """list of :class:`~trimesh.base.Trimesh` : The triangular meshes
        that represent this object.
        """
        if self.lazy_filename is not None and self._meshes is None:
            self.meshes = self._load_and_combine_meshes(
                self.lazy_filename, self.combine
            )
        return self._meshes

    @meshes.setter
    def meshes(self, value):
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
                    raise TypeError(
                        "Mesh requires a trimesh.Trimesh or a " "list of them"
                    )
        elif isinstance(value, trimesh.Trimesh):
            value = [value]
        else:
            raise TypeError("Mesh requires a trimesh.Trimesh")
        self._meshes = value

    @classmethod
    def _load_and_combine_meshes(cls, fn, combine):
        meshes = load_meshes(fn)
        if combine:
            # Delete visuals for simplicity
            for m in meshes:
                m.visual = trimesh.visual.ColorVisuals(mesh=m)
            meshes = [meshes[0] + meshes[1:]]
        return meshes

    @classmethod
    def _from_xml(cls, node, path, lazy_load_meshes):
        kwargs = cls._parse(node, path, lazy_load_meshes)

        # Load the mesh, combining collision geometry meshes but keeping
        # visual ones separate to preserve colors and textures
        fn = get_filename(path, kwargs["filename"])
        combine = node.getparent().getparent().tag == Collision._TAG
        if not lazy_load_meshes:
            meshes = cls._load_and_combine_meshes(fn, combine)
            kwargs["lazy_filename"] = None
        else:
            meshes = None
            kwargs["lazy_filename"] = fn
        kwargs["meshes"] = meshes
        kwargs["combine"] = combine

        return cls(**kwargs)

    def _to_xml(self, parent, path):
        # Get the filename
        fn = get_filename(path, self.filename, makedirs=True)

        # Export the meshes as a single file
        if self._meshes is not None:
            meshes = self.meshes
            if len(meshes) == 1:
                meshes = meshes[0]
            elif os.path.splitext(fn)[1] == ".glb":
                meshes = trimesh.scene.Scene(geometry=meshes)
            trimesh.exchange.export.export_mesh(meshes, fn)

        # Unparse the node
        node = self._unparse(path)
        return node

    def copy(self, prefix="", scale=None):
        """Create a deep copy with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all names.

        Returns
        -------
        :class:`.Mesh`
            A deep copy.
        """
        meshes = [m.copy() for m in self.meshes]
        if scale is not None:
            sm = np.eye(4)
            if isinstance(scale, (list, np.ndarray)):
                sm[:3, :3] = np.diag(scale)
            else:
                sm[:3, :3] = np.diag(np.repeat(scale, 3))
            for i, m in enumerate(meshes):
                meshes[i] = m.apply_transform(sm)
        base, fn = os.path.split(self.filename)
        fn = "{}{}".format(prefix, self.filename)
        m = self.__class__(
            filename=os.path.join(base, fn),
            combine=self.combine,
            scale=(self.scale.copy() if self.scale is not None else None),
            meshes=meshes,
            lazy_filename=self.lazy_filename,
        )
        return m


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

    def __init__(self, box=None, cylinder=None, sphere=None, mesh=None):
        if box is None and cylinder is None and sphere is None and mesh is None:
            raise ValueError("At least one geometry element must be set")
        self.box = box
        self.cylinder = cylinder
        self.sphere = sphere
        self.mesh = mesh

    @property
    def box(self):
        """:class:`.Box` : Box geometry."""
        return self._box

    @box.setter
    def box(self, value):
        if value is not None and not isinstance(value, Box):
            raise TypeError("Expected Box type")
        self._box = value

    @property
    def cylinder(self):
        """:class:`.Cylinder` : Cylinder geometry."""
        return self._cylinder

    @cylinder.setter
    def cylinder(self, value):
        if value is not None and not isinstance(value, Cylinder):
            raise TypeError("Expected Cylinder type")
        self._cylinder = value

    @property
    def sphere(self):
        """:class:`.Sphere` : Spherical geometry."""
        return self._sphere

    @sphere.setter
    def sphere(self, value):
        if value is not None and not isinstance(value, Sphere):
            raise TypeError("Expected Sphere type")
        self._sphere = value

    @property
    def mesh(self):
        """:class:`.Mesh` : Mesh geometry."""
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        if value is not None and not isinstance(value, Mesh):
            raise TypeError("Expected Mesh type")
        self._mesh = value

    @property
    def geometry(self):
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
    def meshes(self):
        """list of :class:`~trimesh.base.Trimesh` : The geometry's triangular
        mesh representation(s).
        """
        return self.geometry.meshes

    def copy(self, prefix="", scale=None):
        """Create a deep copy with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all names.

        Returns
        -------
        :class:`.Geometry`
            A deep copy.
        """
        v = self.__class__(
            box=(self.box.copy(prefix=prefix, scale=scale) if self.box else None),
            cylinder=(
                self.cylinder.copy(prefix=prefix, scale=scale)
                if self.cylinder
                else None
            ),
            sphere=(
                self.sphere.copy(prefix=prefix, scale=scale) if self.sphere else None
            ),
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

    def __init__(self, name, origin, geometry):
        self.geometry = geometry
        self.name = name
        self.origin = origin

    @property
    def geometry(self):
        """:class:`.Geometry` : The geometry of this element."""
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        if not isinstance(value, Geometry):
            raise TypeError("Must set geometry with Geometry object")
        self._geometry = value

    @property
    def name(self):
        """str : The name of this collision element."""
        return self._name

    @name.setter
    def name(self, value):
        if value is not None:
            value = str(value)
        self._name = value

    @property
    def origin(self):
        """(4,4) float : The pose of this element relative to the link frame."""
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = configure_origin(value)

    @classmethod
    def _from_xml(cls, node, path, lazy_load_meshes):
        kwargs = cls._parse(node, path, lazy_load_meshes)
        kwargs["origin"] = parse_origin(node)
        return cls(**kwargs)

    def _to_xml(self, parent, path):
        node = self._unparse(path)
        node.append(unparse_origin(self.origin))
        return node

    def copy(self, prefix="", scale=None):
        """Create a deep copy of the visual with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all joint and link names.

        Returns
        -------
        :class:`.Visual`
            A deep copy of the visual.
        """
        origin = self.origin.copy()
        if scale is not None:
            if not isinstance(scale, (list, np.ndarray)):
                scale = np.repeat(scale, 3)
            origin[:3, 3] *= scale
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

    def __init__(self, geometry, name=None, origin=None, material=None):
        self.geometry = geometry
        self.name = name
        self.origin = origin
        self.material = material

    @property
    def geometry(self):
        """:class:`.Geometry` : The geometry of this element."""
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        if not isinstance(value, Geometry):
            raise TypeError("Must set geometry with Geometry object")
        self._geometry = value

    @property
    def name(self):
        """str : The name of this visual element."""
        return self._name

    @name.setter
    def name(self, value):
        if value is not None:
            value = str(value)
        self._name = value

    @property
    def origin(self):
        """(4,4) float : The pose of this element relative to the link frame."""
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = configure_origin(value)

    @property
    def material(self):
        """:class:`.Material` : The material for this element."""
        return self._material

    @material.setter
    def material(self, value):
        if value is not None:
            if not isinstance(value, Material):
                raise TypeError("Must set material with Material object")
        self._material = value

    @classmethod
    def _from_xml(cls, node, path, lazy_load_meshes):
        kwargs = cls._parse(node, path, lazy_load_meshes)
        kwargs["origin"] = parse_origin(node)
        return cls(**kwargs)

    def _to_xml(self, parent, path):
        node = self._unparse(path)
        node.append(unparse_origin(self.origin))
        return node

    def copy(self, prefix="", scale=None):
        """Create a deep copy of the visual with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all joint and link names.

        Returns
        -------
        :class:`.Visual`
            A deep copy of the visual.
        """
        origin = self.origin.copy()
        if scale is not None:
            if not isinstance(scale, (list, np.ndarray)):
                scale = np.repeat(scale, 3)
            origin[:3, 3] *= scale
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

    def __init__(self, mass, inertia, origin=None):
        self.mass = mass
        self.inertia = inertia
        self.origin = origin

    @property
    def mass(self):
        """float : The mass of the link in kilograms."""
        return self._mass

    @mass.setter
    def mass(self, value):
        self._mass = float(value)

    @property
    def inertia(self):
        """(3,3) float : The 3x3 symmetric rotational inertia matrix."""
        return self._inertia

    @inertia.setter
    def inertia(self, value):
        value = np.asanyarray(value).astype(np.float64)
        if not np.allclose(value, value.T):
            raise ValueError("Inertia must be a symmetric matrix")
        self._inertia = value

    @property
    def origin(self):
        """(4,4) float : The pose of the inertials relative to the link frame."""
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = configure_origin(value)

    @classmethod
    def _from_xml(cls, node, path):
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

    def _to_xml(self, parent, path):
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

    def copy(self, prefix="", mass=None, origin=None, inertia=None):
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

    def __init__(self, name, inertial, visuals, collisions):
        self.name = name
        self.inertial = inertial
        self.visuals = visuals
        self.collisions = collisions

        self._collision_mesh = None

    @property
    def name(self):
        """str : The name of this link."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def inertial(self):
        """:class:`.Inertial` : Inertial properties of the link."""
        return self._inertial

    @inertial.setter
    def inertial(self, value):
        if value is not None and not isinstance(value, Inertial):
            raise TypeError("Expected Inertial object")
        # Set default inertial
        if value is None:
            value = Inertial(mass=1.0, inertia=np.eye(3))
        self._inertial = value

    @property
    def visuals(self):
        """list of :class:`.Visual` : The visual properties of this link."""
        return self._visuals

    @visuals.setter
    def visuals(self, value):
        if value is None:
            value = []
        else:
            value = list(value)
            for v in value:
                if not isinstance(v, Visual):
                    raise ValueError("Expected list of Visual objects")
        self._visuals = value

    @property
    def collisions(self):
        """list of :class:`.Collision` : The collision properties of this link."""
        return self._collisions

    @collisions.setter
    def collisions(self, value):
        if value is None:
            value = []
        else:
            value = list(value)
            for v in value:
                if not isinstance(v, Collision):
                    raise ValueError("Expected list of Collision objects")
        self._collisions = value

    @property
    def collision_mesh(self):
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
            self._collision_mesh = meshes[0] + meshes[1:]
        return self._collision_mesh

    def copy(self, prefix="", scale=None, collision_only=False):
        """Create a deep copy of the link.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all joint and link names.

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
                if not isinstance(scale, (list, np.ndarray)):
                    scale = np.repeat(scale, 3)
                sm[:3, :3] = np.diag(scale)
                cm = self.collision_mesh.copy()
                cm.density = self.inertial.mass / cm.volume
                cm.apply_transform(sm)
                cmm = np.eye(4)
                cmm[:3, 3] = cm.center_mass
                inertial = Inertial(mass=cm.mass, inertia=cm.moment_inertia, origin=cmm)

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
