from __future__ import annotations

import os
import time
from collections import OrderedDict
from typing import IO, Mapping, Optional, Sequence, Union, cast

import networkx as nx
import numpy as np
import numpy.typing as npt
import trimesh
from lxml import etree as ET

from urchin.base import URDFTypeWithMesh
from urchin.joint import Joint
from urchin.link import Link
from urchin.material import Material
from urchin.transmission import Transmission


class URDF(URDFTypeWithMesh):
    """The top-level URDF specification.

    The URDF encapsulates an articulated object, such as a robot or a gripper.
    It is made of links and joints that tie them together and define their
    relative motions.

    Parameters
    ----------
    name : str
        The name of the URDF.
    links : list of :class:`.Link`
        The links of the URDF.
    joints : list of :class:`.Joint`, optional
        The joints of the URDF.
    transmissions : list of :class:`.Transmission`, optional
        The transmissions of the URDF.
    materials : list of :class:`.Material`, optional
        The materials for the URDF.
    other_xml : str, optional
        A string containing any extra XML for extensions.
    """

    _ATTRIBS = {
        "name": (str, True),
    }
    _ELEMENTS = {
        "links": (Link, True, True),
        "joints": (Joint, False, True),
        "transmissions": (Transmission, False, True),
        "materials": (Material, False, True),
    }
    _TAG = "robot"

    def __init__(
        self,
        name: str,
        links: Sequence[Link],
        joints: Optional[Sequence[Joint]] = None,
        transmissions: Optional[Sequence[Transmission]] = None,
        materials: Optional[Sequence[Material]] = None,
        other_xml: Optional[Union[bytes, str]] = None,
    ) -> None:
        if joints is None:
            joints = []
        if transmissions is None:
            transmissions = []
        if materials is None:
            materials = []

        self.name = name
        self.other_xml = other_xml
        self.mesh_need_to_mirror: list[str] = []

        # No setters for these
        self._links: list[Link] = list(links)
        self._joints: list[Joint] = list(joints)
        self._transmissions: list[Transmission] = list(transmissions)
        self._materials: list[Material] = list(materials)

        # Set up private helper maps from name to value
        self._link_map: dict[str, Link] = {}
        self._joint_map: dict[str, Joint] = {}
        self._transmission_map: dict[str, Transmission] = {}
        self._material_map: dict[str, Material] = {}

        for link_obj in self._links:
            if link_obj.name in self._link_map:
                raise ValueError("Two links with name {} found".format(link_obj.name))
            self._link_map[link_obj.name] = link_obj

        for joint_obj in self._joints:
            if joint_obj.name in self._joint_map:
                raise ValueError("Two joints with name {} found".format(joint_obj.name))
            self._joint_map[joint_obj.name] = joint_obj

        for trans_obj in self._transmissions:
            if trans_obj.name in self._transmission_map:
                raise ValueError("Two transmissions with name {} found".format(trans_obj.name))
            self._transmission_map[trans_obj.name] = trans_obj

        for mat_obj in self._materials:
            if mat_obj.name in self._material_map:
                raise ValueError("Two materials with name {} found".format(mat_obj.name))
            self._material_map[mat_obj.name] = mat_obj

        # Synchronize materials between links and top-level set
        self._merge_materials()

        # Validate the joints and transmissions
        actuated_joints = self._validate_joints()
        self._validate_transmissions()

        # Create the link graph and base link/end link sets
        self._G = nx.DiGraph()

        # Add all links
        for link in self.links:
            self._G.add_node(link)

        # Add all edges from CHILDREN TO PARENTS, with joints as their object
        for joint in self.joints:
            parent = self._link_map[joint.parent]
            child = self._link_map[joint.child]
            self._G.add_edge(child, parent, joint=joint)

        # Validate the graph and get the base and end links
        self._base_link, self._end_links = self._validate_graph()

        # Cache the paths to the base link
        self._paths_to_base = nx.shortest_path(self._G, target=self._base_link)

        self._actuated_joints = self._sort_joints(actuated_joints)

        # Cache the reverse topological order (useful for speeding up FK,
        # as we want to start at the base and work outward to cache
        # computation.
        self._reverse_topo = list(reversed(list(nx.topological_sort(self._G))))

    @property
    def name(self) -> str:
        """str : The name of the URDF."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)

    @property
    def links(self) -> list[Link]:
        """list of :class:`.Link` : The links of the URDF.

        This returns a copy of the links array which cannot be edited
        directly. If you want to add or remove links, use
        the appropriate functions.
        """
        return list(self._links)

    @property
    def link_map(self) -> dict[str, Link]:
        """dict : Map from link names to the links themselves.

        This returns a copy of the link map which cannot be edited
        directly. If you want to add or remove links, use
        the appropriate functions.
        """
        return dict(self._link_map)

    @property
    def joints(self) -> list[Joint]:
        """list of :class:`.Joint` : The links of the URDF.

        This returns a copy of the joints array which cannot be edited
        directly. If you want to add or remove joints, use
        the appropriate functions.
        """
        return list(self._joints)

    @property
    def joint_map(self) -> dict[str, Joint]:
        """dict : Map from joint names to the joints themselves.

        This returns a copy of the joint map which cannot be edited
        directly. If you want to add or remove joints, use
        the appropriate functions.
        """
        return dict(self._joint_map)

    @property
    def transmissions(self) -> list[Transmission]:
        """list of :class:`.Transmission` : The transmissions of the URDF.

        This returns a copy of the transmissions array which cannot be edited
        directly. If you want to add or remove transmissions, use
        the appropriate functions.
        """
        return list(self._transmissions)

    @property
    def transmission_map(self) -> dict[str, Transmission]:
        """dict : Map from transmission names to the transmissions themselves.

        This returns a copy of the transmission map which cannot be edited
        directly. If you want to add or remove transmissions, use
        the appropriate functions.
        """
        return dict(self._transmission_map)

    @property
    def materials(self) -> list[Material]:
        """list of :class:`.Material` : The materials of the URDF.

        This returns a copy of the materials array which cannot be edited
        directly. If you want to add or remove materials, use
        the appropriate functions.
        """
        return list(self._materials)

    @property
    def material_map(self) -> dict[str, Material]:
        """dict : Map from material names to the materials themselves.

        This returns a copy of the material map which cannot be edited
        directly. If you want to add or remove materials, use
        the appropriate functions.
        """
        return dict(self._material_map)

    @property
    def other_xml(self) -> Optional[Union[bytes, str]]:
        """str : Any extra XML that belongs with the URDF."""
        return self._other_xml

    @other_xml.setter
    def other_xml(self, value: Optional[Union[bytes, str]]) -> None:
        self._other_xml = value

    @property
    def actuated_joints(self) -> list[Joint]:
        """list of :class:`.Joint` : The joints that are independently
        actuated.

        This excludes mimic joints and fixed joints. The joints are listed
        in topological order, starting from the base-most joint.
        """
        return self._actuated_joints

    @property
    def actuated_joint_names(self) -> list[str]:
        """list of :class:`.Joint` : The names of joints that are independently
        actuated.

        This excludes mimic joints and fixed joints. The joints are listed
        in topological order, starting from the base-most joint.
        """
        return [j.name for j in self._actuated_joints]

    def cfg_to_vector(
        self,
        cfg: Union[
            Mapping[str, float],
            Sequence[float],
            npt.ArrayLike,
            None,
        ],
    ) -> Optional[npt.NDArray[np.float64]]:
        """Convert a configuration dictionary into a configuration vector.

        Parameters
        ----------
        cfg : dict or None
            The configuration value.

        Returns
        -------
        vec : (n,) float
            The configuration vector, or None if no actuated joints present.
        """
        if cfg is None:
            if len(self.actuated_joints) > 0:
                return np.zeros(len(self.actuated_joints))
            else:
                return None
        elif isinstance(cfg, (list, tuple, np.ndarray)):
            return np.asanyarray(cfg)
        elif isinstance(cfg, dict):
            vec = np.zeros(len(self.actuated_joints))
            for i, jn in enumerate(self.actuated_joint_names):
                if jn in cfg:
                    vec[i] = cfg[jn]
            return vec
        else:
            raise ValueError(f"Invalid configuration: {cfg!r}")

    @property
    def base_link(self) -> Link:
        """:class:`.Link`: The base link for the URDF.

        The base link is the single link that has no parent.
        """
        return self._base_link

    @property
    def end_links(self) -> list[Link]:
        """list of :class:`.Link`: The end links for the URDF.

        The end links are the links that have no children.
        """
        return self._end_links

    @property
    def joint_limit_cfgs(self) -> tuple[dict[Joint, float], dict[Joint, float]]:
        """tuple of dict : The lower-bound and upper-bound joint configuration
        maps.

        The first map is the lower-bound map, which maps limited joints to
        their lower joint limits.
        The second map is the upper-bound map, which maps limited joints to
        their upper joint limits.
        """
        lb: dict[Joint, float] = {}
        ub: dict[Joint, float] = {}
        for joint in self.actuated_joints:
            if joint.limit is not None:
                if joint.limit.lower is not None:
                    lb[joint] = joint.limit.lower
                if joint.limit.upper is not None:
                    ub[joint] = joint.limit.upper
        return (lb, ub)

    @property
    def joint_limits(self) -> npt.NDArray[np.float64]:
        """(n,2) float : A lower and upper limit for each joint."""
        limits = []
        for joint in self.actuated_joints:
            limit = [-np.inf, np.inf]
            if joint.limit is not None:
                if joint.limit.lower is not None:
                    limit[0] = joint.limit.lower
                if joint.limit.upper is not None:
                    limit[1] = joint.limit.upper
            limits.append(limit)
        return np.array(limits)

    def link_fk(
        self,
        cfg: Union[
            Mapping[str, float],
            Sequence[float],
            npt.ArrayLike,
            None,
        ] = None,
        link: Optional[Union[str, Link]] = None,
        links: Optional[Sequence[Union[str, Link]]] = None,
        use_names: bool = False,
    ) -> Union[
        dict[Link, npt.NDArray[np.float64]],
        dict[str, npt.NDArray[np.float64]],
        npt.NDArray[np.float64],
    ]:
        """Computes the poses of the URDF's links via forward kinematics.

        Parameters
        ----------
        cfg : dict or (n), float
            A map from joints or joint names to configuration values for
            each joint, or a list containing a value for each actuated joint
            in sorted order from the base link.
            If not specified, all joints are assumed to be in their default
            configurations.
        link : str or :class:`.Link`
            A single link or link name to return a pose for.
        links : list of str or list of :class:`.Link`
            The links or names of links to perform forward kinematics on.
            Only these links will be in the returned map. If neither
            link nor links are specified all links are returned.
        use_names : bool
            If True, the returned dictionary will have keys that are string
            link names rather than the links themselves.

        Returns
        -------
        fk : dict or (4,4) float
            A map from links to 4x4 homogenous transform matrices that
            position them relative to the base link's frame, or a single
            4x4 matrix if ``link`` is specified.
        """
        # Process config value
        joint_cfg = self._process_cfg(cfg)

        # Process link set
        link_set: set[Link] = set()
        if link is not None:
            if isinstance(link, str):
                link_set.add(self._link_map[link])
            elif isinstance(link, Link):
                link_set.add(link)
        elif links is not None:
            for lnk in links:
                if isinstance(lnk, str):
                    link_set.add(self._link_map[lnk])
                elif isinstance(lnk, Link):
                    link_set.add(lnk)
                else:
                    raise TypeError("Got object of type {} in links list".format(type(lnk)))
        else:
            link_set = set(self.links)

        # Compute forward kinematics in reverse topological order
        fk: "OrderedDict[Link, npt.NDArray[np.float64]]" = OrderedDict()
        for lnk in self._reverse_topo:
            if lnk not in link_set:
                continue
            pose = np.eye(4, dtype=np.float64)
            path = cast(list[Link], self._paths_to_base[lnk])
            for i in range(len(path) - 1):
                child = path[i]
                parent = path[i + 1]
                joint = self._G.get_edge_data(child, parent)["joint"]

                cfg = None
                if joint.mimic is not None:
                    mimic_joint = self._joint_map[joint.mimic.joint]
                    if mimic_joint in joint_cfg:
                        cfg = joint_cfg[mimic_joint]
                        cfg = joint.mimic.multiplier * cfg + joint.mimic.offset
                elif joint in joint_cfg:
                    cfg = joint_cfg[joint]
                pose = joint.get_child_pose(cfg).dot(pose)

                # Check existing FK to see if we can exit early
                if parent in fk:
                    pose = fk[parent].dot(pose)
                    break
            fk[lnk] = pose

        if link:
            if isinstance(link, str):
                return fk[self._link_map[link]]
            else:
                return fk[link]
        if use_names:
            return {ell.name: fk[ell] for ell in fk}
        return fk

    def link_fk_batch(self, cfgs=None, link=None, links=None, use_names=False):
        """Computes the poses of the URDF's links via forward kinematics in a batch.

        Parameters
        ----------
        cfgs : dict, list of dict, or (n,m), float
            One of the following: (A) a map from joints or joint names to vectors
            of joint configuration values, (B) a list of maps from joints or joint names
            to single configuration values, or (C) a list of ``n`` configuration vectors,
            each of which has a vector with an entry for each actuated joint.
        link : str or :class:`.Link`
            A single link or link name to return a pose for.
        links : list of str or list of :class:`.Link`
            The links or names of links to perform forward kinematics on.
            Only these links will be in the returned map. If neither
            link nor links are specified all links are returned.
        use_names : bool
            If True, the returned dictionary will have keys that are string
            link names rather than the links themselves.

        Returns
        -------
        fk : dict or (n,4,4) float
            A map from links to a (n,4,4) vector of homogenous transform matrices that
            position the links relative to the base link's frame, or a single
            nx4x4 matrix if ``link`` is specified.
        """
        joint_cfgs, n_cfgs = self._process_cfgs(cfgs)

        # Process link set
        link_set = set()
        if link is not None:
            if isinstance(link, str):
                link_set.add(self._link_map[link])
            elif isinstance(link, Link):
                link_set.add(link)
        elif links is not None:
            for lnk in links:
                if isinstance(lnk, str):
                    link_set.add(self._link_map[lnk])
                elif isinstance(lnk, Link):
                    link_set.add(lnk)
                else:
                    raise TypeError("Got object of type {} in links list".format(type(lnk)))
        else:
            link_set = self.links

        # Compute FK mapping each link to a vector of matrices, one matrix per cfg
        fk = OrderedDict()
        for lnk in self._reverse_topo:
            if lnk not in link_set:
                continue
            poses = np.tile(np.eye(4, dtype=np.float64), (n_cfgs, 1, 1))
            path = self._paths_to_base[lnk]
            for i in range(len(path) - 1):
                child = path[i]
                parent = path[i + 1]
                joint = self._G.get_edge_data(child, parent)["joint"]

                cfg_vals = None
                if joint.mimic is not None:
                    mimic_joint = self._joint_map[joint.mimic.joint]
                    if mimic_joint in joint_cfgs:
                        cfg_vals = joint_cfgs[mimic_joint]
                        cfg_vals = joint.mimic.multiplier * cfg_vals + joint.mimic.offset
                elif joint in joint_cfgs:
                    cfg_vals = joint_cfgs[joint]
                poses = np.matmul(joint.get_child_poses(cfg_vals, n_cfgs), poses)

                if parent in fk:
                    poses = np.matmul(fk[parent], poses)
                    break
            fk[lnk] = poses

        if link:
            if isinstance(link, str):
                return fk[self._link_map[link]]
            else:
                return fk[link]
        if use_names:
            return {ell.name: fk[ell] for ell in fk}
        return fk

    def visual_geometry_fk(
        self,
        cfg: Union[
            Mapping[str, float],
            Sequence[float],
            npt.ArrayLike,
            None,
        ] = None,
        links: Optional[Sequence[Union[str, Link]]] = None,
    ) -> dict:
        """Computes the poses of the URDF's visual geometries using fk.

        Parameters
        ----------
        cfg : dict or (n), float
            A map from joints or joint names to configuration values for
            each joint, or a list containing a value for each actuated joint
            in sorted order from the base link.
            If not specified, all joints are assumed to be in their default
            configurations.
        links : list of str or list of :class:`.Link`
            The links or names of links to perform forward kinematics on.
            Only geometries from these links will be in the returned map.
            If not specified, all links are returned.

        Returns
        -------
        fk : dict
            A map from :class:`Geometry` objects that are part of the visual
            elements of the specified links to the 4x4 homogenous transform
            matrices that position them relative to the base link's frame.
        """
        lfk = cast(dict[Link, npt.NDArray[np.float64]], self.link_fk(cfg=cfg, links=links))

        fk = OrderedDict()
        for link in lfk:
            for visual in link.visuals:
                fk[visual.geometry] = lfk[link].dot(visual.origin)
        return fk

    def visual_geometry_fk_batch(
        self,
        cfgs: Union[
            Mapping[str, Sequence[float]],
            Sequence[Union[Mapping[str, float], None]],
            npt.ArrayLike,
            None,
        ] = None,
        links: Optional[Sequence[Union[str, Link]]] = None,
    ) -> dict:
        """Computes the poses of the URDF's visual geometries using fk.

        Parameters
        ----------
        cfgs : dict, list of dict, or (n,m), float
            One of the following: (A) a map from joints or joint names to vectors
            of joint configuration values, (B) a list of maps from joints or joint names
            to single configuration values, or (C) a list of ``n`` configuration vectors,
            each of which has a vector with an entry for each actuated joint.
        links : list of str or list of :class:`.Link`
            The links or names of links to perform forward kinematics on.
            Only geometries from these links will be in the returned map.
            If not specified, all links are returned.

        Returns
        -------
        fk : dict
            A map from :class:`Geometry` objects that are part of the visual
            elements of the specified links to the 4x4 homogenous transform
            matrices that position them relative to the base link's frame.
        """
        lfk: dict[Link, npt.NDArray[np.float64]] = self.link_fk_batch(cfgs=cfgs, links=links)

        fk = OrderedDict()
        for link in lfk:
            for visual in link.visuals:
                fk[visual.geometry] = np.matmul(lfk[link], visual.origin)
        return fk

    def visual_trimesh_fk(
        self,
        cfg: Union[
            Mapping[str, float],
            Sequence[float],
            npt.ArrayLike,
            None,
        ] = None,
        links: Optional[Sequence[Union[str, Link]]] = None,
    ) -> dict[trimesh.Trimesh, npt.NDArray[np.float64]]:
        """Computes the poses of the URDF's visual trimeshes using fk.

        Parameters
        ----------
        cfg : dict or (n), float
            A map from joints or joint names to configuration values for
            each joint, or a list containing a value for each actuated joint
            in sorted order from the base link.
            If not specified, all joints are assumed to be in their default
            configurations.
        links : list of str or list of :class:`.Link`
            The links or names of links to perform forward kinematics on.
            Only trimeshes from these links will be in the returned map.
            If not specified, all links are returned.

        Returns
        -------
        fk : dict
            A map from :class:`~trimesh.base.Trimesh` objects that are
            part of the visual geometry of the specified links to the
            4x4 homogenous transform matrices that position them relative
            to the base link's frame.
        """
        lfk = cast(dict[Link, npt.NDArray[np.float64]], self.link_fk(cfg=cfg, links=links))
        self.mesh_name_list = []
        fk = OrderedDict()
        for link in lfk:
            for visual in link.visuals:
                for i, mesh in enumerate(visual.geometry.meshes):
                    pose = lfk[link].dot(visual.origin)
                    if visual.geometry.mesh is not None:
                        self.mesh_name_list.append(visual.geometry.mesh.filename)
                        if visual.geometry.mesh.scale is not None:
                            if (
                                np.sum(
                                    visual.geometry.mesh.scale != abs(visual.geometry.mesh.scale)
                                )
                                > 0
                            ):
                                if visual.geometry.mesh.filename not in self.mesh_need_to_mirror:
                                    print(
                                        f"[urchin]: {visual.geometry.mesh.filename} needs to mirror"
                                    )
                                    self.mesh_need_to_mirror.append(visual.geometry.mesh.filename)
                                    mesh_vertices = np.copy(mesh.vertices)
                                    mesh_faces = np.copy(mesh.faces)
                                    mesh_faces_new = np.hstack(
                                        [
                                            mesh_faces[:, 2].reshape(-1, 1),
                                            mesh_faces[:, 1].reshape(-1, 1),
                                            mesh_faces[:, 0].reshape(-1, 1),
                                        ]
                                    )
                                    mesh = trimesh.Trimesh()
                                    mirror_axis = np.where(visual.geometry.mesh.scale < 0)[0][0]
                                    mesh_vertices[:, mirror_axis] = -mesh_vertices[:, mirror_axis]
                                    mesh.vertices = mesh_vertices
                                    mesh.faces = mesh_faces_new
                                    visual.geometry.meshes[i] = mesh
                            S = np.eye(4, dtype=np.float64)
                            S[:3, :3] = np.abs(np.diag(visual.geometry.mesh.scale))
                            pose = pose.dot(S)
                    else:
                        self.mesh_name_list.append("")
                    if visual.material is not None:
                        mesh.visual.face_colors = visual.material.color
                    fk[mesh] = pose
        return fk

    def visual_trimesh_fk_batch(
        self,
        cfgs: Union[
            Mapping[str, Sequence[float]],
            Sequence[Union[Mapping[str, float], None]],
            npt.ArrayLike,
            None,
        ] = None,
        links: Optional[Sequence[Union[str, Link]]] = None,
    ) -> dict[trimesh.Trimesh, npt.NDArray[np.float64]]:
        """Computes the poses of the URDF's visual trimeshes using fk.

        Parameters
        ----------
        cfgs : dict, list of dict, or (n,m), float
            One of the following: (A) a map from joints or joint names to vectors
            of joint configuration values, (B) a list of maps from joints or joint names
            to single configuration values, or (C) a list of ``n`` configuration vectors,
            each of which has a vector with an entry for each actuated joint.
        links : list of str or list of :class:`.Link`
            The links or names of links to perform forward kinematics on.
            Only trimeshes from these links will be in the returned map.
            If not specified, all links are returned.

        Returns
        -------
        fk : dict
            A map from :class:`~trimesh.base.Trimesh` objects that are
            part of the visual geometry of the specified links to the
            4x4 homogenous transform matrices that position them relative
            to the base link's frame.
        """
        lfk = self.link_fk_batch(cfgs=cfgs, links=links)

        fk = OrderedDict()
        for link in lfk:
            for visual in link.visuals:
                for mesh in visual.geometry.meshes:
                    poses = np.matmul(lfk[link], visual.origin)
                    if visual.geometry.mesh is not None:
                        if visual.geometry.mesh.scale is not None:
                            S = np.eye(4, dtype=np.float64)
                            S[:3, :3] = np.diag(visual.geometry.mesh.scale)
                            poses = np.matmul(poses, S)
                    fk[mesh] = poses
        return fk

    def collision_geometry_fk(self, cfg=None, links=None):
        """Computes the poses of the URDF's collision geometries using fk.

        Parameters
        ----------
        cfg : dict or (n), float
            A map from joints or joint names to configuration values for
            each joint, or a list containing a value for each actuated joint
            in sorted order from the base link.
            If not specified, all joints are assumed to be in their default
            configurations.
        links : list of str or list of :class:`.Link`
            The links or names of links to perform forward kinematics on.
            Only geometries from these links will be in the returned map.
            If not specified, all links are returned.

        Returns
        -------
        fk : dict
            A map from :class:`Geometry` objects that are part of the collision
            elements of the specified links to the 4x4 homogenous transform
            matrices that position them relative to the base link's frame.
        """
        lfk = self.link_fk(cfg=cfg, links=links)

        fk = OrderedDict()
        for link in lfk:
            for collision in link.collisions:
                fk[collision] = lfk[link].dot(collision.origin)
        return fk

    def collision_geometry_fk_batch(self, cfgs=None, links=None):
        """Computes the poses of the URDF's collision geometries using fk.

        Parameters
        ----------
        cfgs : dict, list of dict, or (n,m), float
            One of the following: (A) a map from joints or joint names to vectors
            of joint configuration values, (B) a list of maps from joints or joint names
            to single configuration values, or (C) a list of ``n`` configuration vectors,
            each of which has a vector with an entry for each actuated joint.
        links : list of str or list of :class:`.Link`
            The links or names of links to perform forward kinematics on.
            Only geometries from these links will be in the returned map.
            If not specified, all links are returned.

        Returns
        -------
        fk : dict
            A map from :class:`Geometry` objects that are part of the collision
            elements of the specified links to the 4x4 homogenous transform
            matrices that position them relative to the base link's frame.
        """
        lfk = self.link_fk_batch(cfgs=cfgs, links=links)

        fk = OrderedDict()
        for link in lfk:
            for collision in link.collisions:
                fk[collision] = np.matmul(lfk[link], collision.origin)
        return fk

    def collision_trimesh_fk(self, cfg=None, links=None):
        """Computes the poses of the URDF's collision trimeshes using fk.

        Parameters
        ----------
        cfg : dict or (n), float
            A map from joints or joint names to configuration values for
            each joint, or a list containing a value for each actuated joint
            in sorted order from the base link.
            If not specified, all joints are assumed to be in their default
            configurations.
        links : list of str or list of :class:`.Link`
            The links or names of links to perform forward kinematics on.
            Only trimeshes from these links will be in the returned map.
            If not specified, all links are returned.

        Returns
        -------
        fk : dict
            A map from :class:`~trimesh.base.Trimesh` objects that are
            part of the collision geometry of the specified links to the
            4x4 homogenous transform matrices that position them relative
            to the base link's frame.
        """
        lfk = self.link_fk(cfg=cfg, links=links)

        fk = OrderedDict()
        for link in lfk:
            pose = lfk[link]
            cm = link.collision_mesh
            if cm is not None:
                fk[cm] = pose
        return fk

    def collision_trimesh_fk_batch(self, cfgs=None, links=None):
        """Computes the poses of the URDF's collision trimeshes using fk.

        Parameters
        ----------
        cfgs : dict, list of dict, or (n,m), float
            One of the following: (A) a map from joints or joint names to vectors
            of joint configuration values, (B) a list of maps from joints or joint names
            to single configuration values, or (C) a list of ``n`` configuration vectors,
            each of which has a vector with an entry for each actuated joint.
        links : list of str or list of :class:`.Link`
            The links or names of links to perform forward kinematics on.
            Only trimeshes from these links will be in the returned map.
            If not specified, all links are returned.

        Returns
        -------
        fk : dict
            A map from :class:`~trimesh.base.Trimesh` objects that are
            part of the collision geometry of the specified links to the
            4x4 homogenous transform matrices that position them relative
            to the base link's frame.
        """
        lfk = self.link_fk_batch(cfgs=cfgs, links=links)

        fk = OrderedDict()
        for link in lfk:
            poses = lfk[link]
            cm = link.collision_mesh
            if cm is not None:
                fk[cm] = poses
        return fk

    def animate(self, cfg_trajectory=None, loop_time=3.0, use_collision=False):
        """Animate the URDF through a configuration trajectory.

        Parameters
        ----------
        cfg_trajectory : dict or (m,n) float
            A map from joints or joint names to lists of configuration values
            for each joint along the trajectory, or a vector of
            vectors where the second dimension contains a value for each joint.
            If not specified, all joints will articulate from limit to limit.
            The trajectory steps are assumed to be equally spaced out in time.
        loop_time : float
            The time to loop the animation for, in seconds. The trajectory
            will play fowards and backwards during this time, ending
            at the inital configuration.
        use_collision : bool
            If True, the collision geometry is visualized instead of
            the visual geometry.

        Examples
        --------

        You can run this without specifying a ``cfg_trajectory`` to view
        the full articulation of the URDF

        >>> robot = URDF.load("ur5.urdf")
        >>> robot.animate()

        .. image:: /_static/ur5.gif

        >>> ct = {"shoulder_pan_joint": [0.0, 2 * np.pi]}
        >>> robot.animate(cfg_trajectory=ct)

        .. image:: /_static/ur5_shoulder.gif

        >>> ct = {
        ...     "shoulder_pan_joint": [-np.pi / 4, np.pi / 4],
        ...     "shoulder_lift_joint": [0.0, -np.pi / 2.0],
        ...     "elbow_joint": [0.0, np.pi / 2.0],
        ... }
        >>> robot.animate(cfg_trajectory=ct)

        .. image:: /_static/ur5_three_joints.gif

        """
        import pyribbit  # Save pyribbit import for here for CI

        ct = cfg_trajectory

        traj_len = None  # Length of the trajectory in steps
        ct_np = {}  # Numpyified trajectory

        # If trajectory not specified, articulate between the limits.
        if ct is None:
            lb, ub = self.joint_limit_cfgs
            if len(lb) > 0:
                traj_len = 2
                ct_np = {k: np.array([lb[k], ub[k]]) for k in lb}

        # If it is specified, parse it and extract the trajectory length.
        elif isinstance(ct, dict):
            if len(ct) > 0:
                for k in ct:
                    val = np.asanyarray(ct[k]).astype(np.float64)
                    if traj_len is None:
                        traj_len = len(val)
                    elif traj_len != len(val):
                        raise ValueError("Trajectories must be same length")
                    ct_np[k] = val
        elif isinstance(ct, (list, tuple, np.ndarray)):
            traj_len = len(ct)
            ct = np.asanyarray(ct).astype(np.float64)
            if ct.ndim == 1:
                ct = ct.reshape(-1, 1)
            if ct.ndim != 2 or ct.shape[1] != len(self.actuated_joints):
                raise ValueError("Cfg trajectory must have entry for each joint")
            ct_np = {j: ct[:, i] for i, j in enumerate(self.actuated_joints)}
        else:
            raise TypeError("Invalid type for cfg_trajectory: {}".format(type(cfg_trajectory)))

        # If there isn't a trajectory to render, just show the model and exit
        if len(ct_np) == 0 or traj_len < 2:
            self.show(use_collision=use_collision)
            return

        # Create an array of times that loops from 0 to 1 and back to 0
        fps = 30.0
        n_steps = int(loop_time * fps / 2.0)
        times = np.linspace(0.0, 1.0, n_steps)
        times = np.hstack((times, np.flip(times)))

        # Create bin edges in the range [0, 1] for each trajectory step
        bins = np.arange(traj_len) / (float(traj_len) - 1.0)

        # Compute alphas for each time
        right_inds = np.digitize(times, bins, right=True)
        right_inds[right_inds == 0] = 1
        alphas = (bins[right_inds] - times) / (bins[right_inds] - bins[right_inds - 1])

        # Create the new interpolated trajectory
        new_ct = {}
        for k in ct_np:
            new_ct[k] = alphas * ct_np[k][right_inds - 1] + (1.0 - alphas) * ct_np[k][right_inds]

        # Create the scene
        if use_collision:
            fk = self.collision_trimesh_fk()
        else:
            fk = self.visual_trimesh_fk()

        node_map = {}
        scene = pyribbit.Scene()
        for tm in fk:
            pose = fk[tm]
            mesh = pyribbit.Mesh.from_trimesh(tm, smooth=False)
            node = scene.add(mesh, pose=pose)
            node_map[tm] = node

        # Get base pose to focus on
        blp = self.link_fk(links=[self.base_link])[self.base_link]

        # Pop the visualizer asynchronously
        v = pyribbit.Viewer(
            scene, run_in_thread=True, use_raymond_lighting=True, view_center=blp[:3, 3]
        )

        # Now, run our loop
        i = 0
        while v.is_active:
            cfg = {k: new_ct[k][i] for k in new_ct}
            i = (i + 1) % len(times)

            if use_collision:
                fk = self.collision_trimesh_fk(cfg=cfg)
            else:
                fk = self.visual_trimesh_fk(cfg=cfg)

            v.render_lock.acquire()
            for mesh in fk:
                pose = fk[mesh]
                node_map[mesh].matrix = pose
            v.render_lock.release()

            time.sleep(1.0 / fps)

    def show(
        self,
        cfg: Union[
            Mapping[str, float],
            Sequence[float],
            npt.ArrayLike,
            None,
        ] = None,
        use_collision: bool = False,
    ) -> None:
        """Visualize the URDF in a given configuration.

        Parameters
        ----------
        cfg : dict or (n), float
            A map from joints or joint names to configuration values for
            each joint, or a list containing a value for each actuated joint
            in sorted order from the base link.
            If not specified, all joints are assumed to be in their default
            configurations.
        use_collision : bool
            If True, the collision geometry is visualized instead of
            the visual geometry.
        """
        import pyribbit  # Save pyribbit import for here for CI

        if use_collision:
            fk = self.collision_trimesh_fk(cfg=cfg)
        else:
            fk = self.visual_trimesh_fk(cfg=cfg)

        scene = pyribbit.Scene()
        for tm in fk:
            pose = fk[tm]
            mesh = pyribbit.Mesh.from_trimesh(tm, smooth=False)
            scene.add(mesh, pose=pose)
        pyribbit.Viewer(scene, use_raymond_lighting=True)

    def copy(
        self,
        name: Optional[str] = None,
        prefix: str = "",
        scale: Union[float, Sequence[float], None] = None,
        collision_only: bool = False,
    ) -> "URDF":
        """Make a deep copy of the URDF.

        Parameters
        ----------
        name : str, optional
            A name for the new URDF. If not specified, ``self.name`` is used.
        prefix : str, optional
            A prefix to apply to all names except for the base URDF name.
        scale : float or (3,) float, optional
            A scale to apply to the URDF.
        collision_only : bool, optional
            If True, all visual geometry is redirected to the collision geometry.

        Returns
        -------
        copy : :class:`.URDF`
            The copied URDF.
        """
        return self.__class__(
            name=(name if name else self.name),
            links=[v.copy(prefix, scale, collision_only) for v in self.links],
            joints=[v.copy(prefix, scale) for v in self.joints],
            transmissions=[v.copy(prefix, scale) for v in self.transmissions],
            materials=[v.copy(prefix) for v in self.materials],
            other_xml=self.other_xml,
        )

    def save(self, file_obj: Union[str, IO[bytes], IO[str]]) -> None:
        """Save this URDF to a file.

        Parameters
        ----------
        file_obj : str or file-like object
            The file to save the URDF to. Should be the path to the
            ``.urdf`` XML file. Any paths in the URDF should be specified
            as relative paths to the ``.urdf`` file instead of as ROS
            resources.

        Returns
        -------
        None
            Nothing. Writes the URDF XML to ``file_obj``.
        """
        if isinstance(file_obj, str):
            path, _ = os.path.split(file_obj)
        else:
            path, _ = os.path.split(os.path.realpath(file_obj.name))

        node = self._to_xml(None, path)
        tree = ET.ElementTree(node)
        tree.write(file_obj, pretty_print=True, xml_declaration=True, encoding="utf-8")

    def join(
        self,
        other: "URDF",
        link: Union[Link, str],
        origin: Optional[npt.ArrayLike] = None,
        name: Optional[str] = None,
        prefix: str = "",
    ) -> "URDF":
        """Join another URDF to this one by rigidly fixturing the two at a link.

        Parameters
        ----------
        other : :class:.`URDF`
            Another URDF to fuze to this one.
        link : :class:`.Link` or str
            The link of this URDF to attach the other URDF to.
        origin : (4,4) float, optional
            The location in this URDF's link frame to attach the base link of the other
            URDF at.
        name : str, optional
            A name for the new URDF.
        prefix : str, optional
            If specified, all joints and links from the (other) mesh will be pre-fixed
            with this value to avoid name clashes.

        Returns
        -------
        :class:`.URDF`
            The new URDF.
        """
        myself = self.copy()
        other = other.copy(prefix=prefix)

        # Validate
        link_names = set(myself.link_map.keys())
        other_link_names = set(other.link_map.keys())
        if len(link_names.intersection(other_link_names)) > 0:
            raise ValueError("Cannot merge two URDFs with shared link names")

        joint_names = set(myself.joint_map.keys())
        other_joint_names = set(other.joint_map.keys())
        if len(joint_names.intersection(other_joint_names)) > 0:
            raise ValueError("Cannot merge two URDFs with shared joint names")

        links = myself.links + other.links
        joints = myself.joints + other.joints
        transmissions = myself.transmissions + other.transmissions
        materials = myself.materials + other.materials

        if name is None:
            name = self.name

        # Create joint that links the two rigidly
        joints.append(
            Joint(
                name="{}_join_{}{}_joint".format(self.name, prefix, other.name),
                joint_type="fixed",
                parent=link if isinstance(link, str) else link.name,
                child=other.base_link.name,
                origin=origin,
            )
        )

        return self.__class__(
            name=name,
            links=links,
            joints=joints,
            transmissions=transmissions,
            materials=materials,
        )

    def _merge_materials(self) -> None:
        """Merge the top-level material set with the link materials."""
        for link in self.links:
            for v in link.visuals:
                if v.material is None:
                    continue
                if v.material.name in self.material_map:
                    v.material = self._material_map[v.material.name]
                else:
                    self._materials.append(v.material)
                    self._material_map[v.material.name] = v.material

    @classmethod
    def load(
        cls, file_obj: Union[str, IO[bytes], IO[str]], lazy_load_meshes: bool = False
    ) -> "URDF":
        """Load a URDF from a file.

        Parameters
        ----------
        file_obj : str or file-like object
            The file to load the URDF from. Should be the path to the
            ``.urdf`` XML file. Any paths in the URDF should be specified
            as relative paths to the ``.urdf`` file instead of as ROS
            resources.
        lazy_load_meshes : bool
            If true, meshes will only loaded when requested by a function call.
            This dramatically speeds up loading time for the URDF but may lead
            to unexpected timing mid-program when the meshes have to be loaded

        Returns
        -------
        urdf : :class:`.URDF`
            The parsed URDF.
        """
        if isinstance(file_obj, str):
            if os.path.isfile(file_obj):
                parser = ET.XMLParser(remove_comments=True, remove_blank_text=True)
                tree = ET.parse(file_obj, parser=parser)
                path, _ = os.path.split(file_obj)
            else:
                raise ValueError("{} is not a file".format(file_obj))
        else:
            parser = ET.XMLParser(remove_comments=True, remove_blank_text=True)
            tree = ET.parse(file_obj, parser=parser)
            path, _ = os.path.split(file_obj.name)

        node = tree.getroot()
        return cls._from_xml(node, path, lazy_load_meshes)

    def _validate_joints(self):
        """Raise an exception of any joints are invalidly specified.

        Checks for the following:

        - Joint parents are valid link names.
        - Joint children are valid link names that aren't the same as parent.
        - Joint mimics have valid joint names that aren't the same joint.

        Returns
        -------
        actuated_joints : list of :class:`.Joint`
            The joints in the model that are independently controllable.
        """
        actuated_joints = []
        for joint in self.joints:
            if joint.parent not in self._link_map:
                raise ValueError(
                    "Joint {} has invalid parent link name {}".format(joint.name, joint.parent)
                )
            if joint.child not in self._link_map:
                raise ValueError(
                    "Joint {} has invalid child link name {}".format(joint.name, joint.child)
                )
            if joint.child == joint.parent:
                raise ValueError("Joint {} has matching parent and child".format(joint.name))
            if joint.mimic is not None:
                if joint.mimic.joint not in self._joint_map:
                    raise ValueError(
                        "Joint {} has an invalid mimic joint name {}".format(
                            joint.name, joint.mimic.joint
                        )
                    )
                if joint.mimic.joint == joint.name:
                    raise ValueError("Joint {} set up to mimic itself".format(joint.mimic.joint))
            elif joint.joint_type != "fixed":
                actuated_joints.append(joint)

        # Do a depth-first search
        return actuated_joints

    def _sort_joints(self, joints: list[Joint]) -> list[Joint]:
        """Sort joints by ascending distance from the base link (topologically).

        Parameters
        ----------
        joints : list of :class:`.Joint`
            The joints to sort.

        Returns
        -------
        joints : list of :class:`.Joint`
            The sorted joints.
        """
        lens = []
        for joint in joints:
            child_link = self._link_map[joint.child]
            lens.append(len(self._paths_to_base[child_link]))
        order = np.argsort(lens)
        return np.array(joints)[order].tolist()

    def _validate_transmissions(self) -> None:
        """Raise an exception of any transmissions are invalidly specified.

        Checks for the following:

        - Transmission joints have valid joint names.
        """
        for t in self.transmissions:
            for joint in t.joints:
                if joint.name not in self._joint_map:
                    raise ValueError(
                        "Transmission {} has invalid joint name {}".format(t.name, joint.name)
                    )

    def _validate_graph(self) -> tuple[Link, list[Link]]:
        """Raise an exception if the link-joint structure is invalid.

        Checks for the following:

        - The graph is connected in the undirected sense.
        - The graph is acyclic in the directed sense.
        - The graph has only one base link.

        Returns
        -------
        base_link : :class:`.Link`
            The base link of the URDF.
        end_links : list of :class:`.Link`
            The end links of the URDF.
        """

        # Check that the link graph is weakly connected
        if not nx.is_weakly_connected(self._G):
            link_clusters = []
            for cc in nx.weakly_connected_components(self._G):
                cluster = []
                for n in cc:
                    cluster.append(n.name)
                link_clusters.append(cluster)
            message = "Links are not all connected. Connected components are:"
            for lc in link_clusters:
                message += "\n\t"
                for n in lc:
                    message += " {}".format(n)
            raise ValueError(message)

        # Check that link graph is acyclic
        if not nx.is_directed_acyclic_graph(self._G):
            raise ValueError("There are cycles in the link graph")

        # Ensure that there is exactly one base link, which has no parent
        base_link: Optional[Link] = None
        end_links: list[Link] = []
        for n in self._G:
            if len(nx.descendants(self._G, n)) == 0:
                if base_link is None:
                    base_link = n
                else:
                    raise ValueError(
                        "Links {} and {} are both base links!".format(n.name, base_link.name)
                    )
            if len(nx.ancestors(self._G, n)) == 0:
                end_links.append(n)
        if base_link is None:
            raise ValueError("URDF has no base link")
        return base_link, end_links

    def _process_cfg(
        self,
        cfg: Union[
            Mapping[str, float],
            Sequence[float],
            npt.ArrayLike,
            None,
        ],
    ) -> dict[Joint, float]:
        """Process a joint configuration spec into a dictionary mapping
        joints to configuration values.
        """
        joint_cfg: dict[Joint, float] = {}
        if cfg is None:
            return joint_cfg
        if isinstance(cfg, dict):
            for joint in cfg:
                if isinstance(joint, str):
                    joint_cfg[self._joint_map[joint]] = cfg[joint]
                elif isinstance(joint, Joint):
                    joint_cfg[joint] = cfg[joint]
        elif isinstance(cfg, (list, tuple, np.ndarray)):
            if len(cfg) != len(self.actuated_joints):
                raise ValueError(
                    "Cfg must have same length as actuated joints if specified as a numerical array"
                )
            for joint, value in zip(self.actuated_joints, cfg):
                joint_cfg[joint] = value
        else:
            raise TypeError("Invalid type for config")
        return joint_cfg

    def _process_cfgs(
        self,
        cfgs: Union[
            Mapping[str, Sequence[float]],
            Sequence[Union[Mapping[str, float], None]],
            npt.ArrayLike,
            None,
        ],
    ) -> tuple[
        dict[Joint, Union[Sequence[float], npt.NDArray[np.float64], None]],
        Optional[int],
    ]:
        """Process a list of joint configurations into a dictionary mapping joints to
        configuration values.

        This should result in a dict mapping each joint to a list of cfg values, one
        per joint.
        """
        joint_cfg: dict[
            Joint,
            Union[list[float], npt.NDArray[np.float64], None],
        ] = {j: [] for j in self.actuated_joints}
        n_cfgs = None
        if isinstance(cfgs, dict):
            for joint in cfgs:
                if isinstance(joint, str):
                    joint_cfg[self._joint_map[joint]] = cfgs[joint]
                else:
                    joint_cfg[joint] = cfgs[joint]
                if n_cfgs is None:
                    n_cfgs = len(cfgs[joint])
        elif isinstance(cfgs, (list, tuple, np.ndarray)):
            n_cfgs = len(cfgs)
            if isinstance(cfgs[0], dict):
                for cfg in cfgs:
                    for joint in cfg:
                        if isinstance(joint, str):
                            v = joint_cfg[self._joint_map[joint]]
                            assert isinstance(v, list)
                            v.append(cfg[joint])
                            joint_cfg[self._joint_map[joint]] = v
                        else:
                            v2 = joint_cfg[joint]
                            assert isinstance(v2, list)
                            v2.append(cfg[joint])
                            joint_cfg[joint] = v2
            elif cfgs[0] is None:
                pass
            else:
                cfgs = np.asanyarray(cfgs, dtype=np.float64)
                for i, j in enumerate(self.actuated_joints):
                    joint_cfg[j] = cast(npt.NDArray[np.float64], cfgs[:, i])
        else:
            raise ValueError("Incorrectly formatted config array")

        for j in joint_cfg:
            if isinstance(joint_cfg[j], list):
                from typing import cast as _cast

                if len(_cast(list[float], joint_cfg[j])) == 0:
                    joint_cfg[j] = None
                elif n_cfgs is not None and len(_cast(list[float], joint_cfg[j])) != n_cfgs:
                    raise ValueError("Inconsistent number of configurations for joints")

        from typing import cast as _cast

        return _cast(
            dict[
                Joint,
                Union[Sequence[float], npt.NDArray[np.float64], None],
            ],
            joint_cfg,
        ), n_cfgs

    @classmethod
    def _from_xml(
        cls, node: ET._Element, path: str, lazy_load_meshes: Optional[bool] = None
    ) -> "URDF":
        # Explicit parse of URDF components for typing clarity
        name = str(node.attrib.get("name", ""))
        links = [Link._from_xml(n, path, lazy_load_meshes) for n in node.findall("link")]
        joints = [Joint._from_xml(n, path) for n in node.findall("joint")]
        transmissions = [Transmission._from_xml(n, path) for n in node.findall("transmission")]
        materials = [Material._from_xml(n, path) for n in node.findall("material")]

        # Capture any extra XML
        valid_tags = {"joint", "link", "transmission", "material"}
        extra_xml_node = ET.Element("extra")
        for child in node:
            if child.tag not in valid_tags:
                extra_xml_node.append(child)
        other_xml = ET.tostring(extra_xml_node)

        return cls(
            name=name,
            links=links,
            joints=joints,
            transmissions=transmissions,
            materials=materials,
            other_xml=other_xml,
        )

    def _to_xml(self, parent: Optional[ET._Element], path: str) -> ET._Element:
        node = self._unparse(path)
        if self.other_xml:
            extra_tree = ET.fromstring(self.other_xml)
            for child in extra_tree:
                node.append(child)
        return node
