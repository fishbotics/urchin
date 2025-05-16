import numpy as np
import trimesh
from lxml import etree as ET

from urchin.base import URDFType
from urchin.utils import configure_origin, parse_origin, unparse_origin


class SafetyController(URDFType):
    """A controller for joint movement safety.

    Parameters
    ----------
    k_velocity : float
        An attribute specifying the relation between the effort and velocity
        limits.
    k_position : float, optional
        An attribute specifying the relation between the position and velocity
        limits. Defaults to 0.0.
    soft_lower_limit : float, optional
        The lower joint boundary where the safety controller kicks in.
        Defaults to 0.0.
    soft_upper_limit : float, optional
        The upper joint boundary where the safety controller kicks in.
        Defaults to 0.0.
    """

    _ATTRIBS = {
        "k_velocity": (float, True),
        "k_position": (float, False),
        "soft_lower_limit": (float, False),
        "soft_upper_limit": (float, False),
    }
    _TAG = "safety_controller"

    def __init__(
        self, k_velocity, k_position=None, soft_lower_limit=None, soft_upper_limit=None
    ):
        self.k_velocity = k_velocity
        self.k_position = k_position
        self.soft_lower_limit = soft_lower_limit
        self.soft_upper_limit = soft_upper_limit

    @property
    def soft_lower_limit(self):
        """float : The soft lower limit where the safety controller kicks in."""
        return self._soft_lower_limit

    @soft_lower_limit.setter
    def soft_lower_limit(self, value):
        if value is not None:
            value = float(value)
        else:
            value = 0.0
        self._soft_lower_limit = value

    @property
    def soft_upper_limit(self):
        """float : The soft upper limit where the safety controller kicks in."""
        return self._soft_upper_limit

    @soft_upper_limit.setter
    def soft_upper_limit(self, value):
        if value is not None:
            value = float(value)
        else:
            value = 0.0
        self._soft_upper_limit = value

    @property
    def k_position(self):
        """float : A relation between the position and velocity limits."""
        return self._k_position

    @k_position.setter
    def k_position(self, value):
        if value is not None:
            value = float(value)
        else:
            value = 0.0
        self._k_position = value

    @property
    def k_velocity(self):
        """float : A relation between the effort and velocity limits."""
        return self._k_velocity

    @k_velocity.setter
    def k_velocity(self, value):
        self._k_velocity = float(value)

    def copy(self, prefix="", scale=None):
        """Create a deep copy of the visual with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all joint and link names.

        Returns
        -------
        :class:`.SafetyController`
            A deep copy of the visual.
        """
        return self.__class__(
            k_velocity=self.k_velocity,
            k_position=self.k_position,
            soft_lower_limit=self.soft_lower_limit,
            soft_upper_limit=self.soft_upper_limit,
        )


class JointCalibration(URDFType):
    """The reference positions of the joint.

    Parameters
    ----------
    rising : float, optional
        When the joint moves in a positive direction, this position will
        trigger a rising edge.
    falling :
        When the joint moves in a positive direction, this position will
        trigger a falling edge.
    """

    _ATTRIBS = {"rising": (float, False), "falling": (float, False)}
    _TAG = "calibration"

    def __init__(self, rising=None, falling=None):
        self.rising = rising
        self.falling = falling

    @property
    def rising(self):
        """float : description."""
        return self._rising

    @rising.setter
    def rising(self, value):
        if value is not None:
            value = float(value)
        self._rising = value

    @property
    def falling(self):
        """float : description."""
        return self._falling

    @falling.setter
    def falling(self, value):
        if value is not None:
            value = float(value)
        self._falling = value

    def copy(self, prefix="", scale=None):
        """Create a deep copy of the visual with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all joint and link names.

        Returns
        -------
        :class:`.JointCalibration`
            A deep copy of the visual.
        """
        return self.__class__(
            rising=self.rising,
            falling=self.falling,
        )


class JointDynamics(URDFType):
    """The dynamic properties of the joint.

    Parameters
    ----------
    damping : float
        The damping value of the joint (Ns/m for prismatic joints,
        Nms/rad for revolute).
    friction : float
        The static friction value of the joint (N for prismatic joints,
        Nm for revolute).
    """

    _ATTRIBS = {
        "damping": (float, False),
        "friction": (float, False),
    }
    _TAG = "dynamics"

    def __init__(self, damping, friction):
        self.damping = damping
        self.friction = friction

    @property
    def damping(self):
        """float : The damping value of the joint."""
        return self._damping

    @damping.setter
    def damping(self, value):
        if value is not None:
            value = float(value)
        self._damping = value

    @property
    def friction(self):
        """float : The static friction value of the joint."""
        return self._friction

    @friction.setter
    def friction(self, value):
        if value is not None:
            value = float(value)
        self._friction = value

    def copy(self, prefix="", scale=None):
        """Create a deep copy of the visual with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all joint and link names.

        Returns
        -------
        :class:`.JointDynamics`
            A deep copy of the visual.
        """
        return self.__class__(
            damping=self.damping,
            friction=self.friction,
        )


class JointLimit(URDFType):
    """The limits of the joint.

    Parameters
    ----------
    effort : float
        The maximum joint effort (N for prismatic joints, Nm for revolute).
    velocity : float
        The maximum joint velocity (m/s for prismatic joints, rad/s for
        revolute).
    lower : float, optional
        The lower joint limit (m for prismatic joints, rad for revolute).
    upper : float, optional
        The upper joint limit (m for prismatic joints, rad for revolute).
    """

    _ATTRIBS = {
        "effort": (float, True),
        "velocity": (float, True),
        "lower": (float, False),
        "upper": (float, False),
    }
    _TAG = "limit"

    def __init__(self, effort, velocity, lower=None, upper=None):
        self.effort = effort
        self.velocity = velocity
        self.lower = lower
        self.upper = upper

    @property
    def effort(self):
        """float : The maximum joint effort."""
        return self._effort

    @effort.setter
    def effort(self, value):
        self._effort = float(value)

    @property
    def velocity(self):
        """float : The maximum joint velocity."""
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        self._velocity = float(value)

    @property
    def lower(self):
        """float : The lower joint limit."""
        return self._lower

    @lower.setter
    def lower(self, value):
        if value is not None:
            value = float(value)
        self._lower = value

    @property
    def upper(self):
        """float : The upper joint limit."""
        return self._upper

    @upper.setter
    def upper(self, value):
        if value is not None:
            value = float(value)
        self._upper = value

    def copy(self, prefix="", scale=None):
        """Create a deep copy of the visual with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all joint and link names.

        Returns
        -------
        :class:`.JointLimit`
            A deep copy of the visual.
        """
        return self.__class__(
            effort=self.effort,
            velocity=self.velocity,
            lower=self.lower,
            upper=self.upper,
        )


class JointMimic(URDFType):
    """A mimicry tag for a joint, which forces its configuration to
    mimic another joint's.

    This joint's configuration value is set equal to
    ``multiplier * other_joint_cfg + offset``.

    Parameters
    ----------
    joint : str
        The name of the joint to mimic.
    multiplier : float
        The joint configuration multiplier. Defaults to 1.0.
    offset : float, optional
        The joint configuration offset. Defaults to 0.0.
    """

    _ATTRIBS = {
        "joint": (str, True),
        "multiplier": (float, False),
        "offset": (float, False),
    }
    _TAG = "mimic"

    def __init__(self, joint, multiplier=None, offset=None):
        self.joint = joint
        self.multiplier = multiplier
        self.offset = offset

    @property
    def joint(self):
        """float : The name of the joint to mimic."""
        return self._joint

    @joint.setter
    def joint(self, value):
        self._joint = str(value)

    @property
    def multiplier(self):
        """float : The multiplier for the joint configuration."""
        return self._multiplier

    @multiplier.setter
    def multiplier(self, value):
        if value is not None:
            value = float(value)
        else:
            value = 1.0
        self._multiplier = value

    @property
    def offset(self):
        """float : The offset for the joint configuration"""
        return self._offset

    @offset.setter
    def offset(self, value):
        if value is not None:
            value = float(value)
        else:
            value = 0.0
        self._offset = value

    def copy(self, prefix="", scale=None):
        """Create a deep copy of the joint mimic with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all joint and link names.

        Returns
        -------
        :class:`.JointMimic`
            A deep copy of the joint mimic.
        """
        return self.__class__(
            joint="{}{}".format(prefix, self.joint),
            multiplier=self.multiplier,
            offset=self.offset,
        )


class Joint(URDFType):
    """A connection between two links.

    There are several types of joints, including:

    - ``fixed`` - a joint that cannot move.
    - ``prismatic`` - a joint that slides along the joint axis.
    - ``revolute`` - a hinge joint that rotates about the axis with a limited
      range of motion.
    - ``continuous`` - a hinge joint that rotates about the axis with an
      unlimited range of motion.
    - ``planar`` - a joint that moves in the plane orthogonal to the axis.
    - ``floating`` - a joint that can move in 6DoF.

    Parameters
    ----------
    name : str
        The name of this joint.
    parent : str
        The name of the parent link of this joint.
    child : str
        The name of the child link of this joint.
    joint_type : str
        The type of the joint. Must be one of :obj:`.Joint.TYPES`.
    axis : (3,) float, optional
        The axis of the joint specified in joint frame. Defaults to
        ``[1,0,0]``.
    origin : (4,4) float, optional
        The pose of the child link with respect to the parent link's frame.
        The joint frame is defined to be coincident with the child link's
        frame, so this is also the pose of the joint frame with respect to
        the parent link's frame.
    limit : :class:`.JointLimit`, optional
        Limit for the joint. Only required for revolute and prismatic
        joints.
    dynamics : :class:`.JointDynamics`, optional
        Dynamics for the joint.
    safety_controller : :class`.SafetyController`, optional
        The safety controller for this joint.
    calibration : :class:`.JointCalibration`, optional
        Calibration information for the joint.
    mimic : :class:`JointMimic`, optional
        Joint mimicry information.
    """

    TYPES = ["fixed", "prismatic", "revolute", "continuous", "floating", "planar"]
    _ATTRIBS = {
        "name": (str, True),
    }
    _ELEMENTS = {
        "dynamics": (JointDynamics, False, False),
        "limit": (JointLimit, False, False),
        "mimic": (JointMimic, False, False),
        "safety_controller": (SafetyController, False, False),
        "calibration": (JointCalibration, False, False),
    }
    _TAG = "joint"

    def __init__(
        self,
        name,
        joint_type,
        parent,
        child,
        axis=None,
        origin=None,
        limit=None,
        dynamics=None,
        safety_controller=None,
        calibration=None,
        mimic=None,
    ):
        self.name = name
        self.parent = parent
        self.child = child
        self.joint_type = joint_type
        self.axis = axis
        self.origin = origin
        self.limit = limit
        self.dynamics = dynamics
        self.safety_controller = safety_controller
        self.calibration = calibration
        self.mimic = mimic

    @property
    def name(self):
        """str : Name for this joint."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def joint_type(self):
        """str : The type of this joint."""
        return self._joint_type

    @joint_type.setter
    def joint_type(self, value):
        value = str(value)
        if value not in Joint.TYPES:
            raise ValueError("Unsupported joint type {}".format(value))
        self._joint_type = value

    @property
    def parent(self):
        """str : The name of the parent link."""
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = str(value)

    @property
    def child(self):
        """str : The name of the child link."""
        return self._child

    @child.setter
    def child(self, value):
        self._child = str(value)

    @property
    def axis(self):
        """(3,) float : The joint axis in the joint frame."""
        return self._axis

    @axis.setter
    def axis(self, value):
        if value is None:
            value = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        elif np.linalg.norm(value) < 1e-4:
            value = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            value = np.asanyarray(value, dtype=np.float64)
            if value.shape != (3,):
                raise ValueError("Invalid shape for axis, should be (3,)")
            value = value / np.linalg.norm(value)
        self._axis = value

    @property
    def origin(self):
        """(4,4) float : The pose of child and joint frames relative to the
        parent link's frame.
        """
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = configure_origin(value)

    @property
    def limit(self):
        """:class:`.JointLimit` : The limits for this joint."""
        return self._limit

    @limit.setter
    def limit(self, value):
        if value is None:
            if self.joint_type in ["prismatic", "revolute"]:
                raise ValueError(
                    "Require joint limit for prismatic and " "revolute joints"
                )
        elif not isinstance(value, JointLimit):
            raise TypeError("Expected JointLimit type")
        self._limit = value

    @property
    def dynamics(self):
        """:class:`.JointDynamics` : The dynamics for this joint."""
        return self._dynamics

    @dynamics.setter
    def dynamics(self, value):
        if value is not None:
            if not isinstance(value, JointDynamics):
                raise TypeError("Expected JointDynamics type")
        self._dynamics = value

    @property
    def safety_controller(self):
        """:class:`.SafetyController` : The safety controller for this joint."""
        return self._safety_controller

    @safety_controller.setter
    def safety_controller(self, value):
        if value is not None:
            if not isinstance(value, SafetyController):
                raise TypeError("Expected SafetyController type")
        self._safety_controller = value

    @property
    def calibration(self):
        """:class:`.JointCalibration` : The calibration for this joint."""
        return self._calibration

    @calibration.setter
    def calibration(self, value):
        if value is not None:
            if not isinstance(value, JointCalibration):
                raise TypeError("Expected JointCalibration type")
        self._calibration = value

    @property
    def mimic(self):
        """:class:`.JointMimic` : The mimic for this joint."""
        return self._mimic

    @mimic.setter
    def mimic(self, value):
        if value is not None:
            if not isinstance(value, JointMimic):
                raise TypeError("Expected JointMimic type")
        self._mimic = value

    def is_valid(self, cfg):
        """Check if the provided configuration value is valid for this joint.

        Parameters
        ----------
        cfg : float, (2,) float, (6,) float, or (4,4) float
            The configuration of the joint.

        Returns
        -------
        is_valid : bool
            True if the configuration is valid, and False otherwise.
        """
        if self.joint_type not in ["fixed", "revolute"]:
            return True
        if self.joint_limit is None:
            return True
        cfg = float(cfg)
        lower = -np.infty
        upper = np.infty
        if self.limit.lower is not None:
            lower = self.limit.lower
        if self.limit.upper is not None:
            upper = self.limit.upper
        return cfg >= lower and cfg <= upper

    def get_child_pose(self, cfg=None):
        """Computes the child pose relative to a parent pose for a given
        configuration value.

        Parameters
        ----------
        cfg : float, (2,) float, (6,) float, or (4,4) float
            The configuration values for this joint. They are interpreted
            based on the joint type as follows:

            - ``fixed`` - not used.
            - ``prismatic`` - a translation along the axis in meters.
            - ``revolute`` - a rotation about the axis in radians.
            - ``continuous`` - a rotation about the axis in radians.
            - ``planar`` - the x and y translation values in the plane.
            - ``floating`` - the xyz values followed by the rpy values,
              or a (4,4) matrix.

            If ``cfg`` is ``None``, then this just returns the joint pose.

        Returns
        -------
        pose : (4,4) float
            The pose of the child relative to the parent.
        """
        if cfg is None:
            return self.origin
        elif self.joint_type == "fixed":
            return self.origin
        elif self.joint_type in ["revolute", "continuous"]:
            if cfg is None:
                cfg = 0.0
            else:
                cfg = float(cfg)
            R = trimesh.transformations.rotation_matrix(cfg, self.axis)
            return self.origin.dot(R)
        elif self.joint_type == "prismatic":
            if cfg is None:
                cfg = 0.0
            else:
                cfg = float(cfg)
            translation = np.eye(4, dtype=np.float64)
            translation[:3, 3] = self.axis * cfg
            return self.origin.dot(translation)
        elif self.joint_type == "planar":
            if cfg is None:
                cfg = np.zeros(2, dtype=np.float64)
            else:
                cfg = np.asanyarray(cfg, dtype=np.float64)
            if cfg.shape != (2,):
                raise ValueError("(2,) float configuration required for planar joints")
            translation = np.eye(4, dtype=np.float64)
            translation[:3, 3] = self.origin[:3, :2].dot(cfg)
            return self.origin.dot(translation)
        elif self.joint_type == "floating":
            if cfg is None:
                cfg = np.zeros(6, dtype=np.float64)
            else:
                cfg = configure_origin(cfg)
            if cfg is None:
                raise ValueError("Invalid configuration for floating joint")
            return self.origin.dot(cfg)
        else:
            raise ValueError("Invalid configuration")

    def get_child_poses(self, cfg, n_cfgs):
        """Computes the child pose relative to a parent pose for a given set of
        configuration values.

        Parameters
        ----------
        cfg : (n,) float or None
            The configuration values for this joint. They are interpreted
            based on the joint type as follows:

            - ``fixed`` - not used.
            - ``prismatic`` - a translation along the axis in meters.
            - ``revolute`` - a rotation about the axis in radians.
            - ``continuous`` - a rotation about the axis in radians.
            - ``planar`` - Not implemented.
            - ``floating`` - Not implemented.

            If ``cfg`` is ``None``, then this just returns the joint pose.

        Returns
        -------
        poses : (n,4,4) float
            The poses of the child relative to the parent.
        """
        if cfg is None:
            return np.tile(self.origin, (n_cfgs, 1, 1))
        elif self.joint_type == "fixed":
            return np.tile(self.origin, (n_cfgs, 1, 1))
        elif self.joint_type in ["revolute", "continuous"]:
            if cfg is None:
                cfg = np.zeros(n_cfgs)
            return np.matmul(self.origin, self._rotation_matrices(cfg, self.axis))
        elif self.joint_type == "prismatic":
            if cfg is None:
                cfg = np.zeros(n_cfgs)
            translation = np.tile(np.eye(4), (n_cfgs, 1, 1))
            translation[:, :3, 3] = self.axis * cfg[:, np.newaxis]
            return np.matmul(self.origin, translation)
        elif self.joint_type == "planar":
            raise NotImplementedError()
        elif self.joint_type == "floating":
            raise NotImplementedError()
        else:
            raise ValueError("Invalid configuration")

    @classmethod
    def _from_xml(cls, node, path):
        kwargs = cls._parse(node, path)
        kwargs["joint_type"] = str(node.attrib["type"])
        kwargs["parent"] = node.find("parent").attrib["link"]
        kwargs["child"] = node.find("child").attrib["link"]
        axis = node.find("axis")
        if axis is not None:
            axis = np.fromstring(axis.attrib["xyz"], sep=" ")
        kwargs["axis"] = axis
        kwargs["origin"] = parse_origin(node)
        return cls(**kwargs)

    def _to_xml(self, parent, path):
        node = self._unparse(path)
        parent = ET.Element("parent")
        parent.attrib["link"] = self.parent
        node.append(parent)
        child = ET.Element("child")
        child.attrib["link"] = self.child
        node.append(child)
        if self.axis is not None:
            axis = ET.Element("axis")
            axis.attrib["xyz"] = np.array2string(self.axis)[1:-1]
            node.append(axis)
        node.append(unparse_origin(self.origin))
        node.attrib["type"] = self.joint_type
        return node

    def _rotation_matrices(self, angles, axis):
        """Compute rotation matrices from angle/axis representations.

        Parameters
        ----------
        angles : (n,) float
            The angles.
        axis : (3,) float
            The axis.

        Returns
        -------
        rots : (n,4,4)
            The rotation matrices
        """
        axis = axis / np.linalg.norm(axis)
        sina = np.sin(angles)
        cosa = np.cos(angles)
        M = np.tile(np.eye(4), (len(angles), 1, 1))
        M[:, 0, 0] = cosa
        M[:, 1, 1] = cosa
        M[:, 2, 2] = cosa
        M[:, :3, :3] += (
            np.tile(np.outer(axis, axis), (len(angles), 1, 1))
            * (1.0 - cosa)[:, np.newaxis, np.newaxis]
        )
        M[:, :3, :3] += (
            np.tile(
                np.array(
                    [
                        [0.0, -axis[2], axis[1]],
                        [axis[2], 0.0, -axis[0]],
                        [-axis[1], axis[0], 0.0],
                    ]
                ),
                (len(angles), 1, 1),
            )
            * sina[:, np.newaxis, np.newaxis]
        )
        return M

    def copy(self, prefix="", scale=None):
        """Create a deep copy of the joint with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all joint and link names.

        Returns
        -------
        :class:`.Joint`
            A deep copy of the joint.
        """
        origin = self.origin.copy()
        if scale is not None:
            if not isinstance(scale, (list, np.ndarray)):
                scale = np.repeat(scale, 3)
            origin[:3, 3] *= scale
        cpy = self.__class__(
            name="{}{}".format(prefix, self.name),
            joint_type=self.joint_type,
            parent="{}{}".format(prefix, self.parent),
            child="{}{}".format(prefix, self.child),
            axis=self.axis.copy(),
            origin=origin,
            limit=(self.limit.copy(prefix, scale) if self.limit else None),
            dynamics=(self.dynamics.copy(prefix, scale) if self.dynamics else None),
            safety_controller=(
                self.safety_controller.copy(prefix, scale)
                if self.safety_controller
                else None
            ),
            calibration=(
                self.calibration.copy(prefix, scale) if self.calibration else None
            ),
            mimic=(self.mimic.copy(prefix=prefix, scale=scale) if self.mimic else None),
        )
        return cpy
