from lxml import etree as ET

from urchin.base import URDFType


class Actuator(URDFType):
    """An actuator.

    Parameters
    ----------
    name : str
        The name of this actuator.
    mechanicalReduction : str, optional
        A specifier for the mechanical reduction at the joint/actuator
        transmission.
    hardwareInterfaces : list of str, optional
        The supported hardware interfaces to the actuator.
    """

    _ATTRIBS = {
        "name": (str, True),
    }
    _TAG = "actuator"

    def __init__(self, name, mechanicalReduction=None, hardwareInterfaces=None):
        self.name = name
        self.mechanicalReduction = mechanicalReduction
        self.hardwareInterfaces = hardwareInterfaces

    @property
    def name(self):
        """str : The name of this actuator."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def mechanicalReduction(self):
        """str : A specifier for the type of mechanical reduction."""
        return self._mechanicalReduction

    @mechanicalReduction.setter
    def mechanicalReduction(self, value):
        if value is not None:
            value = str(value)
        self._mechanicalReduction = value

    @property
    def hardwareInterfaces(self):
        """list of str : The supported hardware interfaces."""
        return self._hardwareInterfaces

    @hardwareInterfaces.setter
    def hardwareInterfaces(self, value):
        if value is None:
            value = []
        else:
            value = list(value)
            for i, v in enumerate(value):
                value[i] = str(v)
        self._hardwareInterfaces = value

    @classmethod
    def _from_xml(cls, node, path):
        kwargs = cls._parse(node, path)
        mr = node.find("mechanicalReduction")
        if mr is not None:
            mr = float(mr.text)
        kwargs["mechanicalReduction"] = mr
        hi = node.findall("hardwareInterface")
        if len(hi) > 0:
            hi = [h.text for h in hi]
        kwargs["hardwareInterfaces"] = hi
        return cls(**kwargs)

    def _to_xml(self, parent, path):
        node = self._unparse(path)
        if self.mechanicalReduction is not None:
            mr = ET.Element("mechanicalReduction")
            mr.text = str(self.mechanicalReduction)
            node.append(mr)
        if len(self.hardwareInterfaces) > 0:
            for hi in self.hardwareInterfaces:
                h = ET.Element("hardwareInterface")
                h.text = hi
                node.append(h)
        return node

    def copy(self, prefix="", scale=None):
        """Create a deep copy of the visual with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all joint and link names.

        Returns
        -------
        :class:`.Actuator`
            A deep copy of the visual.
        """
        return self.__class__(
            name="{}{}".format(prefix, self.name),
            mechanicalReduction=self.mechanicalReduction,
            hardwareInterfaces=self.hardwareInterfaces.copy(),
        )


class TransmissionJoint(URDFType):
    """A transmission joint specification.

    Parameters
    ----------
    name : str
        The name of this actuator.
    hardwareInterfaces : list of str, optional
        The supported hardware interfaces to the actuator.
    """

    _ATTRIBS = {
        "name": (str, True),
    }
    _TAG = "joint"

    def __init__(self, name, hardwareInterfaces):
        self.name = name
        self.hardwareInterfaces = hardwareInterfaces

    @property
    def name(self):
        """str : The name of this transmission joint."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def hardwareInterfaces(self):
        """list of str : The supported hardware interfaces."""
        return self._hardwareInterfaces

    @hardwareInterfaces.setter
    def hardwareInterfaces(self, value):
        if value is None:
            value = []
        else:
            value = list(value)
            for i, v in enumerate(value):
                value[i] = str(v)
        self._hardwareInterfaces = value

    @classmethod
    def _from_xml(cls, node, path):
        kwargs = cls._parse(node, path)
        hi = node.findall("hardwareInterface")
        if len(hi) > 0:
            hi = [h.text for h in hi]
        kwargs["hardwareInterfaces"] = hi
        return cls(**kwargs)

    def _to_xml(self, parent, path):
        node = self._unparse(path)
        if len(self.hardwareInterfaces) > 0:
            for hi in self.hardwareInterfaces:
                h = ET.Element("hardwareInterface")
                h.text = hi
                node.append(h)
        return node

    def copy(self, prefix="", scale=None):
        """Create a deep copy with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all names.

        Returns
        -------
        :class:`.TransmissionJoint`
            A deep copy.
        """
        return self.__class__(
            name="{}{}".format(prefix, self.name),
            hardwareInterfaces=self.hardwareInterfaces.copy(),
        )


###############################################################################
# Top-level types
###############################################################################


class Transmission(URDFType):
    """An element that describes the relationship between an actuator and a
    joint.

    Parameters
    ----------
    name : str
        The name of this transmission.
    trans_type : str
        The type of this transmission.
    joints : list of :class:`.TransmissionJoint`
        The joints connected to this transmission.
    actuators : list of :class:`.Actuator`
        The actuators connected to this transmission.
    """

    _ATTRIBS = {
        "name": (str, True),
    }
    _ELEMENTS = {
        "joints": (TransmissionJoint, True, True),
        "actuators": (Actuator, True, True),
    }
    _TAG = "transmission"

    def __init__(self, name, trans_type, joints=None, actuators=None):
        self.name = name
        self.trans_type = trans_type
        self.joints = joints
        self.actuators = actuators

    @property
    def name(self):
        """str : The name of this transmission."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def trans_type(self):
        """str : The type of this transmission."""
        return self._trans_type

    @trans_type.setter
    def trans_type(self, value):
        self._trans_type = str(value)

    @property
    def joints(self):
        """:class:`.TransmissionJoint` : The joints the transmission is
        connected to.
        """
        return self._joints

    @joints.setter
    def joints(self, value):
        if value is None:
            value = []
        else:
            value = list(value)
            for v in value:
                if not isinstance(v, TransmissionJoint):
                    raise TypeError("Joints expects a list of TransmissionJoint")
        self._joints = value

    @property
    def actuators(self):
        """:class:`.Actuator` : The actuators the transmission is connected to."""
        return self._actuators

    @actuators.setter
    def actuators(self, value):
        if value is None:
            value = []
        else:
            value = list(value)
            for v in value:
                if not isinstance(v, Actuator):
                    raise TypeError("Actuators expects a list of Actuator")
        self._actuators = value

    @classmethod
    def _from_xml(cls, node, path):
        kwargs = cls._parse(node, path)
        trans_type = node.attrib.get("type")
        if trans_type is None:
            trans_type = node.find("type").text
        kwargs["trans_type"] = trans_type
        return cls(**kwargs)

    def _to_xml(self, parent, path):
        node = self._unparse(path)
        ttype = ET.Element("type")
        ttype.text = self.trans_type
        node.append(ttype)
        return node

    def copy(self, prefix="", scale=None):
        """Create a deep copy with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all names.

        Returns
        -------
        :class:`.Transmission`
            A deep copy.
        """
        return self.__class__(
            name="{}{}".format(prefix, self.name),
            trans_type=self.trans_type,
            joints=[j.copy(prefix) for j in self.joints],
            actuators=[a.copy(prefix) for a in self.actuators],
        )
