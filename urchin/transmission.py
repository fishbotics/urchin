from __future__ import annotations

from typing import Optional, Sequence, Union

from lxml import etree as ET

from urchin.base import URDFType


class Actuator(URDFType):
    """An actuator.

    Parameters
    ----------
    name : str
        The name of this actuator.
    mechanicalReduction : float, optional
        Mechanical reduction (ratio) at the joint/actuator transmission.
    hardwareInterfaces : list of str, optional
        The supported hardware interfaces to the actuator.
    """

    _ATTRIBS = {
        "name": (str, True),
    }
    _TAG = "actuator"

    def __init__(
        self,
        name: str,
        mechanicalReduction: Optional[float] = None,
        hardwareInterfaces: Optional[Sequence[str]] = None,
    ):
        self.name = name
        self.mechanicalReduction = mechanicalReduction
        self.hardwareInterfaces = hardwareInterfaces

    @property
    def name(self) -> str:
        """str : The name of this actuator."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)

    @property
    def mechanicalReduction(self) -> Optional[float]:
        """float | None : Mechanical reduction (ratio)."""
        return self._mechanicalReduction

    @mechanicalReduction.setter
    def mechanicalReduction(self, value: Union[float, str, None]) -> None:
        if value is not None:
            value = float(value)
        self._mechanicalReduction = value

    @property
    def hardwareInterfaces(self) -> list[str]:
        """list of str : The supported hardware interfaces."""
        return self._hardwareInterfaces

    @hardwareInterfaces.setter
    def hardwareInterfaces(self, value: Optional[Sequence[str]]) -> None:
        if value is None:
            value = []
        else:
            value = list(value)
            for i, v in enumerate(value):
                value[i] = str(v)
        self._hardwareInterfaces = value

    @classmethod
    def _from_xml(cls, node: ET._Element, path: str, lazy_load_meshes: Optional[bool] = None):
        name = str(node.attrib["name"]) if "name" in node.attrib else ""
        mr_node = node.find("mechanicalReduction")
        mr_val = float(mr_node.text) if mr_node is not None and mr_node.text else None
        hi_nodes = node.findall("hardwareInterface")
        hi_list = [str(h.text) for h in hi_nodes if h is not None and h.text]
        return cls(name=name, mechanicalReduction=mr_val, hardwareInterfaces=hi_list)

    def _to_xml(self, parent: Optional[ET._Element], path: str) -> ET._Element:
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

    def copy(self, prefix: str = "", scale: Optional[float] = None) -> "Actuator":
        """Create a deep copy with the prefix applied to all names.

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

    def __init__(self, name: str, hardwareInterfaces: Optional[Sequence[str]]):
        self.name = name
        self.hardwareInterfaces = hardwareInterfaces

    @property
    def name(self) -> str:
        """str : The name of this transmission joint."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)

    @property
    def hardwareInterfaces(self) -> list[str]:
        """list of str : The supported hardware interfaces."""
        return self._hardwareInterfaces

    @hardwareInterfaces.setter
    def hardwareInterfaces(self, value: Optional[Sequence[str]]) -> None:
        if value is None:
            value = []
        else:
            value = list(value)
            for i, v in enumerate(value):
                value[i] = str(v)
        self._hardwareInterfaces = value

    @classmethod
    def _from_xml(cls, node: ET._Element, path: str, lazy_load_meshes: Optional[bool] = None):
        name = str(node.attrib["name"]) if "name" in node.attrib else ""
        hi_nodes = node.findall("hardwareInterface")
        hi_list = [str(h.text) for h in hi_nodes if h is not None and h.text]
        return cls(name=name, hardwareInterfaces=hi_list)

    def _to_xml(self, parent: Optional[ET._Element], path: str) -> ET._Element:
        node = self._unparse(path)
        if len(self.hardwareInterfaces) > 0:
            for hi in self.hardwareInterfaces:
                h = ET.Element("hardwareInterface")
                h.text = hi
                node.append(h)
        return node

    def copy(self, prefix: str = "", scale: Optional[float] = None) -> "TransmissionJoint":
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

    def __init__(
        self,
        name: str,
        trans_type: str,
        joints: Optional[Sequence["TransmissionJoint"]] = None,
        actuators: Optional[Sequence[Actuator]] = None,
    ):
        self.name = name
        self.trans_type = trans_type
        self.joints = joints
        self.actuators = actuators

    @property
    def name(self) -> str:
        """str : The name of this transmission."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)

    @property
    def trans_type(self) -> str:
        """str : The type of this transmission."""
        return self._trans_type

    @trans_type.setter
    def trans_type(self, value: str) -> None:
        self._trans_type = str(value)

    @property
    def joints(self) -> list["TransmissionJoint"]:
        """:class:`.TransmissionJoint` : The joints the transmission is
        connected to.
        """
        return self._joints

    @joints.setter
    def joints(self, value: Optional[Sequence["TransmissionJoint"]]) -> None:
        if value is None:
            value = []
        else:
            value = list(value)
            for v in value:
                if not isinstance(v, TransmissionJoint):
                    raise TypeError("Joints expects a list of TransmissionJoint")
        self._joints = value

    @property
    def actuators(self) -> list[Actuator]:
        """:class:`.Actuator` : The actuators the transmission is connected to."""
        return self._actuators

    @actuators.setter
    def actuators(self, value: Optional[Sequence[Actuator]]) -> None:
        if value is None:
            value = []
        else:
            value = list(value)
            for v in value:
                if not isinstance(v, Actuator):
                    raise TypeError("Actuators expects a list of Actuator")
        self._actuators = value

    @classmethod
    def _from_xml(cls, node: ET._Element, path: str, lazy_load_meshes: Optional[bool] = None):
        name = str(node.attrib["name"]) if "name" in node.attrib else ""
        ttype = node.attrib.get("type")
        if ttype is None:
            t_node = node.find("type")
            ttype = t_node.text if t_node is not None else ""
        joints = [TransmissionJoint._from_xml(n, path) for n in node.findall("joint")]
        actuators = [Actuator._from_xml(n, path) for n in node.findall("actuator")]
        return cls(name=name, trans_type=str(ttype), joints=joints, actuators=actuators)

    def _to_xml(self, parent: Optional[ET._Element], path: str) -> ET._Element:
        node = self._unparse(path)
        ttype = ET.Element("type")
        ttype.text = self.trans_type
        node.append(ttype)
        return node

    def copy(
        self, prefix: str = "", scale: Union[float, Sequence[float], None] = None
    ) -> "Transmission":
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
