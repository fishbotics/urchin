from __future__ import annotations

from typing import Optional, Type, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
from lxml import etree as ET

ParsedAttribute = Union[bool, float, int, str, npt.NDArray[np.float64], None]
ParsedElement = Union["URDFType", list["URDFType"], None]
ParsedValue = Union[ParsedAttribute, ParsedElement]
ParsedAttributeDict = dict[str, ParsedAttribute]
ParsedElementDict = dict[str, ParsedElement]
ParsedValueDict = dict[str, ParsedValue]
T = TypeVar("T", bound="URDFType")


class URDFType:
    """Abstract base class for all URDF types.

    This has useful class methods for automatic parsing/unparsing
    of XML trees.

    There are three overridable class variables:

    - ``_ATTRIBS`` - This is a dictionary mapping attribute names to a tuple,
      ``(type, required)`` where ``type`` is the Python type for the
      attribute and ``required`` is a boolean stating whether the attribute
      is required to be present.
    - ``_ELEMENTS`` - This is a dictionary mapping element names to a tuple,
      ``(type, required, multiple)`` where ``type`` is the Python type for the
      element, ``required`` is a boolean stating whether the element
      is required to be present, and ``multiple`` is a boolean indicating
      whether multiple elements of this type could be present.
      Elements are child nodes in the XML tree, and their type must be a
      subclass of :class:`.URDFType`.
    - ``_TAG`` - This is a string that represents the XML tag for the node
      containing this type of object.
    """

    _ATTRIBS: dict[str, tuple[type, bool]] = {}  # Map from attrib name to (type, required)
    _ELEMENTS: dict[
        str, tuple[Type["URDFType"], bool, bool]
    ] = {}  # Map from element name to (type, required, multiple)
    _TAG: str = ""  # XML tag for this element

    def __init__(self):
        pass

    @classmethod
    def _parse_attrib(cls, val_type: type, val: str) -> ParsedAttribute:
        """Parse an XML attribute into a python value.

        Parameters
        ----------
        val_type : :class:`type`
            The type of value to create.
        val : str
            The string value to parse.

        Returns
        -------
        val : ParsedAttribute
            The parsed attribute value.
        """
        if val_type == np.ndarray:
            array_value = cast(npt.NDArray[np.float64], np.fromstring(val, sep=" "))
            return array_value
        return cast(ParsedAttribute, val_type(val))

    @classmethod
    def _parse_simple_attribs(cls, node: ET._Element) -> ParsedAttributeDict:
        """Parse all attributes in the _ATTRIBS array for this class.

        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse attributes for.

        Returns
        -------
        kwargs : ParsedAttributeDict
            Map from attribute name to value. If the attribute is not
            required and is not present, that attribute's name will map to
            ``None``.
        """
        kwargs: ParsedAttributeDict = {}
        for attrib_name, (val_type, required) in cls._ATTRIBS.items():
            if required:
                try:
                    value = cls._parse_attrib(val_type, node.attrib[attrib_name])
                except Exception:
                    raise ValueError(
                        "Missing required attribute {} when parsing an object of type {}".format(
                            attrib_name, cls.__name__
                        )
                    )
            else:
                value = None
                if attrib_name in node.attrib:
                    value = cls._parse_attrib(val_type, node.attrib[attrib_name])
            kwargs[attrib_name] = value
        return kwargs

    @classmethod
    def _parse_simple_elements(
        cls, node: ET._Element, path: str, lazy_load_meshes: Optional[bool] = None
    ) -> ParsedElementDict:
        """Parse all elements in the _ELEMENTS array from the children of
        this node.

        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse children for.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        lazy_load_meshes : bool
            Whether a mesh element should be immediately loaded or loaded when
            needed

        Returns
        -------
        kwargs : ParsedElementDict
            Map from element names to the :class:`URDFType` subclass (or list,
            if ``multiple`` was set) created for that element.
        """
        kwargs: ParsedElementDict = {}
        for element_name, (element_type, required, multiple) in cls._ELEMENTS.items():
            value: ParsedElement
            if not multiple:
                element_node = node.find(element_type._TAG)
                if required or element_node is not None:
                    value = cast(
                        ParsedElement,
                        element_type._from_xml(element_node, path),
                    )
                else:
                    value = None
            else:
                element_nodes = node.findall(element_type._TAG)
                if len(element_nodes) == 0 and required:
                    print(
                        f"Missing required subelement(s) of type {element_type.__name__} when "
                        f"parsing an object of type {cls.__name__}."
                    )
                value = [element_type._from_xml(child, path) for child in element_nodes]
            kwargs[element_name] = value
        return kwargs

    @classmethod
    def _parse(
        cls, node: ET._Element, path: str, lazy_load_meshes: Optional[bool] = None
    ) -> ParsedValueDict:
        """Parse all elements and attributes in the _ELEMENTS and _ATTRIBS
        arrays for a node.

        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).

        Returns
        -------
        kwargs : ParsedValueDict
            Map from names to Python values created from the attributes
            and elements in the class arrays.
        """
        kwargs: ParsedValueDict = {}
        kwargs.update(cls._parse_simple_attribs(node))
        kwargs.update(cls._parse_simple_elements(node, path, lazy_load_meshes))
        return kwargs

    @classmethod
    def _from_xml(
        cls: type[T],
        node: ET._Element,
        path: str,
        lazy_load_meshes: Optional[bool] = None,
    ) -> T:
        """Create an instance of this class from an XML node.

        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).

        Returns
        -------
        obj : :class:`URDFType`
            An instance of this class parsed from the node.
        """
        return cls(**cls._parse(node, path, lazy_load_meshes))

    def _unparse_attrib(self, val_type: type, val: ParsedAttribute) -> str:
        """Convert a Python value into a string for storage in an
        XML attribute.

        Parameters
        ----------
        val_type : :class:`type`
            The type of the Python object.
        val : ParsedAttribute
            The actual value.

        Returns
        -------
        s : str
            The attribute string.
        """
        if val_type == np.ndarray:
            array_value = cast(npt.NDArray[np.float64], val)
            return np.array2string(array_value)[1:-1]
        return str(val)

    def _unparse_simple_attribs(self, node: ET._Element) -> None:
        """Convert all Python types from the _ATTRIBS array back into attributes
        for an XML node.

        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The XML node to add the attributes to.
        """
        for attrib_name, (val_type, required) in self._ATTRIBS.items():
            value = cast(ParsedAttribute, getattr(self, attrib_name, None))
            if required or value is not None:
                node.attrib[attrib_name] = self._unparse_attrib(val_type, value)

    def _unparse_simple_elements(self, node: ET._Element, path: str) -> None:
        """Unparse all Python types from the _ELEMENTS array back into child
        nodes of an XML node.

        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The XML node for this object. Elements will be added as children
            of this node.
        path : str
            The string path where the XML file is being written to (used for
            writing out meshes and image files).
        """
        for element_name, (element_type, _required, multiple) in self._ELEMENTS.items():
            value = getattr(self, element_name, None)
            if not multiple:
                element_value = cast(Optional[URDFType], value)
                if element_value is not None:
                    node.append(element_value._to_xml(node, path))
            else:
                element_values = cast(Optional[list[URDFType]], value)
                for child in element_values or []:
                    node.append(child._to_xml(node, path))

    def _unparse(self, path: str) -> ET._Element:
        """Create a node for this object and unparse all elements and
        attributes in the class arrays.

        Parameters
        ----------
        path : str
            The string path where the XML file is being written to (used for
            writing out meshes and image files).

        Returns
        -------
        node : :class:`lxml.etree.Element`
            The newly-created node.
        """
        node = ET.Element(self._TAG)
        self._unparse_simple_attribs(node)
        self._unparse_simple_elements(node, path)
        return node

    def _to_xml(self, parent: Optional[ET._Element], path: str) -> ET._Element:
        """Create and return an XML node for this object.

        Parameters
        ----------
        parent : :class:`lxml.etree.Element`
            The parent node that this element will eventually be added to.
            This base implementation doesn't use this information, but
            classes that override this function may use it.
        path : str
            The string path where the XML file is being written to (used for
            writing out meshes and image files).

        Returns
        -------
        node : :class:`lxml.etree.Element`
            The newly-created node.
        """
        return self._unparse(path)


class URDFTypeWithMesh(URDFType):
    @classmethod
    def _parse_simple_elements(
        cls, node: ET._Element, path: str, lazy_load_meshes: Optional[bool] = None
    ) -> ParsedElementDict:
        """Parse all elements in the _ELEMENTS array from the children of
        this node.

        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse children for.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        lazy_load_meshes : bool
            Whether a mesh element should be immediately loaded or loaded when
            needed

        Returns
        -------
        kwargs : ParsedElementDict
            Map from element names to the :class:`URDFType` subclass (or list,
            if ``multiple`` was set) created for that element.
        """
        kwargs: ParsedElementDict = {}
        for element_name, (element_type, required, multiple) in cls._ELEMENTS.items():
            value: ParsedElement
            if not multiple:
                element_node = node.find(element_type._TAG)
                if required or element_node is not None:
                    if issubclass(element_type, URDFTypeWithMesh):
                        value = cast(
                            ParsedElement,
                            element_type._from_xml(element_node, path, lazy_load_meshes),
                        )
                    else:
                        value = cast(
                            ParsedElement,
                            element_type._from_xml(element_node, path),
                        )
                else:
                    value = None
            else:
                element_nodes = node.findall(element_type._TAG)
                if len(element_nodes) == 0 and required:
                    raise ValueError(
                        "Missing required subelement(s) of type {} when "
                        "parsing an object of type {}".format(element_type.__name__, cls.__name__)
                    )
                if issubclass(element_type, URDFTypeWithMesh):
                    value = [
                        element_type._from_xml(child, path, lazy_load_meshes)
                        for child in element_nodes
                    ]
                else:
                    value = [element_type._from_xml(child, path) for child in element_nodes]
            kwargs[element_name] = value
        return kwargs

    @classmethod
    def _parse(
        cls, node: ET._Element, path: str, lazy_load_meshes: Optional[bool] = None
    ) -> ParsedValueDict:
        """Parse all elements and attributes in the _ELEMENTS and _ATTRIBS
        arrays for a node.

        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        lazy_load_meshes : bool
            Whether meshes should be loaded immediately or upon their first use

        Returns
        -------
        kwargs : ParsedValueDict
            Map from names to Python values created from the attributes
            and elements in the class arrays.
        """
        kwargs: ParsedValueDict = {}
        kwargs.update(cls._parse_simple_attribs(node))
        kwargs.update(cls._parse_simple_elements(node, path, lazy_load_meshes))
        return kwargs

    @classmethod
    def _from_xml(
        cls: type[T],
        node: ET._Element,
        path: str,
        lazy_load_meshes: Optional[bool] = None,
    ) -> T:
        """Create an instance of this class from an XML node.

        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        lazy_load_meshes : bool
            Whether meshes should be loaded immediately or upon their first use

        Returns
        -------
        obj : :class:`URDFType`
            An instance of this class parsed from the node.
        """
        return cls(**cls._parse(node, path, lazy_load_meshes))
