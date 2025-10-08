from __future__ import annotations

from os import PathLike, fspath
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import PIL
from lxml import etree as ET

from urchin.base import URDFType
from urchin.utils import get_filename


class Texture(URDFType):
    """An image-based texture.

    Parameters
    ----------
    filename : str
        The path to the image that contains this texture. This can be
        relative to the top-level URDF or an absolute path.
    image : :class:`PIL.Image.Image` or ``numpy.ndarray`` or ``str``, optional
        The image for the texture. If a ``str`` path or a numpy array is
        provided, it is converted to a PIL image. If not specified, it is
        loaded automatically from ``filename``.
    """

    _ATTRIBS = {"filename": (str, True)}
    _TAG = "texture"

    def __init__(
        self,
        filename: str,
        image: Union[PIL.Image.Image, str, np.ndarray, None] = None,
    ):
        if image is None:
            image = PIL.Image.open(filename)
        self.filename = filename
        self.image = image

    @property
    def filename(self) -> str:
        """str : Path to the image for this texture."""
        return self._filename

    @filename.setter
    def filename(self, value: Union[str, PathLike[str]]) -> None:
        self._filename = fspath(value)

    @property
    def image(self) -> PIL.Image.Image:
        """:class:`PIL.Image.Image` : The image for this texture."""
        return self._image

    @image.setter
    def image(self, value: Union[PIL.Image.Image, str, np.ndarray]) -> None:
        if isinstance(value, str):
            value = PIL.Image.open(value)
        if isinstance(value, np.ndarray):
            value = PIL.Image.fromarray(value)
        elif not isinstance(value, PIL.Image.Image):
            raise ValueError("Texture only supports numpy arrays or PIL images")
        self._image = value

    @classmethod
    def _from_xml(cls, node: ET._Element, path: str, lazy_load_meshes: Optional[bool] = None):
        # Explicitly parse fields for typing clarity
        filename = str(node.attrib["filename"]) if "filename" in node.attrib else ""
        fn = get_filename(path, filename)
        image = PIL.Image.open(fn)
        return cls(filename=filename, image=image)

    def _to_xml(self, parent: Optional[ET._Element], path: str) -> ET._Element:
        # Save the image
        filepath = get_filename(path, self.filename, makedirs=True)
        self.image.save(filepath)

        return self._unparse(path)

    def copy(self, prefix: str = "", scale: Union[float, np.ndarray, None] = None) -> "Texture":
        """Create a deep copy with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all names.

        Returns
        -------
        :class:`.Texture`
            A deep copy.
        """
        v = self.__class__(filename=self.filename, image=self.image.copy())
        return v


class Material(URDFType):
    """A material for some geometry.

    Parameters
    ----------
    name : str
        The name of the material.
    color : (4,) float, optional
        The RGBA color of the material in the range [0,1].
    texture : :class:`.Texture`, optional
        A texture for the material.
    """

    _ATTRIBS = {"name": (str, True)}
    _ELEMENTS = {
        "texture": (Texture, False, False),
    }
    _TAG = "material"

    def __init__(
        self,
        name: str,
        color: Optional[npt.ArrayLike] = None,
        texture: Union[Texture, str, None] = None,
    ):
        self.name = name
        self.color = color
        self.texture = texture

    @property
    def name(self) -> str:
        """str : The name of the material."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)

    @property
    def color(self) -> Optional[np.ndarray]:
        """(4,) float : The RGBA color of the material, in the range [0,1]."""
        return self._color

    @color.setter
    def color(self, value: Optional[npt.ArrayLike]) -> None:
        if value is not None:
            value = np.asanyarray(value).astype(float)
            value = np.clip(value, 0.0, 1.0)
            if value.shape != (4,):
                raise ValueError("Color must be a (4,) float")
        self._color = value

    @property
    def texture(self) -> Optional[Texture]:
        """:class:`.Texture` : The texture for the material."""
        return self._texture

    @texture.setter
    def texture(self, value: Union[Texture, str, None]) -> None:
        if value is not None:
            if isinstance(value, str):
                image = PIL.Image.open(value)
                value = Texture(filename=value, image=image)
            elif not isinstance(value, Texture):
                raise ValueError("Invalid type for texture -- expect path to image or Texture")
        self._texture = value

    @classmethod
    def _from_xml(cls, node: ET._Element, path: str, lazy_load_meshes: Optional[bool] = None):
        name = str(node.attrib["name"]) if "name" in node.attrib else ""
        color_arr = None
        color_node = node.find("color")
        if color_node is not None and "rgba" in color_node.attrib:
            color_arr = np.fromstring(color_node.attrib["rgba"], sep=" ", dtype=np.float64)
        texture_node = node.find("texture")
        texture = Texture._from_xml(texture_node, path) if texture_node is not None else None
        return cls(name=name, color=color_arr, texture=texture)

    def _to_xml(self, parent: ET._Element, path: str) -> ET._Element:
        # Simplify materials by collecting them at the top level.

        # For top-level elements, save the full material specification
        if parent.tag == "robot":
            node = self._unparse(path)
            if self.color is not None:
                color = ET.Element("color")
                color.attrib["rgba"] = np.array2string(self.color)[1:-1]
                node.append(color)

        else:
            node = ET.Element("material")
            node.attrib["name"] = self.name
            if self.color is not None:
                color = ET.Element("color")
                color.attrib["rgba"] = np.array2string(self.color)[1:-1]
                node.append(color)
        return node

    def copy(self, prefix: str = "", scale: Union[float, np.ndarray, None] = None) -> "Material":
        """Create a deep copy of the material with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all joint and link names.

        Returns
        -------
        :class:`.Material`
            A deep copy of the material.
        """
        return self.__class__(
            name="{}{}".format(prefix, self.name),
            color=self.color,
            texture=self.texture,
        )
