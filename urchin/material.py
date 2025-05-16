from urchin.base import URDFType
from urchin.utils import get_filename

import numpy as np
import PIL
from lxml import etree as ET

class Texture(URDFType):
    """An image-based texture.

    Parameters
    ----------
    filename : str
        The path to the image that contains this texture. This can be
        relative to the top-level URDF or an absolute path.
    image : :class:`PIL.Image.Image`, optional
        The image for the texture.
        If not specified, it is loaded automatically from the filename.
    """

    _ATTRIBS = {"filename": (str, True)}
    _TAG = "texture"

    def __init__(self, filename, image=None):
        if image is None:
            image = PIL.image.open(filename)
        self.filename = filename
        self.image = image

    @property
    def filename(self):
        """str : Path to the image for this texture."""
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = str(value)

    @property
    def image(self):
        """:class:`PIL.Image.Image` : The image for this texture."""
        return self._image

    @image.setter
    def image(self, value):
        if isinstance(value, str):
            value = PIL.Image.open(value)
        if isinstance(value, np.ndarray):
            value = PIL.Image.fromarray(value)
        elif not isinstance(value, PIL.Image.Image):
            raise ValueError("Texture only supports numpy arrays " "or PIL images")
        self._image = value

    @classmethod
    def _from_xml(cls, node, path):
        kwargs = cls._parse(node, path)

        # Load image
        fn = get_filename(path, kwargs["filename"])
        kwargs["image"] = PIL.Image.open(fn)

        return cls(**kwargs)

    def _to_xml(self, parent, path):
        # Save the image
        filepath = get_filename(path, self.filename, makedirs=True)
        self.image.save(filepath)

        return self._unparse(path)

    def copy(self, prefix="", scale=None):
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

    def __init__(self, name, color=None, texture=None):
        self.name = name
        self.color = color
        self.texture = texture

    @property
    def name(self):
        """str : The name of the material."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def color(self):
        """(4,) float : The RGBA color of the material, in the range [0,1]."""
        return self._color

    @color.setter
    def color(self, value):
        if value is not None:
            value = np.asanyarray(value).astype(float)
            value = np.clip(value, 0.0, 1.0)
            if value.shape != (4,):
                raise ValueError("Color must be a (4,) float")
        self._color = value

    @property
    def texture(self):
        """:class:`.Texture` : The texture for the material."""
        return self._texture

    @texture.setter
    def texture(self, value):
        if value is not None:
            if isinstance(value, str):
                image = PIL.Image.open(value)
                value = Texture(filename=value, image=image)
            elif not isinstance(value, Texture):
                raise ValueError(
                    "Invalid type for texture -- expect path to " "image or Texture"
                )
        self._texture = value

    @classmethod
    def _from_xml(cls, node, path):
        kwargs = cls._parse(node, path)

        # Extract the color -- it's weirdly an attribute of a subelement
        color = node.find("color")
        if color is not None:
            color = np.fromstring(color.attrib["rgba"], sep=" ", dtype=np.float64)
        kwargs["color"] = color

        return cls(**kwargs)

    def _to_xml(self, parent, path):
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

    def copy(self, prefix="", scale=None):
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
