import gdextension
from godot.enums import VariantType, PropertyHint

class PropertyInfo(gdextension.PropertyInfo):
    pass


class PropertyInfoRange(PropertyInfo):
    def __new__(cls, name: str, range: slice) -> PropertyInfo:
        return super().__new__(
            cls,
            VariantType.TYPE_FLOAT,
            name,
            PropertyHint.PROPERTY_HINT_RANGE,
            f"{range.start},{range.stop},{range.step}"
        )
