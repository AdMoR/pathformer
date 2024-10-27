import abc
from unittest import TestCase
from abc import ABC
from typing import List, Optional, Any
import enum
import re
import dataclasses

@dataclasses.dataclass
class SVGCommand(ABC):
    fill: Optional[str]
    stroke: Optional[str]
    stroke_width: Optional[int]

    @abc.abstractmethod
    def from_xml_node(self, node):
        pass


class Rectangle(SVGCommand):
    x: float
    width: float
    y: float
    height: float




class PathCommand(enum.Enum):
    MoveTo = "m"
    LineTo = "l"
    Vertical = "v"
    Horizontal = "h"
    BackTo = "z"
    CubicBezier = "c"
    SeveralCubicBezier = "s"
    QuadraticBezier = "q"
    SeveralQuadraticBezier = "t"
    Arc = "a"


@dataclasses.dataclass
class PathInstruction:
    command: PathCommand
    coordinates: List[float]
    extra: Optional[List[Any]]

    def __post_init__(self):
        if self.command == PathCommand.Arc:
            assert len(self.coordinates) == 7
        elif self.command == PathCommand.QuadraticBezier:
            assert len(self.coordinates) == 4
        elif self.command == PathCommand.SeveralQuadraticBezier:
            assert len(self.coordinates) == 2
        elif self.command == PathCommand.CubicBezier:
            assert len(self.coordinates) == 6
        elif self.command == PathCommand.SeveralCubicBezier:
            assert len(self.coordinates) == 4
        elif self.command == PathCommand.BackTo:
            assert len(self.coordinates) == 0
        elif self.command == PathCommand.Horizontal:
            assert  len(self.coordinates) == 1
        elif self.command == PathCommand.Vertical:
            assert  len(self.coordinates) == 1
        elif self.command == PathCommand.MoveTo:
            assert  len(self.coordinates) == 2
        elif self.command == PathCommand.LineTo:
            assert  len(self.coordinates) == 2

    @classmethod
    def from_str(cls, svg_path):
        pattern = re.compile(r'([MLHVCSQTAZmlhvcsqtaz])([^MLHVCSQTAZmlhvcsqtaz]*)?')
        matches = pattern.findall(svg_path)

        instructions = list()
        for match in matches:
            command = match[0]
            parameters = match[1].strip()
            parameters = parameters.replace(" ", ",").replace("-", ",-")
            instructions.append(PathInstruction(PathCommand(command.lower()),
                                                [float(x) for x in parameters.split(",") if len(x) > 0], None))
        return instructions

class TestParser(TestCase):

    def test(self):
        svg_path= "M224,44H32A20.02229,20.02229,0,0,0,12,64V88a20.03265,20.03265,0,0,0,16,19.59668V192a20.02229,20.02229,0,0,0,20,20H208a20.02229,20.02229,0,0,0,20-20V107.59668A20.03265,20.03265,0,0,0,244,88V64A20.02229,20.02229,0,0,0,224,44ZM36,68H220V84H36ZM52,188V108H204v80Zm112-52a12.00028,12.00028,0,0,1-12,12H104a12,12,0,0,1,0-24h48A12.00028,12.00028,0,0,1,164,136Z"

        #svg_path = "M10 10 H 90 V 90 H 10 L 10 10 Z"
        rez = PathInstruction.from_str(svg_path)

        print(rez)
        assert False