from abc import ABC
from typing import List, Optional, Any, Tuple
import enum
from functools import reduce
import dataclasses
import xml.etree.ElementTree as ET
import pylab as p
import svgpathtools
from svgpathtools.parser import parse_path, parse_transform
from svgpathtools.path import transform
from svgpathtools import svg2paths, wsvg
import numpy as np


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

class SpecialTokens(enum.Enum):
    INSTR_START = "<istart>"
    INSTR_END = "<iend>"
    PATH_START = "<pstart>"
    PATH_END = "<pend>"
    START = "<start>"
    END = "<end>"
    OPTION_START = "<ostart>"
    OPTION_END = "<oend>"


def create_merged_2D_command(points: List[complex]):
    all_cmds = list()
    for p in points:
        all_cmds += [{"command": "x", "value": p.real}, {"command": "y", "value": p.imag}]
    return all_cmds


def create_str_command(str_list: List[str]) -> List[dict]:
    return [{"command": s, "value": 0} for s in str_list]


def create_scalar_command(name: str, r: float) -> List[dict]:
    return [{"command": name, "value": r}]


def create_color_command(name: str, rgb: Tuple[int]) -> List[dict]:
    if rgb is None:
        return []
    return [{"command": f"{name}_c={c}", "value": v} for c, v in zip("rgb", rgb)]


def command_to_pre_tensor(c):
    rot_command = list()
    if type(c) == svgpathtools.path.CubicBezier:
        str_commands = [PathCommand.CubicBezier]
        float_cmds = [c.start, c.control1, c.control2, c.end]
    #elif type(c) == svgpathtools.path.Move:
    #    str_commands = [PathCommand.MoveTo]
    #    float_cmds = [c.end]
    elif type(c) == svgpathtools.path.Line:
        str_commands = [PathCommand.LineTo]
        float_cmds = [c.start, c.end]
    elif type(c) == svgpathtools.path.Arc:
        # Arc(start=(84.477706+715.609836j), radius=(5.880173+5.880173j), rotation=0.0, arc=False, sweep=False, end=(90.370167+709.717376j)),
        str_commands = [PathCommand.Arc,  f"arc={c.large_arc}", f"sweep={c.sweep}"]
        float_cmds = [c.start, c.radius, c.end]
        rot_command = create_scalar_command("rot", c.rotation)
    elif type(c) == svgpathtools.path.QuadraticBezier:
        str_commands = [PathCommand.QuadraticBezier]
        float_cmds = [c.start, c.control, c.end]
    else:
        raise Exception(f"{c}, {type(c)}, {str(type(c)) == "Move"}")
    return (create_str_command([SpecialTokens.INSTR_START.value]) + create_str_command(str_commands) + rot_command +
            create_merged_2D_command(float_cmds) + create_str_command([SpecialTokens.INSTR_END.value]))


@dataclasses.dataclass
class SVGPath:
    commands: List[Any]
    fill: Optional[Tuple[float, float, float]]
    stroke_width: Optional[float]
    stroke_color: Optional[Tuple[float, float, float]]

    @classmethod
    def parse_node(cls, commands, attrs):

        if len(commands) == 0:
            return list()

        fill = parse_color(attrs, "fill")
        stroke_color = parse_color(attrs, "stroke-color")
        stroke_width = attrs.get("stroke-width", 1)

        path_commands = reduce(lambda x, y: x+y, [command_to_pre_tensor(c) for c in commands])

        option_commands = (create_str_command([SpecialTokens.OPTION_START.value]) +
                           create_color_command("fill", fill) +
                           create_color_command("stroke", stroke_color) +
                           create_scalar_command("stroke_width", stroke_width) +
                           create_str_command([SpecialTokens.OPTION_END.value]))

        figure_command = (create_str_command([SpecialTokens.PATH_START.value]) +
                          path_commands + option_commands +
                          create_str_command([SpecialTokens.PATH_END.value]))

        return figure_command


def parse_color(node, name):
    """
    Parsed as RGB
    """
    color_dict = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "blue": (0, 0, 255),
        "red": (255, 0, 0),
        "green": (0, 255, 0)
    }
    color = node.get("fill")
    if color is None or color.lower() in {"none"}:
        return None
    elif color in color_dict:
        return color_dict[color]
    h = color.lstrip("#")
    try:
        parsed_color = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        parsed_color = tuple(16 * int(h[i:i + 1], 16) for i in (0, 1, 2))
    return parsed_color


def parse_document(path):
    """
    '/home/amor/Documents/code_dw/pathformer/dataset/www.svgrepo.com/show/475393/cigarette.svg'
    """
    my_paths = create_str_command([SpecialTokens.START.value])

    paths, attributes, *other_attrs = svg2paths(path)
    for pa, attrs in zip(paths, attributes):
        if "transform" in attrs:
            tf = parse_transform(attrs["transform"])
            new_path = transform(pa, tf)
        else:
            new_path = pa
        my_paths.extend(SVGPath.parse_node(new_path, attrs))

    my_paths.extend(create_str_command([SpecialTokens.END.value]))

    return my_paths

