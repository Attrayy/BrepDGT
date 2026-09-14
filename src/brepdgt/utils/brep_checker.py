from brepdgt.postprocessing.occ_wrapper import write_stl_with_timeout
from brepdgt.utils.brep_sampler import get_n_faces_from_step_path
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire, ShapeAnalysis_Shell
from OCC.Core.TopoDS import topods
from OCC.Core.TopAbs import TopAbs_ShapeEnum
from OCC.Core.TopExp import TopExp_Explorer
from typing import Tuple
import tempfile
import os
import trimesh


# Function to extract wires from the shape
def check_wires(shape):
    faces_and_wires = []
    # Traverse faces in the shape
    face_explorer = TopExp_Explorer(shape, TopAbs_ShapeEnum.TopAbs_FACE)
    while face_explorer.More():
        # Get the current face
        face = topods.Face(face_explorer.Current())

        # Traverse wires in the current face
        wire_explorer = TopExp_Explorer(face, TopAbs_ShapeEnum.TopAbs_WIRE)
        while wire_explorer.More():
            # Get the current wire
            wire = topods.Wire(wire_explorer.Current())
            wire_analysis = ShapeAnalysis_Wire(wire, face, 0.01)
            not_ordered = wire_analysis.CheckOrder(False)
            if not_ordered:
                return False
            has_self_intersection = wire_analysis.CheckSelfIntersection()
            if has_self_intersection:
                return False

            # Store the face and the wire as a tuple
            faces_and_wires.append((face, wire))

            wire_explorer.Next()
        face_explorer.Next()
    return True


def check_edges(shape):
    explorer = TopExp_Explorer(shape, TopAbs_ShapeEnum.TopAbs_SHELL)
    shell_analysis = ShapeAnalysis_Shell()
    while explorer.More():
        shell = topods.Shell(explorer.Current())
        shell_analysis.LoadShells(shell)
        explorer.Next()

    has_bad_edges = shell_analysis.HasBadEdges()
    return not has_bad_edges


# Function to extract shells from a shape
def extract_shells(shape):
    shells = []
    explorer = TopExp_Explorer(shape, TopAbs_ShapeEnum.TopAbs_SHELL)
    while explorer.More():
        shell = topods.Shell(explorer.Current())
        shells.append(shell)
        explorer.Next()
    return shells


def check_solid(
    step_path: str, stl_out_path: str = None, timeout: int = 20
) -> Tuple[bool, int]:
    """
    Given a step, checks if the shape is a solid.
    :param step_path: path to the step
    :param stl_out_path: path to the output stl.
        If none, saves the stl in a temporary directory.
    :param timeout: timeout in seconds
        Sometimes python write_stl freezes, thus we need a timeout

    :returns
        - is_solid: boolean indicating if the step is a solid
        - n_faces: number of faces of the brep

    """
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.TopAbs import TopAbs_SOLID
    from OCC.Extend.DataExchange import read_step_file

    try:
        brep = read_step_file(step_path)
    except Exception as e:
        print(f"Error reading step: {e}")
        return False, 0

    # Step 1: Check if the top-level shape is a solid
    if brep.ShapeType() != TopAbs_SOLID:
        # print("The BRep is not a single solid.")
        return False, 0

    # Step 2: Use BRepCheck_Analyzer to validate the solid
    analyzer = BRepCheck_Analyzer(brep)
    if not analyzer.IsValid():
        # print("The BRep is not a valid solid.")
        return False, 0

    # Step 3. Check wire (as in SoldGen)
    correct_wire = check_wires(brep)
    if not correct_wire:
        return False, 0

    # Step 4. check bad edges
    correct_edge = check_edges(brep)
    if not correct_edge:
        return False, 0

    # Step 4: Test whether stl is exportable within reasonable amount of time.
    # Sometimes python occ freezes when writing stl path (don't know why :( ).
    # If this happens, we won't be able to visualize/evaluate them, so we just regard as not watertight
    if stl_out_path is None:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_path = os.path.join(tmp_dir, "out.stl")
                write_stl_with_timeout(brep, temp_path, timeout)
                mesh = trimesh.load_mesh(temp_path)
                if len(mesh.faces) <= 0:
                    return False, 0
        except:
            return False, 0
    else:
        write_stl_with_timeout(brep, stl_out_path, timeout)

    # Use occwl to get the number of faces
    try:
        n_faces = get_n_faces_from_step_path(step_path)
    except Exception as e:
        print(f"{step_path} not valid: {e}")
        return False, 0

    # If all checks pass, it's a single valid solid
    return True, n_faces
