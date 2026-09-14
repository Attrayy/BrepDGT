import os
import typer
import numpy as np
import subprocess
import sys
import tempfile
import trimesh
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
from OCC.Core.GCPnts import GCPnts_QuasiUniformDeflection
from OCC.Extend.DataExchange import write_stl_file

from brepdgt.primitives.uvgrid import UvGrid
from brepdgt.utils.convert_utils import brep_to_uvgrid

app = typer.Typer(pretty_exceptions_enable=False)


def discretize_edge(edge, deflection=0.01):
    """Convert an edge to a series of points with given deflection."""
    curve_handle = BRep_Tool.Curve(edge)[0]
    curve = GeomAdaptor_Curve(curve_handle)

    discretizer = GCPnts_QuasiUniformDeflection(curve, deflection)
    if not discretizer.IsDone():
        return []

    points = []
    for i in range(1, discretizer.NbPoints() + 1):
        p = discretizer.Value(i)
        points.append((p.X(), p.Y(), p.Z()))
    return points


def extract_faces(
    shape,
    linear_deflection=0.1,
    angular_deflection=0.1,
    rotate=False,
    flip=False,
):
    """Extract faces as triangulated mesh."""

    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(tmpdir, exist_ok=True)
        stl_path = os.path.join(tmpdir, "brep.stl")
        write_stl_file(
            shape,
            stl_path,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
        )
        mesh = trimesh.load_mesh(stl_path)
    v, f = np.array(mesh.vertices), np.array(mesh.faces)

    if rotate:
        # Rotate 90 degrees around X axis before normalization
        theta = np.pi / 2
        rotation_matrix = np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )
        v = v @ rotation_matrix.T

    return v, f


def process_step_file(
    step_file: Path,
    output_dir: Path,
    edge_deflection: float = 0.001,
    mesh_linear_deflection: float = 0.1,
    mesh_angular_deflection: float = 0.1,
    rotate: bool = False,
    flip: bool = False,
) -> Path:
    """Process a single STEP file and return the path to the NPZ file"""
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define output paths with _rotate suffix if rotation is applied
        suffix = "_rotate" if rotate else ""
        suffix = "_flip" if flip else suffix
        brep_npz_path = output_dir / f"{step_file.stem}{suffix}.npz"

        # Read STEP file
        step_reader = STEPControl_Reader()
        status = step_reader.ReadFile(str(step_file))

        if status == IFSelect_RetDone:
            step_reader.TransferRoots()
            shape = step_reader.OneShape()

            # Process edges
            all_edges = []
            edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
            while edge_explorer.More():
                edge = edge_explorer.Current()
                points = discretize_edge(edge, edge_deflection)
                if points:
                    edge_points = np.stack(points, axis=0)
                    if rotate:
                        # Apply rotation before normalization
                        theta = np.pi / 2
                        rotation_matrix = np.array(
                            [
                                [1, 0, 0],
                                [0, np.cos(theta), -np.sin(theta)],
                                [0, np.sin(theta), np.cos(theta)],
                            ]
                        )
                        edge_points = edge_points @ rotation_matrix.T
                    all_edges.append(edge_points)
                edge_explorer.Next()

            # Process faces
            vertices, triangles = extract_faces(
                shape, mesh_linear_deflection, mesh_angular_deflection, rotate, flip
            )

            # Normalize and center
            v_min, v_max = vertices.min(axis=0, keepdims=True), vertices.max(
                axis=0, keepdims=True
            )
            center = (v_max + v_min) / 2
            scale = np.max(v_max - v_min)
            vertices = (vertices - center) / scale

            # Move to ground plane at z=-0.5
            z_min = vertices[:, 2].min()
            vertices[:, 2] -= z_min + 0.5

            if flip:
                vertices[:, 1] = -vertices[:, 1]

            # used to move edges to normal direction
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            normal_eps, edge_normal_noise_std, noise_iter = 5e-4, 5e-3, 10

            # Process edges
            all_edges_rescaled = []
            for edge in all_edges:
                # Normalize and center edges
                edge = (edge - center) / scale
                edge[:, 2] -= z_min + 0.5

                if flip:
                    edge[:, 1] = -edge[:, 1]

                # Move edges a little bit towards outwards
                # Compute average normal near the edges
                avg_normals = np.zeros_like(edge)
                for i in range(noise_iter):
                    noisy_edge = edge + edge_normal_noise_std * np.random.randn(
                        *edge.shape
                    )
                    (
                        closest_points,
                        distances,
                        triangle_ids,
                    ) = trimesh.proximity.closest_point(mesh, noisy_edge)
                    avg_normals = mesh.face_normals[triangle_ids]
                avg_normals = avg_normals / np.linalg.norm(
                    avg_normals, keepdims=True, axis=1
                )
                # slightly move edges towards normals
                edge = edge + normal_eps * avg_normals

                all_edges_rescaled.append(edge)

            # Save to numpy file
            np.savez(
                brep_npz_path,
                edges=all_edges_rescaled,
                vertices=vertices,
                triangles=triangles,
                scale=scale,
                center=center,
            )

            # save uvgrid
            existing_uvgrid_path = step_file.parent / f"{step_file.stem}_uvgrid.npz"
            if existing_uvgrid_path.exists():
                # Load and transform existing uvgrid
                npz_data = dict(np.load(existing_uvgrid_path))
                uvgrid = UvGrid.load_from_npz_data(npz_data)
            else:
                # Create new uvgrid from shape
                uvgrid = brep_to_uvgrid(shape, normalize=False)

            uvgrid_coord = uvgrid.coord.reshape(-1, 3)
            uvgrid_coord = (uvgrid_coord - center) / scale
            uvgrid_coord[:, 2] -= z_min + 0.5  # Apply same z-shift to uvgrid
            uvgrid.coord = uvgrid_coord.reshape(uvgrid.coord.shape)
            uvgrid_path = str(brep_npz_path).replace(".npz", "_uvgrid.npz")
            uvgrid.export_npz(uvgrid_path)

            return brep_npz_path

        else:
            print(f"Error: Failed to read STEP file {step_file}")
            return None

    except Exception as e:
        print(f"Error processing file {step_file}: {e}")
        return None


def get_content_bbox(img):
    # Convert to RGBA if not already
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # Get non-white pixels (threshold at 250)
    data = img.getdata()
    non_white = [
        (x, y)
        for y in range(img.height)
        for x in range(img.width)
        if any(v < 240 for v in data[y * img.width + x][:3])
    ]

    if not non_white:
        return None

    # Get bounding box
    left = min(x for x, y in non_white)
    top = min(y for x, y in non_white)
    right = max(x for x, y in non_white)
    bottom = max(y for x, y in non_white)

    return (left, top, right, bottom)


def crop_with_margin(img):
    original_size = img.size
    bbox = get_content_bbox(img)
    if not bbox:
        return img

    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top

    # Make it square based on the larger dimension
    size = max(width, height)

    # Add 10% margin
    margin = int(size * 0.1)

    # Calculate new bounds keeping aspect ratio 1:1 and centering the content
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    half_size = (size + 2 * margin) // 2

    new_left = max(0, center_x - half_size)
    new_right = min(img.width, center_x + half_size)
    new_top = max(0, center_y - half_size)
    new_bottom = min(img.height, center_y + half_size)

    # Crop and resize back to original size
    cropped = img.crop((new_left, new_top, new_right, new_bottom))
    return cropped.resize(original_size, Image.Resampling.LANCZOS)


def render_blender_step(
    npz_file: Path, output_dir: Path, blender_script: Path, color: str = "blue"
) -> Path:
    """Render NPZ file using Blender and return the path to the rendered image"""
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define output image path
        output_image = output_dir / f"{npz_file.stem}.png"

        # Construct Blender command
        cmd = [
            "./blender",
            "--background",
            "--python",
            str(blender_script),
            "--",
            str(npz_file),
            str(output_image),
            color,
        ]

        # Run Blender
        subprocess.run(cmd, check=True, capture_output=True)

        return output_image

    except subprocess.CalledProcessError as e:
        print(f"Error rendering {npz_file}: {e.stderr.decode()}")
        return None
    except Exception as e:
        print(f"Error rendering {npz_file}: {e}")
        return None


def render_blender_uvgrid(uvgrid_path: Path, output_dir: Path) -> Path:
    """Render NPZ file using Blender and return the path to the rendered image"""
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define output image path
        out_path = output_dir / f"{uvgrid_path.stem}.png"

        # Construct Blender command
        cmd = [
            "./blender",
            "--background",
            "--python",
            "./scripts/blender/render_uvgrid_paper_figure.py",
            "--",
            str(uvgrid_path),
            str(out_path),
            "coord_with_mesh",
        ]

        # Run Blender
        subprocess.run(cmd, check=True, capture_output=False)

        return out_path

    except subprocess.CalledProcessError as e:
        print(f"Error rendering uvgrid {uvgrid_path}: {e.stderr.decode()}")
        return None
    except Exception as e:
        print(f"Error rendering uvgrid {uvgrid_path}: {e}")
        return None


def process_file(args):
    """Helper function for parallel processing"""
    (
        step_file,
        output_npz_dir,
        output_render_dir,
        blender_script,
        edge_deflection,
        mesh_linear_deflection,
        mesh_angular_deflection,
        color,
        render_uvgrid,
        rotate,
        flip,
    ) = args

    # Process STEP file to NPZ
    npz_path = process_step_file(
        step_file,
        output_npz_dir,
        edge_deflection,
        mesh_linear_deflection,
        mesh_angular_deflection,
        rotate,
        flip,
    )

    if npz_path:
        # Render NPZ file
        render_path = render_blender_step(
            npz_path, output_render_dir, blender_script, color
        )

        if render_uvgrid:
            uvgrid_path = Path(str(npz_path).replace(".npz", "_uvgrid.npz"))
            if os.path.exists(uvgrid_path):
                render_blender_uvgrid(uvgrid_path, output_render_dir)

        return npz_path, render_path
    return None, None


@app.command()
def main(
    input_dir_or_path: Path = typer.Argument(
        ..., help="Directory or file path containing STEP file(s)"
    ),
    output_dir: Path = typer.Option(None, help="Base output directory"),
    n_steps: int = typer.Option(1000, help="Number of step files to render"),
    render_uvgrid: bool = typer.Option(
        False, help="Whether to render uvgrid alongside step files"
    ),
    edge_deflection: float = typer.Option(0.001, help="Edge discretization deflection"),
    mesh_linear_deflection: float = typer.Option(0.1, help="Mesh linear deflection"),
    mesh_angular_deflection: float = typer.Option(0.1, help="Mesh angular deflection"),
    color: str = typer.Option(
        "blue", help="Color for rendering (blue/pink/orange/green)"
    ),
    max_workers: int = typer.Option(4, help="Maximum number of parallel workers"),
    crop: bool = typer.Option(False, help="Crop and resize rendered images"),
    rotate: bool = typer.Option(False, help="Rotate model 90 degrees around Y axis"),
    flip: bool = typer.Option(False, help="Flip model along XZ plane"),
):
    """
    Process STEP file(s):
    1. Convert STEP files to NPZ format
    2. Generate Blender renders for each file
    3. Generate uvgrid renders for each file
    4. Optionally crop and resize rendered images
    """
    # Validate input path
    if not input_dir_or_path.exists():
        typer.echo(f"Error: Input path {input_dir_or_path} does not exist")
        raise typer.Exit(1)

    # Set up output directory
    if output_dir is None:
        output_dir = (
            input_dir_or_path.parent
            if input_dir_or_path.is_file()
            else input_dir_or_path
        )

    output_npz_dir = output_dir / "npz_for_vis"
    output_render_dir = output_dir

    # Get list of STEP files
    if input_dir_or_path.is_file():
        if not input_dir_or_path.suffix.lower() == ".step":
            typer.echo("Input file must be a STEP file")
            raise typer.Exit(1)
        step_files = [input_dir_or_path]
    else:
        step_files = list(sorted(input_dir_or_path.glob("*.step")))
        if len(step_files) > n_steps:
            print(
                f"Warning: {len(step_files)} STEP files found, only rendering {n_steps} steps"
            )
            step_files = step_files[:n_steps]

    if not step_files:
        typer.echo("No STEP files found")
        raise typer.Exit(1)

    # Get path to blender_visualize.py script
    blender_script = Path(__file__).parent.parent / "blender" / "render_step.py"
    if not blender_script.exists():
        typer.echo(f"Error: Blender script not found at {blender_script}")
        raise typer.Exit(1)

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepare arguments for each file
        args_list = [
            (
                step_file,
                output_npz_dir,
                output_render_dir,
                blender_script,
                edge_deflection,
                mesh_linear_deflection,
                mesh_angular_deflection,
                color,
                render_uvgrid,
                rotate,
                flip,
            )
            for step_file in step_files
        ]

        # Submit all tasks and track progress
        futures = [executor.submit(process_file, args) for args in args_list]

        # Process results with progress bar
        successful_npz = []
        successful_renders = []

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing files"
        ):
            npz_path, render_path = future.result()
            if npz_path:
                successful_npz.append(npz_path)
            if render_path:
                successful_renders.append(render_path)

        # If crop is enabled, process all rendered images
        if crop and successful_renders:
            crop_dir = output_render_dir / "crop"
            crop_dir.mkdir(parents=True, exist_ok=True)

            for render_path in tqdm(successful_renders, desc="Cropping renders"):
                try:
                    # Load and crop image
                    img = Image.open(render_path)
                    cropped_img = crop_with_margin(img)

                    # Save cropped image
                    crop_path = crop_dir / render_path.name
                    cropped_img.save(crop_path)
                except Exception as e:
                    print(f"Error cropping {render_path}: {e}")

    # Print summary
    print("\nProcessing complete!")
    print(f"Processed {len(successful_npz)} NPZ files")
    print(f"Rendered {len(successful_renders)} images")
    print(f"\nOutput locations:")
    print(f"NPZ files: {output_npz_dir}")
    print(f"Renders: {output_render_dir}")
    if crop:
        print(f"Cropped renders: {crop_dir}")


if __name__ == "__main__":
    app()
