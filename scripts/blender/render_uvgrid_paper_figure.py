"""
Renders a uvgrid exported npz path
    $ blender --background --python scripts/blender/render_uvgrid_paper_figure.py -- {npz_path} {render_path} {render_object}
where
    - npz_path:
        path to the npz created from UvGrid.export()
    - render_path:
        path to render
    - render_object: either one of "coord", "coord_normal", or "uv_mesh"
        - coord: renders coordinates with spheres
        - coord_normal: renders coordinates + normals with cones
        - uv_mesh: renders uv grid meshes
or if blender is installed locally, run
    $ {path_to_blender_3.4} --background --python scripts/blender/render_uvgrid_paper_figure.py -- {npz_path} {render_path} coord
Try to use absolute paths to the arguments cause, blender messes up the rendering path
"""

import os
import bpy
from mathutils import Vector

# Get the Blender version
blender_version = bpy.app.version
# Check if it's Blender 3.4.x
assert (
    blender_version[0] == 3 and blender_version[1] == 4
), f"The Blender version is {blender_version[0]}.{blender_version[1]}.x, expected 3.4"
import sys
import numpy as np
from typing import List, Tuple, Union, Dict
from dataclasses import dataclass
from copy import deepcopy


@dataclass(frozen=True)
class Arguments:
    npz_path: str
    render_path: str
    render_object: str  # one of [coord, coord_normal, uv_mesh]


def arg_parser() -> Arguments:
    # obtain parameters for the parametric model
    # should be in form of -- {npz_path}
    # $ blender --background --python scripts/blender/render_uvgrid_paper_figure.py -- {npz_path} {render_path} {render_object}
    parameters = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    npz_path = parameters[0]
    render_path = parameters[1]
    render_object = parameters[2]

    args = Arguments(
        npz_path=npz_path, render_path=render_path, render_object=render_object
    )
    return args


def load_npz(npz_path: str) -> Dict:
    data = dict(np.load(npz_path))
    return data


def color_objects(color=(1, 1, 1, 1)):
    """
    Colors all selected object with Principled BSDF material
    :param color: RGBA colors of mesh, where each value is in range [0, 1]
    :return:
    """

    # Create and assign the Principled BSDF material
    mat = bpy.data.materials.new(name="SphereMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    obj = bpy.context.object
    if obj.data.materials:
        obj.data.materials[0] = mat  # Replace the existing material
    else:
        obj.data.materials.append(mat)  # Add the new material


def create_sphere_from_pc(
    coords, name="PointCloudSpheres", radius=0.1, color=(1, 1, 1, 1)
):
    """
    Create spheres at specified coordinates and join them into a single object.

    Args:
        coordinates (numpy array or list of n x 3): Array of n x 3 coordinates for the sphere positions.
        name (str): Name of the final combined object.
        radius (float): Radius of the spheres.
        color (tuple): RGBA color of the spheres, where each value is between 0 and 1.
    """

    # Create a base sphere at the origin (0, 0, 0)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(0, 0, 0))
    base_sphere = bpy.context.object

    # List to hold the spheres
    all_spheres = []

    # Iterate through the coordinates and create a sphere at each point
    for point in coords:
        # Duplicate the base sphere
        new_sphere = base_sphere.copy()
        new_sphere.data = (
            base_sphere.data.copy()
        )  # Ensure unique mesh data for each sphere
        new_sphere.location = point  # Set the location of the new sphere
        bpy.context.collection.objects.link(
            new_sphere
        )  # Add the new sphere to the scene
        all_spheres.append(new_sphere)

    if len(all_spheres) == 0:
        return

    # Now join all the spheres into a single object
    bpy.context.view_layer.objects.active = all_spheres[
        0
    ]  # Set one sphere as the active object
    bpy.ops.object.select_all(action="DESELECT")  # Deselect everything
    for sphere in all_spheres:
        sphere.select_set(True)  # Select each sphere

    # Join selected spheres into one object
    bpy.ops.object.join()

    # Optionally, delete the original base sphere if not needed
    bpy.data.objects.remove(base_sphere)

    # Rename the combined object
    bpy.context.object.name = name

    color_objects(color)


def create_uv_mesh(
    coord: np.ndarray,
    grid_mask: np.ndarray,
    name="UvMesh",
    color=(1, 1, 1, 1),
    alpha=0.0,
):
    """
    Creates uv mesh from
    :param coord: array of n_grid x n_grid x 3
        UV coordinates
    :param grid_mask: bool array of n_grid x n_grid
        Grid masks
    :param name: Name of the final mesh
    :param color: RGBA colors of mesh, where each value is in range [0, 1]
    :return:
    """

    # create vertices and faces of mesh

    # create valid vertices from grid_mask and mapping from uv to vertex idx
    vertices, faces = [], []
    uv2vertex_idx = -1 * np.ones(coord.shape[:2], dtype=np.int32)
    for i in range(coord.shape[0]):
        for j in range(coord.shape[1]):
            if not grid_mask[i, j]:
                continue
            uv2vertex_idx[i, j] = len(vertices)
            vertices.append(coord[i, j])
    # create faces
    for i in range(coord.shape[0] - 1):
        for j in range(coord.shape[1] - 1):
            if not grid_mask[i, j]:
                continue
            if not grid_mask[i + 1, j + 1]:
                continue
            if grid_mask[i + 1, j]:
                faces.append(
                    (
                        uv2vertex_idx[i, j],
                        uv2vertex_idx[i + 1, j],
                        uv2vertex_idx[i + 1, j + 1],
                    )
                )
            if grid_mask[i, j + 1]:
                faces.append(
                    (
                        uv2vertex_idx[i, j],
                        uv2vertex_idx[i, j + 1],
                        uv2vertex_idx[i + 1, j + 1],
                    )
                )

    # no triangles to render
    if len(faces) < 1:
        return

    vertices = np.stack(vertices, axis=0)
    faces = np.stack(faces, axis=0)

    # Create a new mesh and a new object
    mesh_data = bpy.data.meshes.new(name)  # Create a new mesh
    mesh_obj = bpy.data.objects.new(
        name, mesh_data
    )  # Create a new object with the mesh

    # Link the object to the current collection
    bpy.context.collection.objects.link(mesh_obj)

    # Switch to object mode if not already in it
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.select_all(action="DESELECT")  # Deselect all objects
    mesh_obj.select_set(True)

    # Create the mesh from the given vertices and faces
    mesh_data.from_pydata(vertices, [], faces)

    # Update the mesh (necessary to ensure it's visible in the viewport)
    mesh_data.update()

    # Create a new material
    mat = bpy.data.materials.new(name="TransparentMaterial")
    mat.use_nodes = True

    # Get the material's node tree
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)

    # Add a Principled BSDF node
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    # Set transparency (Alpha value)
    bsdf.inputs["Alpha"].default_value = (
        alpha  # Adjust transparency (0 is fully transparent, 1 is opaque)
    )
    bsdf.inputs["Base Color"].default_value = color

    # Add a Material Output node
    material_output = nodes.new(type="ShaderNodeOutputMaterial")
    material_output.location = (200, 0)

    # Link the BSDF to the Material Output
    links.new(bsdf.outputs["BSDF"], material_output.inputs["Surface"])

    # Assign the material to the object
    if len(mesh_obj.data.materials) == 0:
        mesh_obj.data.materials.append(mat)
    else:
        mesh_obj.data.materials[0] = mat


def create_mesh_from_npz(npz_data: Dict, render_object: str):
    cmap = get_cmap()
    if npz_data.get("coord") is not None:
        radius = 0.02
        # brep-diff uvgrid format
        coords = npz_data["coord"]
        grid_masks = npz_data.get("grid_mask")
        empty_masks = npz_data["empty_mask"]

        if len(coords.shape) == 5:
            coords = coords[0]
            if grid_masks is not None:
                grid_masks = grid_masks[0]
            else:
                grid_masks = np.ones_like(coords)[:, :, :, 0].astype(bool)
            empty_masks = empty_masks[0]

    elif npz_data.get("face_uvgrids") is not None:
        # brepgen format
        radius = 0.01
        coords = npz_data["face_uvgrids"]  # n_face x 32 * 32 x 3
        coords = coords / 6  # brepgen is normalized from [-3, 3]
        coords[:, :, 2] -= coords[:, :, 2].min() + 0.5
        coords = coords.reshape(coords.shape[0], 32, 32, 3)
        grid_masks = np.ones_like(coords)[:, :, :, 0].astype(bool)
        empty_masks = np.zeros(coords.shape[0]).astype(bool)
    else:
        raise ValueError("npz_data does not have coord or face_uvgrids key")

    # deduplicate grid masks so that we do not have an exactly same mesh which does not look good
    deduplicate_masks = grid_masks.copy()
    for face_idx in range(coords.shape[0]):
        for i in range(coords.shape[1]):
            for j in range(coords.shape[2]):
                if not grid_masks[face_idx, i, j]:
                    continue
                diff = np.linalg.norm(
                    coords - coords[face_idx : face_idx + 1, i : i + 1, j : j + 1],
                    axis=-1,
                )
                same_mask = diff < 5e-3
                same_mask[face_idx, i, j] = False
                deduplicate_masks[np.where(same_mask)] = False

    for i in range(len(coords)):
        name = f"points_{str(i).zfill(4)}"
        color = cmap[i % len(cmap)] + (1,)  # RGBA

        # skip if face is empty
        if empty_masks[i]:
            continue

        coord = coords[i][deduplicate_masks[i]]  # N x 3
        if (render_object == "coord") or (render_object == "coord_with_mesh"):
            create_sphere_from_pc(
                coords=coord, name=name, radius=radius, color=deepcopy(color)
            )
        if (render_object == "mesh") or (render_object == "coord_with_mesh"):
            create_uv_mesh(
                coord=coords[i],
                grid_mask=grid_masks[i],
                name=name,
                color=deepcopy(color),
                alpha=0.5,
            )


# Function to create a ground plane assuming point clouds are in range [-1, 1]
def create_ground_plane(size=2, color=(0.8, 0.8, 0.8, 1.0), name="Ground"):
    bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False)
    ground = bpy.context.object
    ground.name = name

    # Move the ground plane slightly below the point clouds
    ground.location = (0, 0, -1.1)
    ground.is_shadow_catcher = True


def create_light(loc, name: str):
    light_data = bpy.data.lights.new(name, type="POINT")
    light_obj = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = loc
    light_obj.data.energy = 100  # Increase the energy for brighter light


def set_env_objects():
    # Clear existing mesh objects in the scene
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="MESH")
    bpy.ops.object.delete()

    # Add a ground plane to the scene, scaled to fit the [-1, 1] range
    # create_ground_plane(size=10, color=(0.8, 0.8, 0.8, 1.0), name="GroundPlane")

    # Set the camera and lighting, assuming point clouds in range [-1, 1]
    # cam_data = bpy.data.cameras.new("Camera")
    # cam_obj = bpy.data.objects.new("Camera", cam_data)
    # bpy.context.collection.objects.link(cam_obj)
    # cam_obj.location = (2.5, -2.5, 2)  # Adjust camera for [-1, 1] range
    # cam_obj.rotation_euler = (1.04, 0, 0.78)  # 60, 0, 45 degrees

    # Setup camera with closer position and centered view
    # bpy.ops.object.camera_add(location=(2, -2, 1.5))
    # camera = bpy.context.active_object
    # camera.rotation_euler = (
    #     0.9,
    #     0.0,
    #     0.785398,
    # )
    # bpy.context.scene.camera = camera

    # # Adjust camera settings for better framing
    # camera.data.lens = 60

    # # Track camera to scene center
    # empty = bpy.data.objects.new("CameraTarget", None)
    # bpy.context.scene.collection.objects.link(empty)
    # empty.location = (0, 0, 0)

    # track = camera.constraints.new(type="TRACK_TO")
    # track.target = empty
    # track.track_axis = "TRACK_NEGATIVE_Z"
    # track.up_axis = "UP_Y"

    # set lighting
    create_light((1.0, 1.0, 4.0), "Light1")
    create_light((-1.0, 1.0, 4.0), "Light2")
    create_light((1.0, -1.0, 4.0), "Light3")
    create_light((-1.0, -1.0, 4.0), "Light4")


def get_cmap():
    cmap = [
        (0.894, 0.102, 0.110),  # Red
        (0.216, 0.494, 0.722),  # Blue
        (0.302, 0.686, 0.290),  # Green
        (0.596, 0.306, 0.639),  # Purple
        (1.000, 0.498, 0.000),  # Orange
        (1.000, 1.000, 0.200),  # Yellow
        (0.651, 0.337, 0.157),  # Brown
        (0.969, 0.506, 0.749),  # Pink
        (0.600, 0.600, 0.600),  # Grey
        (0.090, 0.745, 0.812),  # Cyan
        (0.980, 0.506, 0.353),  # Salmon
        (0.850, 0.372, 0.007),  # Dark Orange
        (0.550, 0.090, 0.290),  # Burgundy
        (0.294, 0.000, 0.510),  # Indigo
        (0.628, 0.745, 0.245),  # Olive Green
        (0.976, 0.306, 0.906),  # Magenta
        (0.255, 0.412, 0.882),  # Royal Blue
        (0.329, 0.510, 0.208),  # Forest Green
        (0.890, 0.258, 0.447),  # Crimson
        (0.835, 0.150, 0.585),  # Hot Pink
    ]
    return cmap


def set_white_background():

    # Set image output settings to RGB (without alpha)
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGB"  # RGB only, no alpha

    # Enable compositing and clear existing nodes
    bpy.context.scene.use_nodes = True
    compositor = bpy.context.scene.node_tree
    compositor.nodes.clear()

    # Add Render Layers node (input)
    render_layers = compositor.nodes.new(type="CompositorNodeRLayers")

    # Add a white background using an RGB node
    white_background = compositor.nodes.new(type="CompositorNodeRGB")
    white_background.outputs["RGBA"].default_value = (1, 1, 1, 1)  # White color (RGBA)

    # Add an Alpha Over node to overlay the render on the white background
    alpha_over = compositor.nodes.new(type="CompositorNodeAlphaOver")

    # Connect nodes in the correct order
    compositor.links.new(
        render_layers.outputs["Image"], alpha_over.inputs[2]
    )  # Render result to Alpha Over bottom input
    compositor.links.new(
        white_background.outputs["RGBA"], alpha_over.inputs[1]
    )  # White background to Alpha Over top input

    # Set up the Composite node (output)
    composite_output = compositor.nodes.new(type="CompositorNodeComposite")
    compositor.links.new(alpha_over.outputs["Image"], composite_output.inputs["Image"])

    # # Enable shadow pass
    # bpy.context.view_layer.cycles.use_pass_shadow_catcher = True
    # Adjust color management settings to ensure a bright white background
    bpy.context.scene.view_settings.view_transform = (
        "Standard"  # Set to Standard to avoid Filmic adjustments
    )


def main():
    args: Arguments = arg_parser()

    # set_env_objects()

    # Enable GPU rendering
    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = (
        "CUDA"  # or 'OPTIX' for compatible NVIDIA GPUs, 'OPENCL' for AMD
    )

    # Specify the devices to use (e.g., select your GPU as the device)
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    for device in bpy.context.preferences.addons["cycles"].preferences.devices:
        if device.type == "CUDA" or device.type == "OPTIX":
            device.use = True  # Set to True to enable GPU

    # Set the scene to use GPU compute
    bpy.context.scene.cycles.device = "GPU"

    # Clear existing scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    npz_data = load_npz(args.npz_path)
    create_mesh_from_npz(npz_data=npz_data, render_object=args.render_object)

    # Create ground plane
    # ground = create_ground_plane()
    # ground.location.z = -0.5

    # Setup camera with closer position and centered view
    bpy.ops.object.camera_add(location=(2, -2, 1.5))
    camera = bpy.context.active_object
    camera.rotation_euler = (
        0.9,
        0.0,
        0.785398,
    )  # 0.785398 is pi/4 for perfect 45-degree angle

    # Create lights with adjusted positions
    bpy.ops.object.light_add(type="SUN", location=(2.5, 2.5, 4))
    sun = bpy.context.active_object
    sun.rotation_euler = (0.5, 0.2, -0.3)
    sun.data.energy = 2.0

    # Add fill light
    bpy.ops.object.light_add(type="AREA", location=(-1.5, -1.5, 2.5))
    fill_light = bpy.context.active_object
    fill_light.data.energy = 10.0
    fill_light.scale = (3, 3, 3)

    # Add rim light
    bpy.ops.object.light_add(type="AREA", location=(2, -1, 2))
    rim_light = bpy.context.active_object
    rim_light.data.energy = 15.0
    rim_light.scale = (2, 2, 2)

    # Setup render settings
    scene = bpy.context.scene
    scene.camera = camera

    # Enhanced render settings
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.film_transparent = True

    # Anti-aliasing improvements
    scene.render.filter_size = 1.5  # Increased from default 1.5
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01  # Lower for better quality
    scene.cycles.adaptive_min_samples = 64
    scene.cycles.samples = 256  # Increased from 128
    scene.cycles.use_denoising = True

    # Shadow and light path settings
    scene.cycles.caustics_reflective = True
    scene.cycles.caustics_refractive = True
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 8
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 8

    # Enhance shadow darkness
    for light in bpy.data.lights:
        if light.type == "SUN":
            light.shadow_soft_size = 0.3
        elif light.type == "AREA":
            light.shadow_soft_size = 0.3
            light.cycles.cast_shadow = True

    # World settings for better ambient occlusion
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Clear existing nodes
    nodes.clear()

    # Add nodes for better background and ambient occlusion
    bg = nodes.new("ShaderNodeBackground")
    ao = nodes.new("ShaderNodeAmbientOcclusion")
    mix = nodes.new("ShaderNodeMixShader")
    output = nodes.new("ShaderNodeOutputWorld")

    # Set background color
    bg.inputs["Color"].default_value = (0.95, 0.95, 0.95, 1)
    bg.inputs["Strength"].default_value = 0.5

    # Force white background
    set_white_background()

    # Set AO settings
    ao.inputs["Distance"].default_value = 0.5
    ao.inputs["Color"].default_value = (0.1, 0.1, 0.1, 1)  # Darker shadows

    # Connect nodes
    links.new(ao.outputs["Color"], mix.inputs[1])
    links.new(bg.outputs["Background"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])

    # Adjust camera settings for better framing
    camera.data.lens = 60

    # Track camera to scene center
    empty = bpy.data.objects.new("CameraTarget", None)
    bpy.context.scene.collection.objects.link(empty)
    empty.location = (0, 0, 0)

    track = camera.constraints.new(type="TRACK_TO")
    track.target = empty
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    # Render first view
    scene.render.filepath = args.render_path
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
