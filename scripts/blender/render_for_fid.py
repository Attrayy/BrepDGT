"""
Renders a mesh in unit bbox from given angle
    $ blender --background --python scripts/blender/render_for_fid.py -- {mesh_path} {render_path} {theta}
    $ blender --background --python scripts/blender/render_for_fid.py -- {mesh_path} {render_path} {theta} normalize
where
    - ply_path:
        mesh
    - render_path:
        path to render
or if blender is installed locally, run
    $ blender --background --python scripts/blender/render_for_fid.py -- {mesh_path} {render_path} {theta}
Try to use absolute paths to the arguments cause, blender messes up the rendering path
"""

import bpy

# Get the Blender version
blender_version = bpy.app.version
# Check if it's Blender 3.4.x
assert (
    blender_version[0] == 3 and blender_version[1] == 4
), f"The Blender version is {blender_version[0]}.{blender_version[1]}.x, expected 3.4"
import bmesh
import sys
from typing import List, Tuple, Union, Dict
from dataclasses import dataclass
from mathutils import Vector, Matrix
from math import radians
import numpy as np
import math


@dataclass(frozen=True)
class Arguments:
    mesh_path: str
    render_path: str
    theta: float  # location of camera
    normalize: bool  # normalize mesh to fit in [-1, 1]
    bottom: bool  # render from bottom
    z_axis_up: bool  # z axis is up, else y axis, depends on obj
    flip: bool  # flip object upside down


def arg_parser() -> Arguments:
    # obtain parameters for the parametric model
    # should be in form of -- {npz_path}
    # $ blender --background --python scripts/blender/render_for_fid.py -- {mesh_path} {render_path} {theta}
    parameters = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    mesh_path = parameters[0]
    render_path = parameters[1]
    theta = float(parameters[2])

    # if normalize option exists
    normalize = "normalize" in parameters
    bottom = "bottom" in parameters
    z_axis_up = "z-axis-up" in parameters
    flip = "flip" in parameters

    args = Arguments(
        mesh_path=mesh_path,
        render_path=render_path,
        theta=theta,
        normalize=normalize,
        bottom=bottom,
        z_axis_up=z_axis_up,
        flip=flip,
    )
    return args


def enable_ply_addon():
    """
    Enable the PLY Import/Export add-on in Blender.
    """
    # Check if the PLY add-on is already enabled
    if not bpy.ops.preferences.addon_enable(module="io_mesh_ply"):
        # Enable the PLY add-on
        bpy.ops.preferences.addon_enable(module="io_mesh_ply")


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


def create_mesh_from_mesh_path(
    mesh_path: str,
    name: str,
    color: Tuple,
    normalize: bool,
    z_axis_up: bool,
    flip: bool = False,
):
    # Import the PLY file
    if mesh_path.endswith(".ply"):
        bpy.ops.import_mesh.ply(filepath=mesh_path)
    elif mesh_path.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=mesh_path)
    elif mesh_path.endswith(".stl"):
        bpy.ops.import_mesh.stl(filepath=mesh_path)
    else:
        raise ValueError(f"format {mesh_path} not allowed")

    # Select the imported object
    imported_object = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = imported_object
    imported_object.select_set(True)

    # Translate mesh to stick to ground (z=-1)
    # Enter edit mode to access vertices directly
    bpy.ops.object.mode_set(mode="EDIT")
    # Create a BMesh to work with the mesh data
    bm = bmesh.from_edit_mesh(imported_object.data)

    if normalize:
        # normalize within [-1, 1] range
        # Extract vertex coordinates
        vertices = np.array([v.co[:] for v in bm.verts])

        # Compute the bounding box dimensions
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)

        # Compute the center and scale
        center = Vector((min_coords + max_coords) / 2)
        scale = (max_coords - min_coords).max() / 2

        # Normalize coordinates
        for v in bm.verts:
            v.co = (v.co - center) / scale

    # Flip the object if requested
    if flip:
        if z_axis_up:
            # Flip around x-axis by negating z coordinates
            for v in bm.verts:
                v.co.z = -v.co.z
        else:
            # Flip around x-axis by negating y coordinates
            for v in bm.verts:
                v.co.y = -v.co.y

    # Translate all the coordinates to face down
    if z_axis_up:
        min_z = min(v.co.z for v in bm.verts)
        translation_z = -1 - min_z
        for v in bm.verts:
            v.co.z += translation_z
    else:
        min_y = min(v.co.y for v in bm.verts)
        translation_y = -1 - min_y
        for v in bm.verts:
            v.co.y += translation_y

    # Make it active
    bpy.context.view_layer.objects.active = imported_object

    # Ensure it's visible in the viewport and ready for rendering
    imported_object.select_set(True)

    # Optionally, apply transformations to the object (scale, rotation, location)
    imported_object.scale = (1.0, 1.0, 1.0)  # Adjust if necessary
    imported_object.location = (0.0, 0.0, 0.0)  # Adjust if necessary
    imported_object.data.name = name
    color_objects(color)


def create_ground_plane(size=2):
    """
    Create shadow catcher ground plane
    :param size:
    :return:
    """
    bpy.ops.mesh.primitive_plane_add(
        size=size, enter_editmode=False, align="WORLD", location=(0, 0, -1.0)
    )
    plane = bpy.context.active_object
    plane.name = "ShadowCatcherPlane"

    # Add a material to the plane
    material = bpy.data.materials.new(name="ShadowCatcherMaterial")
    material.use_nodes = True
    plane.data.materials.append(material)

    # Set the plane to be a shadow catcher
    plane.is_shadow_catcher = True


def create_light(loc, name: str):
    light_data = bpy.data.lights.new(name, type="POINT")
    light_obj = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = loc
    light_obj.data.energy = 100  # Increase the energy for brighter light


def set_env_objects(theta: float, bottom: bool):
    """
    :param theta: in radian
        Location of camera
    :param bottom
        Render from bottom
    :return:
    """
    # Clear existing mesh objects in the scene
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="MESH")
    bpy.ops.object.delete()

    # # Add a ground plane to the scene, scaled to fit the [-1, 1] range
    create_ground_plane(size=10)

    # Set the camera and lighting, assuming point clouds in range [-1, 1]
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_radius = 4.5
    cam_z = 2.5
    if bottom:
        cam_z *= -1
    cam_obj_loc = Vector(
        [cam_radius * math.cos(theta), cam_radius * math.sin(theta), cam_z]
    )
    cam_obj.location = cam_obj_loc
    object_center = Vector([0.0, 0.0, 0.0])
    direction = cam_obj_loc - object_center
    cam_obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    bpy.context.scene.camera = cam_obj

    light_x, light_y = 3.0, 3.0
    # set lighting
    create_light((light_x, light_y, 4.0), "Light1")
    create_light((-light_x, light_y, 4.0), "Light2")
    create_light((light_x, -light_y, 4.0), "Light3")
    create_light((-light_x, -light_y, 4.0), "Light4")


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

    # Render settings
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.filepath = args.render_path
    bpy.context.scene.render.resolution_x = 256
    bpy.context.scene.render.resolution_y = 256
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.cycles.samples = (
        128  # Set a lower number of samples for faster render
    )

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

    # Enable transparent background
    bpy.context.scene.render.film_transparent = True
    set_white_background()

    # Render!
    enable_ply_addon()
    set_env_objects(args.theta, bottom=args.bottom)
    color = (0.5, 0.5, 0.5, 0)
    create_mesh_from_mesh_path(
        args.mesh_path,
        name="Mesh",
        color=color,
        normalize=args.normalize,
        z_axis_up=args.z_axis_up,
        flip=args.flip,
    )

    # Render the scene
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
