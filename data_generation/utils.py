
import os
import re
import sys
import bpy
import cv2
import math
import copy
import json
import random
import numpy as np
from mathutils import Vector
from typing import Union, List

from data_generation.comfort_ball_config import *
from data_generation.constants import *


def get_rotation_path(var_obj, radius, angle_range, num_steps):
    start_angle, end_angle = angle_range
    angle_step = (end_angle - start_angle) / (num_steps-1)

    var_obj_path = []
    for step in range(num_steps):
        angle = math.radians(start_angle + step * angle_step)
        var_obj.location = (
            radius * math.cos(angle),
            radius * math.sin(angle),
            var_obj.location.z,
        )
        angle = step * angle_step - 180
        var_obj_path.append((list(var_obj.location.copy()), angle))

    return var_obj_path

def add_distractor_objects(ref_shape, var_shape, ref_color, var_color, num_distractors, ref_obj, var_obj, var_obj_path, ref_obj_path=None, distractors=None, comfort_ball=True):
    # Collect existing positions and dimensions from the ref_obj and var_obj
    existing_positions = [
        (ref_obj.location.copy(), ref_obj.dimensions.copy()),
        (var_obj.location.copy(), var_obj.dimensions.copy())
    ]

    # Add positions from var_obj's path to the list
    for entry in var_obj_path:
        if isinstance(entry, tuple):
            position = entry[0]  # Assuming the first element is the position if it's a tuple
        else:
            position = entry  # Directly use the entry if it's already a position
        existing_positions.append((position, var_obj.dimensions.copy()))

    if ref_obj_path is not None:
        for entry in ref_obj_path:
            existing_positions.append((entry, ref_obj.dimensions.copy()))

    if distractors is not None:
        for distractor in distractors:
            existing_positions.append((distractor['location'], distractor['dimensions']))
            distractor_obj = add_object(
                    SHAPE_DIR,
                    distractor['shape'],
                    distractor['size'],
                    distractor['position'],
                    comfort_ball=comfort_ball
                )
            distractor_color = distractor['color']
            distractor_mat = bpy.data.materials.new(name="DistractorMaterial")
            distractor_mat.use_nodes = True
            distractor_bsdf_node = distractor_mat.node_tree.nodes.get("Principled BSDF")
            distractor_bsdf_node.inputs["Base Color"].default_value = distractor_color
            distractor_obj.data.materials.append(distractor_mat)

    excluded_shapes = {ref_shape, var_shape}
    excluded_colors = {ref_color, var_color}

    # Now add distractors
    iter = num_distractors if distractors is None else num_distractors - len(distractors)
    for _ in range(iter):

        if comfort_ball:
            available_colors = [GREEN]
            available_shapes = [SPHERE]
    
        else:
            available_shapes = [BICYCLE_MOUNTAIN]
            available_colors = [CAR_BLUE]

        placed = False
        while not placed:
            distractor_shape = random.choice(available_shapes)

            if comfort_ball:
                # distractor_size = random.uniform(0.4, 0.7)
                distractor_size = 0.5
                distractor_position = (
                    random.uniform(0, 3.5),
                    random.uniform(-2.0, 2.0),
                    distractor_size,
                )
            else:
                # distractor_size = random.uniform(2.5,3)
                distractor_size = 2.0
                distractor_position = (
                    random.uniform(-2.5, 5.5),
                    random.uniform(-2.5, 2.5),
                    0.5,
                )

            # if distractor_shape not in AEROPLANES:
            #     continue

            # Check for collisions
            collision = False
            for position, dimensions in existing_positions:
                
                offset = dimensions[0] / 1.5 if comfort_ball else dimensions[0] / distractor_size / 2
                
                distractor_offset = distractor_size / 1.25 if not comfort_ball else distractor_size
                if (
                    abs(distractor_position[0] - position[0]) < (distractor_offset + offset) and
                    abs(distractor_position[1] - position[1]) < (distractor_offset + offset) and
                    abs(distractor_position[2] - position[2]) < (distractor_offset + offset)
                ):
                    collision = True
                    break
            
            if not collision:
                placed = True
                distractor_obj = add_object(
                    SHAPE_DIR,
                    distractor_shape,
                    distractor_size,
                    distractor_position,
                    comfort_ball=comfort_ball
                )
                distractor_color = random.choice(
                    available_colors
                )
                distractor_mat = bpy.data.materials.new(name="DistractorMaterial")
                distractor_mat.use_nodes = True
                distractor_bsdf_node = distractor_mat.node_tree.nodes.get("Principled BSDF")
                distractor_bsdf_node.inputs["Base Color"].default_value = distractor_color
                distractor_obj.data.materials.append(distractor_mat)
                existing_positions.append(
                    (distractor_obj.location, (distractor_size, distractor_size, distractor_size))
                )
                # Check if the object has multiple parts
                if distractor_obj.type == 'MESH':
                    for slot in distractor_obj.material_slots:
                        part_name = slot.name
                        # printing part names to not color the wheels or other parts
                        # print("Part name:", part_name, file=sys.stderr)
                        if "wheel" in part_name.lower():
                            slot.material.node_tree.nodes["Diffuse BSDF"].inputs['Color'].default_value = BLACK
                        else:
                            if slot.material.node_tree.nodes.get("Diffuse BSDF", None) is not None:
                                # print(slot.material.node_tree.nodes.keys(), file=sys.stderr)
                                slot.material.node_tree.nodes["Diffuse BSDF"].inputs['Color'].default_value = distractor_color


        if distractors is None:
            distractors = []
        distractors.append({'location': distractor_obj.location.copy(), 'dimensions': distractor_obj.dimensions.copy(), 'shape': distractor_shape, 'color': distractor_color, 'size': distractor_size, 'position': distractor_position})

        return distractors

def add_object(object_dir, name, scale, loc, theta=0, relation=None, comfort_ball=True):
    """
    Load an object from a file. We assume that in the directory object_dir, there
    is a file named "$name.blend" which contains a single object named "$name"
    that has unit size and is centered at the origin.

    - scale: scalar giving the size that the object should be in the scene
    - loc: tuple (x, y) giving the coordinates on the ground plane where the
      object should be placed.
    """
    # First figure out how many of this object are already in the scene so we can
    # give the new object a unique name
    count = 0
    for obj in bpy.data.objects:
        if obj.name.startswith(name):
            count += 1

    if comfort_ball:
        filename = os.path.join(object_dir, "%s.blend" % name, "Object", name)
        bpy.ops.wm.append(filename=filename)

        # Give it a new name to avoid conflicts
        new_name = "%s_%d" % (name, count)
        bpy.data.objects[name].name = new_name
    else:
        filename = os.path.join(object_dir, f"{name}.blend", "Object", name.split("_")[1])
        bpy.ops.wm.append(filename=filename)
        new_name = "%s_%d" % (name, count)
        bpy.data.objects[name.split("_")[1]].name = new_name

    # Set the new object as active, then rotate, scale, and translate it
    x, y, z = loc

    bpy.context.view_layer.objects.active = bpy.data.objects[new_name]
    bpy.context.object.rotation_euler[2] = theta
    bpy.ops.transform.resize(value=(scale, scale, scale))
    bpy.ops.transform.translate(value=(x, y, z))

    return bpy.context.object

def load_materials(material_dir):
    """
    Load materials from a directory. We assume that the directory contains .blend
    files with one material each. The file X.blend has a single NodeTree item named
    X; this NodeTree item must have a "Color" input that accepts an RGBA value.
    """
    for fn in os.listdir(material_dir):
        if not fn.endswith(".blend"):
            continue
        name = os.path.splitext(fn)[0]
        filepath = os.path.join(material_dir, fn, "NodeTree", name)
        bpy.ops.wm.append(filename=filepath)

def ensure_object_mode():
    if bpy.context.object:
        if bpy.context.object.mode != 'OBJECT':
            # Attempt to change to object mode
            bpy.ops.object.mode_set(mode='OBJECT')

def select_objects_for_join(objects):
    # Deselect all to start clean
    bpy.ops.object.select_all(action='DESELECT')

    # Ensure the context is correct by explicitly setting the active object
    for obj in objects:
        obj.select_set(True)
    
    # Set the first object as the active object
    bpy.context.view_layer.objects.active = objects[0] if objects else None

def join_objects(objects):
    # Select objects and ensure correct context
    select_objects_for_join(objects)

    # Perform the join operation
    bpy.ops.object.join()

def make_single_user(objects):
    for obj in objects:
        if obj.data and obj.data.users > 1:  # Check if data is shared
            obj.data = obj.data.copy()  # Make a single-user copy of the data

def create_and_setup_object(
        shape_dir, 
        shape, 
        size, 
        position, 
        material_name=None, 
        color=None, 
        relation=None, 
        comfort_ball=True
    ):
    # print("comfort_ball:", comfort_ball, file=sys.stderr)
    if not comfort_ball:
        if shape in SPECIAL:
            with bpy.data.libraries.load(os.path.join(shape_dir, f"{shape}.blend"), link=False) as (data_from, data_to):
                data_to.objects = data_from.objects
                # print("Objects loaded:", data_from.objects, file=sys.stderr)  # Debugging line
            # Link objects to the collection and apply transformations
        # Link objects to the collection
        # Ensure Blender is in object mode
            ensure_object_mode()
            make_single_user(data_to.objects)
            # Link objects to the scene
            for obj in data_to.objects:
                if obj is not None:
                    bpy.context.collection.objects.link(obj)

            # Try to join objects
            try:
                join_objects(data_to.objects)
                print("Objects successfully joined.")
            except RuntimeError as e:
                print(f"Failed to join objects: {e}")

            # Set position and optionally scale
            if data_to.objects:
                active_object = bpy.context.active_object
                bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
                active_object.location = position
                # active_object.rotation
                # addressee_obj.rotation_euler = tuple(np.array(addressee_rotation) / 180. * math.pi)
                # active_object.rotation_euler = tuple(np.array([0, 0, 90]) / 180. * math.pi)
                # if size:
                #     # active_object.scale = size
                #     bpy.ops.object.transform_apply(scale=True)
            
            return bpy.context.active_object

        else:
            print("Creating object", shape, file=sys.stderr)
            obj = add_object(shape_dir, shape, size, position, relation=relation, comfort_ball=comfort_ball)
            print(obj.scale, file=sys.stderr)
        
        # print(obj, file=sys.stderr)
        # Set up the object's material and transformations
        # Ensure the object is of a type that supports materials, like 'MESH'
        if obj and obj.type == 'MESH':
            mat = bpy.data.materials.new(name=material_name)
            mat.use_nodes = True
            bsdf_node = mat.node_tree.nodes.get("Principled BSDF")
            bsdf_node.inputs["Base Color"].default_value = color
            obj.data.materials.append(mat)

            bpy.context.view_layer.objects.active = obj
            # bpy.context.object.rotation_euler[2] = 0  # Adjust if you need rotation
            bpy.ops.transform.resize(value=(size, size, size))
            bpy.ops.transform.translate(value=position)
        # Calculate offset to ground and set position
            obj_dimensions = obj.dimensions
            z_offset = obj_dimensions.z / 2  # Assuming the origin is at the center
            new_position = (position[0], position[1], position[2] - z_offset)
            bpy.ops.transform.translate(value=new_position)
            

            for slot in obj.material_slots:
                part_name = slot.name
                if "wheel" in part_name.lower():
                    slot.material.node_tree.nodes["Diffuse BSDF"].inputs['Color'].default_value = (0, 0, 0, 1)  # Assuming BLACK is defined
                else:
                    diff_bsdf = slot.material.node_tree.nodes.get("Diffuse BSDF", None)
                    if diff_bsdf:
                        diff_bsdf.inputs['Color'].default_value = color
        else:
            print("The imported or created object is not compatible with materials.")
    else:
        obj = add_object(shape_dir, shape, size, position, relation=relation, comfort_ball=comfort_ball)
        mat = bpy.data.materials.new(name=material_name)
        mat.use_nodes = True
        bsdf_node = mat.node_tree.nodes.get("Principled BSDF")
        bsdf_node.inputs["Base Color"].default_value = color
        obj.data.materials.append(mat)

        # Check if the object has multiple parts
        if obj.type == 'MESH':
            for slot in obj.material_slots:
                part_name = slot.name
                # printing part names to not color the wheels or other parts
                # print("Part name:", part_name, file=sys.stderr)
                if "wheel" in part_name.lower():
                    slot.material.node_tree.nodes["Diffuse BSDF"].inputs['Color'].default_value = BLACK
                else:
                    if slot.material.node_tree.nodes.get("Diffuse BSDF", None) is not None:
                        # print(slot.material.node_tree.nodes.keys(), file=sys.stderr)
                        slot.material.node_tree.nodes["Diffuse BSDF"].inputs['Color'].default_value = color

    return obj

def render_scene_config(
        variation: str,
        relation: str,
        path_type: str,
        num_steps: int,
        save_path: str,
        
        ref_shape: str,
        ref_color: tuple,
        ref_size: float,
        ref_position: tuple,
        ref_rotation: Union[tuple, List[tuple]],

        var_shape: str,
        var_color: tuple,
        var_size: float,
        var_position: tuple,
        var_rotation: tuple,
        start_point: tuple = None,
        end_point: tuple = None,

        radius: int = None,
        angle_range: tuple = None,

        num_distractors: int = 0,
        cam_position: tuple = None,

        addressee: bool = False,
        addressee_shape: str = None,
        addressee_position: tuple = None,
        addressee_size: float = None,
        addressee_rotation: tuple = None,

        distractors: list = [],

        name: str = None,
        dataset_name: str = None,
        render_shadow: bool = True,
        cuda: bool = True
) -> dict:
    bpy.context.scene.render.engine = "CYCLES"
    if True:
        preferences = bpy.context.preferences.addons['cycles'].preferences
        preferences.get_devices()

        if cuda:
            for device in preferences.devices:
                device.use = True
            preferences.compute_device_type = 'CUDA'
            bpy.context.scene.cycles.device = 'GPU'
    bpy.ops.wm.open_mainfile(filepath=BASE_SCENE)
    bpy.context.scene.render.resolution_x = IM_SIZE
    bpy.context.scene.render.resolution_y = IM_SIZE
    bpy.context.scene.render.resolution_percentage = 100

    print(relation, file=sys.stderr)

    if dataset_name == "comfort_ball":
        comfort_ball = True
    else:
        comfort_ball = False


    # Disable shadow casting for all lights
    camera = bpy.data.objects['Camera']
    if comfort_ball:
        camera.location = (7.8342, 0, 3.6126)
    else:
        camera.location = (14.0, 0, 7.0)

    if cam_position is not None:
        camera.location = cam_position

    variation_name = variation
    SAVE_DIR = os.path.join(save_path, relation, variation_name)
    mapping = {}

    if dataset_name == "comfort_ball":
        comfort_ball = True
    else:
        comfort_ball = False

    if path_type == "rotate":
        # addressee object
        if addressee:
            addressee_obj = create_and_setup_object(
                SHAPE_DIR, 
                addressee_shape, 
                addressee_size, 
                addressee_position,
                comfort_ball=comfort_ball
            )
            
            if addressee_shape in SPECIAL:
                bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
                bpy.context.view_layer.objects.active = addressee_obj
                addressee_obj.rotation_euler = tuple(np.array(addressee_rotation) / 180. * math.pi)
            addressee_obj.scale = (addressee_size, addressee_size, addressee_size)
            bpy.ops.object.transform_apply(scale=True)

        # reference object
        ref_obj = create_and_setup_object(
            SHAPE_DIR, 
            ref_shape, 
            ref_size, 
            ref_position, 
            "RefMaterial", 
            ref_color, 
            relation=None,
            comfort_ball=comfort_ball
        )
        if ref_shape in SPECIAL:
            bpy.context.view_layer.objects.active = ref_obj
            ref_obj.rotation_euler = tuple(np.array(ref_rotation) / 180. * math.pi)
            if ref_shape == BED:
                ref_obj.scale = (ref_size, ref_size, ref_size*1/2)
            else:
                ref_obj.scale = (ref_size, ref_size, ref_size)
            bpy.ops.object.transform_apply(rotation=True, scale=True)
        else:
            ref_obj.rotation_euler = tuple(np.array(ref_rotation) / 180. * math.pi)

        # target object
        var_obj = create_and_setup_object(
            SHAPE_DIR, 
            var_shape, 
            var_size, 
            var_position, 
            "VarMaterial", 
            var_color, 
            relation=None,
            comfort_ball=comfort_ball
        )
        if var_shape in SPECIAL:
            var_obj.rotation_euler = tuple(np.array(var_rotation) / 180. * math.pi)
            bpy.context.view_layer.objects.active = var_obj
            var_obj.scale = (var_size, var_size, var_size)
            bpy.ops.object.transform_apply(scale=True)
        else:
            var_obj.rotation_euler = tuple(np.array(var_rotation) / 180. * math.pi)
        var_obj_path = get_rotation_path(var_obj, radius, angle_range, num_steps)

        added_distractors = add_distractor_objects(
            ref_shape, 
            var_shape, 
            ref_color, 
            var_color, 
            num_distractors, 
            ref_obj, 
            var_obj, 
            var_obj_path, 
            distractors=distractors, 
            comfort_ball=comfort_ball
        )

        for i, (pos, angle) in enumerate(var_obj_path):
            var_obj.location = pos
            output_image = os.path.join(SAVE_DIR, f'{i}.png')
            bpy.context.scene.render.filepath = output_image
            bpy.ops.render.render(write_still=True)
            print(f"Rendered image path: {output_image}", file=sys.stderr)
            # print(f"Angle: {angle}", file=sys.stderr)
            mapping[f'{i}.png'] = int(angle)
        
    return mapping, added_distractors