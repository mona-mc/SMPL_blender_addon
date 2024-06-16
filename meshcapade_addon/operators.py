import bpy
import os
import numpy as np
import json
import copy
import pickle
import math
import mathutils

from bpy.props import (
    BoolProperty,
    StringProperty,
    EnumProperty,
    IntProperty
)
from bpy_extras.io_utils import (
    ImportHelper,
    ExportHelper,
)
from .globals import (
    SMPLX_MODELFILE,
    SMPLH_MODELFILE,
    SUPR_MODELFILE,
    PATH,
    LEFT_HAND_RELAXED,
    RIGHT_HAND_RELAXED,
    MODEL_JOINT_NAMES,
    MODEL_BODY_JOINTS,
    MODEL_HAND_JOINTS,
)
from .blender import (
    set_pose_from_rodrigues,
    rodrigues_from_pose,
    setup_bone,
    correct_for_anim_format,
    key_all_pose_correctives,
)

from mathutils import Vector, Quaternion
from math import radians

class OP_LoadAvatar(bpy.types.Operator, ImportHelper):
    bl_idname = "object.load_avatar"
    bl_label = "Load Avatar"
    bl_description = ("Load a file that contains all the parameters for a SMPL family body")
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: StringProperty(
        default="*.npz",
        options={'HIDDEN'}
    )

    anim_format: EnumProperty(
        name="Format",
        items=(
            ("AMASS", "AMASS (Y-up)", ""),
            ("blender", "Blender (Z-up)", ""),
        ),
    )

    SMPL_version: EnumProperty(
        name="SMPL Version",
        items=(
            #("guess", "Guess", ""),
            ("SMPLX", "SMPL-X", ""),
            ("SMPLH", "SMPL-H", ""),
            ("SUPR", "SUPR", ""),
        ),
    )

    gender_override: EnumProperty(
        name="Gender Override",
        items=(
            ("disabled", "Disabled", ""),
            ("female", "Female", ""),
            ("male", "Male", ""),
            ("neutral", "Neutral", ""),
        ),
    )

    hand_pose: EnumProperty(
        name="Hand Pose Override",
        items=[
            ("disabled", "Disabled", ""),
            ("relaxed", "Relaxed", ""),
            ("flat", "Flat", ""),
        ]
    )

    keyframe_corrective_pose_weights: BoolProperty(
        name="Use keyframed corrective pose weights",
        description="Keyframe the weights of the corrective pose shapes for each frame. This increases animation load time and slows down editor real-time playback.",
        default=False
    )

    target_framerate: IntProperty(
        name="Target framerate [fps]",
        description="Target framerate for animation in frames-per-second. Lower values will speed up import time.",
        default=30,
        min = 1,
        max = 120
    )

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        target_framerate = self.target_framerate

        # Load .npz file
        print("Loading: " + self.filepath)
        with np.load(self.filepath) as data:
            # Check for valid AMASS file
            error_string = ""
            if "trans" not in data:
                error_string += "\n -trans"

            if "gender" not in data:
                error_string += "\n -gender"

            if "mocap_frame_rate" in data:
                fps_key = "mocap_frame_rate"
            elif "mocap_framerate" in data:
                fps_key = "mocap_framerate"
            elif "fps" in data:
                fps_key = "fps"

            if not fps_key:
                error_string += "\n -fps or mocap_framerate or mocap_frame_rate"
            else: 
                fps = int(data[fps_key])

            if "betas" not in data:
                error_string += "\n -betas"

            if "poses" not in data:
                error_string += "\n -poses"
        
            if error_string:
                self.report({"ERROR"}, "the following keys are missing from the .npz: " + error_string)
                return {"CANCELLED"}

            trans = data["trans"]

            if self.gender_override != "disabled":
                gender = self.gender_override
            else:
                gender = str(data["gender"])

            betas = data["betas"]
            poses = data["poses"]

            if fps < target_framerate:
                self.report({"ERROR"}, f"Mocap framerate ({fps}) below target framerate ({target_framerate})")
                return {"CANCELLED"}
            
            SMPL_version = self.SMPL_version

        if context.active_object is not None:
            bpy.ops.object.mode_set(mode='OBJECT')

        print ("gender: " + gender)
        print ("fps: " + str(fps))

        # Add gender specific model
        context.window_manager.smpl_tool.gender = gender
        context.window_manager.smpl_tool.SMPL_version = SMPL_version

        if self.hand_pose != 'disabled':
            context.window_manager.smpl_tool.hand_pose = self.hand_pose

        bpy.ops.scene.create_avatar()

        obj = context.view_layer.objects.active
        armature = obj.parent

        # Append animation name to armature name
        armature.name = armature.name + "_" + os.path.basename(self.filepath).replace(".npz", "")

        context.scene.render.fps = target_framerate
        context.scene.frame_start = 1

        # Set shape and update joint locations
        # TODO once we have the regressor for SMPLH, we can remove this condition
        if SMPL_version != 'SMPLH':
            bpy.ops.object.mode_set(mode='OBJECT')
            for index, beta in enumerate(betas):
                key_block_name = f"Shape{index:03}"

                if key_block_name in obj.data.shape_keys.key_blocks:
                    obj.data.shape_keys.key_blocks[key_block_name].value = beta
                else:
                    print(f"ERROR: No key block for: {key_block_name}")

        bpy.ops.object.update_joint_locations('EXEC_DEFAULT')

        # Keyframe poses
        step_size = int(fps / target_framerate)

        num_frames = trans.shape[0]
        num_keyframes = int(num_frames / step_size)

        if self.keyframe_corrective_pose_weights:
            print(f"Adding pose keyframes with keyframed corrective pose weights: {num_keyframes}")
        else:
            print(f"Adding pose keyframes: {num_keyframes}")

        # Set end frame if we don't have any previous animations in the scene
        if (len(bpy.data.actions) == 0) or (num_keyframes > context.scene.frame_end):
            context.scene.frame_end = num_keyframes

        joints_to_use = MODEL_JOINT_NAMES[SMPL_version].value

        # override hand pose if it's selected
        # don't pose the hands every frame if we're overriding it
        if self.hand_pose != 'disabled':
            bpy.ops.object.set_hand_pose('EXEC_DEFAULT')

            if SMPL_version == 'SMPLH':
                joints_to_use = joints_to_use[:22]
            else:
                joints_to_use = joints_to_use[:25]

        for index, frame in enumerate(range(0, num_frames, step_size)):
            if (index % 100) == 0:
                print(f"  {index}/{num_keyframes}")
            current_pose = poses[frame].reshape(-1, 3)
            current_trans = trans[frame]

            for bone_index, bone_name in enumerate(joints_to_use):
                if bone_name == "pelvis":
                    # there's a scale mismatch somewhere and the global translation is off by a factor of 100
                    armature.pose.bones[bone_name].location = current_trans*100
                    armature.pose.bones[bone_name].keyframe_insert('location', frame=index+1)

                # Keyframe bone rotation
                set_pose_from_rodrigues(armature, bone_name, current_pose[bone_index], frame=index+1)

            if self.keyframe_corrective_pose_weights:
                # Calculate corrective poseshape weights for current pose and keyframe them.
                # Note: This significantly increases animation load time and also reduces real-time playback speed in Blender viewport.
                bpy.ops.object.set_pose_correctives('EXEC_DEFAULT')
                key_all_pose_correctives(obj=obj, index=index+1)

        print(f"  {num_keyframes}/{num_keyframes}")
        context.scene.frame_set(1)

        correct_for_anim_format(self.anim_format, armature)
        bpy.ops.object.snap_to_ground_plane('EXEC_DEFAULT')
        armature.keyframe_insert(data_path="location", frame=bpy.data.scenes[0].frame_current)

        return {'FINISHED'}


class OP_CreateAvatar(bpy.types.Operator):
    bl_idname = "scene.create_avatar"
    bl_label = "Create Avatar"
    bl_description = ("Create a SMPL family avatar at the scene origin.  \nnote: SMPLH is missing the joint regressor so you can't modify it's shape")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if in Object Mode
            return (context.active_object is None) or (context.active_object.mode == 'OBJECT')
        except: return False

    def execute(self, context):
        gender = context.window_manager.smpl_tool.gender
        SMPL_version = context.window_manager.smpl_tool.SMPL_version

        if SMPL_version == 'SMPLX':
            # Use 300 shape model by default if available
            model_file = SMPLX_MODELFILE

        elif SMPL_version == 'SUPR':
            model_file = SUPR_MODELFILE

        # for now, the SMPLH option has been removed from the properties because we don't have regressors for it, 
        # so afm and a bunch of other stuff doesn't work
        elif SMPL_version == "SMPLH":
            model_file = SMPLH_MODELFILE

        else:
            model_file = "error bad SMPL_version"

        objects_path = os.path.join(PATH, "data", model_file, "Object")
        object_name = SMPL_version + "-mesh-" + gender

        bpy.ops.wm.append(filename=object_name, directory=str(objects_path))

        # Select imported mesh
        object_name = context.selected_objects[0].name
        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = bpy.data.objects[object_name]
        bpy.data.objects[object_name].select_set(True)

        #define custom properties on the avatar itself to store this kind of data so we can use it whenever we need to
        bpy.context.object['gender'] = gender
        bpy.context.object['SMPL_version'] = SMPL_version

        # add a texture and change the texture option based on the gender
        # male texture if it's a male, female texture if it's female or neutral
        if gender == 'male':
            context.window_manager.smpl_tool.texture = "m"
        else:
            context.window_manager.smpl_tool.texture = "f"

        bpy.ops.object.set_texture()
        bpy.ops.object.reset_body_shape('EXEC_DEFAULT')

        return {'FINISHED'}


class OP_SetTexture(bpy.types.Operator):
    bl_idname = "object.set_texture"
    bl_label = "Set"
    bl_description = ("Set selected texture")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if in active object is mesh
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context):
        selection = context.window_manager.smpl_tool.texture
        obj = bpy.context.object

        # if there's no material, add one
        if len(obj.data.materials) == 0:
            # the incoming name of the selected object is in the format of "SUPR-mesh-male"
            # this line turns "SUPR-mesh-male" into "SUPR-male", which is the naming format of the materials
            split_name = obj.name.split("-")
            new_material_name = split_name[0] + "-" + split_name[2]
            material = bpy.data.materials.new(name=new_material_name)
            bpy.context.object.data.materials.append(material)

        else: 
            material = obj.data.materials[0]

        # Enable the use of nodes for the material
        material.use_nodes = True
        node_tree = material.node_tree
        nodes = node_tree.nodes

        # if they selected the male or female texture, we add the normal map and roughness map as well
        if selection in ('m', 'f'):
            # Set the path to the texture files
            albedo_map_path = os.path.join(PATH, "data", selection + "_albedo.png")
            normal_map_path = os.path.join(PATH, "data", selection + "_normal.png")
            roughness_map_path = os.path.join(PATH, "data", selection + "_roughness.png")
            ao_map_path = os.path.join(PATH, "data", "ao.png")
            thickness_map_path = os.path.join(PATH, "data", "thickness.png")

            # Clear default nodes
            for node in nodes:
                nodes.remove(node)

            # Create a new Principled BSDF node
            principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
            principled_node.location = 0, 0

            # Add a texture node for the albedo map
            albedo_map_node = nodes.new(type="ShaderNodeTexImage")
            albedo_map_node.location = -400, 200
            albedo_map_node.image = bpy.data.images.load(albedo_map_path)
            albedo_map_node.image.colorspace_settings.name = 'sRGB'
            node_tree.links.new(albedo_map_node.outputs["Color"], principled_node.inputs["Base Color"])

            # Add a texture node for the roughness map
            roughness_map_node = nodes.new(type="ShaderNodeTexImage")
            roughness_map_node.location = -400, -200
            roughness_map_node.image = bpy.data.images.load(roughness_map_path)
            roughness_map_node.image.colorspace_settings.name = 'Non-Color'
            node_tree.links.new(roughness_map_node.outputs["Color"], principled_node.inputs["Roughness"])

            # Add a texture node for the normal map
            normal_map_node = nodes.new(type="ShaderNodeTexImage")
            normal_map_node.location = -800, -600
            normal_map_node.image = bpy.data.images.load(normal_map_path)
            normal_map_node.image.colorspace_settings.name = 'Non-Color'
            noamel_map_adjustment = material.node_tree.nodes.new('ShaderNodeNormalMap')
            noamel_map_adjustment.location = -400, -600
            node_tree.links.new(normal_map_node.outputs["Color"], noamel_map_adjustment.inputs["Color"])
            node_tree.links.new(noamel_map_adjustment.outputs["Normal"], principled_node.inputs["Normal"])

            '''
            # TODO add AO
            # Add a texture node for the ambient occlusion map
            ambient_occlusion_node = nodes.new(type="ShaderNodeTexImage")
            ambient_occlusion_node.location = -400, 200
            ambient_occlusion_node.image = bpy.data.images.load(ao_map_path)
            ambient_occlusion_node.image.colorspace_settings.name = 'Non-Color'
            node_tree.links.new(ambient_occlusion_node.outputs["Color"], principled_node.inputs["Ambient Occlusion"])
            #'''
            
            '''
            # TODO add thickness
            # Add a texture node for the thickness map
            thickness_map_node = nodes.new(type="ShaderNodeTexImage")
            thickness_map_node.location = -400, -200
            thickness_map_node.image = bpy.data.images.load(thickness_map_path)
            thickness_map_node.image.colorspace_settings.name = 'Non-Color'
            node_tree.links.new(thickness_map_node.outputs["Color"], principled_node.inputs["Transmission"])
            #'''

            # Set the subsurface properties
            principled_node.inputs["Subsurface"].default_value = 0.001
            principled_node.inputs["Subsurface Color"].default_value = (1, 0, 0, 1)

            # Link the output of the Principled BSDF node to the material output
            output_node = nodes.new(type="ShaderNodeOutputMaterial")
            output_node.location = 400, 0
            node_tree.links.new(principled_node.outputs["BSDF"], output_node.inputs["Surface"])

        else:
            texture_name = context.window_manager.smpl_tool.texture

            if (len(obj.data.materials) == 0) or (obj.data.materials[0] is None):
                self.report({'WARNING'}, "Selected mesh has no material: %s" % obj.name)
                return {'CANCELLED'}

            # Find texture node
            node_texture = None
            for node in nodes:
                if node.type == 'TEX_IMAGE':
                    node_texture = node
                    break

            # Find shader node
            node_shader = None
            for node in nodes:
                if node.type.startswith('BSDF'):
                    node_shader = node
                    break

            if texture_name == 'NONE':
                # Unlink texture node
                if node_texture is not None:
                    for link in node_texture.outputs[0].links:
                        node_tree.links.remove(link)

                    nodes.remove(node_texture)

                    # 3D Viewport still shows previous texture when texture link is removed via script.
                    # As a workaround we trigger desired viewport update by setting color value.
                    node_shader.inputs[0].default_value = node_shader.inputs[0].default_value
            else:
                if node_texture is None:
                    node_texture = nodes.new(type="ShaderNodeTexImage")

                if (texture_name == 'UV_GRID') or (texture_name == 'COLOR_GRID'):
                    if texture_name not in bpy.data.images:
                        bpy.ops.image.new(name=texture_name, generated_type=texture_name)
                    image = bpy.data.images[texture_name]
                else:
                    if texture_name not in bpy.data.images:
                        texture_path = os.path.join(PATH, "data", texture_name)
                        image = bpy.data.images.load(texture_path)
                    else:
                        image = bpy.data.images[texture_name]

                node_texture.image = image

                # Link texture node to shader node if not already linked
                if len(node_texture.outputs[0].links) == 0:
                    node_tree.links.new(node_texture.outputs[0], node_shader.inputs[0])

        # Switch viewport shading to Material Preview to show texture
        if bpy.context.space_data:
            if bpy.context.space_data.type == 'VIEW_3D':
                bpy.context.space_data.shading.type = 'MATERIAL'

        return {'FINISHED'}


class OP_MeasurementsToShape(bpy.types.Operator):
    bl_idname = "object.measurements_to_shape"
    bl_label = "Measurements To Shape"
    bl_description = ("Calculate and set shape parameters for specified measurements")
    bl_options = {'REGISTER', 'UNDO'}

    betas_regressor_female = None
    betas_regressor_male = None
    betas_regressor_neutral = None

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')

        if self.betas_regressor_female is None:
            regressor_path = os.path.join(PATH, "data", "measurements_to_betas_female.json")
            with open(regressor_path) as f:
                data = json.load(f)
                self.betas_regressor_female = (
                    np.asarray(data["A"]).reshape(-1, 2), 
                    np.asarray(data["B"]).reshape(-1, 1)
                )

        if self.betas_regressor_male is None:
            regressor_path = os.path.join(PATH, "data", "measurements_to_betas_male.json")
            with open(regressor_path) as f:
                data = json.load(f)
                self.betas_regressor_male = (
                    np.asarray(data["A"]).reshape(-1, 2),
                    np.asarray(data["B"]).reshape(-1, 1)
                )

        if self.betas_regressor_neutral is None:
            regressor_path = os.path.join(PATH, "data", "measurements_to_betas_neutral.json")
            with open(regressor_path) as f:
                data = json.load(f)
                self.betas_regressor_neutral = (
                    np.asarray(data["A"]).reshape(-1, 2), 
                    np.asarray(data["B"]).reshape(-1, 1)
                )

        if "female" in obj.name.lower():
            (A, B) = self.betas_regressor_female
        elif "male" in obj.name.lower():
            (A, B) = self.betas_regressor_male
        elif "neutral" in obj.name.lower():
            (A, B) = self.betas_regressor_neutral
        else:
            self.report({"ERROR"}, f"Cannot derive gender from mesh object name: {obj.name}")
            return {"CANCELLED"}

        # Calculate beta values from measurements
        height_cm = context.window_manager.smpl_tool.height
        weight_kg = context.window_manager.smpl_tool.weight

        v_root = pow(weight_kg, 1.0/3.0)
        measurements = np.asarray([[height_cm], [v_root]])
        betas = A @ measurements + B

        num_betas = betas.shape[0]
        for i in range(num_betas):
            name = f"Shape{i:03d}"
            key_block = obj.data.shape_keys.key_blocks[name]
            value = betas[i, 0]

            # Adjust key block min/max range to value
            if value < key_block.slider_min:
                key_block.slider_min = value
            elif value > key_block.slider_max:
                key_block.slider_max = value

            key_block.value = value

        bpy.ops.object.update_joint_locations('EXEC_DEFAULT')

        return {'FINISHED'}


class OP_RandomBodyShape(bpy.types.Operator):
    bl_idname = "object.random_body_shape"
    bl_label = "Random Body Shape"
    bl_description = ("Sets all shape blendshape keys to a random value")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')

        for i in range(0, 10):
            key_name = f"Shape{'%0.3d' % i}"
            key_block = obj.data.shape_keys.key_blocks.get(key_name)
            beta = np.random.normal(0.0, 1.0) * .75 * context.window_manager.smpl_tool.random_body_mult
            key_block.value = beta

        bpy.ops.object.update_joint_locations('EXEC_DEFAULT')

        context.window_manager.smpl_tool.alert = True

        return {'FINISHED'}
    
    def draw(self, context):
        context.window_manager.smpl_tool.alert = True
    

class OP_RandomFaceShape(bpy.types.Operator):
    bl_idname = "object.random_face_shape"
    bl_label = "Random Face Shape"
    bl_description = ("Sets all shape blendshape keys to a random value")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')

        for i in range(10,299):
            key_name = f"Shape{'%0.3d' % i}"
            key_block = obj.data.shape_keys.key_blocks.get(key_name)
            beta = np.random.normal(0.0, 1.0) * .75 * context.window_manager.smpl_tool.random_face_mult
            key_block.value = beta

        bpy.ops.object.update_joint_locations('EXEC_DEFAULT')
        
        return {'FINISHED'}


class OP_ResetBodyShape(bpy.types.Operator):
    bl_idname = "object.reset_body_shape"
    bl_label = "Reset Body Shape"
    bl_description = ("Resets all blendshape keys for shape")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')

        gender = bpy.context.object['gender']

        # These are the default height and weight values for the three different templates.
        # There is some rounding error that makes applying these numbers not give you exactly 0.0 on the shape keys,
        # but they're so close that you can't tell unless if you look at the numbers. 
        # There are cases where you really want all the shape keys to be 0.0, 
        # so my workaround here is to first apply these height and weight values, then afterwards, manually 0 out the shape keys.
        # This results in a mismatch between the shape keys and the height and weight sliders, 
        # but the shape keys at 0.0 is what's actually being represented in the model, and when you slide the sliders, you can't tell.
        # I think this is the right way to do it.  
        if gender == "male":
            context.window_manager.smpl_tool.height = 178.40813305675982
            context.window_manager.smpl_tool.weight = 84.48267403991704
        elif gender == "female":
            context.window_manager.smpl_tool.height = 165.58187348544598
            context.window_manager.smpl_tool.weight = 69.80320278887571
        elif gender == "neutral":
            context.window_manager.smpl_tool.height = 172.05153398364783
            context.window_manager.smpl_tool.weight = 77.51340327590397

        # this is the step that manually 0's out the shape keys
        for i in range(0,10):
            key_name = f"Shape{'%0.3d' % i}"
            key_block = obj.data.shape_keys.key_blocks.get(key_name)
            key_block.value = 0

        bpy.ops.object.update_joint_locations('EXEC_DEFAULT')
        context.window_manager.smpl_tool.alert = False

        return {'FINISHED'}
    
    def draw(self, context):
        context.window_manager.smpl_tool.alert = False


class OP_ResetFaceShape(bpy.types.Operator):
    bl_idname = "object.reset_face_shape"
    bl_label = "Reset Face Shape"
    bl_description = ("Resets all blendshape keys for shape")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return context.object.type == 'MESH'
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')

        for i in range(10,299):
            key_name = f"Shape{'%0.3d' % i}"
            key_block = obj.data.shape_keys.key_blocks.get(key_name)
            key_block.value = 0

        bpy.ops.object.update_joint_locations('EXEC_DEFAULT')

        return {'FINISHED'}


class OP_RandomExpressionShape(bpy.types.Operator):
    bl_idname = "object.random_expression_shape"
    bl_label = "Random Facial Expression"
    bl_description = ("Sets all face expression blendshape keys to a random value")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return ((context.object.type == 'MESH') and (bpy.context.object['SMPL_version'] != "SMPLH"))
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')

        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith('Exp'):
                key_block.value = np.random.uniform(-1.5, 1.5)

        return {'FINISHED'}


class OP_ResetExpressionShape(bpy.types.Operator):
    bl_idname = "object.reset_expression_shape"
    bl_label = "Reset"
    bl_description = ("Resets all blendshape keys for face expression")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return ((context.object.type == 'MESH') and (bpy.context.object['SMPL_version'] != "SMPLH"))
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')

        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith('Exp'):
                key_block.value = 0.0

        return {'FINISHED'}


class OP_SnapToGroundPlane(bpy.types.Operator):
    bl_idname = "object.snap_to_ground_plane"
    bl_label = "Snap To Ground Plane"
    bl_description = ("Snaps mesh to the XY ground plane")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh or armature is active object
            return context.object.type in ('MESH', 'ARMATURE')
        except: return False

    def execute(self, context):
        bpy.ops.object.mode_set(mode='OBJECT')

        obj = bpy.context.object
        if obj.type == 'ARMATURE':
            armature = obj
            obj = bpy.context.object.children[0]
        else:
            armature = obj.parent

        # Get vertices with applied skin modifier in object coordinates
        depsgraph = context.evaluated_depsgraph_get()
        object_eval = obj.evaluated_get(depsgraph)
        mesh_from_eval = object_eval.to_mesh()

        # Get vertices in world coordinates
        matrix_world = obj.matrix_world
        vertices_world = [matrix_world @ vertex.co for vertex in mesh_from_eval.vertices]
        z_min = (min(vertices_world, key=lambda item: item.z)).z
        object_eval.to_mesh_clear() # Remove temporary mesh

        # Adjust height of armature so that lowest vertex is on ground plane.
        # Do not apply new armature location transform so that we are later able to show loaded poses at their desired height.
        armature.location.z = armature.location.z - z_min

        return {'FINISHED'}


class OP_UpdateJointLocations(bpy.types.Operator):
    bl_idname = "object.update_joint_locations"
    bl_label = "Update Joint Locations"
    bl_description = ("You only need to click this button if you change the shape keys from the object data tab (not using the plugin)")
    bl_options = {'REGISTER', 'UNDO'}

    j_regressor_female = {}
    j_regressor_male = {}
    j_regressor_neutral = {}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE'))
        except Exception: 
            return False

    def load_regressor(self, gender, betas, SMPL_version):
        # TODO recreate the SUPR joint regressor so that it doesn't include the 100 expression shape keys.  There are two `if SMPL_version == 'supr'` that we will be able to get rid of as a result
        if betas == 10:
            suffix = ""
        elif betas == 300:
            suffix = "_300"
        elif betas == 400:
            suffix = "_400"
        else:
            print(f"ERROR: No betas-to-joints regressor for desired beta shapes [{betas}]")
            return (None, None)

        regressor_path = os.path.join(PATH, "data", f"{SMPL_version}_betas_to_joints_{gender}{suffix}.json")
        with open(regressor_path) as f:
            data = json.load(f)
            return (np.asarray(data["betasJ_regr"]), np.asarray(data["template_J"]))

    def execute(self, context):
        obj = bpy.context.object
        bpy.ops.object.mode_set(mode='OBJECT')

        SMPL_version = bpy.context.object['SMPL_version']

        # SMPLH is missing the joint regressor so we just want to exit
        if SMPL_version == 'SMPLH':
            return {'CANCELLED'}

        gender = bpy.context.object['gender']
        joint_names = MODEL_JOINT_NAMES[SMPL_version].value
        num_joints = len(joint_names)

        # Get beta shapes
        betas = []
        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Shape"):
                if not key_block.mute:
                    betas.append(key_block.value)
                else: #if the value is unchecked in the gui, it's regarded as zero
                    betas.append(0.0)
        num_betas = len(betas)
        betas = np.array(betas)

        # "Cache regressor files"
        # I think whoever wrote this thought they were caching everything, but they're really just doing it every time this is used, which is bad.
        # TODO we need to actually cache all the files
        self.j_regressor = { num_betas: None }
        self.j_regressor[num_betas] = self.load_regressor(gender, num_betas, SMPL_version.lower())
        (betas_to_joints, template_j) = self.j_regressor[num_betas]
        joint_locations = betas_to_joints @ betas + template_j

        # Set new bone joint locations
        armature = obj.parent
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='EDIT')

        armature.data.edit_bones[0].tail = armature.data.edit_bones[0].head - Vector((0.0,0.0,10.0))  # make the root joint stick out backwards so that it's not blocking anything.  purely visual

        for index in range(num_joints):
            bone = armature.data.edit_bones[joint_names[index]]
            setup_bone(bone, SMPL_version)

            # Convert joint locations to Blender joint locations
            joint_location = joint_locations[index]

            if SMPL_version in ['SMPLX', 'SUPR']:
                bone_start = Vector((joint_location[0]*100, joint_location[1]*100, joint_location[2]*100))

            bone.translate(bone_start)

            # orient the joints
            if len(bone.children) == 1:     # if there's only one child, then just set the tail to the head of the child
                bone.tail = bone.children[0].head
            elif len(bone.children) == 0:   # if there's no child:
                if "jaw" in bone.name:      # jaw bone sticks forward and down 
                    bone.tail = bone.head + Vector((0.0,-5.0,10.0))
                elif "eye" in bone.name:    # eye bones point forward
                    bone.tail = bone.head + Vector((0.0,0,10.0))
                else:                       # just stick out in the direction the parent bone is sticking out
                    bone.tail = bone.parent.tail - bone.parent.head + bone.head
            elif len(bone.children) == 5:   # if it's the hands, set it to point at the middle finger
                for c in bone.children:
                    if "middle" in c.name:
                        middle_finger = c.head
                
                bone.tail = middle_finger

        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.objects.active = obj

        return {'FINISHED'}


class OP_CalculatePoseCorrectives(bpy.types.Operator):
    bl_idname = "object.set_pose_correctives"
    bl_label = "Calculate Pose Correctives"
    bl_description = ("Computes pose correctives for the current frame")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object and parent is armature
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False

    # https://github.com/gulvarol/surreal/blob/master/datageneration/main_part1.py
    # Computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
    def rodrigues_to_mat(self, rotvec):
        theta = np.linalg.norm(rotvec)
        r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
        cost = np.cos(theta)
        mat = np.asarray([[0, -r[2], r[1]],
                        [r[2], 0, -r[0]],
                        [-r[1], r[0], 0]], dtype=object)
        return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)
    
    def rodrigues_to_quat(self, rotvec):
        theta = np.linalg.norm(rotvec)
        r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
        return(Quaternion(r, theta))

    # https://github.com/gulvarol/surreal/blob/master/datageneration/main_part1.py
    # Calculate weights of pose corrective blendshapes
    # Input is pose of all 55 joints, output is weights for all joints except pelvis
    def rodrigues_to_posecorrective_weight(self, context, pose):
        SMPL_version = bpy.context.object['SMPL_version']
        joint_names = MODEL_JOINT_NAMES[SMPL_version].value
        num_joints = len(joint_names)
        
        if SMPL_version in ('SMPLX', 'SMPLH'):
            rod_rots = np.asarray(pose).reshape(num_joints, 3)
            mat_rots = [self.rodrigues_to_mat(rod_rot) for rod_rot in rod_rots]
            bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel() for mat_rot in mat_rots[1:]])
            return(bshapes)

        elif SMPL_version == 'SUPR':
            rod_rots = np.asarray(pose).reshape(num_joints, 3)
            quats = [self.rodrigues_to_quat(rod_rot) for rod_rot in rod_rots]
            for q in quats:
                qcopy = copy.deepcopy(q)
                q.w = qcopy.x
                q.x = qcopy.y
                q.y = qcopy.z
                q.z = qcopy.w - 1 # same as (1 - qcopy.w) * -1
            bshapes = np.concatenate([quat for quat in quats[0:]])

            return(bshapes)
            
        else:
            return("error")


    def execute(self, context):
        obj = bpy.context.object
        SMPL_version = bpy.context.object['SMPL_version']
        joint_names = MODEL_JOINT_NAMES[SMPL_version].value
        num_joints = len(joint_names)

        # Get armature pose in rodrigues representation
        if obj.type == 'ARMATURE':
            armature = obj
            obj = bpy.context.object.children[0]
        else:
            armature = obj.parent

        pose = [0.0] * (num_joints * 3)

        for index in range(num_joints):
            joint_name = joint_names[index]
            joint_pose = rodrigues_from_pose(armature, joint_name)
            pose[index*3 + 0] = joint_pose[0]
            pose[index*3 + 1] = joint_pose[1]
            pose[index*3 + 2] = joint_pose[2]

        poseweights = self.rodrigues_to_posecorrective_weight(context, pose)

        # TODO for the time being, the SMPLX pose correctives only go to 0-206.  
        # It should be 0-485, but we're not sure why the fingers aren't being written out of the blender-worker  
        if SMPL_version in ['SMPLH', 'SMPLX']:
            poseweights_to_use = poseweights[0:207]
        else:
            poseweights_to_use = poseweights

        # Set weights for pose corrective shape keys
        for index, weight in enumerate(poseweights_to_use):
            obj.data.shape_keys.key_blocks["Pose%03d" % index].value = weight

        return {'FINISHED'}


class OP_CalculatePoseCorrectivesForSequence(bpy.types.Operator):
    bl_idname = "object.set_pose_correctives_for_sequence"
    bl_label = "Calculate Pose Correctives for Entire Sequence"
    bl_description = ("Computes pose correctives for the current time slider range")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object and parent is armature
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False
    
    def execute(self, context):
         # Get the start and end frames from the scene's render settings
        start_frame = bpy.context.scene.frame_start
        end_frame = bpy.context.scene.frame_end

        # Get the object you want to animate
        obj = bpy.context.object

        # Iterate over each frame
        for frame in range(start_frame, end_frame + 1):
            # Set the current frame
            bpy.context.scene.frame_set(frame)
            
            # Update pose shapes
            bpy.ops.object.set_pose_correctives('EXEC_DEFAULT')
            
            # Insert a keyframe
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_euler", frame=frame)
            obj.keyframe_insert(data_path="scale", frame=frame)

        return {"FINISHED"}


class OP_ZeroOutPoseCorrectives(bpy.types.Operator):
    bl_idname = "object.zero_out_pose_correctives"
    bl_label = "Zero Out Pose Correctives"
    bl_description = ("Removes pose correctives for current frame")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object and parent is armature
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        obj = bpy.context.object

        if obj.type == 'ARMATURE':
            obj = bpy.context.object.children[0]

        for key_block in obj.data.shape_keys.key_blocks:
            if key_block.name.startswith("Pose"):
                key_block.value = 0.0

        return {'FINISHED'}


class OP_SetHandpose(bpy.types.Operator):
    bl_idname = "object.set_hand_pose"
    bl_label = "Set"
    bl_description = ("Set selected hand pose")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh or armature is active object
            return (
                ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or
                (context.object.type == 'ARMATURE')
            )
        except Exception:
            return False

    def execute(self, context):
        obj = bpy.context.object
        if obj.type == 'MESH':
            armature = obj.parent
        else:
            armature = obj

        hand_pose_name = context.window_manager.smpl_tool.hand_pose

        # flat is just an array of 45 0's so we don't want to load it from a file 
        if hand_pose_name == 'flat':
            left_hand_pose = np.zeros(45)
            right_hand_pose = np.zeros(45)
        
        elif hand_pose_name == 'relaxed':
            left_hand_pose = LEFT_HAND_RELAXED
            right_hand_pose = RIGHT_HAND_RELAXED

        else:
            self.report({"ERROR"}, f"Desired hand pose not existing: {hand_pose_name}")
            return {"CANCELLED"}
        
        hand_pose = np.concatenate((left_hand_pose, right_hand_pose)).reshape(-1, 3)

        SMPL_version = bpy.context.object['SMPL_version']
        joint_names = MODEL_JOINT_NAMES[SMPL_version].value
        num_body_joints = MODEL_BODY_JOINTS[SMPL_version].value
        num_hand_joints = MODEL_HAND_JOINTS[SMPL_version].value

        hand_joint_start_index = 1 + num_body_joints

        # SMPLH doesn't have the jaw and eyes, so we leave it alone
        # we +3 for SUPR and SMPLX because they do
        if (SMPL_version == "SUPR" or SMPL_version == "SMPLX"):
            hand_joint_start_index += 3

        for index in range(2 * num_hand_joints):
            pose_rodrigues = hand_pose[index]
            bone_name = joint_names[index + hand_joint_start_index]
            set_pose_from_rodrigues(armature, bone_name, pose_rodrigues)

        return {'FINISHED'}


class OP_WritePoseToConsole(bpy.types.Operator):
    bl_idname = "object.write_pose_to_console"
    bl_label = "Write Pose To Console"
    bl_description = ("Writes pose to console window")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh or armature is active object
            return context.object.type in ('MESH', 'ARMATURE')
        except: return False

    def execute(self, context):
        obj = bpy.context.object
        SMPL_version = bpy.context.object['SMPL_version']
        joint_names = MODEL_JOINT_NAMES[SMPL_version].value
        num_joints = len(joint_names)

        if obj.type == 'MESH':
            armature = obj.parent
        else:
            armature = obj

        # Get armature pose in rodrigues representation
        pose = [0.0] * (num_joints * 3)

        for index in range(num_joints):
            joint_name = joint_names[index]
            joint_pose = rodrigues_from_pose(armature, joint_name)
            pose[index*3 + 0] = joint_pose[0]
            pose[index*3 + 1] = joint_pose[1]
            pose[index*3 + 2] = joint_pose[2]

        print("\npose = ")
        pose_by_joint = [pose[i:i+3] for i in range(0,len(pose),3)]
        print (*pose_by_joint, sep="\n")

        print ("\npose = " + str(pose))

        print ("\npose = ")
        print (*pose, sep="\n")

        return {'FINISHED'}


class OP_WritePoseToJSON(bpy.types.Operator, ExportHelper):
    bl_idname = "object.write_pose_to_json"
    bl_label = "Write Pose To .json File"
    bl_description = ("Writes pose to a .json file")
    bl_options = {'REGISTER', 'UNDO'}

    # ExportHelper mixin class uses this
    filename_ext = ".json"

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh or armature is active object
            return context.object.type in ('MESH', 'ARMATURE')
        except Exception:
            return False

    def execute(self, context):
        obj = bpy.context.object
        SMPL_version = bpy.context.object['SMPL_version']
        joint_names = MODEL_JOINT_NAMES[SMPL_version].value
        num_joints = len(joint_names)

        if obj.type == 'MESH':
            armature = obj.parent
        else:
            armature = obj

        # Get armature pose in rodrigues representation
        pose = [0.0] * (num_joints * 3)

        for index in range(num_joints):
            joint_name = joint_names[index]
            joint_pose = rodrigues_from_pose(armature, joint_name)
            pose[index*3 + 0] = joint_pose[0]
            pose[index*3 + 1] = joint_pose[1]
            pose[index*3 + 2] = joint_pose[2]

        pose_data = {
            "pose": pose,
        }

        with open(self.filepath, "w") as f:
            json.dump(pose_data, f)

        return {'FINISHED'}


class OP_ResetPose(bpy.types.Operator):
    bl_idname = "object.reset_pose"
    bl_label = "Reset Pose"
    bl_description = ("Resets pose to default zero pose")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return (
                ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or
                (context.object.type == 'ARMATURE')
            )
        except Exception:
            return False

    def execute(self, context):
        obj = bpy.context.object

        if obj.type == 'MESH':
            armature = obj.parent
        else:
            armature = obj

        for bone in armature.pose.bones:
            if bone.rotation_mode != 'QUATERNION':
                bone.rotation_mode = 'QUATERNION'
            bone.rotation_quaternion = Quaternion()

        # Reset corrective pose shapes
        bpy.ops.object.zero_out_pose_correctives('EXEC_DEFAULT')

        return {'FINISHED'}


#######################################################################################################################################################################################

def find_bone_world_location(bone_name, armature):
    # Access the armature's pose bones
    for bone in armature.pose.bones:
        # Check if the bone's name matches
        if bone.name == bone_name:
            # Get the bone's head location in world coordinates
            world_location = armature.matrix_world @ bone.head
            return world_location
    return None

def set_custom_shape_properties(DRV_bone, scale=None, rotation=None, translation=None):
    if scale:
        DRV_bone.custom_shape_scale_xyz = scale
    if rotation:
        DRV_bone.custom_shape_rotation_euler = rotation
    if translation:
        DRV_bone.custom_shape_translation = translation

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))



    

class OP_AutoRig(bpy.types.Operator):
    bl_idname = "object.auto_rig"
    bl_label = "Auto Rig"
    bl_description = "Duplicate bones, set hierarchy, constraints, create bone groups with conditional colors, and set custom shapes"
    bl_options = {'REGISTER', 'UNDO'}
    @classmethod
    def poll(cls, context):
        return context.object and ((context.object.type == 'MESH' and context.object.parent and context.object.parent.type == 'ARMATURE') or (context.object.type == 'ARMATURE'))
    def execute(self, context):
        armature = context.object if context.object.type == 'ARMATURE' else context.object.parent
        context.view_layer.objects.active = armature

        # Ensure we are in object mode
        bpy.ops.object.mode_set(mode='OBJECT')


        # Create a circle mesh for custom shape
        bpy.ops.mesh.primitive_circle_add(radius=0.5, fill_type='NOTHING')
        circle_mesh = bpy.context.object
        circle_mesh.name = "DRV_Circle_Shape"
        circle_mesh.display_type = 'WIRE'
        circle_mesh.rotation_euler[0] = math.pi / 2
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        circle_mesh.hide_render = True
        circle_mesh.hide_select = True
        circle_mesh.hide_viewport = False


        # Create a cube mesh for custom shape
        bpy.ops.mesh.primitive_cube_add(size=0.5)
        cube_mesh = bpy.context.object
        cube_mesh.name = "DRV_Cube_Shape"
        cube_mesh.display_type = 'WIRE'
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        cube_mesh.hide_render = True
        cube_mesh.hide_select = True
        cube_mesh.hide_viewport = False

        # Create a plane mesh for custom shape
        bpy.ops.mesh.primitive_plane_add(size=0.5)
        plane_mesh = bpy.context.object
        plane_mesh.name = "DRV_Plane_Shape"
        plane_mesh.display_type = 'WIRE'
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        plane_mesh.hide_render = True
        plane_mesh.hide_select = True
        plane_mesh.hide_viewport = False


        # Create a sphere shape using three circles
        bpy.ops.object.empty_add(type='SPHERE', radius=0.5)
        sphere_mesh = bpy.context.object
        sphere_mesh.name = "DRV_Sphere_Shape"
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        sphere_mesh.hide_render = True
        sphere_mesh.hide_select = True
        sphere_mesh.hide_viewport = False


        # Create a line mesh for custom shape
        bpy.ops.mesh.primitive_cube_add(size=0.1)
        line_mesh = bpy.context.object
        line_mesh.name = "DRV_Line_Shape"
        line_mesh.display_type = 'WIRE'
        line_mesh.scale[0] = 10  # Make it a long thin line
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        line_mesh.hide_render = True
        line_mesh.hide_select = True
        line_mesh.hide_viewport = False


        # Create a text shape mesh for custom shape
        #bpy.ops.object.text_add()
        #text_mesh = bpy.context.object
        #text_mesh.data.body = "IK/FK"
        #text_mesh.name = "DRV_Text_Shape"
        #text_mesh.display_type = 'WIRE'
        #text_mesh.scale = (0.5, 0.5, 0.5)
        #bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)  # Only apply scale
        #text_mesh.hide_render = True
        #text_mesh.hide_select = True
        #text_mesh.hide_viewport = False


        # Create a NURBS path for custom shape
        bpy.ops.curve.primitive_nurbs_path_add()
        # Get the new NURBS path object
        nurbs_path = bpy.context.object
        # Set the name of the NURBS path
        nurbs_path.name = "DRV_NURBS_Path_Shape"
        # Set display type to 'WIRE'
        nurbs_path.display_type = 'WIRE'
        # Scale the NURBS path
        nurbs_path.scale = (1, 1, 1)
        # Apply the scale transformation
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        # Set object visibility settings
        nurbs_path.hide_render = True
        nurbs_path.hide_select = True
        nurbs_path.hide_viewport = False


        # Make sure the armature is re-selected and active before creating DRV_ bones
        context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='EDIT')
        bone_map = {}
        for bone in list(armature.data.edit_bones):
            new_bone_name = "DRV_" + bone.name
            if new_bone_name not in armature.data.edit_bones:
                new_bone = armature.data.edit_bones.new(new_bone_name)
                new_bone.head = bone.head
                new_bone.tail = bone.tail
                new_bone.roll = bone.roll
                new_bone.layers = bone.layers
                new_bone.use_deform = False
                bone_map[bone.name] = new_bone_name
        for orig_name, DRV_name in bone_map.items():
            DRV_bone = armature.data.edit_bones[DRV_name]
            orig_bone = armature.data.edit_bones[orig_name]
            if orig_bone.parent and orig_bone.parent.name in bone_map:
                DRV_bone.parent = armature.data.edit_bones[bone_map[orig_bone.parent.name]]


        # Define a list of necessary joints for the IK FK system
        ik_fk_joints = [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ]

        # Create the IK bones
        ik_bone_map = {}
        for joint in ik_fk_joints:
            orig_bone = armature.data.edit_bones.get(joint)
            if orig_bone:
                ik_bone_name = "IK_" + joint
                if ik_bone_name not in armature.data.edit_bones:
                    ik_bone = armature.data.edit_bones.new(ik_bone_name)
                    ik_bone.head = orig_bone.head
                    ik_bone.tail = orig_bone.tail
                    ik_bone.roll = orig_bone.roll
                    ik_bone.use_deform = False  # Uncheck deform for the IK bones
                    ik_bone.layers = orig_bone.layers  # Ensure the bone layers are maintained
                    ik_bone_map[joint] = ik_bone_name

        # Set the hierarchy for the IK bones
        for joint, ik_bone_name in ik_bone_map.items():
            orig_bone = armature.data.edit_bones.get(joint)
            ik_bone = armature.data.edit_bones.get(ik_bone_name)
            if orig_bone and ik_bone and orig_bone.parent and orig_bone.parent.name in ik_bone_map:
                ik_bone.parent = armature.data.edit_bones[ik_bone_map[orig_bone.parent.name]]

        # Set the hierarchy for specific IK bones
        ik_hierarchy = {
            "IK_left_shoulder": "DRV_left_collar",
            "IK_right_shoulder": "DRV_right_collar",
            "IK_left_hip": "DRV_pelvis",
            "IK_right_hip": "DRV_pelvis",
        }

        for ik_bone_name, parent_name in ik_hierarchy.items():
            ik_bone = armature.data.edit_bones.get(ik_bone_name)
            parent_bone = armature.data.edit_bones.get(parent_name)
            if ik_bone and parent_bone:
                ik_bone.parent = parent_bone


        # Create the FK bones
        fk_bone_map = {}
        for joint in ik_fk_joints:
            orig_bone = armature.data.edit_bones.get(joint)
            if orig_bone:
                fk_bone_name = "FK_" + joint
                if fk_bone_name not in armature.data.edit_bones:
                    fk_bone = armature.data.edit_bones.new(fk_bone_name)
                    fk_bone.head = orig_bone.head
                    fk_bone.tail = orig_bone.tail
                    fk_bone.roll = orig_bone.roll
                    fk_bone.use_deform = False  # Uncheck deform for the FK bones
                    fk_bone.layers = orig_bone.layers  # Ensure the bone layers are maintained
                    fk_bone_map[joint] = fk_bone_name

        # Set the hierarchy for the FK bones
        for joint, fk_bone_name in fk_bone_map.items():
            orig_bone = armature.data.edit_bones.get(joint)
            fk_bone = armature.data.edit_bones.get(fk_bone_name)
            if orig_bone and fk_bone and orig_bone.parent and orig_bone.parent.name in fk_bone_map:
                fk_bone.parent = armature.data.edit_bones[fk_bone_map[orig_bone.parent.name]]

        # Set the hierarchy for specific FK bones
        fk_hierarchy = {
            "FK_left_shoulder": "DRV_left_collar",
            "FK_right_shoulder": "DRV_right_collar",
            "FK_left_hip": "DRV_pelvis",
            "FK_right_hip": "DRV_pelvis",
        }

        for fk_bone_name, parent_name in fk_hierarchy.items():
            fk_bone = armature.data.edit_bones.get(fk_bone_name)
            parent_bone = armature.data.edit_bones.get(parent_name)
            if fk_bone and parent_bone:
                fk_bone.parent = parent_bone


        # Create the new unparented joint CTRL_IK_left_leg from IK_left_knee
        IK_left_knee_bone = armature.data.edit_bones.get("IK_left_knee")
        if IK_left_knee_bone:
            new_bone = armature.data.edit_bones.new("CTRL_IK_left_leg")
            new_bone.head = IK_left_knee_bone.tail
            new_bone.tail = IK_left_knee_bone.tail + mathutils.Vector((0, 0, -30))
            new_bone.use_connect = False
            new_bone.use_deform = False
            new_bone.parent = None


        # Create the left leg pole vector
            pole_vector_bone = armature.data.edit_bones.new("CTRL_PV_left_leg")
            pole_vector_bone.head = IK_left_knee_bone.head + mathutils.Vector((0, 0, 100))
            pole_vector_bone.tail = pole_vector_bone.head + mathutils.Vector((0, 0, 15))
            pole_vector_bone.use_connect = False
            pole_vector_bone.use_deform = False
            pole_vector_bone.parent = None


        # Create the new unparented joint CTRL_IK_right_leg from IK_right_knee
        IK_right_knee_bone = armature.data.edit_bones.get("IK_right_knee")
        if IK_right_knee_bone:
            new_bone = armature.data.edit_bones.new("CTRL_IK_right_leg")
            new_bone.head = IK_right_knee_bone.tail
            new_bone.tail = IK_right_knee_bone.tail + mathutils.Vector((0, 0, -30))
            new_bone.use_connect = False
            new_bone.use_deform = False
            new_bone.parent = None


        # Create the right leg pole vector
            pole_vector_bone = armature.data.edit_bones.new("CTRL_PV_right_leg")
            pole_vector_bone.head = IK_right_knee_bone.head + mathutils.Vector((0, 0, 100))
            pole_vector_bone.tail = pole_vector_bone.head + mathutils.Vector((0, 0, 15))
            pole_vector_bone.use_connect = False
            pole_vector_bone.use_deform = False
            pole_vector_bone.parent = None

        # Create the right leg pole vector
        #IK_right_knee_bone = armature.data.edit_bones.get("IK_right_knee")
        #if IK_right_knee_bone:
            #pole_vector_bone = armature.data.edit_bones.new("CTRL_PV_right_leg")
            # Calculate the local Z direction in world coordinates
            #local_z = IK_right_knee_bone.matrix.to_3x3() @ mathutils.Vector((0, 0, 1))
            #pole_vector_bone.head = IK_right_knee_bone.head + local_z * -100
            #pole_vector_bone.tail = pole_vector_bone.head + local_z * -15
            #pole_vector_bone.use_connect = False
            #pole_vector_bone.use_deform = False
            #pole_vector_bone.parent = None



    # Create the new unparented joint CTRL_IK_left_arm from IK_left_elbow
        IK_left_elbow_bone = armature.data.edit_bones.get("IK_left_elbow")
        if IK_left_elbow_bone:
            new_bone = armature.data.edit_bones.new("CTRL_IK_left_arm")
            new_bone.head = IK_left_elbow_bone.tail
            new_bone.tail = IK_left_elbow_bone.tail + mathutils.Vector((0, 0, -30))
            new_bone.use_connect = False
            new_bone.use_deform = False
            new_bone.parent = None

        # Create the left arm pole vector
        IK_left_elbow_bone = armature.data.edit_bones.get("IK_left_elbow")
        if IK_left_elbow_bone:
            pole_vector_bone = armature.data.edit_bones.new("CTRL_PV_left_arm")
            # Calculate the local Z direction in world coordinates
            local_z = IK_left_elbow_bone.matrix.to_3x3() @ mathutils.Vector((0, 0, 1))
            pole_vector_bone.head = IK_left_elbow_bone.head + local_z * -50
            pole_vector_bone.tail = pole_vector_bone.head + local_z * -15
            pole_vector_bone.use_connect = False
            pole_vector_bone.use_deform = False
            pole_vector_bone.parent = None

        # Create the new unparented joint CTRL_IK_right_arm from IK_right_elbow
        IK_right_elbow_bone = armature.data.edit_bones.get("IK_right_elbow")
        if IK_right_elbow_bone:
            new_bone = armature.data.edit_bones.new("CTRL_IK_right_arm")
            new_bone.head = IK_right_elbow_bone.tail
            new_bone.tail = IK_right_elbow_bone.tail + mathutils.Vector((0, 0, -30))
            new_bone.use_connect = False
            new_bone.use_deform = False
            new_bone.parent = None

        # Create the right arm pole vector
        IK_right_elbow_bone = armature.data.edit_bones.get("IK_right_elbow")
        if IK_right_elbow_bone:
            pole_vector_bone = armature.data.edit_bones.new("CTRL_PV_right_arm")
            # Calculate the local Z direction in world coordinates
            local_z = IK_right_elbow_bone.matrix.to_3x3() @ mathutils.Vector((0, 0, 1))
            pole_vector_bone.head = IK_right_elbow_bone.head + local_z * -50
            pole_vector_bone.tail = pole_vector_bone.head + local_z * -15
            pole_vector_bone.use_connect = False
            pole_vector_bone.use_deform = False
            pole_vector_bone.parent = None

        # Create the new joint IKFK_SWITCH_left_arm parented to left_arm
        left_middle3_bone = armature.data.edit_bones.get("left_middle3")
        left_wrist_bone = armature.data.edit_bones.get("left_wrist")
        if left_middle3_bone:
            new_bone = armature.data.edit_bones.new("IKFK_SWITCH_left_arm")
            new_bone.head = left_middle3_bone.tail + mathutils.Vector((10, 0, 0))
            new_bone.tail = left_middle3_bone.tail + mathutils.Vector((20, 0, 0))
            new_bone.use_connect = False
            new_bone.use_deform = False
            new_bone.parent = left_wrist_bone

        
        # Create the new joint IKFK_SWITCH_right_arm parented to right_arm
        right_middle3_bone = armature.data.edit_bones.get("right_middle3")
        right_wrist_bone = armature.data.edit_bones.get("right_wrist")
        if right_middle3_bone:
            new_bone = armature.data.edit_bones.new("IKFK_SWITCH_right_arm")
            new_bone.head = right_middle3_bone.tail + mathutils.Vector((-10, 0, 0))
            new_bone.tail = right_middle3_bone.tail + mathutils.Vector((-20, 0, 0))
            new_bone.use_connect = False
            new_bone.use_deform = False
            new_bone.parent = right_wrist_bone


        # Create the new joint IKFK_SWITCH_left_leg parented to IK_left_leg
        left_knee_bone = armature.data.edit_bones.get("left_knee")
        left_ankle_bone = armature.data.edit_bones.get("left_ankle")
        if left_knee_bone:
            new_bone = armature.data.edit_bones.new("IKFK_SWITCH_left_leg")
            new_bone.head = left_knee_bone.tail + mathutils.Vector((0, -7, 0))
            new_bone.tail = left_ankle_bone.tail + mathutils.Vector((0, -3, 0))
            new_bone.use_connect = False
            new_bone.use_deform = False
            new_bone.parent = left_ankle_bone
        
        # Create the new joint IKFK_SWITCH_right_leg parented to IK_right_leg
        right_knee_bone = armature.data.edit_bones.get("right_knee")
        right_ankle_bone = armature.data.edit_bones.get("right_ankle")
        if right_knee_bone:
            new_bone = armature.data.edit_bones.new("IKFK_SWITCH_right_leg")
            new_bone.head = right_knee_bone.tail + mathutils.Vector((0, -7, 0))
            new_bone.tail = right_ankle_bone.tail + mathutils.Vector((0, -3, 0))
            new_bone.use_connect = False
            new_bone.use_deform = False
            new_bone.parent = right_ankle_bone

        
        # Create a new joint LINE_PV_left_leg from IK_left_knee (VISIBLE LINE FROM POLE VECTOR TO KNEE)
        ik_left_knee_bone = armature.data.edit_bones.get("IK_left_knee")
        ctrl_pv_left_leg_bone = armature.data.edit_bones.get("CTRL_PV_left_leg")
        if ik_left_knee_bone and ctrl_pv_left_leg_bone:
            new_bone = armature.data.edit_bones.new("LINE_PV_left_leg")
            new_bone.head = ik_left_knee_bone.head
            new_bone.tail = ctrl_pv_left_leg_bone.head
            new_bone.parent = ik_left_knee_bone
            new_bone.use_connect = False
            new_bone.use_deform = False


        # Create a new joint LINE_PV_right_leg from IK_right_knee (VISIBLE LINE FROM POLE VECTOR TO KNEE)
        ik_right_knee_bone = armature.data.edit_bones.get("IK_right_knee")
        ctrl_pv_right_leg_bone = armature.data.edit_bones.get("CTRL_PV_right_leg")
        if ik_right_knee_bone and ctrl_pv_right_leg_bone:
            new_bone = armature.data.edit_bones.new("LINE_PV_right_leg")
            new_bone.head = ik_right_knee_bone.head 
            new_bone.tail = ctrl_pv_right_leg_bone.head
            new_bone.parent = ik_right_knee_bone
            new_bone.use_connect = False
            new_bone.use_deform = False

    
        # Create a new joint LINE_PV_left_arm from IK_left_elbow (VISIBLE LINE FROM POLE VECTOR TO KNEE)
        ik_left_elbow_bone = armature.data.edit_bones.get("IK_left_elbow")
        ctrl_pv_left_arm_bone = armature.data.edit_bones.get("CTRL_PV_left_arm")
        if ik_left_elbow_bone and ctrl_pv_left_arm_bone:
            new_bone = armature.data.edit_bones.new("LINE_PV_left_arm")
            new_bone.head = ik_left_elbow_bone.head 
            new_bone.tail = ctrl_pv_left_arm_bone.head
            new_bone.parent = ik_left_elbow_bone
            new_bone.use_connect = False
            new_bone.use_deform = False

        # Create a new joint LINE_PV_right_arm from IK_right_elbow (VISIBLE LINE FROM POLE VECTOR TO KNEE)
        ik_right_elbow_bone = armature.data.edit_bones.get("IK_right_elbow")
        ctrl_pv_right_arm_bone = armature.data.edit_bones.get("CTRL_PV_right_arm")
        if ik_right_elbow_bone and ctrl_pv_right_arm_bone:
            new_bone = armature.data.edit_bones.new("LINE_PV_right_arm")
            new_bone.head = ik_right_elbow_bone.head 
            new_bone.tail = ctrl_pv_right_arm_bone.head
            new_bone.parent = ik_right_elbow_bone
            new_bone.use_connect = False
            new_bone.use_deform = False

        bpy.ops.object.mode_set(mode='POSE')


        # Create and assign bone groups
        active_color = hex_to_rgb('ffffff') #White
        select_color = hex_to_rgb('FFA500') #Orange

        deform_group = armature.pose.bone_groups.new(name="DEFORM")
        deform_group.color_set = 'THEME01'

        driver_left_group = armature.pose.bone_groups.new(name="DRIVER_LEFT")
        driver_left_group_color = hex_to_rgb('0000FF') #Blue
        driver_left_group.color_set = 'CUSTOM'
        driver_left_group.colors.normal = driver_left_group_color
        driver_left_group.colors.select = select_color
        driver_left_group.colors.active = active_color

        driver_right_group = armature.pose.bone_groups.new(name="DRIVER_RIGHT")
        driver_right_group_color = hex_to_rgb('ff0000') #Red
        driver_right_group.color_set = 'CUSTOM'
        driver_right_group.colors.normal = driver_right_group_color
        driver_right_group.colors.select = select_color
        driver_right_group.colors.active = active_color
        
        driver_other_group = armature.pose.bone_groups.new(name="DRIVER_OTHER")
        driver_other_group_color = hex_to_rgb('ffff00') #Yellow
        driver_other_group.color_set = 'CUSTOM'
        driver_other_group.colors.normal = driver_other_group_color
        driver_other_group.colors.select = select_color
        driver_other_group.colors.active = active_color
        
        ik_system_group = armature.pose.bone_groups.new(name="IK_SYSTEM")
        ik_color = hex_to_rgb('00FFFF')  # Custom color 00FFFF
        ik_system_group.color_set = 'CUSTOM'
        ik_system_group.colors.normal = ik_color
        ik_system_group.colors.select = select_color
        ik_system_group.colors.active = active_color

        fk_system_group = armature.pose.bone_groups.new(name="FK_SYSTEM")
        fk_color = hex_to_rgb('FFFF00')  # Custom color FFFF00
        fk_system_group.color_set = 'CUSTOM'
        fk_system_group.colors.normal = fk_color
        fk_system_group.colors.select = select_color
        fk_system_group.colors.active = active_color

        
        # Change custom shapes of the line PVs joints
        line_pv_bones = ["LINE_PV_left_leg", "LINE_PV_right_leg", "LINE_PV_left_arm", "LINE_PV_right_arm",]
        for bone_name in line_pv_bones:
                    bone = armature.pose.bones.get(bone_name)
                    if bone:
                        if 'arm' in bone_name:
                            bone.custom_shape = nurbs_path
                            set_custom_shape_properties(bone, scale=(0.5, 1, 1),
                                                        rotation=(0, 0, math.radians(90)),
                                                        translation=(0, 50, 0))
                        else:
                            bone.custom_shape = nurbs_path
                            set_custom_shape_properties(bone, scale=(0.25, 1, 1),
                                                        rotation=(0, 0, math.radians(90)),
                                                        translation=(0, 50, 0))
                        bone.bone.hide_select = True
                        bone.bone.show_wire = True


        switch_bones = [ "IKFK_SWITCH_left_arm", "IKFK_SWITCH_right_arm", "IKFK_SWITCH_left_leg", "IKFK_SWITCH_right_leg" ]
        for bone_name in switch_bones:
                bone = armature.pose.bones.get(bone_name)
                if bone:
                        if 'left' in bone_name:
                                bone.bone_group = driver_left_group
                        elif 'right' in bone_name:
                                bone.bone_group = driver_right_group

                if bone:
                        if 'arm' in bone_name:
                            bone.custom_shape = sphere_mesh
                            set_custom_shape_properties(bone, scale=(0.25, 0.25, 0.25))
                        elif 'leg' in bone_name:
                            bone.custom_shape = plane_mesh
                            set_custom_shape_properties(bone, scale=(3, 3, 1),
                                                        translation=(0, 5, 0))
                            if 'left_leg' in bone_name:
                                set_custom_shape_properties(bone, rotation=(math.radians(-1.95), math.radians(11.3), 0), 
                                                            translation=(0, 5, -0.95))
                            elif 'right_leg' in bone_name:
                                set_custom_shape_properties(bone, rotation=(math.radians(-4.1), math.radians(-9.9), 0), 
                                                            translation=(0, 5, -0.45))

                        bone.bone.show_wire = True
                    


        # Change custom shapes of the IK and PV joints
        ctrl_bones = [
            "CTRL_IK_left_leg", "CTRL_IK_right_leg",
            "CTRL_PV_left_leg", "CTRL_PV_right_leg",
            "CTRL_IK_left_arm", "CTRL_IK_right_arm",
            "CTRL_PV_left_arm", "CTRL_PV_right_arm",   ]
        for bone_name in ctrl_bones:
                    bone = armature.pose.bones.get(bone_name)
                    if bone:
                        if 'CTRL_PV' in bone_name:
                            bone.custom_shape = sphere_mesh
                            set_custom_shape_properties(bone, scale=(0.5, 0.5, 0.5))
                        elif '_arm' in bone_name:
                            bone.custom_shape = cube_mesh
                            set_custom_shape_properties(bone, scale=(0.5, 0.5, 0.5),
                                                        rotation=(math.radians(45), math.radians(0), math.radians(90)))
                        else:
                            bone.custom_shape = cube_mesh
                            set_custom_shape_properties(bone, scale=(0.5, 0.5, 0.5),
                                                        rotation=(math.radians(45), math.radians(0), math.radians(90)),
                                                        translation=(0, 2, 0))
                            
                    if bone:
                        if 'left' in bone_name:
                            bone.bone_group = driver_left_group
                        elif 'right' in bone_name:
                            bone.bone_group = driver_right_group
                        bone.bone.show_wire = True

        # Assign IK bones to IK_SYSTEM
        ik_bones = [
            "IK_left_shoulder", "IK_left_elbow", "IK_left_wrist",
            "IK_right_shoulder", "IK_right_elbow", "IK_right_wrist",
            "IK_left_hip", "IK_left_knee", "IK_left_ankle",
            "IK_right_hip", "IK_right_knee", "IK_right_ankle",
        ]
        for bone_name in ik_bones:
                    bone = armature.pose.bones.get(bone_name)
                    if bone:
                        if 'left' in bone_name:
                            bone.bone_group = driver_left_group
                        elif 'right' in bone_name:
                            bone.bone_group = driver_right_group
                        bone.bone.show_wire = True
                        #bone.bone_group = ik_system_group

        # Assign FK bones to FK_SYSTEM
        fk_bones = [
            "FK_left_shoulder", "FK_left_elbow", "FK_left_wrist",
            "FK_right_shoulder", "FK_right_elbow", "FK_right_wrist",
            "FK_left_hip", "FK_left_knee", "FK_left_ankle",
            "FK_right_hip", "FK_right_knee", "FK_right_ankle",
        ]
        for bone_name in fk_bones:
                    bone = armature.pose.bones.get(bone_name)
                    if bone:
                        if 'left' in bone_name:
                            bone.bone_group = driver_left_group
                        elif 'right' in bone_name:
                            bone.bone_group = driver_right_group
                        bone.bone.show_wire = True
                        bone.custom_shape = circle_mesh
                        #bone.bone_group = fk_system_group

        for bone_name in fk_bones:
            bone = armature.pose.bones.get(bone_name)
            if 'FK_left_elbow' in bone_name:
                set_custom_shape_properties(bone, 
                                            scale=(0.7, 0.7, 0.7))
            elif 'FK_right_elbow' in bone_name:
                set_custom_shape_properties(bone, 
                                            scale=(0.7, 0.7, 0.7))
            elif 'FK_left_shoulder' in bone_name:
                set_custom_shape_properties(bone, 
                                            scale=(0.8, 0.8, 0.8), 
                                            rotation=(math.radians(6.5), 0, math.radians(8)),
                                            translation=(0, 6.5, 0))
            elif 'FK_right_shoulder' in bone_name:
                set_custom_shape_properties(bone, 
                                            scale=(0.8, 0.8, 0.8), 
                                            rotation=(math.radians(5), 0, math.radians(-8)),
                                            translation=(0, 6.5, 0))
            elif 'FK_left_hip' in bone_name:
                set_custom_shape_properties(bone, 
                                            scale=(0.7, 0.7, 0.7), 
                                            rotation=(math.radians(-2.6), 0, math.radians(-3.88)),
                                            translation=(0, 6.6, 5))
            elif 'FK_right_hip' in bone_name:
                set_custom_shape_properties(bone, 
                                            scale=(0.7, 0.7, 0.7), 
                                            rotation=(math.radians(-2.4), 0, math.radians(4.6)),
                                            translation=(0, 6, 5))
            elif 'FK_left_knee' in bone_name:
                set_custom_shape_properties(bone, 
                                            scale=(0.5, 0.5, 0.5),
                                            rotation=(math.radians(-2.9), 0, math.radians(1.4)))
            elif 'FK_right_knee' in bone_name:
                set_custom_shape_properties(bone, 
                                            scale=(0.5, 0.5, 0.5),
                                            rotation=(math.radians(-2.8), 0, 0))
            elif 'FK_left_ankle' in bone_name:
                set_custom_shape_properties(bone, 
                                            translation=(0, -2.1, -2.1),
                                            rotation=(math.radians(65.5), 0, math.radians(-26.5)))
            elif 'FK_right_ankle' in bone_name:
                set_custom_shape_properties(bone, 
                                            translation=(0, -1.85, -1.45),
                                            rotation=(math.radians(64), 0, math.radians(19)))


        for orig_name, DRV_name in bone_map.items():
            orig_bone = armature.pose.bones[orig_name]
            # Add constraints to original bones
            constraint = orig_bone.constraints.new('COPY_TRANSFORMS')
            constraint.target = armature
            constraint.subtarget = DRV_name
            orig_bone.bone_group = deform_group
            orig_bone.bone.hide = True  # Hide original bones in viewport
            DRV_bone = armature.pose.bones[DRV_name]
            DRV_bone.custom_shape = circle_mesh
            if DRV_name == "DRV_root":  # Apply the custom rotation to the DRV_root bone
                DRV_bone.custom_shape_rotation_euler[0] = math.pi / 2
            DRV_bone.bone_group = driver_left_group if "left" in DRV_name.lower() \
                                  else driver_right_group if "right" in DRV_name.lower() \
                                  else driver_other_group

            # Specific conditions for DRV_bones
            if DRV_name == "DRV_spine1":
                set_custom_shape_properties(DRV_bone, 
                                            scale=(3, 3, 3), 
                                            rotation=(math.radians(5), 0, math.radians(4.9)))
            elif DRV_name == "DRV_spine2":
                set_custom_shape_properties(DRV_bone, 
                                            scale=(6, 6, 6), 
                                            rotation=(math.radians(-36.5), math.radians(-12), 0))
            elif DRV_name == "DRV_spine3":
                set_custom_shape_properties(DRV_bone, 
                                            scale=(3.5, 3.5, 3.5))
            elif DRV_name == "DRV_pelvis":
                set_custom_shape_properties(DRV_bone, 
                                            scale=(4.5, 4.5, 4.5))
            elif DRV_name == "DRV_left_hip":
                set_custom_shape_properties(DRV_bone, 
                                            scale=(0.7, 0.7, 0.7), 
                                            rotation=(math.radians(-2.6), 0, math.radians(-3.88)),
                                            translation=(0, 6.6, 5))
            elif DRV_name == "DRV_right_hip":
                set_custom_shape_properties(DRV_bone, 
                                            scale=(0.7, 0.7, 0.7), 
                                            rotation=(math.radians(-2.4), 0, math.radians(4.6)),
                                            translation=(0, 6, 5))
            elif DRV_name == "DRV_left_knee":
                set_custom_shape_properties(DRV_bone, 
                                            scale=(0.5, 0.5, 0.5),
                                            rotation=(math.radians(-2.9), 0, math.radians(1.4)))
            elif DRV_name == "DRV_right_knee":
                set_custom_shape_properties(DRV_bone, 
                                            scale=(0.5, 0.5, 0.5),
                                            rotation=(math.radians(-2.8), 0, 0))
            elif DRV_name == "DRV_root":
                set_custom_shape_properties(DRV_bone, 
                                            scale=(10, 10, 10))
            elif DRV_name == "DRV_neck":
                set_custom_shape_properties(DRV_bone, 
                                            scale=(1.2, 1.2, 1.2), 
                                            rotation=(math.radians(-8), 0, math.radians(3)),
                                            translation=(0, 5, 0))
            elif DRV_name == "DRV_left_eye":
                set_custom_shape_properties(DRV_bone, 
                                            scale=(0.4, 0.4, 0.4), 
                                            translation=(0, 15, 0))
            elif DRV_name == "DRV_right_eye":
                set_custom_shape_properties(DRV_bone, 
                                            scale=(0.4, 0.4, 0.4), 
                                            translation=(0, 15, 0))
            elif DRV_name == "DRV_head":
                set_custom_shape_properties(DRV_bone, 
                                            scale=(2.2, 2.2, 2.2), 
                                            translation=(0, 5, -0.5))
            elif DRV_name == "DRV_jaw":
                set_custom_shape_properties(DRV_bone, 
                                            scale=(0.3, 0.3, 0.3), 
                                            rotation=(math.radians(-27), 0, 0),
                                            translation=(0, 12.7, -1))
            elif DRV_name == "DRV_left_elbow":
                set_custom_shape_properties(DRV_bone, 
                                            scale=(0.7, 0.7, 0.7))
            elif DRV_name == "DRV_right_elbow":
                set_custom_shape_properties(DRV_bone, 
                                            scale=(0.7, 0.7, 0.7))
            elif DRV_name == "DRV_left_shoulder":
                set_custom_shape_properties(DRV_bone, 
                                            scale=(0.8, 0.8, 0.8), 
                                            rotation=(math.radians(6.5), 0, math.radians(8)),
                                            translation=(0, 6.5, 0))
            elif DRV_name == "DRV_right_shoulder":
                set_custom_shape_properties(DRV_bone, 
                                            scale=(0.8, 0.8, 0.8), 
                                            rotation=(math.radians(5), 0, math.radians(-8)),
                                            translation=(0, 6.5, 0))
            elif DRV_name == "DRV_left_collar":
                set_custom_shape_properties(DRV_bone, 
                                            scale=(1, 1, 1), 
                                            rotation=(math.radians(8), 0, math.radians(-35)),
                                            translation=(-5.7, 11.1, 0))
            elif DRV_name == "DRV_right_collar":
                set_custom_shape_properties(DRV_bone, 
                                            scale=(1, 1, 1), 
                                            rotation=(math.radians(5), 0, math.radians(35)),
                                            translation=(5.7, 11.1, 0))

    

        # Hide the control mesh objects in the viewport
        circle_mesh.hide_set(True)
        cube_mesh.hide_set(True) 
        plane_mesh.hide_set(True)
        sphere_mesh.hide_set(True)
        line_mesh.hide_set(True)
        nurbs_path.hide_set(True)


        # Add constraints to DRV_ bones for IK bones
        for orig_name, ik_name in ik_bone_map.items():
            drv_bone_name = "DRV_" + orig_name
            drv_bone = armature.pose.bones.get(drv_bone_name)
            if drv_bone:
                constraint = drv_bone.constraints.new('COPY_TRANSFORMS')
                constraint.target = armature
                constraint.subtarget = ik_name

        # Add constraints to DRV_ bones for FK bones
        for orig_name, fk_name in fk_bone_map.items():
            drv_bone_name = "DRV_" + orig_name
            drv_bone = armature.pose.bones.get(drv_bone_name)
            if drv_bone:
                constraint = drv_bone.constraints.new('COPY_TRANSFORMS')
                constraint.target = armature
                constraint.subtarget = fk_name


        # Select and rotate IK_right_hip
        bpy.ops.pose.select_all(action='DESELECT')
        IK_right_hip_bone = armature.pose.bones.get("IK_right_hip")
        if IK_right_hip_bone:
            IK_right_hip_bone.bone.select = True
            armature.data.bones.active = IK_right_hip_bone.bone
            #bpy.ops.transform.rotate(value=math.radians(15), orient_axis='X')
            #bpy.ops.transform.rotate(value=math.radians(15), orient_axis='Z')


        # Select and rotate IK_right_knee
        bpy.ops.pose.select_all(action='DESELECT')
        IK_right_knee_bone = armature.pose.bones.get("IK_right_knee")
        if IK_right_knee_bone:
            IK_right_knee_bone.bone.select = True
            armature.data.bones.active = IK_right_knee_bone.bone
            bpy.ops.transform.rotate(value=math.radians(-45), orient_axis='X')
            bpy.ops.transform.rotate(value=math.radians(-3), orient_axis='Y') #Fix knee bending


        # Select and rotate IK_left_hip
        bpy.ops.pose.select_all(action='DESELECT')
        IK_left_hip_bone = armature.pose.bones.get("IK_left_hip")
        if IK_left_hip_bone:
            IK_left_hip_bone.bone.select = True
            armature.data.bones.active = IK_left_hip_bone.bone
            #bpy.ops.transform.rotate(value=math.radians(15), orient_axis='X')
            #bpy.ops.transform.rotate(value=math.radians(-15), orient_axis='Z')


        # Select and rotate IK_left_knee
        bpy.ops.pose.select_all(action='DESELECT')
        IK_left_knee_bone = armature.pose.bones.get("IK_left_knee")
        if IK_left_knee_bone:
            IK_left_knee_bone.bone.select = True
            armature.data.bones.active = IK_left_knee_bone.bone
            bpy.ops.transform.rotate(value=math.radians(-45), orient_axis='X')
            bpy.ops.transform.rotate(value=math.radians(3), orient_axis='Y') #Fix knee bending

        # Select and rotate IK_left_elbow
        bpy.ops.pose.select_all(action='DESELECT')
        IK_left_elbow_bone = armature.pose.bones.get("IK_left_elbow")
        if IK_left_elbow_bone:
            IK_left_elbow_bone.bone.select = True
            armature.data.bones.active = IK_left_elbow_bone.bone
            bpy.ops.transform.rotate(value=math.radians(30), orient_axis='Z')
            bpy.ops.transform.rotate(value=math.radians(-15), orient_axis='X')

        # Select and rotate IK_right_elbow
        bpy.ops.pose.select_all(action='DESELECT')
        IK_right_elbow_bone = armature.pose.bones.get("IK_right_elbow")
        if IK_right_elbow_bone:
            IK_right_elbow_bone.bone.select = True
            armature.data.bones.active = IK_right_elbow_bone.bone
            bpy.ops.transform.rotate(value=math.radians(-30), orient_axis='Z')
            bpy.ops.transform.rotate(value=math.radians(-15), orient_axis='X')


        # Add an IK constraint to IK_left_knee
        IK_left_knee_pose_bone = armature.pose.bones.get("IK_left_knee")
        ctrl_ik_left_leg_bone = armature.pose.bones.get("CTRL_IK_left_leg")
        ctrl_pv_left_leg_bone = armature.pose.bones.get("CTRL_PV_left_leg")
        if IK_left_knee_pose_bone and ctrl_ik_left_leg_bone:
            ik_constraint = IK_left_knee_pose_bone.constraints.new('IK')
            ik_constraint.target = armature
            ik_constraint.subtarget = ctrl_ik_left_leg_bone.name
            ik_constraint.chain_count = 2
            ik_constraint.pole_target = armature
            ik_constraint.pole_subtarget = ctrl_pv_left_leg_bone.name
            ik_constraint.pole_angle = math.radians(13.2)


        # Add an IK constraint to IK_right_knee
        IK_right_knee_pose_bone = armature.pose.bones.get("IK_right_knee")
        ctrl_ik_right_leg_bone = armature.pose.bones.get("CTRL_IK_right_leg")
        ctrl_pv_right_leg_bone = armature.pose.bones.get("CTRL_PV_right_leg")
        if IK_right_knee_pose_bone and ctrl_ik_right_leg_bone:
            ik_constraint = IK_right_knee_pose_bone.constraints.new('IK')
            ik_constraint.target = armature
            ik_constraint.subtarget = ctrl_ik_right_leg_bone.name
            ik_constraint.chain_count = 2
            ik_constraint.pole_target = armature
            ik_constraint.pole_subtarget = ctrl_pv_right_leg_bone.name
            ik_constraint.pole_angle = math.radians(142)


       # Add an IK constraint to IK_left_elbow
        IK_left_elbow_pose_bone = armature.pose.bones.get("IK_left_elbow")
        ctrl_ik_left_arm_bone = armature.pose.bones.get("CTRL_IK_left_arm")
        ctrl_pv_left_arm_bone = armature.pose.bones.get("CTRL_PV_left_arm")
        if IK_left_elbow_pose_bone and ctrl_ik_left_arm_bone:
            ik_constraint = IK_left_elbow_pose_bone.constraints.new('IK')
            ik_constraint.target = armature
            ik_constraint.subtarget = ctrl_ik_left_arm_bone.name
            ik_constraint.chain_count = 2
            ik_constraint.pole_target = armature
            ik_constraint.pole_subtarget = ctrl_pv_left_arm_bone.name
            ik_constraint.pole_angle = math.radians(-90)

         # Add an IK constraint to IK_right_elbow
        IK_right_elbow_pose_bone = armature.pose.bones.get("IK_right_elbow")
        ctrl_ik_right_arm_bone = armature.pose.bones.get("CTRL_IK_right_arm")
        ctrl_pv_right_arm_bone = armature.pose.bones.get("CTRL_PV_right_arm")
        if IK_right_elbow_pose_bone and ctrl_ik_right_arm_bone:
            ik_constraint = IK_right_elbow_pose_bone.constraints.new('IK')
            ik_constraint.target = armature
            ik_constraint.subtarget = ctrl_ik_right_arm_bone.name
            ik_constraint.chain_count = 2
            ik_constraint.pole_target = armature
            ik_constraint.pole_subtarget = ctrl_pv_right_arm_bone.name
            ik_constraint.pole_angle = math.radians(-90)



        bpy.ops.object.mode_set(mode='EDIT')


         # Parent IK_left_ankle to CTRL_IK_left_leg while keeping the offset
        IK_left_ankle_bone = armature.data.edit_bones.get("IK_left_ankle")
        ctrl_ik_left_leg_bone = armature.data.edit_bones.get("CTRL_IK_left_leg")
        if IK_left_ankle_bone and ctrl_ik_left_leg_bone:
            bpy.ops.armature.select_all(action='DESELECT')
            IK_left_ankle_bone.select = True
            ctrl_ik_left_leg_bone.select = True
            armature.data.edit_bones.active = ctrl_ik_left_leg_bone
            bpy.ops.armature.parent_set(type='OFFSET')


         # Parent IK_right_ankle to CTRL_IK_right_leg while keeping the offset
        IK_right_ankle_bone = armature.data.edit_bones.get("IK_right_ankle")
        ctrl_ik_right_leg_bone = armature.data.edit_bones.get("CTRL_IK_right_leg")
        if IK_right_ankle_bone and ctrl_ik_right_leg_bone:
            bpy.ops.armature.select_all(action='DESELECT')
            IK_right_ankle_bone.select = True
            ctrl_ik_right_leg_bone.select = True
            armature.data.edit_bones.active = ctrl_ik_right_leg_bone
            bpy.ops.armature.parent_set(type='OFFSET')

         # Parent IK_left_wrist to CTRL_IK_left_arm while keeping the offset
        IK_left_wrist_bone = armature.data.edit_bones.get("IK_left_wrist")
        ctrl_ik_left_arm_bone = armature.data.edit_bones.get("CTRL_IK_left_arm")
        if IK_left_wrist_bone and ctrl_ik_left_arm_bone:
            bpy.ops.armature.select_all(action='DESELECT')
            IK_left_wrist_bone.select = True
            ctrl_ik_left_arm_bone.select = True
            armature.data.edit_bones.active = ctrl_ik_left_arm_bone
            bpy.ops.armature.parent_set(type='OFFSET')


         # Parent IK_right_wrist to CTRL_IK_right_arm while keeping the offset
        IK_right_wrist_bone = armature.data.edit_bones.get("IK_right_wrist")
        ctrl_ik_right_arm_bone = armature.data.edit_bones.get("CTRL_IK_right_arm")
        if IK_right_wrist_bone and ctrl_ik_right_arm_bone:
            bpy.ops.armature.select_all(action='DESELECT')
            IK_right_wrist_bone.select = True
            ctrl_ik_right_arm_bone.select = True
            armature.data.edit_bones.active = ctrl_ik_right_arm_bone
            bpy.ops.armature.parent_set(type='OFFSET')


        # Parent IK and PV joints to DRV_root while keeping the offset
        DRV_root_bone = armature.data.edit_bones.get("DRV_root")
        if DRV_root_bone:
            ik_pv_bones = ["CTRL_IK_left_leg", "CTRL_IK_right_leg", "CTRL_PV_left_leg", "CTRL_PV_right_leg", "CTRL_IK_left_arm", "CTRL_IK_right_arm",
                           "CTRL_PV_left_arm", "CTRL_PV_right_arm"]
            for bone_name in ik_pv_bones:
                bone = armature.data.edit_bones.get(bone_name)
                if bone:
                    bpy.ops.armature.select_all(action='DESELECT')
                    bone.select = True
                    DRV_root_bone.select = True
                    armature.data.edit_bones.active = DRV_root_bone
                    bpy.ops.armature.parent_set(type='OFFSET')

        bpy.ops.object.mode_set(mode='POSE')


        # Add a Copy Location constraint to IK_left_ankle
        IK_left_ankle_pose_bone = armature.pose.bones.get("IK_left_ankle")
        IK_left_knee_pose_bone = armature.pose.bones.get("IK_left_knee")
        if IK_left_ankle_pose_bone and IK_left_knee_pose_bone:
            copy_loc_constraint = IK_left_ankle_pose_bone.constraints.new('COPY_LOCATION')
            copy_loc_constraint.target = armature
            copy_loc_constraint.subtarget = IK_left_knee_pose_bone.name
            copy_loc_constraint.head_tail = 1.0 


        # Add a Copy Location constraint to IK_right_ankle
        IK_right_ankle_pose_bone = armature.pose.bones.get("IK_right_ankle")
        IK_right_knee_pose_bone = armature.pose.bones.get("IK_right_knee")
        if IK_right_ankle_pose_bone and IK_right_knee_pose_bone:
            copy_loc_constraint = IK_right_ankle_pose_bone.constraints.new('COPY_LOCATION')
            copy_loc_constraint.target = armature
            copy_loc_constraint.subtarget = IK_right_knee_pose_bone.name
            copy_loc_constraint.head_tail = 1.0 


        
        # Add a Copy Location constraint to IK_left_wrist
        IK_left_wrist_pose_bone = armature.pose.bones.get("IK_left_wrist")
        IK_left_elbow_pose_bone = armature.pose.bones.get("IK_left_elbow")
        if IK_left_wrist_pose_bone and IK_left_elbow_pose_bone:
            copy_loc_constraint = IK_left_wrist_pose_bone.constraints.new('COPY_LOCATION')
            copy_loc_constraint.target = armature
            copy_loc_constraint.subtarget = IK_left_elbow_pose_bone.name
            copy_loc_constraint.head_tail = 1.0 


        # Add a Copy Location constraint to IK_right_wrist
        IK_right_wrist_pose_bone = armature.pose.bones.get("IK_right_wrist")
        IK_right_elbow_pose_bone = armature.pose.bones.get("IK_right_elbow")
        if IK_right_wrist_pose_bone and IK_right_elbow_pose_bone:
            copy_loc_constraint = IK_right_wrist_pose_bone.constraints.new('COPY_LOCATION')
            copy_loc_constraint.target = armature
            copy_loc_constraint.subtarget = IK_right_elbow_pose_bone.name
            copy_loc_constraint.head_tail = 1.0 



        ikfk_switch_bone = armature.pose.bones.get("IKFK_SWITCH")
        ikfk_switch_left_arm_bone = armature.pose.bones.get("IKFK_SWITCH_left_arm")
        ikfk_switch_right_arm_bone = armature.pose.bones.get("IKFK_SWITCH_right_arm")
        ikfk_switch_left_leg_bone = armature.pose.bones.get("IKFK_SWITCH_left_leg")
        ikfk_switch_right_leg_bone = armature.pose.bones.get("IKFK_SWITCH_right_leg")

        def set_custom_property(bone, prop_name, min_value, max_value):
            if prop_name not in bone.keys():
                bone[prop_name] = 0.0
            bone['_RNA_UI'] = bone.get('_RNA_UI', {})
            bone['_RNA_UI'][prop_name] = {
                'min': min_value,
                'max': max_value,
                'soft_min': min_value,
                'soft_max': max_value,
                'default': 0
            }

        if ikfk_switch_left_arm_bone:
            set_custom_property(ikfk_switch_left_arm_bone, "IK/FK Arm L", 0.0, 1.0)
            ikfk_switch_left_arm_bone["IK/FK Arm L"] = 0.0
            ikfk_switch_left_arm_bone["_RNA_UI"] = None

        if ikfk_switch_right_arm_bone:
            set_custom_property(ikfk_switch_right_arm_bone, "IK/FK Arm R", 0.0, 1.0)
            ikfk_switch_right_arm_bone["IK/FK Arm R"] = 0.0
            ikfk_switch_right_arm_bone["_RNA_UI"] = None

        if ikfk_switch_left_leg_bone:
            set_custom_property(ikfk_switch_left_leg_bone, "IK/FK Leg L", 0.0, 1.0)
            ikfk_switch_left_leg_bone["IK/FK Leg L"] = 0.0
            ikfk_switch_left_leg_bone["_RNA_UI"] = None

        if ikfk_switch_right_leg_bone:
            set_custom_property(ikfk_switch_right_leg_bone, "IK/FK Leg R", 0.0, 1.0)
            ikfk_switch_right_leg_bone["IK/FK Leg R"] = 0.0
            ikfk_switch_right_leg_bone["_RNA_UI"] = None




        # Function to add driver to the Influence parameter of a constraint
        def add_driver_to_constraint(bone_name, constraint_name, target_name, prop_name):
            bone = armature.pose.bones.get(bone_name)
            if bone:
                for constraint in bone.constraints:
                    if constraint.name == constraint_name:
                        fcurve = constraint.driver_add("influence")
                        driver = fcurve.driver
                        driver.type = 'AVERAGE'
                        var = driver.variables.new()
                        var.name = "var"
                        var.type = 'SINGLE_PROP'
                        target = var.targets[0]
                        target.id_type = 'OBJECT'
                        target.id = armature
                        target.data_path = f'pose.bones["{target_name}"]["{prop_name}"]'
                        driver.expression = "var"


        # Add the driver to the specified joints and constraints for the left arm
        add_driver_to_constraint("DRV_left_shoulder", "Copy Transforms.001", "IKFK_SWITCH_left_arm", "IK/FK Arm L")
        add_driver_to_constraint("DRV_left_elbow", "Copy Transforms.001", "IKFK_SWITCH_left_arm", "IK/FK Arm L")
        add_driver_to_constraint("DRV_left_wrist", "Copy Transforms.001", "IKFK_SWITCH_left_arm", "IK/FK Arm L")

        # Add the driver to the specified joints and constraints for the right arm
        add_driver_to_constraint("DRV_right_shoulder", "Copy Transforms.001", "IKFK_SWITCH_right_arm", "IK/FK Arm R")
        add_driver_to_constraint("DRV_right_elbow", "Copy Transforms.001", "IKFK_SWITCH_right_arm", "IK/FK Arm R")
        add_driver_to_constraint("DRV_right_wrist", "Copy Transforms.001", "IKFK_SWITCH_right_arm", "IK/FK Arm R")

        # Add the driver to the specified joints and constraints for the left leg
        add_driver_to_constraint("DRV_left_hip", "Copy Transforms.001", "IKFK_SWITCH_left_leg", "IK/FK Leg L")
        add_driver_to_constraint("DRV_left_knee", "Copy Transforms.001", "IKFK_SWITCH_left_leg", "IK/FK Leg L")
        add_driver_to_constraint("DRV_left_ankle", "Copy Transforms.001", "IKFK_SWITCH_left_leg", "IK/FK Leg L")

        # Add the driver to the specified joints and constraints for the right leg
        add_driver_to_constraint("DRV_right_hip", "Copy Transforms.001", "IKFK_SWITCH_right_leg", "IK/FK Leg R")
        add_driver_to_constraint("DRV_right_knee", "Copy Transforms.001", "IKFK_SWITCH_right_leg", "IK/FK Leg R")
        add_driver_to_constraint("DRV_right_ankle", "Copy Transforms.001", "IKFK_SWITCH_right_leg", "IK/FK Leg R")


        # Ensure properties are library overridable and set limits
        if ikfk_switch_left_arm_bone: 
            for prop_name in ["IK/FK Arm L"]:
                ikfk_switch_left_arm_bone.id_properties_ui(prop_name).update(min=0.0, max=1.0, soft_min=0.0, soft_max=1.0)

        if ikfk_switch_right_arm_bone: 
            for prop_name in ["IK/FK Arm R"]:
                ikfk_switch_right_arm_bone.id_properties_ui(prop_name).update(min=0.0, max=1.0, soft_min=0.0, soft_max=1.0)

        if ikfk_switch_left_leg_bone: 
            for prop_name in ["IK/FK Leg L"]:
                ikfk_switch_left_leg_bone.id_properties_ui(prop_name).update(min=0.0, max=1.0, soft_min=0.0, soft_max=1.0)

        if ikfk_switch_right_leg_bone: 
            for prop_name in ["IK/FK Leg R"]:
                ikfk_switch_right_leg_bone.id_properties_ui(prop_name).update(min=0.0, max=1.0, soft_min=0.0, soft_max=1.0)


        # Get the left leg pose bone (VISIBLE LINE FOR POLE VECTOR)
        left_leg_pose_bone = armature.pose.bones.get("LINE_PV_left_leg")
        if left_leg_pose_bone:
            # Add a Stretch To constraint
            stretch_to = left_leg_pose_bone.constraints.new(type='STRETCH_TO')
            stretch_to.target = armature
            stretch_to.subtarget = "CTRL_PV_left_leg"
            stretch_to.head_tail = 0.0
            stretch_to.rest_length = 100.0

        # Get the right leg pose bone (VISIBLE LINE FOR POLE VECTOR)
        right_leg_pose_bone = armature.pose.bones.get("LINE_PV_right_leg")
        if right_leg_pose_bone:
            # Add a Stretch To constraint
            stretch_to = right_leg_pose_bone.constraints.new(type='STRETCH_TO')
            stretch_to.target = armature
            stretch_to.subtarget = "CTRL_PV_right_leg"
            stretch_to.head_tail = 0.0
            stretch_to.rest_length = 100.0

        # Get the left arm pose bone (VISIBLE LINE FOR POLE VECTOR)
        left_arm_pose_bone = armature.pose.bones.get("LINE_PV_left_arm")
        if left_arm_pose_bone:
            # Add a Stretch To constraint
            stretch_to = left_arm_pose_bone.constraints.new(type='STRETCH_TO')
            stretch_to.target = armature
            stretch_to.subtarget = "CTRL_PV_left_arm"
            stretch_to.head_tail = 0.0
            stretch_to.rest_length = 100.0

        # Get the right arm pose bone (VISIBLE LINE FOR POLE VECTOR)
        right_arm_pose_bone = armature.pose.bones.get("LINE_PV_right_arm")
        if right_arm_pose_bone:
            # Add a Stretch To constraint
            stretch_to = right_arm_pose_bone.constraints.new(type='STRETCH_TO')
            stretch_to.target = armature
            stretch_to.subtarget = "CTRL_PV_right_arm"
            stretch_to.head_tail = 0.0
            stretch_to.rest_length = 100.0

        bones_to_hide = ['DRV_left_wrist', 'DRV_left_elbow', 'DRV_left_shoulder',
                         'DRV_left_ankle', 'DRV_left_knee', 'DRV_left_hip',
                         'DRV_right_wrist', 'DRV_right_elbow', 'DRV_right_shoulder',
                         'DRV_right_ankle', 'DRV_right_knee', 'DRV_right_hip',
                         'IK_left_wrist', 'IK_left_shoulder', 'IK_left_elbow',
                         'IK_right_wrist', 'IK_right_shoulder', 'IK_right_elbow',
                         'IK_left_ankle', 'IK_left_knee', 'IK_left_hip',
                         'IK_right_ankle', 'IK_right_knee', 'IK_right_hip']  
        for bone_name in bones_to_hide:
            if bone_name in armature.pose.bones:
                bone = armature.pose.bones[bone_name]
                bone.bone.hide = True


        def toggle_ikfk_visibility():
            bone_names = {
                "left_arm": "IKFK_SWITCH_left_arm",
                "right_arm": "IKFK_SWITCH_right_arm",
                "left_leg": "IKFK_SWITCH_left_leg",
                "right_leg": "IKFK_SWITCH_right_leg",
            }

            property_names = {
                "left_arm": "IK/FK Arm L",
                "right_arm": "IK/FK Arm R",
                "left_leg": "IK/FK Leg L",
                "right_leg": "IK/FK Leg R",
            }

            ik_bones = {
                "left_arm": ["CTRL_IK_left_arm", "CTRL_PV_left_arm", "LINE_PV_left_arm"],
                "right_arm": ["CTRL_IK_right_arm", "CTRL_PV_right_arm", "LINE_PV_right_arm"],
                "left_leg": ["CTRL_IK_left_leg", "CTRL_PV_left_leg", "LINE_PV_left_leg"],
                "right_leg": ["CTRL_IK_right_leg", "CTRL_PV_right_leg", "LINE_PV_right_leg"],
            }

            fk_bones = {
                "left_arm": ["FK_left_wrist", "FK_left_shoulder", "FK_left_elbow"],
                "right_arm": ["FK_right_wrist", "FK_right_shoulder", "FK_right_elbow"],
                "left_leg": ["FK_left_ankle", "FK_left_knee", "FK_left_hip"],
                "right_leg": ["FK_right_ankle", "FK_right_knee", "FK_right_hip"],
            }

            def add_visibility_driver(bone_name, armature, property_path, hide_if_value):
                if bone_name in armature.pose.bones:
                    bone = armature.pose.bones[bone_name]
                    fcurve = bone.bone.driver_add("hide")
                    driver = fcurve.driver
                    driver.type = 'SCRIPTED'
                    var = driver.variables.new()
                    var.name = 'switch'
                    var.type = 'SINGLE_PROP'
                    target = var.targets[0]
                    target.id_type = 'OBJECT'
                    target.id = armature
                    target.data_path = property_path
                    if hide_if_value:
                        driver.expression = "switch == 1"
                    else:
                        driver.expression = "switch == 0"

            try:
                for limb, switch_bone in bone_names.items():
                    bone = armature.pose.bones[switch_bone]
                    property_name = property_names[limb]
                    property_value = bone.get(property_name, None)
                    if property_value is None:
                        raise Exception(f"Property '{property_name}' not found on bone '{switch_bone}'")
                    
                    property_path = f'pose.bones["{switch_bone}"]["{property_name}"]'

                    if property_value == 0:
                        for bone_name in ik_bones[limb]:
                            add_visibility_driver(bone_name, armature, property_path, hide_if_value=True)
                        for bone_name in fk_bones[limb]:
                            add_visibility_driver(bone_name, armature, property_path, hide_if_value=False)
                    elif property_value == 1:
                        for bone_name in fk_bones[limb]:
                            add_visibility_driver(bone_name, armature, property_path, hide_if_value=True)
                        for bone_name in ik_bones[limb]:
                            add_visibility_driver(bone_name, armature, property_path, hide_if_value=False)
                    else:
                        for bone_name in fk_bones[limb]:
                            add_visibility_driver(bone_name, armature, property_path, hide_if_value=True)
                        for bone_name in ik_bones[limb]:
                            add_visibility_driver(bone_name, armature, property_path, hide_if_value=True)
                
                print("Script executed successfully")
            
            except KeyError as e:
                raise Exception(f"Error: {e}")

        # Execute the function
        toggle_ikfk_visibility()


        bpy.ops.object.mode_set(mode='POSE')
        return {'FINISHED'}
#######################################################################################################################################################################################





# TODO once we have AFV, we need to replace this with load animation, so you can load any animation onto any body and treat them separately
class OP_LoadPose(bpy.types.Operator, ImportHelper):
    bl_idname = "object.load_pose"
    bl_label = "Load Pose"
    bl_description = ("Load relaxed-hand model pose from file")
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: StringProperty(
        default="*.npz;*.npy;*.json;*.pkl", # this originally worked for .pkl files only, but they have been since removed.  Let us know if that's a problem, we just need a good .pkl file to test against.
        options={'HIDDEN'} 
    )

    anim_format: EnumProperty(
        name="Format",
        items=(
            ("AMASS", "AMASS (Y-up)", ""),
            ("blender", "Blender (Z-up)", ""),
        ),
    )

    hand_pose: EnumProperty(
        name="Hand Pose Override",
        items=[
            ("disabled", "Disabled", ""),
            ("relaxed", "Relaxed", ""),
            ("flat", "Flat", ""),
        ]
    )

    # taking this out for now
    '''
    update_shape: BoolProperty(
        name="Update shape parameters",
        description="Update shape parameters using the beta shape information in the loaded file.  This is hard coded to false for SMPLH.",
        default=False
    )
    '''

    frame_number: IntProperty(
        name="Frame Number",
        description="Select the frame of the animation you'd like to load.  Only for .npz files.",
        default = 0,
        min = 0
    )


    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh or armature is active object
            return ( ((context.object.type == 'MESH') and (context.object.parent.type == 'ARMATURE')) or (context.object.type == 'ARMATURE'))
        except: return False

    def execute(self, context):
        obj = bpy.context.object

        SMPL_version = bpy.context.object['SMPL_version']
        gender = bpy.context.object['gender']
        joint_names = MODEL_JOINT_NAMES[SMPL_version].value
        num_joints = len(joint_names)
        num_body_joints = MODEL_BODY_JOINTS[SMPL_version].value
        num_hand_joints = MODEL_HAND_JOINTS[SMPL_version].value

        if obj.type == 'MESH':
            armature = obj.parent
        else:
            armature = obj
            obj = armature.children[0]
            context.view_layer.objects.active = obj # mesh needs to be active object for recalculating joint locations

        print("Loading: " + self.filepath)

        translation = None
        global_orient = None
        body_pose = None
        jaw_pose = None
        #leye_pose = None
        #reye_pose = None
        left_hand_pose = None
        right_hand_pose = None
        betas = None
        expression = None
        with open(self.filepath, "rb") as f:
            extension = os.path.splitext(self.filepath)[1]
            if extension == ".pkl": 
                data = pickle.load(f, encoding="latin1")
            elif extension == ".npz":
                data = np.load(f, allow_pickle=True)
            elif extension == ".npy":
                data = np.load(f, allow_pickle=True)
            elif extension == ".json":
                data = json.load(f)

            if "global_orient" in data:
                global_orient = np.array(data["global_orient"]).reshape(3)

            # it's not working anymore for some reason, but loading the betas onto a body isn't that useful because you could just load the body instead.  
            '''
            if extension in ['.npz', 'pkl']:
                betas = np.array(data["betas"]).reshape(-1).tolist()
    
            # Update shape if selected
            # TODO once we get the SMPLH regressor, we can take the SMPLH part out of this
            if self.update_shape and SMPL_version != 'SMPLH':
                bpy.ops.object.mode_set(mode='OBJECT')

                if (extension in ['.npz', 'pkl']):
                    for index, beta in enumerate(betas):
                        key_block_name = f"Shape{index:03}"

                        if key_block_name in obj.data.shape_keys.key_blocks:
                            obj.data.shape_keys.key_blocks[key_block_name].value = beta
                        else:
                            print(f"ERROR: No key block for: {key_block_name}")

                bpy.ops.object.update_joint_locations('EXEC_DEFAULT')
            '''

            if extension == '.pkl':
                correct_pose_key = 'pose'

                try: 
                    np.array(data['pose'])

                except KeyError:
                    correct_pose_key = "poses"
                body_pose = np.array(data[correct_pose_key])

                if body_pose.shape != (1, num_body_joints * 3):
                    print(f"Invalid body pose dimensions: {body_pose.shape}")
                    return {'CANCELLED'}

                body_pose = np.array(data[correct_pose_key]).reshape(num_body_joints, 3)

                # jaw_pose = np.array(data["jaw_pose"]).reshape(3)
                # leye_pose = np.array(data["leye_pose"]).reshape(3)
                # reye_pose = np.array(data["reye_pose"]).reshape(3)
                # left_hand_pose = np.array(data["left_hand_pose"]).reshape(-1, 3)
                # right_hand_pose = np.array(data["right_hand_pose"]).reshape(-1, 3)
                # expression = np.array(data["expression"]).reshape(-1).tolist()

                # pose just the body
                for index in range(num_body_joints): 
                    pose_rodrigues = body_pose[index]
                    bone_name = joint_names[index + 1] 
                    set_pose_from_rodrigues(armature, bone_name, pose_rodrigues, frame=bpy.data.scenes[0].frame_current)

            elif extension == '.npz':
                correct_pose_key = 'pose'

                try: 
                    np.array(data['pose'])

                except KeyError:
                    correct_pose_key = "poses"

                print (f"using '{correct_pose_key}'")

                pose_index = max(0, min(self.frame_number, (len(np.array(data[correct_pose_key]))))) # clamp the frame they give you from 0 and the max number of frames in this poses array 
                body_pose = np.array(data[correct_pose_key][pose_index]).reshape(len(joint_names), 3)

                # pose the entire body
                for index in range(len(joint_names)):
                    pose_rodrigues = body_pose[index]
                    bone_name = joint_names[index]
                    set_pose_from_rodrigues(armature, bone_name, pose_rodrigues, frame=bpy.data.scenes[0].frame_current)

            elif extension == '.npy':
                # assuming a .npy containing a single pose
                body_pose = np.array(data).reshape(len(joint_names), 3)
                
                # pose the entire body
                for index in range(len(joint_names)):
                    pose_rodrigues = body_pose[index]
                    bone_name = joint_names[index]
                    set_pose_from_rodrigues(armature, bone_name, pose_rodrigues, frame=bpy.data.scenes[0].frame_current)

            elif extension == '.json':
                with open(self.filepath, "rb") as f:
                    pose_data = json.load(f)
                    
                pose = np.array(pose_data["pose"]).reshape(num_joints, 3)

                for index in range(num_joints):
                    pose_rodrigues = pose[index]
                    bone_name = joint_names[index]
                    set_pose_from_rodrigues(armature, bone_name, pose_rodrigues, frame=bpy.data.scenes[0].frame_current)
                       

        if global_orient is not None:
            set_pose_from_rodrigues(armature, "pelvis", global_orient, frame=bpy.data.scenes[0].frame_current)

        '''
        if translation is not None:
            # Set translation
            armature.location = (translation[0], -translation[2], translation[1])
        '''

        if self.hand_pose != 'disabled':
            context.window_manager.smpl_tool.hand_pose = self.hand_pose
            bpy.ops.object.set_hand_pose('EXEC_DEFAULT')

        # Activate corrective poseshapes
        bpy.ops.object.set_pose_correctives('EXEC_DEFAULT')

        # Set face expression
        if extension == '.pkl':
            set_pose_from_rodrigues(armature, "jaw", jaw_pose, frame=bpy.data.scenes[0].frame_current)

            for index, exp in enumerate(expression):
                key_block_name = f"Exp{index:03}"

                if key_block_name in obj.data.shape_keys.key_blocks:
                    obj.data.shape_keys.key_blocks[key_block_name].value = exp
                else:
                    print(f"ERROR: No key block for: {key_block_name}")

        bpy.ops.object.set_pose_correctives('EXEC_DEFAULT')
        key_all_pose_correctives(obj=obj, index=bpy.data.scenes[0].frame_current)

        correct_for_anim_format(self.anim_format, armature)
        bpy.ops.object.snap_to_ground_plane('EXEC_DEFAULT')
        armature.keyframe_insert(data_path="location", frame=bpy.data.scenes[0].frame_current)

        return {'FINISHED'}


class OP_SetExpressionPreset(bpy.types.Operator):
    bl_idname = "object.set_expression_preset"
    bl_label = "Set Expression Preset"
    bl_description = ("Sets the facial expression to artist created presets")
    bl_options = {"REGISTER", "UNDO"}

    preset: bpy.props.StringProperty()

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if mesh is active object
            return ((context.object.type == 'MESH') and (bpy.context.object['SMPL_version'] != "SMPLH"))
        except: return False

    def execute(self, context):
        SMPL_version = bpy.context.object['SMPL_version']


        obj = context.object
        if not obj or not obj.data.shape_keys:
            self.report(
                {"WARNING"}, "Object has no shape keys. Please select a SMPL family mesh."
            )
            return {"CANCELLED"}

        if SMPL_version == 'SMPLX':
            bpy.ops.object.reset_expression_shape('EXEC_DEFAULT')
            presets = {
                "pleasant": [0, .3, 0, -.892, 0, 0, 0, 0, -1.188, 0, .741, -2.83, 0, -1.48, 0, 0, 0, 0, 0, -.89, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .89, 0, 0, 2.67],
                "happy": [0.9, 0, .741, -2, .27, -.593, -.29, 0, .333, 0, 1.037, -1, 0, .7, .296, 0, 0, -1.037, 0, 0, 0, 1.037, 0, 3],
                "excited": [-.593, .593, .7, -1.55, -.32, -1.186, -.43, -.14, -.26, -.88, 1, -.74, 1, -.593, 0, 0, 0, 0, 0, 0, -.593],
                "sad": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 2,2,-2, 1, 1.6, 2, 1.6],
                "frustrated": [0, 0, -1.33, 1.63, 0, -1.185, 2.519, 0, 0, -.593, -.444],
                "angry": [0, 0, -2.074, 1.185, 1.63, -1.78, 1.63, .444, .89, .74, -4, 1.63, -1.93, -2.37, -4],
            }
        
        elif SMPL_version == 'SUPR':
            bpy.ops.object.reset_expression_shape('EXEC_DEFAULT')
            presets = {
                "pleasant": [0.3, 0, -0.2, 0, 0, 0, 0, 0, 0.3, 0.4],
                "happy":  [1.3, 0, 0, 0, -0.3, 0, 0.7, 0, -1, 0],
                "excited": [0.7, 0, -1.1, 0.9, -0.5, 0, 0, 0, 0, 0],
                "sad": [-0.35, 0, 0, -0.25, 1.75, 0, 0, 0, 0, 1.15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8.5],
                "frustrated": [0, 0, 0.7, -0.25, -1.5, -1, 0, 1.8, 0, 1.3],
                "angry": [0, 0, 1.2, 0, -1, -1, -1.5, 2.3, 0, -3],
            }
            
        preset_values = presets.get(self.preset)

        if not preset_values:
            self.report({"WARNING"}, f"Unknown preset: {self.preset}")
            return {"CANCELLED"}

        for i, value in enumerate(preset_values):
            key_name = f"Exp{i:03}"
            key_block = obj.data.shape_keys.key_blocks.get(key_name)
            if key_block:
                key_block.value = value

        return {"FINISHED"}


class OP_ModifyMetadata(bpy.types.Operator):
    bl_idname = "object.modify_avatar"
    bl_label = "Modify Metadata"
    bl_description = ("Click this button to save the meta data (SMPL_version and gender) on the selected avatar.  The SMPL_version and gender that are selected in the `Create Avatar` section will be assigned to the selected mesh.  This allows the plugin to know what kind of skeleton it's dealing with.  To view the meta data, click `Read Metadata` and check the console, or click `Object Properties` (orange box underneath the scene collection) > `Custom Properties`")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if in Object Mode
            return (context.active_object is None) or (context.active_object.mode == 'OBJECT')
        except: return False

    def execute(self, context):
        gender = context.window_manager.smpl_tool.gender
        SMPL_version = context.window_manager.smpl_tool.SMPL_version

        #define custom properties on the avatar itself to store this kind of data so we can use it whenever we need to
        bpy.context.object['gender'] = gender
        bpy.context.object['SMPL_version'] = SMPL_version

        bpy.ops.object.read_avatar('EXEC_DEFAULT')

        return {'FINISHED'}


class OP_ReadMetadata(bpy.types.Operator):
    bl_idname = "object.read_avatar"
    bl_label = "Read Metadata"
    bl_description = ("Prints the selected Avatar's meta data to the console")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if in Object Mode
            return (context.active_object is None) or (context.active_object.mode == 'OBJECT')
        except: return False

    def execute(self, context):
        print(bpy.context.object['gender'])
        print(bpy.context.object['SMPL_version'])

        return {'FINISHED'}

# this is a work around for a problem with the blender worker's fbx output.  Currently those .fbx's shape keys ranges are limited to 0 and 1.  
# this is a known problem, but I don't know why it's doing that.  For now, we can fix it using this button
class OP_FixBlendShapeRanges(bpy.types.Operator):
    bl_idname = "object.fix_blend_shape_ranges"
    bl_label = "Fix Blendshape Ranges"
    bl_description = ("Click this for any imported .fbx to set the min and max values for all blendshapes to -10 to 10.  At the time of writing this, Blender hardcodes imported .fbx file's blendshape ranges to 0 and 1.  This means that all meshcapade.me and digidoppel .fbx files will have their blendshapes clamped.  Until Blender fixes this issue (they're working on it), this button functions as a workaround.")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        try:
            # Enable button only if in Object Mode
            return (context.active_object is None) or (context.active_object.mode == 'OBJECT')
        except: return False

    def execute(self, context):
        
        for sk in context.active_object.data.shape_keys.key_blocks:
            sk.slider_min = -10
            sk.slider_max = 10

        return {'FINISHED'}


OPERATORS = [
    OP_LoadAvatar,
    OP_CreateAvatar,
    OP_SetTexture,
    OP_MeasurementsToShape,
    OP_RandomBodyShape,
    OP_RandomFaceShape,
    OP_ResetBodyShape,
    OP_ResetFaceShape,
    OP_RandomExpressionShape,
    OP_ResetExpressionShape,
    OP_SetExpressionPreset,
    OP_SnapToGroundPlane,
    OP_UpdateJointLocations,
    OP_CalculatePoseCorrectives,
    OP_CalculatePoseCorrectivesForSequence,
    OP_SetHandpose,
    OP_WritePoseToJSON,
    OP_WritePoseToConsole,
    OP_ResetPose,
    OP_ZeroOutPoseCorrectives,
    OP_LoadPose,
    OP_ModifyMetadata,
    OP_ReadMetadata,
    OP_FixBlendShapeRanges,
    OP_AutoRig,
]
