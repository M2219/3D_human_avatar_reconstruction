import bpy

ROOT_BLENDER = '/app/blender/'

# Import Texture
myobj = bpy.ops.import_scene.obj(filepath=ROOT_BLENDER + "./input/mesh_object.obj")

bpy.context.scene.objects.active = bpy.data.objects[0]


# Enhance
bpy.ops.object.modifier_add(type='SUBSURF')
bpy.context.object.modifiers["Subsurf"].levels = 2
bpy.ops.object.shade_smooth()
bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Subsurf")


# Open Texture
bpy.ops.object.editmode_toggle()
bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.001)

bpy.context.screen.areas[2].type = 'IMAGE_EDITOR'
bpy.data.images.load(ROOT_BLENDER + './input/sample_texture.jpg')
bpy.context.screen.areas[2].spaces.active.image = bpy.data.images['sample_texture.jpg']

# Export Texture
bpy.context.screen.areas[2].type = 'VIEW_3D'
bpy.context.screen.areas[2].spaces.active.viewport_shade="TEXTURED"
bpy.ops.mesh.select_all(action='TOGGLE')

bpy.ops.export_scene.obj(filepath=ROOT_BLENDER + './output/' + 'mesh_texture.obj', path_mode='STRIP')

