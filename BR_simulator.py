bl_info = {
    'name': 'Chemical Reaction Simulator',
    'blender': (3, 0, 0),
    'category': 'Object',
}

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import bpy
import math
import os
import subprocess


#UIで扱う変数の設定
class ReactionSimProperties(bpy.types.PropertyGroup):
    initial_conc_A: bpy.props.IntProperty(name='Initial A', default=50, min=1) #Aの初期濃度
    initial_conc_B: bpy.props.IntProperty(name='Initial B', default=0, min=0) #Bの初期濃度
    coef_a: bpy.props.IntProperty(name='a (A)', default=1, min=1) #反応式中のAの係数(1以上、Aは限定反応成分)
    coef_b: bpy.props.IntProperty(name='b (B)', default=0, min=0) #反応式中のBの係数
    coef_c: bpy.props.IntProperty(name='c (C)', default=1, min=1) #反応式中のCの係数(1以上)
    coef_d: bpy.props.IntProperty(name='d (D)', default=0, min=0) #反応式中のDの係数
    order_A: bpy.props.IntProperty(name='order(A)', default=1, min=0, max=3) #反応速度式でのAの次数
    order_B: bpy.props.IntProperty(name='order(B)', default=0, min=0, max=3) #反応速度式でのBの次数
    reaction_rate: bpy.props.FloatProperty(name='Reaction Rate', default=1, min=0.01) #反応定数
    brown: bpy.props.BoolProperty(name='Brownian Motion', default=True) #ブラウン運動のスイッチ
    
    #A~Dのマテリアルの設定
    material_a: bpy.props.FloatVectorProperty(name='Material A', subtype='COLOR_GAMMA', default=(1.0, 0.0, 0.0), size=3, min=0.0, max=1.0) 
    material_b: bpy.props.FloatVectorProperty(name='Material B', subtype='COLOR_GAMMA', default=(0.5, 0.0, 1.0), size=3, min=0.0, max=1.0)
    material_c: bpy.props.FloatVectorProperty(name='Material C', subtype='COLOR_GAMMA', default=(0.5, 1.0, 0.0), size=3, min=0.0, max=1.0)
    material_d: bpy.props.FloatVectorProperty(name='Material D', subtype='COLOR_GAMMA', default=(0.0, 1.0, 1.0), size=3, min=0.0, max=1.0)

#UI
class ReactionSimPanel(bpy.types.Panel):
    bl_label = 'Reaction Simulator'
    bl_idname = 'OBJECT_reaction'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Reaction Sim'

    def draw(self, context):
        layout = self.layout
        props = context.scene.reaction_sim_props

        layout.label(text='Reaction Rate Constant')
        layout.prop(props, 'reaction_rate')
        layout.separator()

        layout.label(text='Initial Concentrations')
        layout.prop(props, 'initial_conc_A')
        layout.prop(props, 'initial_conc_B')
        layout.separator()

        layout.label(text='Reaction Coefficients')
        layout.label(text='aA + bB → cC + dD')
        layout.prop(props, 'coef_a')
        layout.prop(props, 'coef_b')
        layout.prop(props, 'coef_c')
        layout.prop(props, 'coef_d')
        layout.separator()
        
        layout.label(text='Reaction Order')
        layout.prop(props, 'order_A')
        layout.prop(props, 'order_B')
        layout.separator()
        
        layout.operator('object.material_color')
        layout.separator()

        layout.prop(props, 'brown')
        layout.separator()

        layout.operator('object.run_simulator')

#シミュレーションの実行、条件確認
class RunSimulator(bpy.types.Operator):
    bl_idname = 'object.run_simulator'
    bl_label = 'Run Simulator'
    
    def execute(self, context):
        props = context.scene.reaction_sim_props
        CA0 = props.initial_conc_A
        CB0 = props.initial_conc_B
        a = props.coef_a
        b = props.coef_b
        c = props.coef_c
        d = props.coef_d
        k = props.reaction_rate
        order_A = props.order_A
        order_B = props.order_B
        material_A = props.material_a
        material_B = props.material_b
        material_C = props.material_c
        material_D = props.material_d
        material = [material_A, material_B, material_C, material_D]
        brown = props.brown

        #限定反応成分がAになっているか確認
        if (CA0 * b > CB0 * a) and b!= 0:
            self.report({'ERROR'}, 'A is the limiting reagent: (Initial_A) / a must be <= (Initial_B) / b')
            return {'CANCELLED'}

        else:
            run_simulation(CA0, CB0, a, b, c, d, k, order_A, order_B, material, brown)
            
            self.report({'INFO'}, 'Simulation executed successfully!')
            return {'FINISHED'}
        
#マテリアル設定のダイアログ
class Material(bpy.types.Operator):
    bl_idname = 'object.material_color'
    bl_label = 'Material Menu'
    bl_description = 'Material'
    bl_options = {'REGISTER', 'UNDO'}
    
    def draw(self, context):
        props = context.scene.reaction_sim_props
        layout = self.layout
        
        layout.prop(props, 'material_a', text='Material A')
        layout.prop(props, 'material_b', text='Material B')
        layout.prop(props, 'material_c', text='Material C')
        layout.prop(props, 'material_d', text='Material D')
        
    def execute(self, context):
        self.report({'INFO'}, "Material Properties Updated!")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        scene = context.scene
        wm = context.window_manager
        
        return wm.invoke_props_dialog(self)

#シミュレーション前のBlenderリセット
def reset_blender():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)

    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)

    for obj in bpy.data.objects:
        if obj.animation_data:
            obj.animation_data_clear()
    bpy.context.scene.frame_set(1)
    
#マテリアルを割り当てる
def apply_material(material_name, color):
    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = False 
    mat.diffuse_color = (color[0], color[1], color[2], 1.0) 
    return mat

#微分方程式を立てる
def formula_simple(x_A, t, k, CA0):
    dx_dt = k / CA0
    return dx_dt

def formula_simple_A(x_A, t, k, CA0, order_A):
    dx_dt = k  * (CA0 ** (order_A-1)) * ((1-x_A) ** order_A)
    return dx_dt

def formula_simple_B(x_A, t, k, CA0, order_B, theta_b, b_a):
    dx_dt = k * (CA0 ** (order_B-1)) * ((theta_b - b_a * x_A) ** order_B)
    return dx_dt

def formula_complex(x_A, t, k, CA0, order_A, order_B, theta_b, b_a):
    dx_dt = k * (CA0 ** (order_A+order_B-1)) * ((1-x_A) ** order_A) * ((theta_b - b_a * x_A) ** order_B)
    return dx_dt

#ブラウン運動
def brown(obj, frame, switch_brown):
    if not switch_brown:
        return

    dx = np.random.uniform(-0.1, 0.1)
    dy = np.random.uniform(-0.1, 0.1)
    dz = np.random.uniform(-0.1, 0.1)

    x = obj.location.x + dx
    y = obj.location.y + dy
    z = obj.location.z + dz

    if (x*x + y*y) > 9 : #容器からはみ出ないように制限
        x = obj.location.x - dx
        y = obj.location.y - dy
    if z < 0 or z > 5:
        z = obj.location.z - dz

    obj.location = (x, y, z)
    obj.keyframe_insert(data_path='location', frame=frame)
    
#シミュレーション本体
def run_simulation(CA0, CB0, a, b, c, d, k, order_A, order_B, material, switch_brown):    
    k = k / 10
    #aA + bB → cC + dD
    b_a = b / a
    c_a = c / a
    d_a = d / a
    if b != 0:
        c_b = c / b
        d_b = d / b
        
    theta_b = CB0 / CA0
 
    #アニメーションを有効にするキーフレーム数を決定
    if theta_b != 0 and b != 0:
        if order_A == 0 and order_B == 0:
            t = int(CA0 / k)
           
        elif order_B == 0 and order_A == 1:
            t = int(4.605 / k) 
        else:
            log = math.log(1 - ((b * 0.99) / (a * theta_b)))
            t = int((-a / (b * k)) * log)
    elif b == 0 and order_B != 0:
        t = int(100 / (k * CB0))
    else:
        if order_A <= 1:
            t = int(4.605 / k)
        else:
            t = int(100 / (k * CA0))
        
    dt = np.linspace(0, t, 2*t)  

    bpy.context.scene.frame_end = 10 * t + 100

    #微分方程式を解く
    if order_A == 0 and order_B == 0:
        x_A = odeint(formula_simple, 0, dt, args=(k, CA0)).flatten()
    elif order_A == 0:
        x_A = odeint(formula_simple_B, 0, dt, args=(k, CA0, order_B, theta_b, b_a)).flatten()
    elif order_B == 0:
        x_A = odeint(formula_simple_A, 0, dt, args=(k, CA0, order_A)).flatten()
    else:
        x_A = odeint(formula_complex, 0, dt, args=(k, CA0, order_A, order_B, theta_b, b_a)).flatten()
    
    #A~Dの濃度と時間の関係
    x_A = np.where(x_A > 1, 1, x_A) #反応率x=1で停止
    
    A_t = [CA0*(1-x) for x in x_A]
    
    A_t_int = [int(x) for x in A_t if x != 0]
    A_t_int.append(1)
    
    B_t_int = [int(b_a * (x)) for x in A_t_int]
    C_t_int = [int(c_a * (CA0 - x)) for x in A_t_int]
    D_t_int = [int(d_a* (CA0 - x)) for x in A_t_int]


    #Blender上での操作
    reset_blender()
    
    #反応器の作成
    bpy.ops.mesh.primitive_cylinder_add(radius=3, depth=6, location=(0, 0, 3))
    container = bpy.context.object
    container.name = 'Reactor'
    container.display_type = 'WIRE'  
    container.hide_render = True

    bpy.ops.mesh.primitive_circle_add(radius=3, location=(0, 0, 5))
    bpy.context.object.name = 'Surface'
    bpy.context.object.hide_render = True
    
    #マテリアル適用
    material_A = apply_material('A', material[0])
    material_B = apply_material('B', material[1])
    material_C = apply_material('C', material[2])
    material_D = apply_material('D', material[3])

    #A~Dの粒子を必要な分だけ作成(表示・非表示で操作)
    CAf = 0
    if a != 0:
        particles_A = []
        for i in range(1, CA0+1):
            if not 'A_{}'.format(i) in bpy.data.objects:
                r = np.random.uniform(0, 3)
                theta = np.random.uniform(0, 2 * math.pi)
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                z = np.random.uniform(0, 5)
                bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(x, y, z))
                obj = bpy.context.object
                obj.name = 'A_{}'.format(i)
                obj.keyframe_insert(data_path='hide_viewport', frame=1)
                obj.keyframe_insert(data_path='hide_render', frame=1)
                particles_A.append(obj)
            obj = bpy.data.objects['A_{}'.format(i)]
            if len(obj.data.materials) == 0:
                obj.data.materials.append(material_A)   

    CBf = int(CA0 * (theta_b - b_a))
    if b != 0 or CB0 != 0:   
        particles_B = []    
        for i in range(1, CB0+1):
            if not 'B_{}'.format(i) in bpy.data.objects:
                r = np.random.uniform(0, 3)
                theta = np.random.uniform(0, 2 * math.pi)
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                z = np.random.uniform(0, 5)
                bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(x, y, z))
                obj = bpy.context.object
                obj.name = 'B_{}'.format(i)
                obj.keyframe_insert(data_path='hide_viewport', frame=1)
                obj.keyframe_insert(data_path='hide_render', frame=1)
                particles_B.append(obj)
            obj = bpy.data.objects['B_{}'.format(i)]
            if len(obj.data.materials) == 0:
                obj.data.materials.append(material_B)   

    CCf = int(CA0 * c_a)          
    if c != 0:   
        particles_C = []    
        for i in range(1, CCf + 1):
            if not 'C_{}'.format(i) in bpy.data.objects:
                bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(0, 0, 0))
                obj = bpy.context.object
                obj.name = 'C_{}'.format(i)
                obj.hide_viewport = True
                obj.hide_render = True
                obj.keyframe_insert(data_path='hide_viewport', frame=1)
                obj.keyframe_insert(data_path='hide_render', frame=1)
                particles_C.append(obj)
            obj = bpy.data.objects['C_{}'.format(i)]
            if len(obj.data.materials) == 0:
                obj.data.materials.append(material_C)

    CDf = int(CA0 * d_a)           
    if d != 0:   
        particles_D = []    
        for i in range(1, CDf + 1):
            if not 'D_{}'.format(i) in bpy.data.objects:
                bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(0, 0, 0))
                obj = bpy.context.object
                obj.name = 'D_{}'.format(i)
                obj.hide_viewport = True
                obj.hide_render = True
                obj.keyframe_insert(data_path='hide_viewport', frame=1)
                obj.keyframe_insert(data_path='hide_render', frame=1)
                particles_D.append(obj)
            obj = bpy.data.objects['D_{}'.format(i)]
            if len(obj.data.materials) == 0:
                obj.data.materials.append(material_D)    


    #反応による各粒子の変化を適用
    bpy.context.scene.frame_current = 1   
    for i in range(1,len(A_t_int)):
        frame = i*5

        if A_t_int[i] > 0:
            for j in range(A_t_int[i], A_t_int[i-1] + 1):
                obj_A = 'A_{}'.format(j)
                obj_A_loc = bpy.data.objects[obj_A].location.copy()

                obj = bpy.data.objects[obj_A]
                obj.hide_viewport = True
                obj.hide_render = True
                obj.keyframe_insert(data_path='hide_viewport', frame=frame)
                obj.keyframe_insert(data_path='hide_render', frame=frame)
                
                for l in range(j, int(b_a * j) + 1):
                    obj_B = 'B_{}'.format(j)
                    obj = bpy.data.objects[obj_B]
                    obj.hide_viewport = True
                    obj.hide_render = True
                    obj.keyframe_insert(data_path='hide_viewport', frame=frame)
                    obj.keyframe_insert(data_path='hide_render', frame=frame)

                for l in range(j, CCf + 1, CA0):
                    obj_C = 'C_{}'.format(l)
                    obj = bpy.data.objects[obj_C]
                    obj.location = obj_A_loc  
                    obj.hide_render = False
                    obj.hide_viewport = False
                    obj.keyframe_insert(data_path='hide_viewport', frame=frame)
                    obj.keyframe_insert(data_path='hide_render', frame=frame)
                    
                    if (not switch_brown) and (c >= 2 or (c >= 1 and d>= 1)):
                        dx = np.random.uniform(-0.15, 0.15)
                        dy = np.random.uniform(-0.15, 0.15)
                        dz = np.random.uniform(-0.15, 0.15)

                        x = obj.location.x + dx
                        y = obj.location.y + dy
                        z = obj.location.z + dz
                        
                        if (x*x + y*y) > 9 :
                            x = obj.location.x - dx
                            y = obj.location.y - dy
                        if z < 0 or z > 5:
                            z = obj.location.z - dz

                        obj.location = (x, y, z)

                for l in range(j, CDf + 1,CA0):    
                    obj_D = 'D_{}'.format(l)
                    obj = bpy.data.objects[obj_D]
                    obj.location = obj_A_loc
                    obj.hide_render = False
                    obj.hide_viewport = False
                    obj.keyframe_insert(data_path='hide_viewport', frame=frame)
                    obj.keyframe_insert(data_path='hide_render', frame=frame)
                    
                    if (not switch_brown) and (c >= 2 or (c >= 1 and d>= 1)):
                        dx = np.random.uniform(-0.15, 0.15)
                        dy = np.random.uniform(-0.15, 0.15)
                        dz = np.random.uniform(-0.15, 0.15)

                        x = obj.location.x + dx
                        y = obj.location.y + dy
                        z = obj.location.z + dz
                        
                        if (x*x + y*y) > 9 :
                            x = obj.location.x - dx
                            y = obj.location.y - dy
                        if z < 0 or z > 5:
                            z = obj.location.z - dz

                        obj.location = (x, y, z)

    #キーフレームを1にしておく
    bpy.context.scene.frame_current = 1 
    
    #ブラウン運動のオンオフはここで
    if switch_brown == True:
        print("Applying Brownian Motion")
        for frame in range(1, 10 * t + 100, 5):
            for obj in bpy.data.objects:
                if obj.name.startswith('A_') or obj.name.startswith('B_') or obj.name.startswith('C_') or obj.name.startswith('D_'):
                    brown(obj, frame, switch_brown)
                    
#アドオンに必要なクラスを書いておく                    
classes = [ReactionSimProperties, RunSimulator, ReactionSimPanel, Material]


#アドオンとしての登録
def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.reaction_sim_props = bpy.props.PointerProperty(type=ReactionSimProperties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.reaction_sim_props

if __name__ == '__main__':
    register()
    