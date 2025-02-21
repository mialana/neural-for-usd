from pxr import Usd, UsdShade, Sdf, UsdGeom, Vt




stage: Usd.Stage = Usd.Stage.Open("woodenPlane-orig.usda")

prim: Usd.Prim= stage.GetPrimAtPath("/_materials/japanese_toy")
material = UsdShade.Material = UsdShade.Material.Get(stage, "/_materials/japanese_toy")
varname = material.CreateInput('file1:varname', Sdf.ValueTypeNames.String)
varname.Set("st")

pbrShader: UsdShade.Shader = UsdShade.Shader.Get(stage, "/_materials/japanese_toy/previewShader")

file1Shader = UsdShade.Shader.Define(stage,'/_materials/japanese_toy/file1')
place2dTexture = UsdShade.Shader.Define(stage, '/_materials/japanese_toy/place2dTexture1')

place2dTexture = UsdShade.Shader.Define(stage, '/_materials/japanese_toy/place2dTexture1')
place2dTexture.CreateIdAttr('UsdPrimvarReader_float2')

file1Shader.CreateIdAttr('UsdUVTexture')

file = file1Shader.CreateInput('file', Sdf.ValueTypeNames.Asset)
file.Set('textures/plane_toy_Plane_toy_Diffuse.png')

file1Shader.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
file1Shader.CreateInput('wrapS', Sdf.ValueTypeNames.Token).Set('repeat')
file1Shader.CreateInput('wrapT', Sdf.ValueTypeNames.Token).Set('repeat')



file1Shader.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)

file1Shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(place2dTexture.ConnectableAPI(), 'result')

print(pbrShader.GetInput('diffuseColor').GetPrim().RemoveProperty('inputs:diffuseColor'))
col = pbrShader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Asset)
col.ConnectToSource(file1Shader.ConnectableAPI(), 'rgb')
place2dTexture.CreateInput('varname', Sdf.ValueTypeNames.String).ConnectToSource(varname)

geom = stage.GetPrimAtPath("/japanese_toy/japanese_toy")

geom.ApplyAPI(UsdShade.Tokens.MaterialBindingAPI)

pv_api = UsdGeom.PrimvarsAPI(geom)

pv_api.CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, interpolation='faceVarying').Set(pv_api.GetPrimvar('uv').Get())

nm = pv_api.CreatePrimvar('normals', Sdf.ValueTypeNames.Vector3fArray, interpolation='faceVarying', elementSize=1260)

from PIL import Image
import numpy as np

normal_map = Image.open('./textures/result.png').convert('RGB')
normal_map = np.array(normal_map, dtype=np.float16) / 255.0
normal_map = np.round(normal_map, 3)
normal_map = normal_map * 2.0 - 1.0
nm.Set(Vt.Vec3fArray.FromNumpy(normal_map))

pv_api.RemovePrimvar('uv')

# print(pv_api.GetPrimvar('uv').GetTypeName())
# uv = geom.GetAttribute('primvars:uv')

# geom.CreateAttribute('primvars:st', Sdf.ValueTypeNames.TexCoord2fArray).Set(uv.Get())

# print(uv.Get())
# geom.CreateAttribute('primvars:st', Usd.Tokens.)
            
# UsdShade.MaterialBindingAPI(geom).Bind(material)

stage.Export("/Users/liu.amy05/Documents/Neural-for-USD/src/USDC_Wooden_Plane/woodenPlaneEditedFirst.usda")