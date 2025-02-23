import numpy as np
from pxr import UsdRender

camera_positions = np.zeros([100, 4, 4], dtype=np.float16)

print(camera_positions)




# cam_path = "/World/MyOrthoCam"
# stage: Usd.Stage = Usd.Stage.CreateInMemory()
# root_prim = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
# stage.SetDefaultPrim(root_prim.GetPrim())

# camera = create_orthographic_camera(stage, cam_path)

# usda = stage.GetRootLayer().ExportToString()
# print(usda)

# # Check that the camera was created
# prim = camera.GetPrim()
# assert prim.IsValid()
# assert camera.GetPath() == Sdf.Path(cam_path)
# assert prim.GetTypeName() == "Camera"
# projection = camera.GetProjectionAttr().Get()
# assert projection == UsdGeom.Tokens.orthographic