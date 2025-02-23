# usage: usdrecord [-h] [--mask PRIMPATH[,PRIMPATH...]] [--purposes PURPOSE[,PURPOSE...]] [--sessionLayer SESSIONLAYER] [--disableGpu]
#                  [--disableCameraLight] [--camera CAMERA] [--defaultTime | --frames FRAMESPEC[,FRAMESPEC...]] [--complexity {low,medium,high,veryhigh}]
#                  [--colorCorrectionMode {disabled,sRGB,openColorIO}] [--renderer {Metal}] [--imageWidth IMAGEWIDTH] [--enableDomeLightVisibility]
#                  [--renderPassPrimPath RPPRIMPATH] [--renderSettingsPrimPath RSPRIMPATH] [--traceToFile TRACETOFILE] [--traceFormat {chrome,trace}]
#                  [--memstats]
#                  usdFilePath outputImagePath

# Generates images from a USD file

# positional arguments:
#   usdFilePath           USD file to record
#   outputImagePath       Output image path. For frame ranges, the path must contain exactly one frame number placeholder of the form "###" or "###.###".
#                         Note that the number of hash marks is variable in each group.

# optional arguments:
#   -h, --help            show this help message and exit
#   --mask PRIMPATH[,PRIMPATH...]
#                         Limit stage population to these prims, their descendants and ancestors. To specify multiple paths, either use commas with no
#                         spaces or quote the argument and separate paths by commas and/or spaces.
#   --purposes PURPOSE[,PURPOSE...]
#                         Specify which UsdGeomImageable purposes should be included in the renders. The "default" purpose is automatically included, so
#                         you need specify only the *additional* purposes. If you want more than one extra purpose, either use commas with no spaces or
#                         quote the argument and separate purposes by commas and/or spaces.
#   --sessionLayer SESSIONLAYER
#                         If specified, the stage will be opened with the 'sessionLayer' in place of the default anonymous layer.
#   --disableGpu          Indicates if the GPU should not be used for rendering. If set this not only restricts renderers to those which only run on the
#                         CPU, but additionally it will prevent any tasks that require the GPU from being invoked.
#   --disableCameraLight  Indicates if the default camera lights should not be used for rendering.
#   --camera CAMERA, -cam CAMERA
#                         Which camera to use - may be given as either just the camera's prim name (i.e. just the last element in the prim path), or as a
#                         full prim path. Note that if only the prim name is used and more than one camera exists with that name, which one is used will
#                         effectively be random (default=main_cam)
#   --defaultTime, -d     explicitly operate at the Default time code (the default behavior is to operate at the startTimeCode authored on the UsdStage
#                         which defaults to 0.0)
#   --frames FRAMESPEC[,FRAMESPEC...], -f FRAMESPEC[,FRAMESPEC...]
#                         specify FrameSpec(s) of the time codes to operate on - A FrameSpec consists of up to three floating point values for the start
#                         time code, end time code, and stride of a time code range. A single time code can be specified, or a start and end time code
#                         can be specified separated by a colon (:). When a start and end time code are specified, the stride may optionally be specified
#                         as well, separating it from the start and end time codes with (x). Multiple FrameSpecs can be combined as a comma-separated
#                         list. The following are examples of valid FrameSpecs: 123 - 101:105 - 105:101 - 101:109x2 - 101:110x2 - 101:104x0.5
#   --complexity {low,medium,high,veryhigh}, -c {low,medium,high,veryhigh}
#                         level of refinement to use (default=low)
#   --colorCorrectionMode {disabled,sRGB,openColorIO}, -color {disabled,sRGB,openColorIO}
#                         the color correction mode to use (default=sRGB)
#   --renderer {Metal}, -r {Metal}
#                         Hydra renderer plugin to use when generating images
#   --imageWidth IMAGEWIDTH, -w IMAGEWIDTH
#                         Width of the output image. The height will be computed from this value and the camera's aspect ratio (default=960)
#   --enableDomeLightVisibility
#                         Show the dome light background in the rendered output. If this option is not included and there is a dome light in the stage,
#                         the IBL from it will be used for lighting but not drawn into the background.
#   --renderPassPrimPath RPPRIMPATH, -rp RPPRIMPATH
#                         Specify the Render Pass Prim to use to render the given usdFile. Note that if a renderSettingsPrimPath has been specified in
#                         the stage metadata, using this argument will override that opinion. Furthermore any properties authored on the RenderSettings
#                         will override other arguments (imageWidth, camera, outputImagePath)
#   --renderSettingsPrimPath RSPRIMPATH, -rs RSPRIMPATH
#                         Specify the Render Settings Prim to use to render the given usdFile. Note that if a renderSettingsPrimPath has been specified
#                         in the stage metadata, using this argument will override that opinion. Furthermore any properties authored on the
#                         RenderSettings will override other arguments (imageWidth, camera, outputImagePath)
#   --traceToFile TRACETOFILE
#                         Start tracing at application startup and write --traceFormat specified format output to the specified trace file when the
#                         application quits
#   --traceFormat {chrome,trace}
#                         Output format for trace file specified by --traceToFile. 'chrome' files can be read in chrome, 'trace' files are simple text
#                         reports. (default=chrome)
#   --memstats            Use the Pxr MallocTags memory accounting system to profile USD, saving results to a tmp file, with a summary to the console.
#                         Will have no effect if MallocTags are not supported in the USD installation.

SETTINGS=--traceToFile=trace.txt --traceFormat trace --camera Camera_003 --colorCorrectionMode=openColorIO  --enableDomeLightVisibility

default:
	usdrecord $(SETTINGS) japanesePlaneToy.usda data/tmp/test0000.png

SETTINGS2=--traceToFile=trace.txt --traceFormat trace --camera Camera_003 --complexity veryhigh --colorCorrectionMode=openColorIO  --enableDomeLightVisibility

high:
	usdrecord $(SETTINGS2) japanesePlaneToy.usda data/tmp/test0001.png