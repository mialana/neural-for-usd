#pragma once

#include "scene.h"
#include "openglcontext.h"

#include <pxr/imaging/hd/engine.h>
#include <pxr/imaging/hd/driver.h>
#include <pxr/imaging/hd/rendererPlugin.h>
#include <pxr/imaging/hd/pluginRenderDelegateUniqueHandle.h>
#include <pxr/imaging/hdx/taskController.h>
#include <pxr/imaging/hdx/taskControllerSceneIndex.h>
#include <pxr/imaging/glf/drawTarget.h>
#include <pxr/imaging/hgi/hgi.h>
#include <pxr/imaging/hgiInterop/hgiInterop.h>
#include <pxr/imaging/glf/glContext.h>

#define VALUE(string) #string
#define TO_LITERAL(string) VALUE(string)

PXR_NAMESPACE_USING_DIRECTIVE

class MyEngine
{
public:
    MyEngine(OpenGLContext* context, Scene const* scene, int width, int height);

    ~MyEngine();

    void Render();

    void setRenderSize(int width, int height);
    void setCameraMatrices(GfMatrix4d camView, GfMatrix4d camProj);
    void prepareDefaultLighting();

    SdfPath findIntersection(GfVec2f screenPos);

    void renderToFile();
    GLuint getEngineFrameBuffer();

    static HdPluginRenderDelegateUniqueHandle CreateRenderDelegateForPlugin(TfToken pluginId);
    static TfToken GetRendererPluginId(std::optional<std::string> idString = std::nullopt);

private:
    OpenGLContext* mp_context;
    Scene const* mcp_scene; // contains scene indices

    int m_width, m_height;
    TfToken m_rendererPluginId;  // id of the plugin we want our engine will use. OpenUSD ships with HdStorm, used by UsdView
    GfMatrix4d m_camView, m_camProj;
    SdfPath m_taskControllerPrefix;
    HdRprimCollection m_renderedCollection; // specifier that selects certain prims for our task controller
    HdxSelectionTrackerSharedPtr mp_selectionTracker;

    HdEngine m_hdEngine;
    HdPluginRenderDelegateUniqueHandle mp_renderDelegate; // the "thing" that does the actual render work. handle means it is a uPtr.
    HdxTaskController* mp_taskController;  // container for tasks
    HdxTaskControllerSceneIndexRefPtr mp_taskControllerIndex;
    HdRenderIndex* mp_renderIndex;         // tied to 1 render delegate, but multiple sceneindices. Holds our render-related info + pointers
    GlfDrawTargetRefPtr mp_drawTarget;     // contains information on GL object locations

    HgiUniquePtr mp_hgi; // hydra graphics interface. allows us to execute certain cmds, such as interlopping from Metal to OpenGL
    HdDriver m_hgiDriver; // driver is a c++ object that represents a hardware device. This one will specifically be passed -> render index -> render delegate for HGI usage
    HgiInterop m_interop; // an object that helps transfer data between different graphics APIs / GPUs.
};
