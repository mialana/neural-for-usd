#include "myengine.h"

#include <pxr/imaging/hgi/tokens.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/rprim.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>

#include <pxr/imaging/hio/image.h>
#include <pxr/imaging/hdSt/hioConversions.h>
#include "pxr/base/tf/scoped.h"

#include <QDebug>
#include <QOpenGLFunctions>
#include <QOpenGLFramebufferObject>

MyEngine::MyEngine(OpenGLContext* context, Scene const* scene, int width, int height)
    : mp_context(context)
    , mcp_scene(scene)
    , m_rendererPluginId(MyEngine::GetRendererPluginId())
    , m_width(width)
    , m_height(height)
    , m_taskControllerPrefix(SdfPath("/defaultTaskController"))
    , m_renderedCollection(HdTokens->geometry, HdReprSelector(HdReprTokens->smoothHull))
{
    GlfGLContextScopeHolder contextHolder(mpOpenGLContext);
    qDebug() << mpOpenGLContext->IsInitialized();

    mp_drawTarget = GlfDrawTarget::New(GfVec2i(width, height), false);
    mp_drawTarget->Bind();
    mp_drawTarget->AddAttachment(HdAovTokens->color, GL_RGBA, GL_FLOAT, GL_RGBA);
    mp_drawTarget->AddAttachment(HdAovTokens->depth,
                                 GL_DEPTH_COMPONENT,
                                 GL_FLOAT,
                                 GL_DEPTH_COMPONENT);
    mp_drawTarget->Unbind();

    mp_hgi = Hgi::CreatePlatformDefaultHgi();
    m_hgiDriver = HdDriver({HgiTokens->renderDriver, VtValue(mp_hgi.get())});

    mp_renderDelegate = MyEngine::CreateRenderDelegateForPlugin(m_rendererPluginId);

    mp_renderIndex = HdRenderIndex::New(mp_renderDelegate.Get(), {&m_hgiDriver});
    mp_renderIndex->InsertSceneIndex(mcp_scene->getFinalSceneIndex(), m_taskControllerPrefix);

    // mp_taskControllerIndex = HdxTaskControllerSceneIndex::New(m_taskControllerPrefix, )
    /* set task controller settings */
    mp_taskController = new HdxTaskController(mp_renderIndex, m_taskControllerPrefix);

    HdxRenderTaskParams params;
    params.viewport = GfVec4f(0, 0, m_width, m_height);
    params.enableLighting = true;
    mp_taskController->SetRenderParams(params);

    mp_taskController->SetCollection(m_renderedCollection);
    mp_taskController->SetRenderTags(TfTokenVector());
    mp_taskController->SetOverrideWindowPolicy(CameraUtilFit);

    mp_taskController->SetEnableSelection(true);
    mp_selectionTracker = std::make_shared<HdxSelectionTracker>();

    GfVec4f selectionColor = GfVec4f(1.f, 1.f, 0.f, .5f);
    mp_taskController->SetSelectionColor(selectionColor);

    VtValue selTrackVt(mp_selectionTracker);
    m_hdEngine.SetTaskContextData(HdxTokens->selectionState, selTrackVt);

    TfTokenVector mp_aovOutputs{HdAovTokens->color, HdAovTokens->depth};
    mp_taskController->SetRenderOutputs(mp_aovOutputs);

    // set default clear color of aov
    GfVec4f clearColor = GfVec4f(0.2f, 0.2f, 0.2f, 1.0f);
    HdAovDescriptor colorAovDesc = mp_taskController->GetRenderOutputSettings(HdAovTokens->color);

    if (colorAovDesc.format != HdFormatInvalid) {
        colorAovDesc.clearValue = VtValue(clearColor);
        mp_taskController->SetRenderOutputSettings(HdAovTokens->color, colorAovDesc);
    }

    // set default clear color of aov
    GfVec4f clearColorDepth = GfVec4f(1.f, 0.1f, 0.1f, 1.0f);
    HdAovDescriptor depthAovDesc = mp_taskController->GetRenderOutputSettings(HdAovTokens->depth);

    if (depthAovDesc.format != HdFormatInvalid) {
        depthAovDesc.clearValue = VtValue(clearColorDepth);
        mp_taskController->SetRenderOutputSettings(HdAovTokens->depth, depthAovDesc);
    }

    mp_taskController->SetEnablePresentation(true);
    mp_taskController->SetViewportRenderOutput(HdAovTokens->color);
    /* * */
}

MyEngine::~MyEngine()
{
    mp_drawTarget = GlfDrawTargetRefPtr();
    // Destroy objects in opposite order of construction.
    delete mp_taskController;

    mp_renderIndex->RemoveSceneIndex(mcp_scene->getFinalSceneIndex());
    delete mp_renderIndex;

    mp_renderDelegate = nullptr;
    mcp_scene = nullptr;
}

void MyEngine::Render()
{
    mpOpenGLContext = pxr::GlfGLContextSharedPtr()->GetSharedGLContext();
    GlfGLContextScopeHolder contextHolder(mpOpenGLContext);

    mp_drawTarget->Bind();
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    HdTaskSharedPtrVector tasks = mp_taskController->GetRenderingTasks();
    // HdTaskSharedPtrVector pickingTasks = mp_taskController->GetPickingTasks();
    // tasks.insert(tasks.begin(), pickingTasks.begin(), pickingTasks.end());

    m_hdEngine.Execute(mp_renderIndex, &tasks);

    VtValue aov;
    HgiTextureHandle aovTexture;

    if (m_hdEngine.GetTaskContextData(HdAovTokens->color, &aov)) {
        if (aov.IsHolding<HgiTextureHandle>()) {
            aovTexture = aov.Get<HgiTextureHandle>();
        }
    }

    VtValue depthaov;
    HgiTextureHandle depthTexture;
    if (m_hdEngine.GetTaskContextData(HdAovTokens->depth, &aov)) {
        if (depthaov.IsHolding<HgiTextureHandle>()) {
            depthTexture = depthaov.Get<HgiTextureHandle>();
        }
    }

    uint32_t framebuffer = mp_context->defaultFramebufferObject();
    m_interop.TransferToApp(mp_hgi.get(),
                            aovTexture,
                            depthTexture,
                            HgiTokens->OpenGL,
                            VtValue(framebuffer),
                            GfVec4i(0, 0, m_width * 2, m_height * 2));

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    mp_drawTarget->Unbind();
}

void MyEngine::setRenderSize(int width, int height)
{
    mpOpenGLContext = pxr::GlfGLContextSharedPtr()->GetSharedGLContext();
    GlfGLContextScopeHolder contextHolder(mpOpenGLContext);

    m_width = width;
    m_height = height;

    mp_taskController->SetRenderViewport(GfVec4f(0, 0, width, height));
    mp_taskController->SetRenderBufferSize(GfVec2i(width, height));

    GfRange2f displayWindow(GfVec2f(0, 0), GfVec2f(width, height));
    GfRect2i renderBufferRect(GfVec2i(0, 0), width, height);
    GfRect2i dataWindow = renderBufferRect.GetIntersection(renderBufferRect);
    CameraUtilFraming framing(displayWindow, dataWindow);

    mp_taskController->SetFraming(framing);

    mp_drawTarget->Bind();
    mp_drawTarget->SetSize(GfVec2i(width, height));
    mp_drawTarget->Unbind();
}

void MyEngine::setCameraMatrices(GfMatrix4d camView, GfMatrix4d camProj)
{
    m_camView = camView;
    m_camProj = camProj;
    mp_taskController->SetFreeCameraMatrices(camView, camProj);
}

void MyEngine::prepareDefaultLighting()
{
    // set a spot light to the camera position
    GfVec3d camPos = m_camView.GetInverse().ExtractTranslation();
    GlfSimpleLight l;
    l.SetAmbient(GfVec4f(0, 0, 0, 0));
    l.SetPosition(GfVec4f(camPos[0], camPos[1], camPos[2], 1));

    // create default material
    GlfSimpleMaterial material;
    material.SetAmbient(GfVec4f(2, 2, 2, 1.0));
    material.SetSpecular(GfVec4f(0.1, 0.1, 0.1, 1.0));
    material.SetShininess(32.0);

    // create scene ambience
    GfVec4f sceneAmbient(1.0, 1.0, 1.0, 1.0);

    // configure into lighting context and subsequently the task controller
    GlfSimpleLightingContextRefPtr lightingContextState = GlfSimpleLightingContext::New();

    lightingContextState->SetLights({l});
    lightingContextState->SetMaterial(material);
    lightingContextState->SetSceneAmbient(sceneAmbient);
    lightingContextState->SetUseLighting(true);
    mp_taskController->SetLightingState(lightingContextState);
}

SdfPath MyEngine::findIntersection(GfVec2f screenPos) {
    // create a narrowed frustum on the given position
    float normalizedXPos = screenPos[0] / m_width;
    float normalizedYPos = screenPos[1] / m_height;

    // GfVec2d size(1.0 / m_width, 1.0 / m_height);

    // GfCamera gfCam;
    // gfCam.SetFromViewAndProjectionMatrix(m_camView, m_camProj);
    // GfFrustum frustum = gfCam.GetFrustum();

    // auto nFrustum = frustum.ComputeNarrowedFrustum(
    //     GfVec2d(2.0 * normalizedXPos - 1.0, 2.0 * (1.0 - normalizedYPos) - 1.0),
    //     size);

    // // check the intersection from the narrowed frustum
    // HdxPickHitVector allHits;
    // HdxPickTaskContextParams pickParams;
    // pickParams.resolveMode = HdxPickTokens->resolveNearestToCenter;
    // pickParams.viewMatrix = nFrustum.ComputeViewMatrix();
    // pickParams.projectionMatrix = nFrustum.ComputeProjectionMatrix();
    // pickParams.collection = _collection;
    // pickParams.outHits = &allHits;
    // const VtValue vtPickParams(pickParams);

    // _engine.SetTaskContextData(HdxPickTokens->pickParams, vtPickParams);

    // // render with the picking task
    // HdTaskSharedPtrVector tasks = _taskController->GetPickingTasks();
    // _engine.Execute(_renderIndex, &tasks);

    // // get the hitting point
    // if (allHits.size() != 1)
    //     return SdfPath();

    // const SdfPath path = allHits[0].objectId.ReplacePrefix(
    //     _taskControllerId, SdfPath::AbsoluteRootPath());

    return SdfPath();
}

void MyEngine::renderToFile()
{
    mpOpenGLContext = pxr::GlfGLContextSharedPtr()->GetSharedGLContext();
    GlfGLContextScopeHolder contextHolder(mpOpenGLContext);

    std::string destFile = std::string(PROJECT_SOURCE_DIR) + "/engineDebug.png";
    // bool result = mp_drawTarget->WriteToFile("depth", destFile);
    // qDebug() << "Success:" << result << "Destination file:" << destFile;
    HdRenderBuffer* b = mp_taskController->GetRenderOutput(HdAovTokens->color);
    b->Resolve();

    HioImage::StorageSpec storage;
    storage.width = m_width;
    storage.height = m_height;
    storage.format = HdStHioConversions::GetHioFormat(b->GetFormat());;
    storage.flipped = true;
    storage.data = b->Map();

    TfScoped<> scopedUnmap([b](){ b->Unmap(); });
    {
        TRACE_FUNCTION_SCOPE("writing image");

        const HioImageSharedPtr image = HioImage::OpenForWriting(destFile);
        const bool writeSuccess = image && image->Write(storage);

        if (!writeSuccess) {
            TF_RUNTIME_ERROR("Failed to write image to %s",
                             destFile.c_str());
        }
    }

}

GLuint MyEngine::getEngineFrameBuffer()
{
    return mp_drawTarget->GetFramebufferId();
}

HdPluginRenderDelegateUniqueHandle MyEngine::CreateRenderDelegateForPlugin(TfToken pluginId)
{
    HdRendererPluginRegistry& registry = HdRendererPluginRegistry::GetInstance();

    return registry.CreateRenderDelegate(pluginId);
}

TfToken MyEngine::GetRendererPluginId(std::optional<std::string> idString)
{
    HdRendererPluginRegistry& registry = HdRendererPluginRegistry::GetInstance();
    if (idString.has_value()) {
        TfToken t = TfToken(idString.value());
        if (registry.IsRegisteredPlugin(t)) {  // check if given string is a valid plugin
            qInfo() << "Chosen renderer plugin is" << t.GetString();
            return t;
        } else {  // else warn
            qWarning() << "The renderer plugin" << idString
                       << "is invalid or unregistered. Will use default renderer plugin.";
        }
    }

    TfToken t = registry.GetDefaultPluginId(true);
    qInfo() << "Chosen renderer plugin is" << t.GetString();
    return t;
}
