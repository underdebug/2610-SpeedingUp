#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_xlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "nurbs.hpp"

namespace {

void CheckVkResult(VkResult result, const char* call) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(call) +
                                 " failed: " + std::to_string(result));
    }
}

#define VK_CHECK(call) CheckVkResult((call), #call)

double MaxPositionDeviation(const nurbs::Surface& surface) {
    constexpr int kSampleIntervals = 50;
    double max_error = 0.0;
    for (int j = 0; j <= kSampleIntervals; ++j) {
        for (int i = 0; i <= kSampleIntervals; ++i) {
            const double u = static_cast<double>(i) / kSampleIntervals;
            const double v = static_cast<double>(j) / kSampleIntervals;
            const auto cox = nurbs::coxDeBoor(surface, u, v);
            const auto de_boor = nurbs::deBoor(surface, u, v);
            const double dx = cox.x - de_boor.x;
            const double dy = cox.y - de_boor.y;
            const double dz = cox.z - de_boor.z;
            max_error =
                std::max(max_error, std::sqrt(dx * dx + dy * dy + dz * dz));
        }
    }
    return max_error;
}

int RunNurbsMathTest() {
    const double max_error = MaxPositionDeviation(nurbs::exampleSurface());
    std::cout << "maximum Cox/de Boor position error: " << max_error << '\n';
    return max_error < 1e-10 ? EXIT_SUCCESS : EXIT_FAILURE;
}

struct Vertex {
    float x, y, z, r, g, b;
};

enum class EvaluationMethod {
    kCoxDeBoor,
    kDeBoor,
};

std::vector<Vertex> MakeMesh(const nurbs::Surface& surface,
                             EvaluationMethod method) {
    constexpr int kResolution = 80;
    constexpr int kRowSize = kResolution + 1;
    std::vector<Vertex> grid;
    grid.reserve(kRowSize * kRowSize);
    for (int j = 0; j <= kResolution; ++j) {
        for (int i = 0; i <= kResolution; ++i) {
            const double u = static_cast<double>(i) / kResolution;
            const double v = static_cast<double>(j) / kResolution;
            const auto p = method == EvaluationMethod::kCoxDeBoor
                               ? nurbs::coxDeBoor(surface, u, v)
                               : nurbs::deBoor(surface, u, v);
            // Exaggerated contrast so the surface shape is visible.
            const float t =
                std::clamp(static_cast<float>(0.5 + 0.5 * p.z), 0.0f, 1.0f);
            grid.push_back({static_cast<float>(p.x), static_cast<float>(p.y),
                            static_cast<float>(p.z), 0.12f + 0.52f * t,
                            0.30f + 0.40f * t, 0.85f - 0.48f * t});
        }
    }

    // The vertex shader assigns edge coordinates to each group of three.
    std::vector<Vertex> vertices;
    vertices.reserve(6 * kResolution * kResolution);
    for (int j = 0; j < kResolution; ++j) {
        for (int i = 0; i < kResolution; ++i) {
            const int a = j * kRowSize + i;
            const int b = a + 1;
            const int c = a + kRowSize;
            const int d = c + 1;
            vertices.insert(vertices.end(), {grid[a], grid[b], grid[c], grid[b],
                                             grid[d], grid[c]});
        }
    }
    return vertices;
}

struct V3 {
    float x, y, z;
};

V3 operator-(V3 a, V3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

V3 cross(V3 a, V3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

float dot(V3 a, V3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

V3 normalize(V3 v) {
    float k = 1.f / std::sqrt(dot(v, v));
    return {v.x * k, v.y * k, v.z * k};
}

// Column-major matrices, exactly as GLSL mat4 expects.
struct Mat4 {
    float a[16]{};

    float& at(int row, int col) {
        return a[col * 4 + row];
    }
};

struct DrawConstants {
    Mat4 mvp;
    float clip_per_pixel[2];
};

Mat4 mul(const Mat4& A, const Mat4& B) {
    Mat4 C;
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            for (int k = 0; k < 4; ++k) {
                C.at(r, c) += A.a[k * 4 + r] * B.a[c * 4 + k];
            }
        }
    }
    return C;
}

Mat4 perspective(float fovy, float aspect, float nearZ, float farZ) {
    Mat4 P;
    float f = 1.f / std::tan(fovy * 0.5f);
    P.at(0, 0) = f / aspect;
    P.at(1, 1) = -f;  // Vulkan framebuffer Y direction (GLM-style Y flip)
    P.at(2, 2) = farZ / (nearZ - farZ);
    P.at(2, 3) = farZ * nearZ / (nearZ - farZ);
    P.at(3, 2) = -1;
    return P;
}

Mat4 lookAt(V3 eye, V3 target, V3 up) {
    V3 f = normalize(target - eye), s = normalize(cross(f, up)),
       u = cross(s, f);
    Mat4 M;
    M.at(0, 0) = s.x;
    M.at(0, 1) = s.y;
    M.at(0, 2) = s.z;
    M.at(0, 3) = -dot(s, eye);
    M.at(1, 0) = u.x;
    M.at(1, 1) = u.y;
    M.at(1, 2) = u.z;
    M.at(1, 3) = -dot(u, eye);
    M.at(2, 0) = -f.x;
    M.at(2, 1) = -f.y;
    M.at(2, 2) = -f.z;
    M.at(2, 3) = dot(f, eye);
    M.at(3, 3) = 1.f;
    return M;
}

std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("Cannot open shader: " + path +
                                 " (run from build/)");
    }
    size_t length = static_cast<size_t>(f.tellg());
    std::vector<char> bytes(length);
    f.seekg(0);
    f.read(bytes.data(), length);
    if (!f || (length % 4)) {
        throw std::runtime_error("Invalid SPIR-V: " + path);
    }
    return bytes;
}

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

class App {
public:
    void run() {
        initWindow();
        initVulkan();
        surface_ = nurbs::exampleSurface();
        compareMethods();
        uploadMesh(method_);
        uploadControlPoints();
        std::cout << "Mouse drag: orbit; wheel: zoom; 1: Cox-de Boor; 2: de "
                     "Boor; Esc: exit\n";
        while (running_) {
            while (XPending(display_)) {
                XEvent e{};
                XNextEvent(display_, &e);
                handleEvent(e);
            }
            if (!running_) {
                break;
            }
            if (width_ > 0 && height_ > 0) {
                draw();
            }
        }
        vkDeviceWaitIdle(device_);
        cleanup();
    }

private:
    Display* display_ = nullptr;
    Window window_ = 0;
    Atom wmDelete_{};
    bool running_ = true, dragging_ = false, resize_ = false;
    EvaluationMethod method_ = EvaluationMethod::kDeBoor;
    int width_ = 1000, height_ = 720, lastX_ = 0, lastY_ = 0;
    float yaw_ = 0.65f, pitch_ = 0.55f, distance_ = 7.f;
    nurbs::Surface surface_;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surfaceHandle_ = VK_NULL_HANDLE;
    VkPhysicalDevice gpu_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE, presentQueue_ = VK_NULL_HANDLE;
    uint32_t graphicsFamily_ = UINT32_MAX, presentFamily_ = UINT32_MAX;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat colorFormat_ = VK_FORMAT_UNDEFINED,
             depthFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
    std::vector<VkImage> images_;
    std::vector<VkImageView> imageViews_;
    std::vector<VkFramebuffer> framebuffers_;
    VkImage depthImage_ = VK_NULL_HANDLE;
    VkDeviceMemory depthMemory_ = VK_NULL_HANDLE;
    VkImageView depthView_ = VK_NULL_HANDLE;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipeline control_point_pipeline_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_ = VK_NULL_HANDLE;
    VkSemaphore acquired_ = VK_NULL_HANDLE, rendered_ = VK_NULL_HANDLE;
    Buffer vertex_;
    uint32_t vertex_count_ = 0;
    Buffer control_points_;
    uint32_t control_point_count_ = 0;

    void initWindow() {
        display_ = XOpenDisplay(nullptr);
        if (!display_) {
            throw std::runtime_error(
                "XOpenDisplay failed: set DISPLAY and run under XWayland/X11");
        }
        int screen = DefaultScreen(display_);
        window_ = XCreateSimpleWindow(
            display_, RootWindow(display_, screen), 80, 80, width_, height_, 0,
            BlackPixel(display_, screen), WhitePixel(display_, screen));
        XStoreName(display_, window_, "NURBS Surface - Vulkan + Xlib/XWayland");
        XSelectInput(display_, window_,
                     ExposureMask | KeyPressMask | StructureNotifyMask |
                         ButtonPressMask | ButtonReleaseMask |
                         PointerMotionMask);
        wmDelete_ = XInternAtom(display_, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(display_, window_, &wmDelete_, 1);
        XMapWindow(display_, window_);
        XFlush(display_);
    }

    void handleEvent(XEvent& e) {
        if (e.type == ClientMessage &&
            static_cast<Atom>(e.xclient.data.l[0]) == wmDelete_) {
            running_ = false;
        } else if (e.type == DestroyNotify) {
            running_ = false;
        } else if (e.type == ConfigureNotify) {
            if (width_ != e.xconfigure.width ||
                height_ != e.xconfigure.height) {
                width_ = e.xconfigure.width;
                height_ = e.xconfigure.height;
                resize_ = true;
            }
        } else if (e.type == ButtonPress) {
            if (e.xbutton.button == Button1) {
                dragging_ = true;
                lastX_ = e.xbutton.x;
                lastY_ = e.xbutton.y;
            }
            if (e.xbutton.button == Button4) {
                distance_ = std::max(2.f, distance_ * 0.9f);
            }
            if (e.xbutton.button == Button5) {
                distance_ = std::min(30.f, distance_ * 1.1f);
            }
        } else if (e.type == ButtonRelease && e.xbutton.button == Button1) {
            dragging_ = false;
        } else if (e.type == MotionNotify && dragging_) {
            yaw_ += 0.007f * (e.xmotion.x - lastX_);
            pitch_ = std::clamp(pitch_ + 0.007f * (e.xmotion.y - lastY_), -1.4f,
                                1.4f);
            lastX_ = e.xmotion.x;
            lastY_ = e.xmotion.y;
        } else if (e.type == KeyPress) {
            KeySym key = XLookupKeysym(&e.xkey, 0);
            if (key == XK_Escape || key == XK_q) {
                running_ = false;
            }
            if (key == XK_1 || key == XK_2) {
                const EvaluationMethod selected =
                    key == XK_1 ? EvaluationMethod::kCoxDeBoor
                                : EvaluationMethod::kDeBoor;
                if (selected != method_) {
                    VK_CHECK(vkDeviceWaitIdle(device_));
                    uploadMesh(selected);
                }
            }
        }
    }

    void compareMethods() {
        const double max_error = MaxPositionDeviation(surface_);
        std::cout << "Maximum Cox/de Boor position deviation = " << max_error
                  << '\n';
        if (max_error > 1e-9) {
            throw std::runtime_error("Cox/de Boor mismatch");
        }
    }

    void initVulkan() {
        VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        ai.pApplicationName = "NURBS Vulkan XWayland";
        ai.apiVersion = VK_API_VERSION_1_0;
        const char* exts[] = {VK_KHR_SURFACE_EXTENSION_NAME,
                              VK_KHR_XLIB_SURFACE_EXTENSION_NAME};
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &ai;
        ici.enabledExtensionCount = 2;
        ici.ppEnabledExtensionNames = exts;
        VK_CHECK(vkCreateInstance(&ici, nullptr, &instance_));
        VkXlibSurfaceCreateInfoKHR sci{
            VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR};
        sci.dpy = display_;
        sci.window = window_;
        VK_CHECK(
            vkCreateXlibSurfaceKHR(instance_, &sci, nullptr, &surfaceHandle_));
        chooseDevice();
        createDevice();
        createCommandPool();
        createSync();
        createSwapchainResources();
    }

    void chooseDevice() {
        uint32_t count = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, nullptr));
        if (!count) {
            throw std::runtime_error("No Vulkan physical device");
        }
        std::vector<VkPhysicalDevice> devices(count);
        VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, devices.data()));
        for (auto dev : devices) {
            uint32_t extCount = 0;
            vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount,
                                                 nullptr);
            std::vector<VkExtensionProperties> extensions(extCount);
            vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount,
                                                 extensions.data());
            bool swapchainSupported = false;
            for (auto& ext : extensions) {
                if (std::strcmp(ext.extensionName,
                                VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
                    swapchainSupported = true;
                }
            }
            if (!swapchainSupported) {
                continue;
            }
            uint32_t n = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(dev, &n, nullptr);
            std::vector<VkQueueFamilyProperties> families(n);
            vkGetPhysicalDeviceQueueFamilyProperties(dev, &n, families.data());
            uint32_t g = UINT32_MAX, p = UINT32_MAX;
            for (uint32_t i = 0; i < n; ++i) {
                VkBool32 supported = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surfaceHandle_,
                                                     &supported);
                if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                    g == UINT32_MAX) {
                    g = i;
                }
                if (supported && p == UINT32_MAX) {
                    p = i;
                }
                if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                    supported) {
                    g = p = i;
                    break;
                }
            }
            if (g == UINT32_MAX || p == UINT32_MAX) {
                continue;
            }
            uint32_t formats = 0;
            vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surfaceHandle_, &formats,
                                                 nullptr);
            uint32_t modes = 0;
            vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surfaceHandle_,
                                                      &modes, nullptr);
            if (!formats || !modes) {
                continue;
            }
            gpu_ = dev;
            graphicsFamily_ = g;
            presentFamily_ = p;
            break;
        }
        if (gpu_ == VK_NULL_HANDLE) {
            throw std::runtime_error(
                "No device supports Vulkan Xlib presentation");
        }
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(gpu_, &props);
        std::cout << "GPU: " << props.deviceName << "\n";
    }

    void createDevice() {
        float priority = 1.f;
        std::vector<VkDeviceQueueCreateInfo> queues;
        std::vector<uint32_t> families = {graphicsFamily_};
        if (presentFamily_ != graphicsFamily_) {
            families.push_back(presentFamily_);
        }
        for (auto family : families) {
            VkDeviceQueueCreateInfo q{
                VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            q.queueFamilyIndex = family;
            q.queueCount = 1;
            q.pQueuePriorities = &priority;
            queues.push_back(q);
        }
        const char* extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        VkDeviceCreateInfo dc{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dc.queueCreateInfoCount = static_cast<uint32_t>(queues.size());
        dc.pQueueCreateInfos = queues.data();
        dc.enabledExtensionCount = 1;
        dc.ppEnabledExtensionNames = extensions;
        VK_CHECK(vkCreateDevice(gpu_, &dc, nullptr, &device_));
        vkGetDeviceQueue(device_, graphicsFamily_, 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, presentFamily_, 0, &presentQueue_);
    }

    void createCommandPool() {
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        ci.queueFamilyIndex = graphicsFamily_;
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK(vkCreateCommandPool(device_, &ci, nullptr, &commandPool_));
        VkCommandBufferAllocateInfo ai{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = commandPool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device_, &ai, &command_));
    }

    void createSync() {
        VkSemaphoreCreateInfo ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(device_, &ci, nullptr, &acquired_));
        VK_CHECK(vkCreateSemaphore(device_, &ci, nullptr, &rendered_));
    }

    uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags flags) {
        VkPhysicalDeviceMemoryProperties props{};
        vkGetPhysicalDeviceMemoryProperties(gpu_, &props);
        for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
            if ((bits & (1u << i)) &&
                (props.memoryTypes[i].propertyFlags & flags) == flags) {
                return i;
            }
        }
        throw std::runtime_error("No suitable Vulkan memory type");
    }

    Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
        Buffer b;
        VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        ci.size = size;
        ci.usage = usage;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(device_, &ci, nullptr, &b.handle));
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device_, b.handle, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = memoryType(
            req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &b.memory));
        VK_CHECK(vkBindBufferMemory(device_, b.handle, b.memory, 0));
        return b;
    }

    void destroyBuffer(Buffer& b) {
        if (b.handle) {
            vkDestroyBuffer(device_, b.handle, nullptr);
        }
        if (b.memory) {
            vkFreeMemory(device_, b.memory, nullptr);
        }
        b = {};
    }

    uint32_t uploadVertices(const std::vector<Vertex>& vertices,
                            Buffer& buffer) {
        const VkDeviceSize size = vertices.size() * sizeof(Vertex);
        if (buffer.handle == VK_NULL_HANDLE) {
            buffer = createBuffer(size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        }
        void* data = nullptr;
        VK_CHECK(vkMapMemory(device_, buffer.memory, 0, size, 0, &data));
        std::memcpy(data, vertices.data(), size);
        vkUnmapMemory(device_, buffer.memory);
        return static_cast<uint32_t>(vertices.size());
    }

    void uploadControlPoints() {
        std::vector<Vertex> points;
        points.reserve(surface_.cp.size());
        for (const auto& control_point : surface_.cp) {
            const auto p = nurbs::project(control_point);
            points.push_back({static_cast<float>(p.x), static_cast<float>(p.y),
                              static_cast<float>(p.z), 1.0f, 0.46f, 0.045f});
        }
        control_point_count_ = uploadVertices(points, control_points_);
    }

    void uploadMesh(EvaluationMethod method) {
        // Both methods tessellate the same grid, so they share one buffer.
        vertex_count_ = uploadVertices(MakeMesh(surface_, method), vertex_);
        method_ = method;
        const char* method_name =
            method == EvaluationMethod::kCoxDeBoor ? "Cox-de Boor" : "de Boor";
        const std::string title = std::string("NURBS Surface - ") + method_name;
        XStoreName(display_, window_, title.c_str());
        std::cout << "Method: " << method_name
                  << ", triangles=" << vertex_count_ / 3 << '\n';
    }

    VkFormat chooseDepthFormat() {
        for (VkFormat format :
             {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT,
              VK_FORMAT_D16_UNORM}) {
            VkFormatProperties props{};
            vkGetPhysicalDeviceFormatProperties(gpu_, format, &props);
            if (props.optimalTilingFeatures &
                VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
                return format;
            }
        }
        throw std::runtime_error("No supported depth format");
    }

    void createDepth() {
        depthFormat_ = chooseDepthFormat();
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = depthFormat_;
        ci.extent = {extent_.width, extent_.height, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VK_CHECK(vkCreateImage(device_, &ci, nullptr, &depthImage_));
        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(device_, depthImage_, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex =
            memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &depthMemory_));
        VK_CHECK(vkBindImageMemory(device_, depthImage_, depthMemory_, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = depthImage_;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = depthFormat_;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(device_, &vi, nullptr, &depthView_));
    }

    void createRenderPass() {
        VkAttachmentDescription attachments[2]{};
        attachments[0].format = colorFormat_;
        attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        attachments[1].format = depthFormat_;
        attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference color{0,
                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depth{
            1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = 1;
        sub.pColorAttachments = &color;
        sub.pDepthStencilAttachment = &depth;
        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstStageMask = dep.srcStageMask;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        ci.attachmentCount = 2;
        ci.pAttachments = attachments;
        ci.subpassCount = 1;
        ci.pSubpasses = &sub;
        ci.dependencyCount = 1;
        ci.pDependencies = &dep;
        VK_CHECK(vkCreateRenderPass(device_, &ci, nullptr, &renderPass_));
    }

    VkShaderModule shader(const char* file) {
        auto bytes = readFile(file);
        VkShaderModuleCreateInfo ci{
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = bytes.size();
        ci.pCode = reinterpret_cast<const uint32_t*>(bytes.data());
        VkShaderModule out;
        VK_CHECK(vkCreateShaderModule(device_, &ci, nullptr, &out));
        return out;
    }

    void createPipelines() {
        VkPushConstantRange pc{VK_SHADER_STAGE_VERTEX_BIT, 0,
                               sizeof(DrawConstants)};
        VkPipelineLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        li.pushConstantRangeCount = 1;
        li.pPushConstantRanges = &pc;
        VK_CHECK(
            vkCreatePipelineLayout(device_, &li, nullptr, &pipelineLayout_));
        pipeline_ = createPipeline("surface.vert.spv", "surface.frag.spv",
                                   VK_VERTEX_INPUT_RATE_VERTEX, VK_TRUE);
        control_point_pipeline_ =
            createPipeline("control_points.vert.spv", "control_points.frag.spv",
                           VK_VERTEX_INPUT_RATE_INSTANCE, VK_FALSE);
    }

    VkPipeline createPipeline(const char* vertex_shader,
                              const char* fragment_shader,
                              VkVertexInputRate input_rate,
                              VkBool32 use_depth) {
        VkShaderModule vert = shader(vertex_shader);
        VkShaderModule frag = shader(fragment_shader);
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;
        stages[1].pName = "main";
        VkVertexInputBindingDescription binding{0, sizeof(Vertex), input_rate};
        VkVertexInputAttributeDescription attrs[2] = {
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, x)},
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, r)}};
        VkPipelineVertexInputStateCreateInfo vertex{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vertex.vertexBindingDescriptionCount = 1;
        vertex.pVertexBindingDescriptions = &binding;
        vertex.vertexAttributeDescriptionCount = 2;
        vertex.pVertexAttributeDescriptions = attrs;
        VkPipelineInputAssemblyStateCreateInfo assembly{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo raster{
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.f;
        VkPipelineMultisampleStateCreateInfo ms{
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo depth{
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth.depthTestEnable = use_depth;
        depth.depthWriteEnable = use_depth;
        depth.depthCompareOp = VK_COMPARE_OP_LESS;
        VkPipelineColorBlendAttachmentState blend{};
        blend.colorWriteMask = 0xf;
        VkPipelineColorBlendStateCreateInfo blending{
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blending.attachmentCount = 1;
        blending.pAttachments = &blend;
        VkDynamicState states[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                   VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dyn.dynamicStateCount = 2;
        dyn.pDynamicStates = states;
        VkGraphicsPipelineCreateInfo pi{
            VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pi.stageCount = 2;
        pi.pStages = stages;
        pi.pVertexInputState = &vertex;
        pi.pInputAssemblyState = &assembly;
        pi.pViewportState = &viewport;
        pi.pRasterizationState = &raster;
        pi.pMultisampleState = &ms;
        pi.pDepthStencilState = &depth;
        pi.pColorBlendState = &blending;
        pi.pDynamicState = &dyn;
        pi.layout = pipelineLayout_;
        pi.renderPass = renderPass_;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkResult result = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1,
                                                    &pi, nullptr, &pipeline);
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        VK_CHECK(result);
        return pipeline;
    }

    void createSwapchainResources() {
        VkSurfaceCapabilitiesKHR caps{};
        VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu_, surfaceHandle_,
                                                           &caps));
        uint32_t count = 0;
        VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_, surfaceHandle_,
                                                      &count, nullptr));
        std::vector<VkSurfaceFormatKHR> formats(count);
        VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_, surfaceHandle_,
                                                      &count, formats.data()));
        VkSurfaceFormatKHR chosen = formats[0];
        for (auto f : formats) {
            if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
                f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                chosen = f;
                break;
            }
        }
        if (chosen.format == VK_FORMAT_UNDEFINED) {
            chosen.format = VK_FORMAT_B8G8R8A8_UNORM;
        }
        colorFormat_ = chosen.format;
        extent_ = caps.currentExtent;
        if (extent_.width == UINT32_MAX) {
            extent_.width = std::clamp(
                static_cast<uint32_t>(std::max(width_, 1)),
                caps.minImageExtent.width, caps.maxImageExtent.width);
            extent_.height = std::clamp(
                static_cast<uint32_t>(std::max(height_, 1)),
                caps.minImageExtent.height, caps.maxImageExtent.height);
        }
        uint32_t imageCount = caps.minImageCount + 1;
        if (caps.maxImageCount) {
            imageCount = std::min(imageCount, caps.maxImageCount);
        }
        VkSwapchainCreateInfoKHR ci{
            VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        ci.surface = surfaceHandle_;
        ci.minImageCount = imageCount;
        ci.imageFormat = colorFormat_;
        ci.imageColorSpace = chosen.colorSpace;
        ci.imageExtent = extent_;
        ci.imageArrayLayers = 1;
        ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        uint32_t families[] = {graphicsFamily_, presentFamily_};
        if (graphicsFamily_ != presentFamily_) {
            ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            ci.queueFamilyIndexCount = 2;
            ci.pQueueFamilyIndices = families;
        } else {
            ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }
        ci.preTransform = caps.currentTransform;
        ci.compositeAlpha =
            (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
                ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                : static_cast<VkCompositeAlphaFlagBitsKHR>(
                      caps.supportedCompositeAlpha &
                      (~caps.supportedCompositeAlpha + 1));
        ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        ci.clipped = VK_TRUE;
        VK_CHECK(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_));
        VK_CHECK(vkGetSwapchainImagesKHR(device_, swapchain_, &count, nullptr));
        images_.resize(count);
        VK_CHECK(vkGetSwapchainImagesKHR(device_, swapchain_, &count,
                                         images_.data()));
        for (auto image : images_) {
            VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vi.image = image;
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = colorFormat_;
            vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vi.subresourceRange.levelCount = 1;
            vi.subresourceRange.layerCount = 1;
            VkImageView view;
            VK_CHECK(vkCreateImageView(device_, &vi, nullptr, &view));
            imageViews_.push_back(view);
        }
        createDepth();
        createRenderPass();
        createPipelines();
        for (auto view : imageViews_) {
            std::array<VkImageView, 2> attachments{view, depthView_};
            VkFramebufferCreateInfo fi{
                VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            fi.renderPass = renderPass_;
            fi.attachmentCount = 2;
            fi.pAttachments = attachments.data();
            fi.width = extent_.width;
            fi.height = extent_.height;
            fi.layers = 1;
            VkFramebuffer fb;
            VK_CHECK(vkCreateFramebuffer(device_, &fi, nullptr, &fb));
            framebuffers_.push_back(fb);
        }
    }

    void destroySwapchainResources() {
        for (auto fb : framebuffers_) {
            vkDestroyFramebuffer(device_, fb, nullptr);
        }
        framebuffers_.clear();
        if (pipeline_) {
            vkDestroyPipeline(device_, pipeline_, nullptr);
        }
        pipeline_ = VK_NULL_HANDLE;
        if (control_point_pipeline_) {
            vkDestroyPipeline(device_, control_point_pipeline_, nullptr);
        }
        control_point_pipeline_ = VK_NULL_HANDLE;
        if (pipelineLayout_) {
            vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        }
        pipelineLayout_ = VK_NULL_HANDLE;
        if (renderPass_) {
            vkDestroyRenderPass(device_, renderPass_, nullptr);
        }
        renderPass_ = VK_NULL_HANDLE;
        if (depthView_) {
            vkDestroyImageView(device_, depthView_, nullptr);
        }
        depthView_ = VK_NULL_HANDLE;
        if (depthImage_) {
            vkDestroyImage(device_, depthImage_, nullptr);
        }
        depthImage_ = VK_NULL_HANDLE;
        if (depthMemory_) {
            vkFreeMemory(device_, depthMemory_, nullptr);
        }
        depthMemory_ = VK_NULL_HANDLE;
        for (auto view : imageViews_) {
            vkDestroyImageView(device_, view, nullptr);
        }
        imageViews_.clear();
        images_.clear();
        if (swapchain_) {
            vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        }
        swapchain_ = VK_NULL_HANDLE;
    }

    void recreateSwapchain() {
        if (width_ <= 0 || height_ <= 0) {
            return;
        }
        VK_CHECK(vkDeviceWaitIdle(device_));
        destroySwapchainResources();
        createSwapchainResources();
        resize_ = false;
    }

    Mat4 cameraMatrix() const {
        V3 eye{distance_ * std::cos(pitch_) * std::sin(yaw_),
               -distance_ * std::cos(pitch_) * std::cos(yaw_),
               distance_ * std::sin(pitch_)};
        Mat4 view = lookAt(eye, {0, 0, 0.15f}, {0, 0, 1});
        Mat4 proj = perspective(45.f * 3.14159265358979323846f / 180.f,
                                float(extent_.width) / float(extent_.height),
                                0.1f, 100.f);
        return mul(proj, view);
    }

    void draw() {
        if (resize_) {
            recreateSwapchain();
            return;
        }
        uint32_t imageIndex = 0;
        VkResult r =
            vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, acquired_,
                                  VK_NULL_HANDLE, &imageIndex);
        if (r == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
            return;
        }
        if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
            VK_CHECK(r);
        }
        VK_CHECK(vkResetCommandBuffer(command_, 0));
        VkCommandBufferBeginInfo begin{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(command_, &begin));
        VkClearValue clears[2]{};
        clears[0].color = {{0.035f, 0.05f, 0.085f, 1.f}};
        clears[1].depthStencil = {1.f, 0};
        VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rp.renderPass = renderPass_;
        rp.framebuffer = framebuffers_.at(imageIndex);
        rp.renderArea.extent = extent_;
        rp.clearValueCount = 2;
        rp.pClearValues = clears;
        vkCmdBeginRenderPass(command_, &rp, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
        VkViewport vp{0, 0, float(extent_.width), float(extent_.height), 0, 1};
        VkRect2D scissor{{0, 0}, extent_};
        vkCmdSetViewport(command_, 0, 1, &vp);
        vkCmdSetScissor(command_, 0, 1, &scissor);
        const DrawConstants constants{
            cameraMatrix(), {2.0f / extent_.width, 2.0f / extent_.height}};
        vkCmdPushConstants(command_, pipelineLayout_,
                           VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(constants),
                           &constants);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(command_, 0, 1, &vertex_.handle, &offset);
        vkCmdDraw(command_, vertex_count_, 1, 0, 0);
        vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          control_point_pipeline_);
        vkCmdBindVertexBuffers(command_, 0, 1, &control_points_.handle,
                               &offset);
        vkCmdDraw(command_, 6, control_point_count_, 0, 0);
        vkCmdEndRenderPass(command_);
        VK_CHECK(vkEndCommandBuffer(command_));
        VkPipelineStageFlags stage =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &acquired_;
        submit.pWaitDstStageMask = &stage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &rendered_;
        VK_CHECK(vkQueueSubmit(graphicsQueue_, 1, &submit, VK_NULL_HANDLE));
        VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &rendered_;
        present.swapchainCount = 1;
        present.pSwapchains = &swapchain_;
        present.pImageIndices = &imageIndex;
        r = vkQueuePresentKHR(presentQueue_, &present);
        // Deliberately one frame in flight: simple and safe for a learning
        // sample.
        VK_CHECK(vkQueueWaitIdle(presentQueue_));
        VK_CHECK(vkQueueWaitIdle(graphicsQueue_));
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR ||
            resize_) {
            recreateSwapchain();
        } else {
            VK_CHECK(r);
        }
    }

    void cleanup() {
        destroyBuffer(control_points_);
        destroyBuffer(vertex_);
        destroySwapchainResources();
        if (acquired_) {
            vkDestroySemaphore(device_, acquired_, nullptr);
        }
        if (rendered_) {
            vkDestroySemaphore(device_, rendered_, nullptr);
        }
        if (commandPool_) {
            vkDestroyCommandPool(device_, commandPool_, nullptr);
        }
        if (device_) {
            vkDestroyDevice(device_, nullptr);
        }
        if (surfaceHandle_) {
            vkDestroySurfaceKHR(instance_, surfaceHandle_, nullptr);
        }
        if (instance_) {
            vkDestroyInstance(instance_, nullptr);
        }
        if (window_) {
            XDestroyWindow(display_, window_);
        }
        if (display_) {
            XCloseDisplay(display_);
        }
    }
};

}  // namespace

int main(int argc, char* argv[]) {
    freopen("log.txt", "w", stdout);
    setvbuf(stdout, nullptr, _IONBF, 0);
    // tail -f log.txt

    try {
        if (argc == 2 && std::string_view(argv[1]) == "--test") {
            return RunNurbsMathTest();
        }
        if (argc != 1) {
            std::cerr << "Usage: " << argv[0] << " [--test]\n";
            return EXIT_FAILURE;
        }
        App app;
        app.run();
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
