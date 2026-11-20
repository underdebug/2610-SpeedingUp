# NURBS Surface: Vulkan + Xlib (XWayland)

All C++ source files and `CMakeLists.txt` are in this root directory. Only shaders live in `shaders/` and VS Code configuration lives in `.vscode/`.

## Dependencies (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y build-essential cmake gdb libvulkan-dev vulkan-tools glslang-tools libx11-dev
```

This sample uses Xlib directly. When running in a Wayland session, the compositor must support XWayland and `$DISPLAY` must be available. No GLFW, SDL, GLM, or CUDA is used.

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
./build/NurbsVulkan --test
(cd build && ./NurbsVulkan)
```

Run the math test without an X11 display using `--test` or `ctest --test-dir build`.
Launch the viewer from `build/` so it can find the compiled shaders.
The viewer and math test append standard output to `log.txt` in their working directory. Errors still go to the terminal.

## Controls

* Left mouse drag: orbit
* Mouse wheel: zoom
* `1`: Cox-de Boor (basis function weighted sum)
* `2`: de Boor (control-point interpolation)
* `Esc` or `q`: quit

The window title shows the active method. The example uses a weighted bicubic (degree 3x3) NURBS surface with 7x7 control points. Orange markers show the control points, including points behind the surface. The app compares both methods on a 51x51 grid and prints the maximum deviation. The viewer tessellates an 80x80 grid and outlines each triangle, including the diagonal across each grid cell.

## VS Code

Install the Microsoft C/C++ extension. Run **Run and Debug -> Debug NURBS Vulkan (XWayland)**. This launches gdb after the CMake build task and uses `${workspaceFolder}/build` as the working directory.

## Implementation notes

* `nurbs.hpp`: algorithms and example surface; homogeneous coordinates support rational NURBS weights
* `main.cpp`: entry point, numerical consistency test, Xlib window, Vulkan surface/swapchain/pipeline, camera, mesh upload, draw loop
* `shaders/`: surface and control point shaders compiled to SPIR-V at build time

This example intentionally waits for GPU queues to become idle per frame to simplify synchronization; optimize with per-frame fences and multiple command buffers for higher throughput.
