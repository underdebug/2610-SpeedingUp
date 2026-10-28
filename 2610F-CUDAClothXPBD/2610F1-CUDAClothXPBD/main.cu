
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <vector>
#include <cmath>
#include <memory>

#include <boost/chrono.hpp>
#include <boost/thread/thread.hpp>
#include <boost/program_options.hpp>

#include "state.h"
#include "app_config.h"
#include "cuda_check.cuh"

State* g_state = nullptr;

void timer_callback_wrapper(int value) {
    g_state->timer_callback(value);
}
void show_screen_cloth_wrapper() {
    g_state->show_screen_cloth();
}
void mouse_button_wrapper(int button, int state, int x, int y) {
    g_state->mouse_button(button, state, x, y);
}
void mouse_drag_wrapper(int x, int y) {
    g_state->mouse_drag(x, y);
}
void key_down_wrapper(unsigned char key, int x, int y) {
    g_state->key_down(key, x, y);
}
void key_up_wrapper(unsigned char key, int x, int y) {
    g_state->key_up(key, x, y);
}
void process_special_keys_wrapper(int key, int x, int y) {
    g_state->process_special_keys(key, x, y);
}
void process_special_keys_up_wrapper(int key, int x, int y) {
    g_state->process_special_keys_up(key, x, y);
}


int main(int argc, char** argv) {

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA-capable GPU found. The cloth simulation requires an NVIDIA GPU."
                  << std::endl;
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaSetDevice(0));

    boost::this_thread::sleep_for(boost::chrono::seconds(2));

    int block_size;
    bool profile_mode = false;

    boost::program_options::options_description desc("Options");
    desc.add_options()
        ("block_size", boost::program_options::value<int>(&block_size)
            ->default_value(DEFAULT_BLOCK_SIZE))
        ("profile_mode", boost::program_options::bool_switch(&profile_mode),
            "Enable profile mode");

    boost::program_options::positional_options_description pos;
    pos.add("block_size", 1);

    boost::program_options::variables_map vm;
    store(boost::program_options::command_line_parser(argc, argv)
        .options(desc).positional(pos).run(), vm);
    notify(vm);

    AppConfig app_config(block_size, profile_mode);

    std::cout << "Block Size: " << app_config.get_block_size() << std::endl;
    std::cout << "Profile Mode: " << app_config.is_profile_mode() << std::endl;

    State state(app_config, argc, argv);
    g_state = &state;

    state.init();

    glutDisplayFunc(show_screen_cloth_wrapper);
    glutMouseFunc(mouse_button_wrapper);
    glutMotionFunc(mouse_drag_wrapper);
    glutKeyboardFunc(key_down_wrapper);
    glutKeyboardUpFunc(key_up_wrapper);
    glutSpecialFunc(process_special_keys_wrapper);
    glutSpecialUpFunc(process_special_keys_up_wrapper);
    glutTimerFunc(0, timer_callback_wrapper, 0);

    glutMainLoop();


    return 0;
}
