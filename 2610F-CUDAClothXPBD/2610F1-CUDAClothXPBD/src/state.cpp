
#include "state.h"
#include "app_config.h"

State::State(const AppConfig& app_config, int argc, char** argv) :
    app_config(app_config),
    ground_verts(3 * 4 * ground_num_tiles * ground_num_tiles, 0.0f),
    ground_colors(3 * 4 * ground_num_tiles * ground_num_tiles, 0.0f) {

    cloth = new Cloth(app_config.get_block_size(), app_config.get_substeps(), 
        cloth_y, cloth_num_x, cloth_num_y, cloth_spacing, sphere_center, sphere_radius);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(screen_width, screen_height);
    glutCreateWindow("CUDA Cloth XPBD");
}

State::~State() {

    delete cloth;
}

void State::init() {

    setup_opengl();
    build_ground();
}

void State::setup_opengl() {

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_CULL_FACE);
    glShadeModel(GL_SMOOTH);
    glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    glLightModelf(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    float ambient_color[] = { 0.2f, 0.2f, 0.2f, 1.0f };
    float diffuse_color[] = { 0.8f, 0.8f, 0.8f, 1.0f };
    float specular_color[] = { 1.0f, 1.0f, 1.0f, 1.0f };

    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_color);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_color);
    glLightfv(GL_LIGHT0, GL_SPECULAR, specular_color);

    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular_color);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50.0f);

    float light_position[] = { 10.0f, 10.0f, 10.0f, 0.0f };
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);

    glEnable(GL_NORMALIZE);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0, 1.0);
}

std::vector<float> State::convert_vec3_to_float_vector(const Vec3<float>* vec3_array, int num_elements) {

    std::vector<float> floatVector;
    floatVector.reserve(num_elements * 3); // Reserve space to avoid multiple allocations

    for (int i = 0; i < num_elements; ++i) {
        floatVector.push_back(vec3_array[i].x);
        floatVector.push_back(vec3_array[i].y);
        floatVector.push_back(vec3_array[i].z);
    }

    return floatVector;
}

std::vector<float> State::convert_vec3_to_float_vector(const float* vec3_x, const float* vec3_y, const float* vec3_z, int num_elements) {

    std::vector<float> floatVector;
    floatVector.reserve(num_elements * 3); // Reserve space to avoid multiple allocations

    for (int i = 0; i < num_elements; ++i) {
        floatVector.push_back(vec3_x[i]);
        floatVector.push_back(vec3_y[i]);
        floatVector.push_back(vec3_z[i]);
    }

    return floatVector;
}

void State::build_ground() {

    int square_verts[4][2] = { {0,0}, {0,1}, {1,1}, {1,0} };
    float r = ground_num_tiles / 2.0f * ground_tile_size;

    for (int xi = 0; xi < ground_num_tiles; ++xi) {
        for (int zi = 0; zi < ground_num_tiles; ++zi) {
            float x = (-ground_num_tiles / 2.0f + xi) * ground_tile_size;
            float z = (-ground_num_tiles / 2.0f + zi) * ground_tile_size;
            int p = xi * ground_num_tiles + zi;
            for (int i = 0; i < 4; ++i) {
                int q = 4 * p + i;
                float px = x + square_verts[i][0] * ground_tile_size;
                float pz = z + square_verts[i][1] * ground_tile_size;
                ground_verts[3 * q] = px;
                ground_verts[3 * q + 2] = pz;
                float col = 0.4f;
                if ((xi + zi) % 2 == 1) {
                    col = 0.8f;
                }
                float pr = std::sqrt(px * px + pz * pz);
                float d = std::max(0.0f, 1.0f - pr / r);
                col = col * d;
                for (int j = 0; j < 3; ++j) {
                    ground_colors[3 * q + j] = col;
                }
            }
        }
    }
}

// OLDCODE

// 原布料绘制代码：保留供对照，当前不执行。
/*
void State::render() {

    cloth->simulate_step();
    cloth->update_mesh();

    std::vector<float> host_pos = convert_vec3_to_float_vector(
        cloth->get_host_pos_x(), cloth->get_host_pos_y(), cloth->get_host_pos_z(), cloth->get_num_particles());
    std::vector<float> host_normals = convert_vec3_to_float_vector(cloth->get_host_normals(), cloth->get_num_particles());

    draw_sphere();

    glColor3f(1.0f, 0.0f, 0.0f);
    glNormal3f(0.0f, 0.0f, -1.0f);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    glVertexPointer(3, GL_FLOAT, 0, host_pos.data());
    glNormalPointer(GL_FLOAT, 0, host_normals.data());

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    glDisable(GL_CULL_FACE);
    glColor3f(1.0f, 0.0f, 0.0f);
    glDrawElements(GL_TRIANGLES, cloth->get_num_tris() * 3, GL_UNSIGNED_INT, cloth->get_host_tri_ids());
    glEnable(GL_CULL_FACE);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
}
*/

// NEWCODE

// 新代码：布料只绘制每个三角形的三条边，不填充面。
void State::render() {
    cloth->simulate_step();
    cloth->update_mesh();

    std::vector<float> host_pos = convert_vec3_to_float_vector(
        cloth->get_host_pos_x(), cloth->get_host_pos_y(), cloth->get_host_pos_z(), cloth->get_num_particles());

    draw_sphere();

    // 保存绘制状态；关闭光照和背面剔除，让布料两侧的线条都清晰可见。
    glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT | GL_CURRENT_BIT | GL_LINE_BIT);
    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glLineWidth(1.0f);
    glColor3f(1.0f, 0.0f, 0.0f);

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, host_pos.data());
    // 继续使用原有三角形索引，因此每个网格的对角线也会显示。
    glDrawElements(GL_TRIANGLES, cloth->get_num_tris() * 3, GL_UNSIGNED_INT, cloth->get_host_tri_ids());
    glDisableClientState(GL_VERTEX_ARRAY);

    // 恢复状态，避免影响地面等后续绘制。
    glPopAttrib();
}

// OLDCODE

// 原球体实体绘制代码：保留供对照，当前不执行。
/*
void State::draw_sphere() {
    GLUquadric* q = gluNewQuadric();

    glColor3f(0.8f, 0.8f, 0.8f);

    glPushMatrix();
    glTranslatef(sphere_center.x, sphere_center.y, sphere_center.z);
    gluSphere(q, sphere_radius, 40, 40);
    glPopMatrix();

    gluDeleteQuadric(q);
}
*/

// NEWCODE

// 新代码：将球面经纬网格拆成三角形，只绘制三角形边框。
void State::draw_sphere() {
    // 原始球体细分为 40 × 40，已保留在上方 OLDCODE 的 gluSphere 调用中。
    // 5% 测试网格：10 × 9 分段；独立顶点为 82，约为原来 1562 的 5.25%。
    // 仅降低显示网格精度，球体半径和碰撞检测仍保持不变。
    const int slices = 10;
    const int stacks = 9;
    const float pi = 3.14159265358979323846f;

    glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT | GL_CURRENT_BIT | GL_LINE_BIT);
    glDisable(GL_LIGHTING);
    glDisable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glLineWidth(1.0f);
    glColor3f(0.8f, 0.8f, 0.8f);

    glPushMatrix();
    glTranslatef(sphere_center.x, sphere_center.y, sphere_center.z);

    // 根据纬度和经度计算球面顶点，球体大小仍使用原来的碰撞半径。
    const auto vertex = [&](float latitude, float longitude) {
        const float ring_radius = sphere_radius * std::cos(latitude);
        glVertex3f(ring_radius * std::cos(longitude),
            sphere_radius * std::sin(latitude), ring_radius * std::sin(longitude));
    };

    glBegin(GL_TRIANGLES);
    for (int j = 0; j < stacks; ++j) {
        const float lat0 = -0.5f * pi + pi * j / stacks;
        const float lat1 = -0.5f * pi + pi * (j + 1) / stacks;
        for (int i = 0; i < slices; ++i) {
            const float lon0 = 2.0f * pi * i / slices;
            const float lon1 = 2.0f * pi * (i + 1) / slices;
            // 每个经纬四边形分成两个三角形；两极跳过退化三角形。
            if (j > 0) {
                vertex(lat0, lon0);
                vertex(lat1, lon0);
                vertex(lat0, lon1);
            }
            if (j < stacks - 1) {
                vertex(lat0, lon1);
                vertex(lat1, lon0);
                vertex(lat1, lon1);
            }
        }
    }
    glEnd();

    glPopMatrix();
    glPopAttrib();
}

void State::show_screen_cloth() {

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Ground
    glColor3f(1.0f, 1.0f, 1.0f);
    glNormal3f(0.0f, 1.0f, 0.0f);

    int num_verts = static_cast<int>(ground_verts.size() / 3);

    glVertexPointer(3, GL_FLOAT, 0, ground_verts.data());
    glColorPointer(3, GL_FLOAT, 0, ground_colors.data());

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glDrawArrays(GL_QUADS, 0, num_verts);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

    render();
    glutSwapBuffers();
}

int frame_counter = 0;
void State::timer_callback(int value) {

    if (app_config.is_profile_mode()) {
        frame_counter += 1;
        if (frame_counter > app_config.profile_loop_count) exit(0);
    }

    show_screen_cloth();

    camera.set_view();
    camera.handle_keys();
    camera.handle_special_key();
    glutTimerFunc(16, timer_callback_wrapper, 0); // 60 FPS
}

void State::key_down(unsigned char key, int x, int y) {
    camera.handle_key_down(key, x, y);
}

void State::key_up(unsigned char key, int x, int y) {
    camera.handle_key_up(key, x, y);
}

void State::process_special_keys(int key, int x, int y) {
    camera.handle_special_key_down(key, x, y);
}

void State::process_special_keys_up(int key, int x, int y) {
    camera.handle_special_key_up(key, x, y);
}

std::pair<Vec3<float>, Vec3<float>> State::get_mouse_ray(int x, int y) {

    GLint viewport[4];
    GLdouble model_matrix[16];
    GLdouble proj_matrix[16];
    GLdouble p0[3], p1[3];

    // Get the current viewport, modelview, and projection matrices
    glGetIntegerv(GL_VIEWPORT, viewport);
    glGetDoublev(GL_MODELVIEW_MATRIX, model_matrix);
    glGetDoublev(GL_PROJECTION_MATRIX, proj_matrix);

    // Adjust y to match OpenGL's coordinate system
    y = viewport[3] - y - 1;

    // Get the point in 3D space for depth 0.0 (near plane)
    gluUnProject(static_cast<GLdouble>(x), static_cast<GLdouble>(y), 0.0, model_matrix, proj_matrix, viewport, &p0[0], &p0[1], &p0[2]);

    // Get the point in 3D space for depth 1.0 (far plane)
    gluUnProject(static_cast<GLdouble>(x), static_cast<GLdouble>(y), 1.0, model_matrix, proj_matrix, viewport, &p1[0], &p1[1], &p1[2]);

    // Convert points to Vec3
    Vec3<float> orig(p0[0], p0[1], p0[2]);
    Vec3<float> dir(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]);

    // Normalize the direction vector
    dir = dir.normalized();

    return std::make_pair(orig, dir);
}

void State::mouse_button(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN && is_dragging == false) {
            is_dragging = true;
            std::pair<Vec3<float>, Vec3<float>> mouse_ray = get_mouse_ray(x, y);
            Vec3<float> ray_origin = mouse_ray.first;
            Vec3<float> ray_direction = mouse_ray.second;
            cloth->start_drag(ray_origin, ray_direction);
        }
        else if (state == GLUT_UP && is_dragging == true) {
            is_dragging = false;
            cloth->end_drag();
        }
    }
    if (button == GLUT_RIGHT_BUTTON) {
        if (state == GLUT_DOWN) { is_orbiting = true;  last_x = x; last_y = y; }
        if (state == GLUT_UP) { is_orbiting = false; }
    }
}

void State::mouse_drag(int x, int y) {

    if (is_dragging) {
        std::pair<Vec3<float>, Vec3<float>> mouse_ray = get_mouse_ray(x, y);
        Vec3<float> ray_origin = mouse_ray.first;
        Vec3<float> ray_direction = mouse_ray.second;
        cloth->drag(ray_origin, ray_direction);
    }
    if (is_orbiting) {
        double dx = double(x - last_x);
        double dy = double(y - last_y);
        last_x = x; last_y = y;

        camera.handle_mouse_orbit(dx, dy, orbit_center);
        glutPostRedisplay();
        return;
    }
}
