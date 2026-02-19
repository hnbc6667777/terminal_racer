/*
 * terminal_racer.c
 * 终极改进版：平滑无限赛道 + 地形自适应相机 + 可调俯仰角(U/I) + 重力势能 + 彩色渲染 + 中心虚线 + 随机景物
 *
 * 主要改进：
 *   - 使用虚拟点处理样条端点，保证首尾段连续。
 *   - 新控制点基于历史趋势生成，避免突变。
 *   - 仅渲染车辆附近区域，提高性能并避免远处裁剪。
 *   - 修正滑动窗口逻辑，确保赛道无限平滑延伸。
 *
 * 可调参数宏定义（位于文件开头）
 *   GRAVITY          重力加速度 (m/s²) 默认 9.8
 *   CAMERA_DIST      相机与车辆距离（默认 3.0）
 *   CAMERA_HEIGHT    相机高度（默认 2.0）
 *   PITCH_SPEED      俯仰调整速度 (rad/s) 默认 1.0
 *   PITCH_LIMIT      俯仰角度限制（度）默认 30.0
 *   TRACK_WIDTH      赛道半宽（默认 4.0）
 *   ACCEL_FORWARD    前进加速度 (m/s²) 默认 10.0
 *   ACCEL_BACK       刹车/倒车加速度 (m/s²) 默认 8.0
 *   FRICTION_COEF    摩擦系数 (1/s) 默认 2.0
 *   CTRL_PT_DIST     控制点间隔（米）默认 10.0
 *   NUM_CTRL_PTS     控制点数量（决定赛道长度）默认 30
 *   NUM_SEGMENTS     每段插值点数 默认 200
 *   RENDER_RANGE     渲染范围（米）默认 200.0
 *
 * 编译: gcc -o terminal_racer terminal_racer.c -lncurses -lm -lpthread
 * 运行: sudo ./terminal_racer /dev/input/eventX
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/input.h>
#include <sys/select.h>
#include <sys/time.h>
#include <ncurses.h>
#include <pthread.h>
#include <stdbool.h>

// ========== 可调参数宏 ==========
#define GRAVITY          9.8f     // 重力加速度 (m/s²)
#define CAMERA_DIST      3.0f     // 相机与车辆距离
#define CAMERA_HEIGHT    2.0f     // 相机高度
#define PITCH_SPEED      1.0f     // 俯仰调整速度 (rad/s)
#define PITCH_LIMIT      30.0f    // 俯仰角度限制（度）
#define TRACK_WIDTH      4.0f     // 赛道半宽
#define ACCEL_FORWARD    40.0f    // 前进加速度 (m/s²)
#define ACCEL_BACK       8.0f     // 刹车/倒车加速度 (m/s²)
#define FRICTION_COEF    2.0f     // 摩擦系数 (1/s)
#define CTRL_PT_DIST     10.0f    // 控制点间隔（米）
#define NUM_CTRL_PTS     30       // 控制点数量
#define NUM_SEGMENTS     200      // 每段插值点数
#define RENDER_RANGE     200.0f   // 渲染范围（车辆前后距离）

// 俯仰限制转换为弧度
#define PITCH_LIMIT_RAD (PITCH_LIMIT * M_PI / 180.0f)

// ========== 3D 数学库 ==========
typedef struct { float x, y, z; } vec3;
typedef struct { float m[4][4]; } mat4;

static vec3 vec3_add(vec3 a, vec3 b) { return (vec3){a.x+b.x, a.y+b.y, a.z+b.z}; }
static vec3 vec3_sub(vec3 a, vec3 b) { return (vec3){a.x-b.x, a.y-b.y, a.z-b.z}; }
static vec3 vec3_mul(vec3 a, float s) { return (vec3){a.x*s, a.y*s, a.z*s}; }
static float vec3_dot(vec3 a, vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static vec3 vec3_cross(vec3 a, vec3 b) { return (vec3){a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x}; }
static float vec3_len(vec3 v) { return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z); }
static vec3 vec3_norm(vec3 v) { float l = vec3_len(v); return l>0 ? (vec3){v.x/l, v.y/l, v.z/l} : (vec3){0,0,0}; }

static mat4 mat4_identity(void) {
    mat4 m = {{{0}}};
    for (int i=0; i<4; i++) m.m[i][i] = 1.0f;
    return m;
}

static mat4 mat4_mul(mat4 a, mat4 b) {
    mat4 res = {{{0}}};
    for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
            for (int k=0; k<4; k++)
                res.m[i][j] += a.m[i][k] * b.m[k][j];
    return res;
}

static vec3 mat4_mul_vec3(mat4 m, vec3 v, float w) {
    vec3 res;
    res.x = m.m[0][0]*v.x + m.m[0][1]*v.y + m.m[0][2]*v.z + m.m[0][3]*w;
    res.y = m.m[1][0]*v.x + m.m[1][1]*v.y + m.m[1][2]*v.z + m.m[1][3]*w;
    res.z = m.m[2][0]*v.x + m.m[2][1]*v.y + m.m[2][2]*v.z + m.m[2][3]*w;
    float ww = m.m[3][0]*v.x + m.m[3][1]*v.y + m.m[3][2]*v.z + m.m[3][3]*w;
    if (ww != 0) { res.x /= ww; res.y /= ww; res.z /= ww; }
    return res;
}

// 透视投影矩阵
static mat4 mat4_frustum(float l, float r, float b, float t, float n, float f) {
    mat4 m = {{{0}}};
    m.m[0][0] = 2*n/(r-l);
    m.m[0][2] = (r+l)/(r-l);
    m.m[1][1] = 2*n/(t-b);
    m.m[1][2] = (t+b)/(t-b);
    m.m[2][2] = -(f+n)/(f-n);
    m.m[2][3] = -2*f*n/(f-n);
    m.m[3][2] = -1;
    return m;
}

// 视图矩阵
static mat4 mat4_lookat(vec3 eye, vec3 center, vec3 up) {
    vec3 f = vec3_norm(vec3_sub(center, eye));
    vec3 s = vec3_norm(vec3_cross(f, up));
    vec3 u = vec3_cross(s, f);
    mat4 m = mat4_identity();
    m.m[0][0] = s.x; m.m[0][1] = s.y; m.m[0][2] = s.z; m.m[0][3] = -vec3_dot(s, eye);
    m.m[1][0] = u.x; m.m[1][1] = u.y; m.m[1][2] = u.z; m.m[1][3] = -vec3_dot(u, eye);
    m.m[2][0] = -f.x; m.m[2][1] = -f.y; m.m[2][2] = -f.z; m.m[2][3] = vec3_dot(f, eye);
    return m;
}

// ========== 无限赛道生成（带法线和切线） ==========
typedef struct {
    vec3* ctrl_pts;          // 控制点数组 [NUM_CTRL_PTS]
    vec3* center;            // 插值中心点数组 [total_points]
    vec3* left;              // 左边界数组 [total_points]
    vec3* right;             // 右边界数组 [total_points]
    vec3* normal;            // 路面法线数组 [total_points]
    vec3* tangent;           // 切线方向数组 [total_points]
    int total_points;        // 插值点总数 = (NUM_CTRL_PTS-1)*NUM_SEGMENTS
} InfiniteTrack;

// Catmull-Rom 样条插值
static vec3 catmull_rom(vec3 p0, vec3 p1, vec3 p2, vec3 p3, float t) {
    float t2 = t*t;
    float t3 = t2*t;
    return vec3_add(
        vec3_add(
            vec3_mul(p0, -t3 + 2*t2 - t),
            vec3_mul(p1, 3*t3 - 5*t2 + 2)),
        vec3_add(
            vec3_mul(p2, -3*t3 + 4*t2 + t),
            vec3_mul(p3, t3 - t2)));
}

// 初始化无限赛道（生成初始控制点）
static InfiniteTrack* create_infinite_track(void) {
    InfiniteTrack* track = malloc(sizeof(InfiniteTrack));
    track->ctrl_pts = malloc(NUM_CTRL_PTS * sizeof(vec3));
    track->total_points = (NUM_CTRL_PTS - 1) * NUM_SEGMENTS;
    track->center = malloc(track->total_points * sizeof(vec3));
    track->left   = malloc(track->total_points * sizeof(vec3));
    track->right  = malloc(track->total_points * sizeof(vec3));
    track->normal = malloc(track->total_points * sizeof(vec3));
    track->tangent = malloc(track->total_points * sizeof(vec3));

    // 生成初始控制点（Z从0开始递增）
    for (int i = 0; i < NUM_CTRL_PTS; i++) {
        float z = i * CTRL_PT_DIST;
        // 使用正弦组合模拟弯道和起伏
        float x = 8.0f * sinf(z * 0.1f) + 5.0f * cosf(z * 0.23f);
        float y = 3.0f * sinf(z * 0.15f) + 2.0f * cosf(z * 0.37f);
        track->ctrl_pts[i] = (vec3){x, y, z};
    }
    return track;
}

// 重新计算所有插值点（改进端点处理）
static void update_track(InfiniteTrack* track) {
    int total = 0;
    for (int i = 0; i < NUM_CTRL_PTS - 1; i++) {
        // 使用虚拟点处理首尾段，保证连续性
        vec3 p0, p1, p2, p3;
        p1 = track->ctrl_pts[i];
        p2 = track->ctrl_pts[i+1];

        if (i == 0) {
            // 首段：虚拟前点 p0 = p1 - (p2 - p1)
            p0 = vec3_sub(p1, vec3_sub(p2, p1));
        } else {
            p0 = track->ctrl_pts[i-1];
        }

        if (i == NUM_CTRL_PTS - 2) {
            // 末段：虚拟后点 p3 = p2 + (p2 - p1)
            p3 = vec3_add(p2, vec3_sub(p2, p1));
        } else {
            p3 = track->ctrl_pts[i+2];
        }

        for (int j = 0; j < NUM_SEGMENTS; j++) {
            float t = (float)j / NUM_SEGMENTS;
            vec3 center = catmull_rom(p0, p1, p2, p3, t);
            track->center[total] = center;

            // 计算切线（使用差分）
            vec3 next = catmull_rom(p0, p1, p2, p3, t + 0.01f);
            vec3 tangent = vec3_norm(vec3_sub(next, center));
            track->tangent[total] = tangent;

            vec3 world_up = {0,1,0};
            vec3 right_dir = vec3_norm(vec3_cross(tangent, world_up));
            if (vec3_len(right_dir) < 0.1f) right_dir = (vec3){1,0,0};

            track->left[total] = vec3_sub(center, vec3_mul(right_dir, TRACK_WIDTH));
            track->right[total] = vec3_add(center, vec3_mul(right_dir, TRACK_WIDTH));

            vec3 normal = vec3_cross(tangent, right_dir);
            normal = vec3_norm(normal);
            if (normal.y < 0) normal = vec3_mul(normal, -1.0f);
            track->normal[total] = normal;

            total++;
        }
    }
}

// 基于历史趋势生成新控制点（二次外推）
static vec3 extrapolate_ctrl_point(vec3* prev_pts, int count) {
    // 使用最后三个点进行二次多项式外推（假设等间距Z）
    if (count >= 3) {
        vec3 p0 = prev_pts[count-3];
        vec3 p1 = prev_pts[count-2];
        vec3 p2 = prev_pts[count-1];
        float dz = CTRL_PT_DIST;
        // 使用有限差分近似二阶导数，然后外推
        // 简单线性外推：p3 = p2 + (p2 - p1) = 2*p2 - p1
        // 但为了更平滑，可以加上二阶项
        vec3 p3_linear = vec3_add(p2, vec3_sub(p2, p1));
        // 二阶修正：假设加速度恒定
        vec3 accel = vec3_sub(vec3_sub(p2, p1), vec3_sub(p1, p0)); // 二阶差分
        vec3 p3 = vec3_add(p3_linear, accel); // 外推一步
        p3.z = p2.z + dz;
        return p3;
    } else if (count >= 2) {
        // 线性外推
        vec3 p1 = prev_pts[count-2];
        vec3 p2 = prev_pts[count-1];
        vec3 p3 = vec3_add(p2, vec3_sub(p2, p1));
        p3.z = p2.z + CTRL_PT_DIST;
        return p3;
    } else {
        // 退化为原始正弦公式
        float z = prev_pts[count-1].z + CTRL_PT_DIST;
        float x = 8.0f * sinf(z * 0.1f) + 5.0f * cosf(z * 0.23f);
        float y = 3.0f * sinf(z * 0.15f) + 2.0f * cosf(z * 0.37f);
        return (vec3){x, y, z};
    }
}

// 根据车辆位置更新赛道（平滑滑动窗口）
static void advance_track(InfiniteTrack* track, float car_z) {
    float last_ctrl_z = track->ctrl_pts[NUM_CTRL_PTS-1].z;
    // 当车辆超过倒数第二个控制点一定距离时触发滑动
    if (car_z > last_ctrl_z - 2.0f * CTRL_PT_DIST) {
        // 移除第一个控制点
        for (int i = 1; i < NUM_CTRL_PTS; i++) {
            track->ctrl_pts[i-1] = track->ctrl_pts[i];
        }
        // 基于历史生成新控制点（使用最后几个点）
        vec3 new_pt = extrapolate_ctrl_point(track->ctrl_pts, NUM_CTRL_PTS-1);
        track->ctrl_pts[NUM_CTRL_PTS-1] = new_pt;

        // 重新计算所有插值点
        update_track(track);
    }
}

static void free_track(InfiniteTrack* track) {
    free(track->ctrl_pts);
    free(track->center);
    free(track->left);
    free(track->right);
    free(track->normal);
    free(track->tangent);
    free(track);
}

// ========== 随机景物 ==========
#define NUM_OBJECTS 200
typedef struct {
    vec3 pos;
    char ch;
    int color_pair;
} SceneObject;

static SceneObject objects[NUM_OBJECTS];

// 生成景物（沿赛道随机分布）
static void generate_objects(InfiniteTrack* track, float min_z, float max_z) {
    for (int i = 0; i < NUM_OBJECTS; i++) {
        float z = min_z + (float)rand() / RAND_MAX * (max_z - min_z);
        // 近似赛道中心X
        float base_x = 8.0f * sinf(z * 0.1f) + 5.0f * cosf(z * 0.23f);
        float offset = (15.0f + (float)rand() / RAND_MAX * 20.0f) * (rand()%2 ? 1 : -1);
        float x = base_x + offset;
        float y = 3.0f * sinf(z * 0.15f) + 2.0f * cosf(z * 0.37f) + ((float)rand()/RAND_MAX-0.5f)*4.0f;
        objects[i].pos = (vec3){x, y, z};

        int r = rand() % 5;
        switch (r) {
            case 0: objects[i].ch = 'T'; objects[i].color_pair = 2; break;
            case 1: objects[i].ch = '#'; objects[i].color_pair = 3; break;
            case 2: objects[i].ch = '*'; objects[i].color_pair = 4; break;
            case 3: objects[i].ch = 'Y'; objects[i].color_pair = 5; break;
            case 4: objects[i].ch = '^'; objects[i].color_pair = 6; break;
        }
    }
}

// ========== 输入处理 ==========
#define MAX_KEYS 256
static volatile int key_state[MAX_KEYS];

static void* input_thread(void* arg) {
    int fd = *(int*)arg;
    struct input_event ev;
    while (1) {
        ssize_t n = read(fd, &ev, sizeof(ev));
        if (n == sizeof(ev)) {
            if (ev.type == EV_KEY && ev.code < MAX_KEYS) {
                key_state[ev.code] = ev.value;
            }
        }
        pthread_testcancel();
    }
    return NULL;
}

static int open_input(const char* dev) {
    int fd = open(dev, O_RDONLY);
    if (fd < 0) {
        perror("open input device");
        return -1;
    }
    return fd;
}

// ========== 渲染器 ==========
static int scr_width, scr_height;

static void init_screen(void) {
    initscr();
    start_color();
    init_pair(1, COLOR_GREEN, COLOR_BLACK);   // 赛道边界
    init_pair(2, COLOR_GREEN, COLOR_BLACK);   // 树木
    init_pair(3, COLOR_YELLOW, COLOR_BLACK);  // 岩石
    init_pair(4, COLOR_RED, COLOR_BLACK);     // 花朵
    init_pair(5, COLOR_YELLOW, COLOR_BLACK);  // 灌木
    init_pair(6, COLOR_CYAN, COLOR_BLACK);    // 小丘
    init_pair(7, COLOR_YELLOW, COLOR_BLACK);  // 车辆
    init_pair(8, COLOR_YELLOW, COLOR_BLACK);  // 中心虚线
    cbreak();
    noecho();
    curs_set(0);
    nodelay(stdscr, TRUE);
    getmaxyx(stdscr, scr_height, scr_width);
}

static void close_screen(void) {
    endwin();
}

static int project(vec3 world, mat4 view, mat4 proj, int* sx, int* sy) {
    vec3 view_pos = mat4_mul_vec3(view, world, 1.0f);
    vec3 clip = mat4_mul_vec3(proj, view_pos, 1.0f);
    if (fabs(clip.x) > 1 || fabs(clip.y) > 1 || fabs(clip.z) > 1)
        return 0;
    *sx = (int)((clip.x + 1) * 0.5 * scr_width);
    *sy = (int)((1 - (clip.y + 1) * 0.5) * scr_height);
    if (*sx < 0 || *sx >= scr_width || *sy < 0 || *sy >= scr_height)
        return 0;
    return 1;
}

static void draw_line(int x0, int y0, int x1, int y1, chtype ch) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;
    while (1) {
        if (x0 >= 0 && x0 < scr_width && y0 >= 0 && y0 < scr_height)
            mvaddch(y0, x0, ch);
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

// ========== 游戏状态 ==========
typedef struct {
    vec3 pos;
    float yaw;
    float speed;
    float pitch;   // 当前俯仰角（弧度）
} Car;

static double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "用法: %s <键盘设备文件>\n", argv[0]);
        return 1;
    }

    int fd = open_input(argv[1]);
    if (fd < 0) return 1;

    pthread_t input_th;
    if (pthread_create(&input_th, NULL, input_thread, &fd) != 0) {
        perror("pthread_create");
        close(fd);
        return 1;
    }

    srand(time(NULL));

    init_screen();

    // 创建无限赛道
    InfiniteTrack* track = create_infinite_track();
    update_track(track);

    // 生成景物
    generate_objects(track, track->ctrl_pts[0].z, track->ctrl_pts[NUM_CTRL_PTS-1].z);

    Car car = {
        .pos = track->center[0],
        .yaw = 0,
        .speed = 0,
        .pitch = 0.0f
    };

    float near = 1.0f, far = 500.0f; // 增大 far 平面
    float fov = 60.0f * M_PI / 180.0f;
    float aspect = (float)scr_width / scr_height;
    float top = near * tanf(fov/2);
    float bottom = -top;
    float right = top * aspect;
    float left = -right;
    mat4 proj = mat4_frustum(left, right, bottom, top, near, far);

    double last_time = get_time();
    double fps = 0.0;

    int running = 1;
    while (running) {
        double current_time = get_time();
        double delta_time = current_time - last_time;
        if (delta_time > 0.1) delta_time = 0.1;
        last_time = current_time;
        fps = 1.0 / delta_time;

        // 输入
        if (key_state[KEY_Q]) running = 0;

        // 俯仰控制（U / I）
        if (key_state[KEY_U]) {
            car.pitch += PITCH_SPEED * delta_time;
            if (car.pitch > PITCH_LIMIT_RAD) car.pitch = PITCH_LIMIT_RAD;
        }
        if (key_state[KEY_I]) {
            car.pitch -= PITCH_SPEED * delta_time;
            if (car.pitch < -PITCH_LIMIT_RAD) car.pitch = -PITCH_LIMIT_RAD;
        }

        float steer = 0;
        if (key_state[KEY_A]) steer += 1;
        if (key_state[KEY_D]) steer -= 1;
        car.yaw += steer * 2.0f * delta_time;

        // 油门加速度
        float throttle_accel = 0;
        if (key_state[KEY_W]) throttle_accel += ACCEL_FORWARD;
        if (key_state[KEY_S]) throttle_accel -= ACCEL_BACK;

        // 找到最近的赛道点
        int best_idx = 0;
        float min_dist = 1e9;
        for (int i=0; i<track->total_points; i++) {
            float dx = car.pos.x - track->center[i].x;
            float dz = car.pos.z - track->center[i].z;
            float dist = dx*dx + dz*dz;
            if (dist < min_dist) {
                min_dist = dist;
                best_idx = i;
            }
        }
        vec3 target = track->center[best_idx];
        car.pos.y = target.y + 0.5f;

        // 获取当前路面的切线方向
        vec3 tangent = track->tangent[best_idx];

        // 重力加速度沿切线的分量
        float gravity_accel = -GRAVITY * tangent.y;

        // 总加速度
        float total_accel = throttle_accel + gravity_accel;

        // 更新速度
        car.speed += total_accel * delta_time;
        car.speed *= (1.0f - FRICTION_COEF * delta_time);
        if (fabs(car.speed) < 0.001f) car.speed = 0;

        // 更新位置（使用当前朝向，但实际应沿切线？这里保持原逻辑以允许漂移感）
        vec3 dir = { sinf(car.yaw), 0, cosf(car.yaw) };
        car.pos.x += dir.x * car.speed * delta_time;
        car.pos.z += dir.z * car.speed * delta_time;

        vec3 normal = track->normal[best_idx];

        // 扩展赛道
        advance_track(track, car.pos.z);

        // 基础相机位置
        vec3 eye_base = vec3_add(car.pos, vec3_add(vec3_mul(dir, -CAMERA_DIST), vec3_mul(normal, CAMERA_HEIGHT)));
        vec3 center_base = car.pos;

        // 应用俯仰
        vec3 view_dir = vec3_norm(vec3_sub(center_base, eye_base));
        vec3 right_vec = vec3_norm(vec3_cross(view_dir, normal));
        float cos_p = cosf(car.pitch);
        float sin_p = sinf(car.pitch);
        vec3 new_dir = vec3_add(
            vec3_mul(view_dir, cos_p),
            vec3_add(
                vec3_mul(vec3_cross(right_vec, view_dir), sin_p),
                vec3_mul(right_vec, vec3_dot(right_vec, view_dir) * (1 - cos_p))
            )
        );
        new_dir = vec3_norm(new_dir);
        float view_dist = vec3_len(vec3_sub(center_base, eye_base));
        vec3 new_center = vec3_add(eye_base, vec3_mul(new_dir, view_dist));
        mat4 view = mat4_lookat(eye_base, new_center, normal);

        // 渲染（仅绘制车辆附近区域）
        erase();

        float min_z = car.pos.z - RENDER_RANGE;
        float max_z = car.pos.z + RENDER_RANGE;

        // 绘制赛道边界（仅当在渲染范围内）
        for (int i=0; i<track->total_points-1; i++) {
            if (track->center[i].z < min_z || track->center[i].z > max_z) continue;
            int x1,y1, x2,y2;
            if (project(track->left[i], view, proj, &x1, &y1) &&
                project(track->left[i+1], view, proj, &x2, &y2))
                draw_line(x1, y1, x2, y2, '#' | COLOR_PAIR(1));
            if (project(track->right[i], view, proj, &x1, &y1) &&
                project(track->right[i+1], view, proj, &x2, &y2))
                draw_line(x1, y1, x2, y2, '|' | COLOR_PAIR(1) | A_BOLD);
        }

        // 绘制中心虚线
        for (int i=0; i<track->total_points; i+=5) {
            if (track->center[i].z < min_z || track->center[i].z > max_z) continue;
            int sx, sy;
            if (project(track->center[i], view, proj, &sx, &sy)) {
                mvaddch(sy, sx, '-' | COLOR_PAIR(8));
            }
        }

        // 绘制景物（仅绘制在可见范围内）
        for (int i=0; i<NUM_OBJECTS; i++) {
            if (objects[i].pos.z < min_z || objects[i].pos.z > max_z) continue;
            int sx, sy;
            if (project(objects[i].pos, view, proj, &sx, &sy)) {
                mvaddch(sy, sx, objects[i].ch | COLOR_PAIR(objects[i].color_pair));
            }
        }

        // 绘制车辆
        int car_x, car_y;
        if (project(car.pos, view, proj, &car_x, &car_y)) {
            chtype car_ch;
            if (fabs(dir.x) > fabs(dir.z)) {
                car_ch = (dir.x > 0) ? '>' : '<';
            } else {
                car_ch = (dir.z > 0) ? 'v' : '^';
            }
            mvaddch(car_y, car_x, car_ch | COLOR_PAIR(7) | A_BOLD);
            vec3 trail_pos = vec3_sub(car.pos, vec3_mul(dir, 0.8f));
            int trail_x, trail_y;
            if (project(trail_pos, view, proj, &trail_x, &trail_y)) {
                mvaddch(trail_y, trail_x, 'o' | COLOR_PAIR(7));
            }
        }

        // 显示信息
        float slope_percent = -tangent.y * 100.0f;
        mvprintw(0, 0, "WASD 控制 | U/I 俯仰 | Q 退出 | 速度: %+6.2f m/s | 坡度: %+5.1f%% | 俯仰: %+3.0f° | FPS: %.1f | Z: %.1f",
                 car.speed, slope_percent, car.pitch * 180.0f / M_PI, fps, car.pos.z);

        refresh();
        usleep(1000);
    }

    close_screen();
    free_track(track);
    pthread_cancel(input_th);
    pthread_join(input_th, NULL);
    close(fd);
    return 0;
}
