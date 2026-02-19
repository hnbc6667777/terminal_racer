/*
 * terminal_racer.c
 * 增强版：拉近视角、彩色渲染、随机景物、车辆图形化
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

// ---------- 3D 数学库 ----------
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

// ---------- 赛道生成 ----------
#define TRACK_POINTS 50
#define TRACK_SEGMENTS 200
#define TRACK_WIDTH 4.0f

typedef struct {
    vec3 center[TRACK_SEGMENTS];
    vec3 left[TRACK_SEGMENTS];
    vec3 right[TRACK_SEGMENTS];
    int count;
} Track;

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

static Track generate_track(void) {
    vec3 points[TRACK_POINTS+3];
    for (int i=0; i<TRACK_POINTS; i++) {
        float angle = 2*M_PI * i / TRACK_POINTS;
        float radius = 60.0f + 15*sinf(angle*3) + 10*cosf(angle*5);
        float x = radius * cosf(angle);
        float z = radius * sinf(angle);
        float y = 8*sinf(angle*4) + 5*cosf(angle*6);
        points[i+1] = (vec3){x, y, z};
    }
    points[0] = points[TRACK_POINTS];
    points[1] = points[TRACK_POINTS+1];
    points[TRACK_POINTS+1] = points[2];
    points[TRACK_POINTS+2] = points[3];

    Track track;
    track.count = TRACK_SEGMENTS;
    for (int i=0; i<TRACK_SEGMENTS; i++) {
        float t = (float)i / TRACK_SEGMENTS * TRACK_POINTS;
        int idx = (int)t;
        float frac = t - idx;
        vec3 p0 = points[idx];
        vec3 p1 = points[idx+1];
        vec3 p2 = points[idx+2];
        vec3 p3 = points[idx+3];
        vec3 center = catmull_rom(p0, p1, p2, p3, frac);
        track.center[i] = center;

        vec3 next = catmull_rom(p0, p1, p2, p3, frac+0.01f);
        vec3 tangent = vec3_norm(vec3_sub(next, center));
        vec3 up = {0,1,0};
        vec3 right = vec3_norm(vec3_cross(tangent, up));
        vec3 left_dir = vec3_mul(right, -1);
        track.left[i] = vec3_add(center, vec3_mul(left_dir, TRACK_WIDTH));
        track.right[i] = vec3_add(center, vec3_mul(right, TRACK_WIDTH));
    }
    return track;
}

// ---------- 随机景物 ----------
#define NUM_OBJECTS 150
typedef struct {
    vec3 pos;
    char ch;          // 显示字符
    int color_pair;   // 颜色对编号
} SceneObject;

static SceneObject objects[NUM_OBJECTS];

// 生成随机景物（在赛道周围较远位置）
static void generate_objects(Track* track) {
    for (int i=0; i<NUM_OBJECTS; i++) {
        // 随机选择一个赛道中心点作为基准
        int idx = rand() % track->count;
        vec3 base = track->center[idx];

        // 随机偏移：径向距离 15~40，角度随机
        float angle = (float)rand() / RAND_MAX * 2*M_PI;
        float dist = 15.0f + (float)rand() / RAND_MAX * 25.0f;
        float x = base.x + dist * cosf(angle);
        float z = base.z + dist * sinf(angle);

        // 高度简单设为赛道高度附近
        float y = base.y + ((float)rand() / RAND_MAX - 0.5f) * 5.0f;

        objects[i].pos = (vec3){x, y, z};

        // 随机选择字符和颜色
        int r = rand() % 5;
        switch (r) {
            case 0: objects[i].ch = 'T'; objects[i].color_pair = 2; break; // 树木（绿色）
            case 1: objects[i].ch = '#'; objects[i].color_pair = 3; break; // 岩石（棕色）
            case 2: objects[i].ch = '*'; objects[i].color_pair = 4; break; // 花朵（红色）
            case 3: objects[i].ch = 'Y'; objects[i].color_pair = 5; break; // 灌木（黄绿色）
            case 4: objects[i].ch = '^'; objects[i].color_pair = 6; break; // 小丘（青色）
        }
    }
}

// ---------- 输入处理 ----------
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

// ---------- 渲染器 ----------
static int scr_width, scr_height;

static void init_screen(void) {
    initscr();
    start_color();  // 启用颜色
    init_pair(1, COLOR_GREEN, COLOR_BLACK);   // 赛道边界
    init_pair(2, COLOR_GREEN, COLOR_BLACK);   // 树木
    init_pair(3, COLOR_YELLOW, COLOR_BLACK);  // 岩石（用棕色无法直接获得，用黄色替代）
    init_pair(4, COLOR_RED, COLOR_BLACK);      // 花朵
    init_pair(5, COLOR_YELLOW, COLOR_BLACK);   // 灌木
    init_pair(6, COLOR_CYAN, COLOR_BLACK);     // 小丘
    init_pair(7, COLOR_YELLOW, COLOR_BLACK);   // 车辆
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

// ---------- 游戏状态 ----------
typedef struct {
    vec3 pos;
    float yaw;
    float speed;
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

    // 初始化随机数种子
    srand(time(NULL));

    init_screen();

    Track track = generate_track();
    generate_objects(&track);  // 生成景物

    Car car = {
        .pos = track.center[0],
        .yaw = 0,
        .speed = 0
    };

    float near = 1.0f, far = 200.0f;
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

        float steer = 0;
        if (key_state[KEY_A]) steer += 1;
        if (key_state[KEY_D]) steer -= 1;
        car.yaw += steer * 2.0f * delta_time;

        float accel = 0;
        if (key_state[KEY_W]) accel += 10.0f;
        if (key_state[KEY_S]) accel -= 8.0f;
        car.speed += accel * delta_time;
        car.speed *= (1.0f - 2.0f * delta_time);
        if (fabs(car.speed) < 0.001f) car.speed = 0;

        vec3 dir = { sinf(car.yaw), 0, cosf(car.yaw) };
        car.pos.x += dir.x * car.speed * delta_time;
        car.pos.z += dir.z * car.speed * delta_time;

        // 高度跟随
        float min_dist = 1e9;
        vec3 target = car.pos;
        for (int i=0; i<track.count; i++) {
            float dx = car.pos.x - track.center[i].x;
            float dz = car.pos.z - track.center[i].z;
            float dist = dx*dx + dz*dz;
            if (dist < min_dist) {
                min_dist = dist;
                target = track.center[i];
            }
        }
        car.pos.y = target.y + 0.5f;

        // 相机设置：拉近视角，距离 8，高度 3.5
        vec3 eye = vec3_add(car.pos, (vec3){ -8*sinf(car.yaw), 3.5f, -8*cosf(car.yaw) });
        vec3 center = car.pos;
        vec3 up = {0,1,0};
        mat4 view = mat4_lookat(eye, center, up);

        erase();

        // 绘制赛道边界（绿色）
        for (int i=0; i<track.count-1; i++) {
            int x1,y1, x2,y2;
            if (project(track.left[i], view, proj, &x1, &y1) &&
                project(track.left[i+1], view, proj, &x2, &y2))
                draw_line(x1, y1, x2, y2, '#' | COLOR_PAIR(1));
            if (project(track.right[i], view, proj, &x1, &y1) &&
                project(track.right[i+1], view, proj, &x2, &y2))
                draw_line(x1, y1, x2, y2, '#' | COLOR_PAIR(1));
        }

        // 绘制景物（带颜色）
        for (int i=0; i<NUM_OBJECTS; i++) {
            int sx, sy;
            if (project(objects[i].pos, view, proj, &sx, &sy)) {
                mvaddch(sy, sx, objects[i].ch | COLOR_PAIR(objects[i].color_pair));
            }
        }

        // 绘制车辆（用两个字符表示车头朝向）
        int car_x, car_y;
        if (project(car.pos, view, proj, &car_x, &car_y)) {
            // 根据车头方向选择不同字符（简单表示）
            // 使用 Unicode 字符（如果终端支持，否则 fallback）
            chtype car_ch;
            if (fabs(dir.x) > fabs(dir.z)) {
                // 左右方向
                car_ch = (dir.x > 0) ? '>' : '<';
            } else {
                // 前后方向
                car_ch = (dir.z > 0) ? 'v' : '^';
            }
            mvaddch(car_y, car_x, car_ch | COLOR_PAIR(7) | A_BOLD);
            // 在后方加一个点表示车身
            int trail_x, trail_y;
            vec3 trail_pos = vec3_sub(car.pos, vec3_mul(dir, 0.8f)); // 车身稍微后移
            if (project(trail_pos, view, proj, &trail_x, &trail_y)) {
                mvaddch(trail_y, trail_x, 'o' | COLOR_PAIR(7));
            }
        }

        // 显示信息
        mvprintw(0, 0, "WASD 控制 | Q 退出 | 速度: %.2f m/s | FPS: %.1f", car.speed, fps);

        refresh();
        usleep(1000);
    }

    close_screen();
    pthread_cancel(input_th);
    pthread_join(input_th, NULL);
    close(fd);
    return 0;
}
