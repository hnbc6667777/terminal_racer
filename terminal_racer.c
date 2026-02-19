/*
 * terminal_racer.c
 * 一个在 Linux 终端中运行的 3D 赛车游戏原型。
 * 使用 ncurses 绘制，通过 /dev/input 获取精确按键输入。
 * 赛道由随机样条生成，包含透视投影。
 *
 * 编译: gcc -o terminal_racer terminal_racer.c -lncurses -lm -lpthread
 * 运行: sudo ./terminal_racer /dev/input/eventX   (X为你的键盘设备号)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/input.h>
#include <sys/select.h>
#include <ncurses.h>
#include <pthread.h>      // 添加 pthread 头文件

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

// 透视投影矩阵 (参数: left, right, bottom, top, near, far)
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

// 视图矩阵 (相机位置, 目标点, 上方向)
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
#define TRACK_POINTS 50      // 控制点数量
#define TRACK_SEGMENTS 200   // 插值点数量
#define TRACK_WIDTH 2.0f     // 赛道半宽

typedef struct {
    vec3 center[TRACK_SEGMENTS];   // 中心线点
    vec3 left[TRACK_SEGMENTS];     // 左边界
    vec3 right[TRACK_SEGMENTS];    // 右边界
    int count;
} Track;

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

// 生成随机闭合赛道
static Track generate_track(void) {
    vec3 points[TRACK_POINTS+3]; // 前后各加一个点用于闭合样条
    // 随机生成控制点（在 XZ 平面起伏，Y 轴为高度）
    for (int i=0; i<TRACK_POINTS; i++) {
        float angle = 2*M_PI * i / TRACK_POINTS;
        float radius = 20.0f + 5*sinf(angle*3) + 3*cosf(angle*5); // 非圆形状
        float x = radius * cosf(angle);
        float z = radius * sinf(angle);
        float y = 3*sinf(angle*4) + 2*cosf(angle*6); // 起伏
        points[i+1] = (vec3){x, y, z};
    }
    // 闭合：复制前两个点到末尾，后两个点到开头
    points[0] = points[TRACK_POINTS];
    points[1] = points[TRACK_POINTS+1];
    points[TRACK_POINTS+1] = points[2];
    points[TRACK_POINTS+2] = points[3];

    Track track;
    track.count = TRACK_SEGMENTS;
    for (int i=0; i<TRACK_SEGMENTS; i++) {
        float t = (float)i / TRACK_SEGMENTS * TRACK_POINTS; // 映射到控制点区间
        int idx = (int)t;
        float frac = t - idx;
        // 索引偏移：因为 points[0..3] 对应控制点 -1..2，所以 idx 从 1 开始对应控制点 0
        vec3 p0 = points[idx];
        vec3 p1 = points[idx+1];
        vec3 p2 = points[idx+2];
        vec3 p3 = points[idx+3];
        vec3 center = catmull_rom(p0, p1, p2, p3, frac);
        track.center[i] = center;

        // 计算近似切线（差分）
        vec3 next = catmull_rom(p0, p1, p2, p3, frac+0.01f);
        vec3 tangent = vec3_norm(vec3_sub(next, center));
        // 选择世界 Y 轴作为上方向参考，计算副切线（右方向）
        vec3 up = {0,1,0};
        vec3 right = vec3_norm(vec3_cross(tangent, up));
        vec3 left_dir = vec3_mul(right, -1);
        track.left[i] = vec3_add(center, vec3_mul(left_dir, TRACK_WIDTH));
        track.right[i] = vec3_add(center, vec3_mul(right, TRACK_WIDTH));
    }
    return track;
}

// ---------- 输入处理 ----------
#define MAX_KEYS 256                 // 改用不冲突的宏名
static volatile int key_state[MAX_KEYS]; // 按键状态 (0=抬起,1=按下)

static void* input_thread(void* arg) {
    int fd = *(int*)arg;
    struct input_event ev;
    while (1) {
        ssize_t n = read(fd, &ev, sizeof(ev));
        if (n == sizeof(ev)) {
            if (ev.type == EV_KEY && ev.code < MAX_KEYS) {
                key_state[ev.code] = ev.value; // 0释放 1按下 2重复
            }
        }
        pthread_testcancel(); // 添加取消点
    }
    return NULL;
}

// 只打开设备，不再创建线程
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
    cbreak();
    noecho();
    curs_set(0);
    nodelay(stdscr, TRUE);  // 非阻塞
    getmaxyx(stdscr, scr_height, scr_width);
}

static void close_screen(void) {
    endwin();
}

// 将 3D 世界坐标映射到屏幕坐标（像素位置）
// 返回 0 表示点在视锥内，1 表示被裁剪（简单丢弃）
static int project(vec3 world, mat4 view, mat4 proj, int* sx, int* sy) {
    // 视图变换
    vec3 view_pos = mat4_mul_vec3(view, world, 1.0f);
    // 投影变换 + 透视除法
    vec3 clip = mat4_mul_vec3(proj, view_pos, 1.0f);
    // 检查裁剪空间范围（粗略）
    if (fabs(clip.x) > 1 || fabs(clip.y) > 1 || fabs(clip.z) > 1)
        return 0;
    // 映射到屏幕坐标
    *sx = (int)((clip.x + 1) * 0.5 * scr_width);
    *sy = (int)((1 - (clip.y + 1) * 0.5) * scr_height); // 翻转 Y
    if (*sx < 0 || *sx >= scr_width || *sy < 0 || *sy >= scr_height)
        return 0;
    return 1;
}

// 画线（简单 Bresenham 实现，仅画点）
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
    vec3 pos;       // 车辆位置
    float yaw;      // 朝向角 (弧度)
    float speed;    // 当前速度
} Car;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "用法: %s <键盘设备文件>\n", argv[0]);
        return 1;
    }

    // 打开输入设备
    int fd = open_input(argv[1]);
    if (fd < 0) return 1;

    // 启动输入线程
    pthread_t input_th;
    if (pthread_create(&input_th, NULL, input_thread, &fd) != 0) {
        perror("pthread_create");
        close(fd);
        return 1;
    }

    // 初始化 ncurses
    init_screen();

    // 生成赛道
    Track track = generate_track();

    // 初始化车辆
    Car car = {
        .pos = track.center[0],
        .yaw = 0,
        .speed = 0
    };

    // 投影参数 (假设视野 60°, 宽高比由终端决定)
    float near = 1.0f, far = 100.0f;
    float fov = 60.0f * M_PI / 180.0f;
    float aspect = (float)scr_width / scr_height;
    float top = near * tanf(fov/2);
    float bottom = -top;
    float right = top * aspect;
    float left = -right;
    mat4 proj = mat4_frustum(left, right, bottom, top, near, far);

    // 主循环
    int running = 1;
    while (running) {
        // 处理输入
        if (key_state[KEY_Q]) running = 0;          // Q 退出
        // 转向
        float steer = 0;
        if (key_state[KEY_A]) steer += 1;
        if (key_state[KEY_D]) steer -= 1;
        car.yaw += steer * 0.05f;

        // 加减速
        if (key_state[KEY_W]) car.speed += 0.1f;
        if (key_state[KEY_S]) car.speed -= 0.1f;
        // 简单摩擦
        car.speed *= 0.98f;

        // 根据速度和方向移动车辆
        vec3 dir = { sinf(car.yaw), 0, cosf(car.yaw) };
        car.pos.x += dir.x * car.speed;
        car.pos.z += dir.z * car.speed;
        // 简单高度跟随赛道（最近点）
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
        car.pos.y = target.y + 0.5f; // 车辆略高于赛道

        // 设置相机 (跟随车辆后方)
        vec3 eye = vec3_add(car.pos, (vec3){ -10*sinf(car.yaw), 3, -10*cosf(car.yaw) });
        vec3 center = car.pos;
        vec3 up = {0,1,0};
        mat4 view = mat4_lookat(eye, center, up);

        // 渲染
        erase();

        // 绘制赛道边界
        for (int i=0; i<track.count-1; i++) {
            int x1,y1, x2,y2;
            if (project(track.left[i], view, proj, &x1, &y1) &&
                project(track.left[i+1], view, proj, &x2, &y2))
                draw_line(x1, y1, x2, y2, '#');
            if (project(track.right[i], view, proj, &x1, &y1) &&
                project(track.right[i+1], view, proj, &x2, &y2))
                draw_line(x1, y1, x2, y2, '#');
        }

        // 绘制车辆
        int car_x, car_y;
        if (project(car.pos, view, proj, &car_x, &car_y))
            mvaddch(car_y, car_x, '@');

        // 显示帮助信息
        mvprintw(0, 0, "WASD 控制 | Q 退出 | 速度: %.2f", car.speed);

        refresh();
        usleep(16000); // 约 60 FPS
    }

    close_screen();
    pthread_cancel(input_th);
    pthread_join(input_th, NULL);
    close(fd);
    return 0;
}
