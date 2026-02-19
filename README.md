# 安装 ncurses 开发库（如果尚未安装）
sudo apt install libncurses-dev

# 编译（需要链接数学库和 ncurses）
gcc -o terminal_racer terminal_racer.c -lncurses -lm

# 以 root 运行（或确保当前用户在 input 组，并有权限读取键盘设备）
sudo ./terminal_racer/dev/input/by-path/*-kbd
