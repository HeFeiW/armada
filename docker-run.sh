#!/bin/bash
# ARMADA Docker 启动脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
IMAGE_NAME="armada:latest"
CONTAINER_NAME="armada-dev"
DOCKERFILE_PATH="./Dockerfile"

# 函数
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker未安装。请访问 https://docs.docker.com/get-docker/"
        exit 1
    fi
    print_info "Docker环境检查通过"
}

# 检查NVIDIA Docker运行时
check_nvidia_docker() {
    if docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        print_info "NVIDIA Docker运行时可用"
        return 0
    else
        print_warning "NVIDIA Docker运行时不可用。GPU支持将不可用。"
        return 1
    fi
}

# 构建镜像
build_image() {
    print_info "构建Docker镜像: $IMAGE_NAME"
    docker build -t $IMAGE_NAME -f $DOCKERFILE_PATH .
    print_info "镜像构建完成"
}

# 启动容器
start_container() {
    print_info "启动容器: $CONTAINER_NAME"

    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            print_warning "容器已在运行"
            return 0
        else
            print_info "删除已停止的容器"
            docker rm $CONTAINER_NAME
        fi
    fi

    docker run -d \
        --gpus all \
        --name $CONTAINER_NAME \
        -v "$(pwd)":/workspace \
        -e CUDA_VISIBLE_DEVICES=0 \
        $IMAGE_NAME tail -f /dev/null

    print_info "容器启动完成"
}

# 进入容器
enter_container() {
    print_info "进入容器: $CONTAINER_NAME"

    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_error "容器未运行。请先运行 'start' 命令"
        exit 1
    fi

    docker exec -it $CONTAINER_NAME bash -c "source activate armada && exec bash"
}

# 运行ManiSkill2测试
test_maniskill() {
    print_info "运行ManiSkill2测试..."

    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_error "容器未运行。请先运行 'start' 命令"
        exit 1
    fi

    docker exec -it $CONTAINER_NAME bash -c "\
        source activate armada && \
        python -m mani_skill2.examples.demo_random_action -e PickCube-v0
    "
}

# 查看容器日志
logs() {
    docker logs -f $CONTAINER_NAME
}

# 停止容器
stop_container() {
    print_info "停止容器..."
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    print_info "容器已停止"
}

# 清理
clean() {
    print_warning "清理所有ARMADA相关的容器和镜像..."

    # 停止并删除容器
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true

    # 删除镜像
    docker rmi $IMAGE_NAME 2>/dev/null || true

    print_info "清理完成"
}

# 显示使用信息
usage() {
    cat << EOF
ARMADA Docker 启动脚本

用法: $0 <命令>

命令:
    build       构建Docker镜像
    start       启动容器
    enter       进入容器交互shell
    test        运行ManiSkill2测试
    logs        查看容器日志
    stop        停止容器
    clean       清理所有容器和镜像
    help        显示此帮助信息

示例:
    $0 build       # 构建镜像
    $0 start       # 启动容器
    $0 enter       # 进入容器
    $0 stop        # 停止容器

EOF
}

# 主流程
main() {
    if [ $# -eq 0 ]; then
        usage
        exit 1
    fi

    # 检查前置条件
    check_docker
    check_nvidia_docker || print_warning "某些功能可能无法正常运行"

    case "$1" in
        build)
            build_image
            ;;
        start)
            build_image
            start_container
            ;;
        enter)
            enter_container
            ;;
        test)
            test_maniskill
            ;;
        logs)
            logs
            ;;
        stop)
            stop_container
            ;;
        clean)
            stop_container
            clean
            ;;
        help)
            usage
            ;;
        *)
            print_error "未知命令: $1"
            usage
            exit 1
            ;;
    esac
}

main "$@"
