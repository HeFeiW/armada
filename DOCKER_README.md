# ARMADA Docker 配置

此目录包含用于ARMADA项目的Docker配置文件，集成了所有依赖项以及ManiSkill2环境。

## 文件说明

- **Dockerfile**: 构建配置，安装ARMADA和ManiSkill2的所有依赖
- **docker-run.sh**: 自动化脚本，简化镜像构建和容器管理
- **.dockerignore**: 优化镜像大小的忽略文件列表

## 前置要求

1. **Docker**: >= 20.10
2. **NVIDIA Docker Runtime**: 用于GPU支持
   ```bash
   # Ubuntu 安装 nvidia-docker
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

## 使用方法

### 方式1：使用自动化脚本（推荐）

```bash
# 构建镜像
./docker-run.sh build

# 启动容器
./docker-run.sh start

# 进入容器交互shell
./docker-run.sh enter

# 运行ManiSkill2测试
./docker-run.sh test

# 查看日志
./docker-run.sh logs

# 停止容器
./docker-run.sh stop

# 清理所有容器和镜像
./docker-run.sh clean
```

### 方式2：使用 Docker 命令

#### 构建镜像
```bash
docker build -t armada:latest .
```

#### 运行容器（交互模式）
```bash
docker run --gpus all \
  -it \
  --name armada-dev \
  -v $(pwd):/workspace \
  -e CUDA_VISIBLE_DEVICES=0 \
  armada:latest bash -c "source activate armada && exec bash"
```

#### 后台运行容器
```bash
docker run --gpus all \
  -d \
  --name armada-dev \
  -v $(pwd):/workspace \
  -e CUDA_VISIBLE_DEVICES=0 \
  armada:latest tail -f /dev/null
```

#### 进入运行的容器
```bash
docker exec -it armada-dev bash -c "source activate armada && exec bash"
```

## 容器内操作

### 激活 conda 环境
```bash
conda activate armada
```

### 验证安装
```bash
# 验证PyTorch和GPU支持
python -c "import torch; print(torch.cuda.is_available())"

# 验证ManiSkill2
python -m mani_skill2.examples.demo_random_action

# 验证ARMADA依赖
python -c "import diffusers; import POT; import kornia; print('All dependencies loaded!')"
```

### 下载ManiSkill2资源
```bash
# 下载所有资源（约10GB）
python -m mani_skill2.utils.download_asset all

# 或下载特定任务资源
python -m mani_skill2.utils.download_asset PickCube-v0
```

### 运行ARMADA示例
```bash
# 进入项目目录
cd /workspace

# 查看可用的示例和配置
ls armada/config/training/

# 运行ManiSkill集成配置
# python -m armada.train <config_file>
```

## 环境变量

- `CUDA_VISIBLE_DEVICES`: 指定可用的GPU ID（默认：0）
- `MS2_ASSET_DIR`: ManiSkill2资源目录（默认：/workspace/data）
- `PYTORCH_CUDA_ALLOC_CONF`: PyTorch CUDA内存配置（可选）

### 自定义环境变量
在run命令时传递：
```bash
docker run --gpus all \
  -it \
  --name armada-dev \
  -v $(pwd):/workspace \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512 \
  armada:latest bash -c "source activate armada && exec bash"
```

### 使用多GPU
```bash
docker run --gpus all \
  -it \
  --name armada-dev \
  -v $(pwd):/workspace \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  armada:latest bash -c "source activate armada && exec bash"
```

## 故障排除

### GPU不可用
```bash
# 检查GPU支持
docker run --gpus all armada:latest nvidia-smi

# 验证nvidia-docker安装
docker run --gpus all ubuntu:22.04 nvidia-smi
```

### 内存不足
- 减少batch size
- 增加Docker容器的内存限制：`docker run -m 32g [其他参数]`
- 清理缓存：`python -c "import torch; torch.cuda.empty_cache()"`

### 容器启动失败
```bash
# 查看错误日志
docker logs armada-dev

# 重新构建镜像
docker build --no-cache -t armada:latest .
```

## 开发工作流

### 挂载本地代码
容器配置将当前目录挂载到 `/workspace`，开发时的任何文件更改都会立即反映在容器内。

### 安装额外包
```bash
docker exec -it armada-dev bash -c "source activate armada && pip install <package-name>"
```

### 进行本地编辑与运行
```bash
# 在宿主机编辑代码
# 在容器内运行
docker exec -it armada-dev bash -c "source activate armada && python -m armada.train ..."
```

## 构建优化建议

如果网络较慢，可以在构建前预配置conda/pip镜像源（在主机上执行）：

```bash
# 创建 .condarc
cat > ~/.condarc << EOF
channels:
  - https://mirrors.tsinghua.edu.cn/anaconda/cloud/pytorch
  - https://mirrors.tsinghua.edu.cn/anaconda/cloud/pytorch3d
  - https://mirrors.tsinghua.edu.cn/anaconda/cloud/nvidia
  - https://mirrors.tsinghua.edu.cn/anaconda/cloud/conda-forge
EOF

# 创建 pip.conf
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf << EOF
[global]
index-url = https://pypi.tsinghua.edu.cn/simple
EOF

# 然后重新构建镜像
docker build -t armada:latest .
```

## 许可证

ARMADA和ManiSkill2均采用各自的开源许可证。详见项目根目录的LICENSE文件。

## 支持

如有问题或建议，请提交Issue到对应的GitHub仓库。
