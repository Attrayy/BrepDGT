FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# 0. Install micromamba (lightweight conda)
# ------------------------------------------------------------
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y curl bzip2 ca-certificates && \
    curl -L https://micromamba.snakepit.net/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# ------------------------------------------------------------
# 1. Create micromamba environment with all Python dependencies
# ------------------------------------------------------------
RUN micromamba create -y -n occenv \
    -c conda-forge -c pytorch -c nvidia \
    python=3.9 pip \
    pytorch=2.2.2 torchvision torchaudio pytorch-cuda=11.8 \
    pythonocc-core=7.8.1=novtk_h5981d98_102 && \
    micromamba clean --all --yes

# Automatically activate occenv in all bash shells
RUN echo "micromamba activate occenv" >> /root/.bashrc

# ------------------------------------------------------------
# 2. Install GUI/X11 packages (Polyscope/GLFW, etc.)
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libxrender1 libsm6 libxext6 libx11-dev \
    libxcomposite1 libxcursor1 libxdamage1 libxi6 libxtst6 libnss3 libxrandr2 \
    libasound2 libatk1.0-0 libcups2 libxss1 libxshmfence-dev \
    libglu1-mesa-dev freeglut3-dev mesa-common-dev x11-xserver-utils \
    libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    git vim build-essential libssl-dev wget \
    libgmp-dev libmpfr-dev libboost-all-dev libeigen3-dev \
    libjpeg-dev libpng-dev libxkbcommon-dev && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 3. Install CMake 3.30.0
# ------------------------------------------------------------
ENV CMAKE_VERSION=3.30.0
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz && \
    tar -zxvf cmake-${CMAKE_VERSION}.tar.gz && \
    cd cmake-${CMAKE_VERSION} && \
    ./bootstrap -- -DCMAKE_USE_OPENSSL=OFF && make -j$(nproc) && make install && \
    cd .. && rm -rf cmake-${CMAKE_VERSION}*

# ------------------------------------------------------------
# 4. Clone brepdgt repo and submodules
# ------------------------------------------------------------
RUN git clone https://github.com/LF-masker/BrepDGT.git brepdgt && \
    cd brepdgt && \
    git submodule update --init --recursive
WORKDIR /brepdgt

# ------------------------------------------------------------
# 5. Install Python requirements inside micromamba (manually activated)
# ------------------------------------------------------------
RUN micromamba run -n occenv pip install -r requirements.txt && \
    micromamba run -n occenv pip install pdbpp && \
    micromamba run -n occenv pip uninstall -y numpy && \
    micromamba run -n occenv pip install numpy==1.23.5 && \
    micromamba run -n occenv pip install -e .

# Torch-geometric and DGL (with CUDA 11.8)
RUN micromamba run -n occenv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cu118.html && \
    micromamba run -n occenv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu118.html && \
    micromamba run -n occenv pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu118/repo.html

# ------------------------------------------------------------
# 6. Build PoissonRecon
# ------------------------------------------------------------
RUN cd deps/PoissonRecon && \
    make poissonrecon && \
    cd ../.. && \
    ln -s deps/PoissonRecon/Bin/Linux/PoissonRecon ./psr

# ------------------------------------------------------------
# 7. Build PyMesh
# ------------------------------------------------------------
RUN micromamba run -n occenv bash -c "\
    cd deps/PyMesh && \
    pip install -r python/requirements.txt && \
    python setup.py build && \
    python setup.py install && \
    python -c 'import pymesh; pymesh.test()'"

# ------------------------------------------------------------
# 8. Install Blender (independent Python)
# ------------------------------------------------------------
RUN wget https://download.blender.org/release/Blender3.4/blender-3.4.0-linux-x64.tar.xz && \
    tar -xvf blender-3.4.0-linux-x64.tar.xz && \
    ln -sr ./blender-3.4.0-linux-x64/blender ./blender && \
    rm -rf blender-3.4.0-linux-x64.tar.xz

# Install numpy into Blender's bundled Python to avoid conflicts
RUN ./blender-3.4.0-linux-x64/3.4/python/bin/python3.10 -m ensurepip && \
    ./blender-3.4.0-linux-x64/3.4/python/bin/python3.10 -m pip install numpy

# ------------------------------------------------------------
# 9. GUI runtime environment variables
# ------------------------------------------------------------
ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1

# ------------------------------------------------------------
# 10. Default command
# ------------------------------------------------------------
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
WORKDIR /workspace

# # Initialize micromamba shell hooks for bash and auto-activate `occenv`
# # Manually add shell hook and auto-activate occenv
# RUN echo 'eval "$(micromamba shell hook --shell bash)"' >> /root/.bashrc && \
#     echo 'micromamba activate occenv' >> /root/.bashrc
#
# WORKDIR /workspace
#
# ENTRYPOINT ["bash", "-c", "\
# eval \"$(micromamba shell hook --shell bash)\" && \
# micromamba activate occenv && \
# if [ ! -e ./psr ]; then ln -s /brepdgt/psr ./psr; fi && \
# if [ ! -e ./blender ]; then ln -s /brepdgt/blender ./blender; fi && \
# exec bash"]
