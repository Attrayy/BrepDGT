# 🛠 Installation

We support both Docker and Python/Conda environments for installing and running **BRepDGT**.

- For **training** and **reproducing quantitative results** on unconditional generation, we recommend using the **Docker container**, which ensures reproducibility and simplifies environment setup.
- For **interactive editing** with BRepDGT (e.g., using the Polyscope viewer), we recommend the **Conda environment**, which better supports GUI applications. Docker does work, but it might be slow.

---
## 📦 **Docker container**

### 🔧 Build the Docker Image
To build the docker environment, run
```
# build docker
docker build -t brepdgt .
```
⚠️ Building the image may take up to an hour (or more) depending on your system and internet speed.

### 🚀 Run the Container with GPU and GUI Support
1. Allow GUI access (X11 forwarding)
Run this once per session on your host:
```
bash
xhost +local:root
```
This allows the container to open GUI windows (e.g., Polyscope).
To revoke this permission later:
```
bash
xhost -local:root
```

2. Launch the container
To run the container interactively with GPU support and X11 GUI forwarding:
```
docker run --gpus all -it \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace \
  --shm-size=8g \
  --name brepdgt \
  brepdgt
```
This command:
* Mounts your current working directory to /workspace inside the container.
* Enables GPU acceleration.
* Enables X11 GUI forwarding (for interactive viewer).
* Ensures that changes inside the container persist on your host (e.g., logs, code edits, checkpoints).

⚠️ You may see a warning like:
`"micromamba is running as a subprocess and can't modify the parent shell..."`
This is expected and harmless.



If you've exited the container and want to re-enter:
```
docker start -ai brepdgt
```

📂 Directory Structure Notes
* `/workspace`: This directory is mounted from your host’s current working directory. Any changes made inside the container (e.g., editing code) will persist on your host.
* `/brepdgt`: This directory is used only during the image build process to install dependencies. It should not be modified during interactive sessions.


---
## 🔧 **Python/Conda environment** 

### Prerequisites
Tested on:
cmake 3.30.5 (> 3.10)
gcc 3.30.5 (> 7)
CUDA 11.8

Follow the order written below to install the dependencies.
Note that order may matter. 
Make sure everything is compiled and installed using consistent versions of cmake,gcc,cuda

Also, keep in mind that after you install the dependencies, check the numpy version because they get double installed.

### Basic Installation
```
conda create -n brepdgt python=3.9
conda activate brepdgt
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.2/cu118/repo.html
pip install -r requirements.txt
pip install -e .
```
If deepdish fails when saving trajectories, install it with conda.
```
conda install -c conda-forge deepdish
```

Check numpy version
```angular2html
python -c "import numpy; print(numpy.__version__)"
```
If the numpy version (output) is not 1.23.5, then remove and reinstall numpy with
```angular2html
pip uninstall numpy
pip install numpy==1.23.5
```


### pythonOCC
Run 
```
 conda install -c conda-forge pythonocc-core=7.8.1=novtk_h5981d98_102
```
Note that we install the exact occ 7.8.1 and not 7.8.1.1, which the above script will ensure to install.

## Submodules
### Poisson Surface Reconstruction
To obtain occupancy grid from the UvGrids, we use Poisson Surface Reconstuction.
After cloning the main repository, run below to update PoissonReconGrid in deps
```
git submodule update --init --recursive
# build poisson recon
cd deps/PoissonRecon
make poissonrecon
```
Now, make a symbolic link to the root of the repository for convienence.
```
# to root
cd ../..
ln -s deps/PoissonRecon/Bin/Linux/PoissonRecon ./psr
```
To check if the psr is sucessfully installed, run the following command after downloading the example uvgrid from [here](https://drive.google.com/drive/folders/1Gl40Xvwypkbpx5uWG01uOQqlSPzLEIdF?usp=drive_link).
```
./psr --in psr_example.ply --out psr_example_out.ply --depth 7
```
Compare with the expected output (`psr_example_out.ply`) in the google drive to check if psr is properly installed.

### Pymesh
Run the following command to install [PyMesh](https://pymesh.readthedocs.io/en/latest/installation.html) as described in the original installation section.
```angular2html
git submodule update --init --recursive
cd deps/PyMesh
export PYMESH_PATH=`pwd`
pip install -r python/requirements.txt
python ./setup.py build
python ./setup.py install --user
python -c "import pymesh; pymesh.test()"
```

#### Troubleshooting PyMesh installation

If `Could NOT find MPFR (missing: MPFR_LIBRARIES MPFR_INCLUDE_DIR)`(https://github.com/PyMesh/PyMesh/issues/96), install CGAL separately.
```
sudo apt install libcgal-dev
```

If draco build fails, follow the replies in (https://github.com/PyMesh/PyMesh/issues/365) or modify:
* `third_party/draco/src/draco/core/hash_utils.h` (added second line below)
```cpp
#include <stdint.h>
#include <cstddef>
#include <functional>
```
* `third_party/draco/src/draco/io/parser_utils.cc` (added fifth line below)
```cpp
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iterator>
#include <limits>
```

If MMG fails with linking errors, add `SET(CMAKE_C_FLAGS " -fcommon ${CMAKE_C_FLAGS}")` to `PyMesh/third_party/mmg/CMakeLists.txt`.

### Blender
During training, we visualize the generated uvgrids which uses blender.
Even if blender is not installed, the training is expected to continue, thus making it optional.
```angular2html
wget https://download.blender.org/release/Blender3.4/blender-3.4.0-linux-x64.tar.xz
tar -xvf blender-3.4.0-linux-x64.tar.xz                                                                                                                                                                           
ln -sr ./blender-3.4.0-linux-x64/blender ./blender
rm -rf blender-3.4.0-linux-x64.tar.xz  
```
Run `$ ./blender --version` and if version 3.4.0 pops up, then you've succesfully installed it!


### Troubleshooting

- Chamfer distance: If you have a problem with importing chamfer distance please check if you have your cuda version (including minor version) matches your cuda version.
