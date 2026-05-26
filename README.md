## Code Structure
The repository is structured as follows:

```bash
    kigumi-project
    ├── meshes
    │   └── *.msh
    │   └── *.obj
    ├── output
    └── scripts
        └── *.json
        └── *.sh
```

In alphabetical order:

meshes/: Stores the 3D geometry files  ```.msh``` used for the simulation. In this project, it specifically contains the ```kigumi``` (Japanese wood joinery) models.

output/: This folder will be populated with simulation results (such as ```.vtu``` files for ParaView) when you run the PolyFEM solver.

scripts/: Contains utility scripts for running simulations, data processing, or automating the workflow.

---

## PolyFEM Simulation Procedures
### 1. Run Simulation
To run the simulation, navigate to your script directory and execute the PolyFEM binary with your configuration file.

```bash
    #Navigate to the project directory
    cd scripts
    #Run PolyFEM 
    ~/polyfem/build/PolyFEM_bin --json  *.json
```

### Usage
Simulations are executed using the helper script located in the ```scripts``` directory.
This script automatically generates a timestamped output folder (format: ```MMDD_HHMM_simulation```) inside the ```output``` directory to organize results and prevent overwriting.

### Basic Execution
To run the default simulation (```*.json```):

```bash
    cd scripts
    ./run_sim.sh
```

### Running with Custom Settings
To run a specific JSON configuration file (e.g., ```tension_test.json```), pass the filename as an argument:

```bash
    cd scripts
    ./run_sim.sh tension_test.json
```
---

### 2. Render Results in Blender
The script `scripts/render_simulation.py` loads deformed surface geometries from PolyFEM `.vtu` output files into Blender and creates a scene with one mesh object per body per step.

Before running, edit the `CONFIG` section at the top of the script:

```python
SIM_DIR = "/path/to/simulation_output"   # folder containing step_N_surf.vtu files
STEPS   = [0, 5, 10]                     # which time steps to import
```

Blender bundles its own Python interpreter, but it needs `numpy` to parse the binary VTU files. The easiest way is to run Blender from within the `kigumi_env` conda environment, which makes the environment's `numpy` visible to Blender's Python:

```bash
conda activate kigumi_env
blender --background --python scripts/render_simulation.py
```

To open the result interactively (with the Blender GUI), omit `--background`:

```bash
conda activate kigumi_env
blender --python scripts/render_simulation.py
```

---

### 3. Visualize Results in ParaView
Once the simulation completes, a ```result.vtu``` file will be generated in the ```output``` folder. Follow these steps to visualize it:

1. Open File: Launch ParaView and go to ```File``` > ```Open``` to select ```result.vtu```.

2. Apply Settings: Click the green Apply button in the Properties panel on the left.

3. Color Mapping:
Change the display mode from Solid Color to solution (or other available fields like ```scalar_value```).

Click the Rescale to Data Range icon (a small rainbow icon with a magnifying glass) to adjust the color scale to your results.

4. Deform the Mesh (Optional):

Go to ```Filters``` > ```Alphabetical``` > Warp By Vector.

Set the "Vectors" to ```solution``` and click Apply to see the physical deformation.

---
両方の変化を確認するには（ParaView）:

1. body_ids で色分け（既に出力ON: kigumi-tension.json (lines 100-102)）
2. Warp By Vector で solution を使って変形表示
3. Threshold を2回使って body_ids=1 と body_ids=2 を分けて表示すると、各部材の変形を個別に比較できます


## Units


## PolyFEM Installation Notes

Most importantly, you need to install [https://github.com/polyfem/polyfem](PolyFEM) on your machine. Some dependencies are needed such as SuiteSparse or Eigen, which you can install through Homebrew or MacPort.

### Mac-specific installation

It is important to notice that you need to disable SPQR when building PolyFEM. This is done by executing in your local PolyFEM repository as

```bash
    mkdir build && cd build
    cmake -DPOLYSOLVE_WITH_SPQR=OFF ..
    build -j4
```

Ninja doesn't work for Mac on the current latest commit (tested 10/03/2026 with a M1 Max chip).