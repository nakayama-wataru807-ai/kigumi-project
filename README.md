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
    cd ~/kigumi-project/scripts
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

### 2. Visualize Results in ParaView
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