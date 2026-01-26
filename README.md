## Code Structure
The repository is structured as follows:

```kigumi-project
    ├── meshes
    │   └── *.msh
    ├── output
    └── scripts
        └── *.json
```

In alphabetical order:

meshes/: Stores the 3D geometry files  ```.msh``` used for the simulation. In this project, it specifically contains the ```kigumi``` (Japanese wood joinery) models.

output/: This folder will be populated with simulation results (such as ```.vtu``` files for ParaView) when you run the PolyFEM solver.

scripts/: Contains utility scripts for running simulations, data processing, or automating the workflow.