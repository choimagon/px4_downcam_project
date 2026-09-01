#!/usr/bin/env python3
"""Convert the checked-in Gazebo X500 visual meshes for MuJoCo use.

MuJoCo accepts OBJ/STL but not Gazebo's Collada (DAE) assets.  This script is
deliberately a conversion step, not a hand-modelled replacement: it reads the
same X500 mesh files referenced by ``x500_base/model.sdf`` and writes a small
MuJoCo asset bundle.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GAZEBO_MESH_DIR = PROJECT_ROOT / "PX4-Autopilot" / "Tools" / "simulation" / "gz" / "models" / "x500_base" / "meshes"
OUTPUT_DIR = PROJECT_ROOT / "assets" / "mujoco_x500"
DAE_MESHES = {
    "NXP-HGD-CF.dae": "x500_frame.obj",
    "5010Base.dae": "x500_motor.obj",
    "5010Bell.dae": "x500_bell.obj",
}
STL_MESHES = ("1345_prop_cw.stl", "1345_prop_ccw.stl")


def convert_dae(source: Path, destination: Path) -> None:
    scene = trimesh.load(source, force="scene")
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(scene)
    mesh = scene.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0:
        raise RuntimeError(f"Could not extract a mesh from {source}")
    mesh.export(destination)
    dimensions = mesh.extents
    print(f"{source.name} -> {destination.name}: {len(mesh.vertices)} vertices, extents={dimensions.round(4).tolist()}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for source_name, destination_name in DAE_MESHES.items():
        convert_dae(GAZEBO_MESH_DIR / source_name, OUTPUT_DIR / destination_name)
    for source_name in STL_MESHES:
        shutil.copy2(GAZEBO_MESH_DIR / source_name, OUTPUT_DIR / source_name)
        print(f"{source_name} -> {source_name}")
    (OUTPUT_DIR / "SOURCE.txt").write_text(
        "Converted from PX4-Autopilot/Tools/simulation/gz/models/x500_base/meshes/\n"
        "The original SDF is x500_mono_cam_down -> x500 -> x500_base plus mono_cam.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
