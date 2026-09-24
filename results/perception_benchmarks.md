# Perception Benchmarks

This document records experimental measurements obtained on the current RGB-D perception setup.

## Hardware

- Intel RealSense D435if
- RGB: 1280x720 @ 30 FPS
- Depth: 848x480 @ 30 FPS
- Depth format: 16UC1
- Depth scale: 0.001 m/unit
- Stereo baseline: approximately 49.9 mm

## ROS 2 middleware

On the current workstation, CycloneDDS provided substantially more stable image delivery than the initially tested Fast DDS configuration.

With CycloneDDS:

- depth 848x480 @ 30 FPS: approximately 30 FPS,
- RGB 1280x720 @ 30 FPS: approximately 30 FPS,
- combined RGB + depth acquisition remained close to the requested rate.

These results are specific to the tested computer and configuration and are not intended as a general comparison between DDS implementations.

## Depth-quality experiment

A planar cardboard target was evaluated at approximately 0.29 m.

With the infrared emitter enabled:

```text
invalid depth pixels : ~0.126 %
mean depth           : ~291.3 mm
global std. dev.     : ~3.0 mm
```

With the emitter disabled:

```text
invalid depth pixels : ~0.215 %
mean depth           : ~291.7 mm
global std. dev.     : ~3.1 mm
```

In this experiment, the emitter reduced the proportion of invalid depth pixels.

## Point-cloud reduction

A 5 mm voxel grid is currently used before plane segmentation and clustering.

Typical processing sequence:

```text
raw point cloud
    |
    v
3D CropBox
    |
    v
VoxelGrid
    |
    v
RANSAC / clustering
```

The purpose is to reduce computational cost while preserving enough geometric resolution for parcel-scale objects.

## Planned benchmark extensions

Future measurements will include:

- ground-truth object dimensions,
- position error in X/Y/Z,
- OBB orientation error,
- temporal jitter,
- segmentation performance,
- grasp-pose stability,
- end-to-end perception latency,
- grasp success rate.
