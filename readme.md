# Radiant Foam Viewer: Real-Time Differentiable Ray Tracing

https://www.youtube.com/watch?v=CDza_K9XNSQ

![example1](./imgs/scene1.png)

Visualization of point indices with Morton reordering enabled.

![morton](./imgs/scene1_morton.png)

## Features

- compute shader based renderer
  - morton reordering of voronoi cells
- standalone shader rendering mode (works in VRChat)
- optimized `.ply` loader (`binary_little_endian`)

## Model Data

Pre-trained models from the original authors:

- https://github.com/theialab/radfoam/releases/tag/v1
- https://drive.google.com/drive/folders/1Gs2cMeurL3l0jy497xoGqiFmlR9r7W8w

## Original Project Page

[Radiant Foam: Real-Time Differentiable Ray Tracing](https://radfoam.github.io/)

Not sure if performance is up to par with the original CUDA implementation, as I could not get it running locally yet (getting around 25-50fps on my RTX 3070, depending on view point).

¯\\\_(ツ)\_/¯
