<p align="center">
  <h1 align="center"><strong>StableHand: Quality-Aware Flow Matching for World-Space Dual-Hand Motion Estimation from Egocentric Video</strong></h1>
  <p align="center">
    <a href="https://huajian-zeng.github.io/">Huajian Zeng</a><sup>1</sup>, Chaohua Yao<sup>2</sup>, <a href="https://scholar.google.com/citations?user=Lh2CthAAAAAJ&hl">Yuantai Zhang</a><sup>1</sup>, <a href="https://scholar.google.com/citations?hl=zh-CN&user=f7ox7CIAAAAJ">Jiaqi Yang</a><sup>1</sup>, <a href="https://rolpotamias.github.io/">Rolandos Alexandros Potamias</a><sup>3</sup>, <a href="https://xingxingzuo.github.io/">Xingxing Zuo</a><sup>1,*</sup>
    <br>
    <sup>1</sup>Mohamed Bin Zayed University of Artificial Intelligence (MBZUAI)
    <br>
    <sup>2</sup>University of Illinois at Urbana-Champaign (UIUC)
    <br>
    <sup>3</sup>Imperial College London
    <br>
    <sup>*</sup>Corresponding author
    <br>
  </p>

  <p align="center"><strong>In Submission</strong></p>
</p>

<div id="top" align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2605.18553-red)](https://arxiv.org/abs/2605.18553)
[![PDF](https://img.shields.io/badge/PDF-%F0%9F%93%84-green)](https://arxiv.org/pdf/2605.18553)
[![Homepage](https://img.shields.io/badge/Homepage-%F0%9F%8C%90-blue)](https://huajian-zeng.github.io/projects/stablehand/)
[![Video](https://img.shields.io/badge/Video-%E2%96%B6-red?logo=youtube&logoColor=red)](https://youtu.be/2UmaYTKQOAM)

</div>


## 🔥 Highlight <a name="highlight"></a>

**StableHand** is a quality-aware flow-matching framework that recovers world-space 4D motion of two interacting hands from egocentric video, even under long missing-hand spans and persistent hand–object occlusions.

We decompose noisy hand observations into four quality channels (wrist global translation and finger articulations of both hands), predicted by a learned quality network. These signals drive a per-channel forward schedule, a quality-adjusted velocity target, AdaLN modulation of a DiT denoiser, and a quality-aware ODE initialization, so that reliable observations are anchored while unreliable ones are regenerated from a learned bimanual motion prior.

On HOT3D and ARCTIC, StableHand achieves state-of-the-art performance across all reported metrics, **reducing W-MPJPE by 20–25%** over the strongest baseline, with the largest gains on heavily occluded ARCTIC sequences.

<div align="center">
    <img src="./asset/teaser.png" alt="teaser" width="90%" style="position: relative;">
</div>

## 📢 The code will be released after the paper is accepted. Stay tuned!
