<div align="center">

<h1>
   
[Buy hardware](https://tenstorrent.com/cards/) | [Discord](https://discord.gg/tenstorrent)

</h1>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/public/images/tt_refresh_forge_w_logo_gray.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/public/images/tt_refresh_forge_w_logo.png">
  <img alt="Logo" src="docs/public/images/tt_refresh_forge_w_logo_purple.png" height="250">
</picture>

</div>
<br>

TT-Forge – is Tenstorrent's MLIR-based compiler integrated into various frameworks, AI/ML models from domain-specific compilers to custom kernel generation.
We are currently still in developer preview for early adopters to check out what we've built and give it a try. 
  
-----
# What is this Repo?
This repository is the central hub for the TT-Forge compiler project, bringing together its various sub-projects into a cohesive product. Here, you'll find releases, demos, model support, roadmaps, and other key resources as the project evolves. Currently in early developer preview, we’ll be providing frequent updates to keep you informed on the latest developments. Please file any issues with questions or feedback you may have.

# Getting Started Guide 
See our individual front end documentations in the [Front end](#current-ai-framework-front-end-projects) section to get started running some tests.

# Project goals
- Provide abstraction of many different frontend frameworks
- Generically compile many kinds of model architectures without modification and with good performance
- Abstract all Tenstorrent device architectures

# Project Overview

TT-Forge is composed of various projects ranging from Frontends to support popular third-party AI Frameworks, MLIR compiler project, performance optimizations and tools to support the project.
TT-Forge lowers to our tt-metalium project providing additional functionality to our AI Sofware ecosystem.

![Tenstorrent Software overview](docs/public/images/tt-sw-overview.png)


## Current AI Framework Front End projects
- [TT-Forge-FE](https://github.com/tenstorrent/tt-forge-fe)
  - A TVM based graph compiler designed to optimize and transform computational graphs for deep learning models. Supports ingestion of PyTorch, ONNX, TensorFlow, PaddlePaddle and similar ML frameworks via TVM ([`tt-tvm`](https://github.com/tenstorrent/tt-tvm)).
  - See [docs pages](https://docs.tenstorrent.com/tt-forge-fe/getting-started.html) for an overview and getting started guide.

- [`tt-torch`](https://github.com/tenstorrent/tt-torch)

  - A MLIR-native, open-source, PyTorch 2.X and torch-mlir based front-end. It provides stableHLO (SHLO) graphs to `tt-mlir`. Supports ingestion of PyTorch models via PT2.X compile and ONNX models via torch-mlir (ONNX->SHLO)
  - See [docs pages](https://docs.tenstorrent.com/tt-torch) for an overview and getting started guide. 

- [`tt-xla`](https://github.com/tenstorrent/tt-xla)
  - Leverages a PJRT interface to integrate JAX (and in the future other frameworks), `tt-mlir` and Tenstorrent hardware. Supports ingestion of JAX models via jit compile, providing StableHLO (SHLO) graph to `tt-mlir` compiler
  - See [README](https://github.com/tenstorrent/tt-xla/blob/main/README.md) for an overview and getting started guide.


## [`tt-mlir`](https://github.com/tenstorrent/tt-mlir) project

At it's core `tt-mlir` is our compiler that is interfacing with tt-metalium our opens source low level AI Hardware SDK. `tt-mlir` provides a solution for optimizing machine learning and other compute workloads for all tenstorrent hardware. `tt-mlir` bridges the gap between all different ML Frameworks and Tenstorrent Hardware. `tt-mlir` is broken into different dialects:

- TTIR Dialect - Our common IR that can then be lowered into multiple different backends

- TTNN Dialect: Our entry point into the TTNN Library of Ops

- TTMetalium Dialect: Our entry point into directly accessing tt-metalium kernels. 

The compiler employs various optimization passes, including layout transformation, operation fusing, decomposition, and sharding, ensuring the efficient lowering to the target dialect.​

### `tt-mlir` tools and capabilities

- ttmlir-opt – tool is used to run the `tt-mlir` compiler passes on a .mlir source files and is central to developing and testing the compiler.​

- ttmlir-translate - ttmlir-translate allows us to ingest something (e.g., code) into MLIR compiler, and produce something (e.g., executable binary, or even code again) from MLIR compiler.​

- ttrt – is a standalone runtime tool that can inspect and run compiler executable files without front-end.​

- tt-explorer - It provides a “Human-In-Loop” interface such that the compiler results can be actively tuned and understood by the person compiling the model.​

- ttnn-standalone - post-compile tuning/debugging tool for C++ TTNN generated code.


-----
# Related Tenstorrent Projects
- [TT-Forge-FE](https://github.com/tenstorrent/tt-forge-fe)
- [tt-xla](https://github.com/tenstorrent/tt-xla)
- [tt-torch](https://github.com/tenstorrent/tt-torch)
- [tt-mlir](https://github.com/tenstorrent/tt-mlir)
- [tt-Metallium](https://github.com/tenstorrent/tt-metal)
- [tt-tvm](https://github.com/tenstorrent/tt-tvm)


### Tenstorrent Bounty Program Terms and Conditions
This repo is a part of Tenstorrent’s bounty program. If you are interested in helping to improve tt-metal, please make sure to read the [Tenstorrent Bounty Program Terms and Conditions](https://docs.tenstorrent.com/bounty_terms.html) before heading to the issues tab. Look for the issues that are tagged with both “bounty” and difficulty level!
