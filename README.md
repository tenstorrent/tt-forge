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

TT-Forge is Tenstorrent's MLIR-based compiler. It integrates into various compiler technologies from AI/ML frameworks, to both enable running models and create custom kernel generation. We are currently still in developer preview for early adopters to check out what we've built and give it a try. 
  
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

### Interactive Architecture Diagram
Click on components to navigate to their repositories:

```mermaid
graph TD
    %% Define styles
    classDef frameworks fill:#f9d6d2,stroke:#e05d44,stroke-width:2px
    classDef compiler fill:#d1e7dd,stroke:#198754,stroke-width:2px
    classDef runtime fill:#cfe2ff,stroke:#0d6efd,stroke-width:2px
    classDef hardware fill:#e2e3e5,stroke:#6c757d,stroke-width:2px
    classDef tools fill:#fff3cd,stroke:#ffc107,stroke-width:2px
    
    %% Main components in strict vertical alignment
    USER[ML/AI Applications]
    FrameworksBox[AI Framework Integration]
    CompilerBox[TT-Forge Compiler Stack]
    RuntimeBox[Runtime & Hardware SDK]
    
    %% Force vertical flow
    USER --> FrameworksBox
    FrameworksBox --> CompilerBox
    CompilerBox --> RuntimeBox
    
    %% Framework components positioned inside their box
    FrameworksBox --- FORGE_FE["TT-Forge-FE<br>(TVM-based)"]
    FrameworksBox --- TORCH["tt-torch<br>(PyTorch 2.X)"]
    FrameworksBox --- XLA["tt-xla<br>(JAX)"]
    FrameworksBox --- TVM["tt-tvm"]
    
    %% Input models
    PYTORCH["PyTorch"] --> FORGE_FE & TORCH
    ONNX["ONNX"] --> FORGE_FE & TORCH
    TF["TensorFlow"] --> FORGE_FE
    PADDLE["PaddlePaddle"] --> FORGE_FE
    JAX["JAX"] --> XLA
    TVM --> FORGE_FE
    
    %% Compiler components
    CompilerBox --- MLIR["tt-mlir"]
    
    %% MLIR Dialects
    MLIR --- TTIR["TTIR Dialect<br>(Common IR)"]
    MLIR --- TTNN["TTNN Dialect<br>(Library of Ops)"]
    MLIR --- TTMETAL["TTMetalium Dialect<br>(Hardware Access)"]
    
    %% Compiler Tools
    MLIR --- EXPLORER["tt-explorer<br>(Human-In-Loop)"]
    MLIR --- TTMLIR_OPT["ttmlir-opt<br>(Compiler Passes)"]
    MLIR --- TTMLIR_TRANSLATE["ttmlir-translate<br>(Format Conversion)"]
    MLIR --- TTRT["ttrt<br>(Runtime Tool)"]
    MLIR --- TTNN_STANDALONE["ttnn-standalone<br>(Tuning/Debugging)"]
    
    %% Direct connection to runtime
    MLIR --> METAL["tt-Metallium<br>(AI Hardware SDK)"]
    
    %% Runtime components
    RuntimeBox --- METAL
    METAL --- GRAYSKULL["Grayskull"]
    METAL --- WORMHOLE["Wormhole"]
    METAL --- BLACKHOLE["BlackHole"]
    
    %% Apply styles
    class FrameworksBox,FORGE_FE,TORCH,XLA,TVM,PYTORCH,ONNX,TF,PADDLE,JAX frameworks
    class CompilerBox,MLIR,TTIR,TTNN,TTMETAL compiler
    class EXPLORER,TTMLIR_OPT,TTMLIR_TRANSLATE,TTRT,TTNN_STANDALONE tools
    class RuntimeBox,METAL,GRAYSKULL,WORMHOLE,BLACKHOLE runtime
    
    %% Click actions
    click FORGE_FE "https://github.com/tenstorrent/tt-forge-fe" "TT-Forge-FE Repository"
    click TORCH "https://github.com/tenstorrent/tt-torch" "tt-torch Repository"
    click XLA "https://github.com/tenstorrent/tt-xla" "tt-xla Repository"
    click MLIR "https://github.com/tenstorrent/tt-mlir" "tt-mlir Repository"
    click METAL "https://github.com/tenstorrent/tt-metal" "tt-Metallium Repository"
    click TVM "https://github.com/tenstorrent/tt-tvm" "tt-tvm Repository"
```

### Current AI Framework Front End Projects
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
This repo is a part of Tenstorrent’s bounty program. If you are interested in helping to improve tt-forge, please make sure to read the [Tenstorrent Bounty Program Terms and Conditions](https://docs.tenstorrent.com/bounty_terms.html) before heading to the issues tab. Look for the issues that are tagged with both “bounty” and difficulty level!
