# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import urllib.parse
import hashlib
import requests
from pathlib import Path
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import cv2
import numpy as np


def get_file(path):
    """Get a file from local filesystem, cache, or URL.

    This function handles both local files and URLs, retrieving from cache when available
    or downloading/fetching as needed. For URLs, it creates a unique cached filename using
    a hash of the URL to prevent collisions.

    Args:
        path: Path to a local file or URL to download

    Returns:
        Path to the file in the cache
    """
    # Check if path is a URL - handle URLs and files differently
    path_is_url = path.startswith(("http://", "https://"))

    if path_is_url:
        # Create a hash from the URL to ensure uniqueness and prevent collisions
        url_hash = hashlib.md5(path.encode()).hexdigest()[:10]

        # Get filename from URL, or create one if not available
        file_name = os.path.basename(urllib.parse.urlparse(path).path)
        if not file_name:
            file_name = f"downloaded_file_{url_hash}"
        else:
            file_name = f"{url_hash}_{file_name}"

        rel_path = Path("url_cache")
        cache_dir_fallback = Path.home() / ".cache/url_cache"
    else:
        rel_dir, file_name = os.path.split(path)
        rel_path = Path("models/tt-ci-models-private") / rel_dir
        cache_dir_fallback = Path.home() / ".cache/lfcache" / rel_dir

    # Determine the base cache directory based on environment variables
    if "DOCKER_CACHE_ROOT" in os.environ and Path(os.environ["DOCKER_CACHE_ROOT"]).exists():
        cache_dir = Path(os.environ["DOCKER_CACHE_ROOT"]) / rel_path
    elif "LOCAL_LF_CACHE" in os.environ:
        cache_dir = Path(os.environ["LOCAL_LF_CACHE"]) / rel_path
    else:
        cache_dir = cache_dir_fallback

    file_path = cache_dir / file_name

    # Support case where shared cache is read only and file not found. Can read files from it, but
    # fall back to home dir cache for storing downloaded files. Common w/ CI cache shared w/ users.
    cache_dir_rdonly = not os.access(cache_dir, os.W_OK)
    if not file_path.exists() and cache_dir_rdonly and cache_dir != cache_dir_fallback:
        print(f"Warning: {cache_dir} is read-only, using {cache_dir_fallback} for {path}")
        cache_dir = cache_dir_fallback
        file_path = cache_dir / file_name

    cache_dir.mkdir(parents=True, exist_ok=True)

    # If file is not found in cache, download URL from web, or get file from IRD_LF_CACHE web server.
    if not file_path.exists():
        if path_is_url:
            try:
                print(f"Downloading file from URL {path} to {file_path}")
                response = requests.get(path, stream=True, timeout=(15, 60))
                response.raise_for_status()  # Raise exception for HTTP errors

                with open(file_path, "wb") as f:
                    f.write(response.content)

            except Exception as e:
                raise RuntimeError(f"Failed to download {path}: {str(e)}")
        elif "DOCKER_CACHE_ROOT" in os.environ:
            raise FileNotFoundError(
                f"File {file_path} is not available, check file path. If path is correct, DOCKER_CACHE_ROOT syncs automatically with S3 bucket every hour so please wait for the next sync."
            )
        else:
            if "IRD_LF_CACHE" not in os.environ:
                raise ValueError(
                    "IRD_LF_CACHE environment variable is not set. Please set it to the address of the IRD LF cache."
                )
            print(f"Downloading file from path {path} to {cache_dir}/{file_name}")
            exit_code = os.system(
                f"wget -nH -np -R \"indexg.html*\" -P {cache_dir} {os.environ['IRD_LF_CACHE']}/{path} --connect-timeout=15 --read-timeout=60 --tries=3"
            )
            # Check for wget failure
            if exit_code != 0:
                raise RuntimeError(
                    f"wget failed with exit code {exit_code} when downloading {os.environ['IRD_LF_CACHE']}/{path}"
                )

            # Ensure file_path exists after wget command
            if not file_path.exists():
                raise RuntimeError(
                    f"Download appears to have failed: File {file_name} not found in {cache_dir} after wget command"
                )

    return file_path


class ForgeModel(ABC):
    """Base class for all model implementations that can be shared across Tenstorrent projects."""

    @classmethod
    @abstractmethod
    def load_model(cls, **kwargs):
        """Load and return the model instance.

        Returns:
            torch.nn.Module: The model instance
        """
        pass

    @classmethod
    @abstractmethod
    def load_inputs(cls, **kwargs):
        """Load and return sample inputs for the model.

        Returns:
            Any: Sample inputs that can be fed to the model
        """
        pass

    @classmethod
    def decode_output(cls, **kwargs):
        """Load and return sample inputs for the model.

        Returns:
            Any: Output will be Decoded from the model
        """
        pass


class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        self.b1 = nn.BatchNorm2d(512)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.c2 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(1024)

        self.c3 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        self.b3 = nn.BatchNorm2d(512)

        # 3 maxpools
        self.p1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
        self.p2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
        self.p3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)
        ####

        self.c4 = nn.Conv2d(2048, 512, 1, 1, 0, bias=False)
        self.b4 = nn.BatchNorm2d(512)

        self.c5 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.b5 = nn.BatchNorm2d(1024)

        self.c6 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        self.b6 = nn.BatchNorm2d(512)

        self.c7 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.b7 = nn.BatchNorm2d(256)

        # 2 upsample2d
        self.u = nn.Upsample(scale_factor=(2, 2), mode="nearest")

        self.c7_2 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.b7_2 = nn.BatchNorm2d(256)

        self.c7_3 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.b7_3 = nn.BatchNorm2d(256)

        self.c8 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.b8 = nn.BatchNorm2d(512)

        self.c7_4 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.b7_4 = nn.BatchNorm2d(256)

        self.c8_2 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.b8_2 = nn.BatchNorm2d(512)

        self.c7_5 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.b7_5 = nn.BatchNorm2d(256)

        self.c9 = nn.Conv2d(256, 128, 1, 1, 0, bias=False)
        self.b9 = nn.BatchNorm2d(128)

        self.c9_2 = nn.Conv2d(256, 128, 1, 1, 0, bias=False)
        self.b9_2 = nn.BatchNorm2d(128)
        self.c9_3 = nn.Conv2d(256, 128, 1, 1, 0, bias=False)
        self.b9_3 = nn.BatchNorm2d(128)

        self.c10 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.b10 = nn.BatchNorm2d(256)

        self.c9_4 = nn.Conv2d(256, 128, 1, 1, 0, bias=False)
        self.b9_4 = nn.BatchNorm2d(128)
        self.c10_2 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.b10_2 = nn.BatchNorm2d(256)
        self.c9_5 = nn.Conv2d(256, 128, 1, 1, 0, bias=False)
        self.b9_5 = nn.BatchNorm2d(128)

    def forward(self, input1, input2, input3):
        # 3 CBN blocks
        x1 = self.c1(input1)
        x1_b = self.b1(x1)
        x1_m = self.relu(x1_b)

        x2 = self.c2(x1_m)
        x2_b = self.b2(x2)
        x2_m = self.relu(x2_b)

        x3 = self.c3(x2_m)
        x3_b = self.b3(x3)
        x3_m = self.relu(x3_b)

        # maxpools
        x4 = self.p1(x3_m)
        x5 = self.p2(x3_m)
        x6 = self.p3(x3_m)

        # concat the outputs of maxpool and x3_m
        conc1 = torch.cat([x6, x5, x4, x3_m], dim=1)

        # 4 back2back CBRs
        # CBR4-1
        x7 = self.c4(conc1)
        x7_b = self.b4(x7)
        x7_m = self.relu(x7_b)

        # CBR4-2
        x8 = self.c5(x7_m)
        x8_b = self.b5(x8)
        x8_m = self.relu(x8_b)

        # CBR4-3
        x9 = self.c6(x8_m)
        x9_b = self.b6(x9)
        x9_m = self.relu(x9_b)

        # CBR4-4
        x10 = self.c7(x9_m)
        x10_b = self.b7(x10)
        x10_m = self.relu(x10_b)

        # upsample
        u1 = self.u(x10_m)

        # Next CBR block to be concatinated with output of u1
        # gets the output of downsample4 module which is dimensions: [1, 512, 20, 20] - make a random tensor with that shape for the purpose of running the neck unit test stand-alone
        outDownSample4 = input2
        # CBR block for conc2
        x11 = self.c7_2(outDownSample4)
        x11_b = self.b7_2(x11)
        x11_m = self.relu(x11_b)

        # concat CBR output with output from u1
        conc2 = torch.cat([x11_m, u1], dim=1)

        # 6 back2back CBRs
        # CBR6_1
        x12 = self.c7_3(conc2)
        x12_b = self.b7_3(x12)
        x12_m = self.relu(x12_b)

        # CBR6_2
        x13 = self.c8(x12_m)
        x13_b = self.b8(x13)
        x13_m = self.relu(x13_b)

        # CBR6_3
        x14 = self.c7_4(x13_m)
        x14_b = self.b7_4(x14)
        x14_m = self.relu(x14_b)

        # CBR6_4
        x15 = self.c8_2(x14_m)
        x15_b = self.b8_2(x15)
        x15_m = self.relu(x15_b)

        # CBR6_5
        x16 = self.c7_5(x15_m)
        x16_b = self.b7_5(x16)
        x16_m = self.relu(x16_b)

        # CBR6_6
        x17 = self.c9(x16_m)
        x17_b = self.b9(x17)
        x17_m = self.relu(x17_b)

        # upsample
        u2 = self.u(x17_m)

        # CBR block for conc3
        outDownSample3 = input3
        x18 = self.c9_2(outDownSample3)
        x18_b = self.b9_2(x18)
        x18_m = self.relu(x18_b)

        # concat CBR output with output from u2
        conc3 = torch.cat([x18_m, u2], dim=1)

        # 5 CBR blocks
        # CBR5_1
        x19 = self.c9_3(conc3)
        x19_b = self.b9_3(x19)
        x19_m = self.relu(x19_b)

        # CBR5_2
        x20 = self.c10(x19_m)
        x20_b = self.b10(x20)
        x20_m = self.relu(x20_b)

        # CBR5_3
        x21 = self.c9_4(x20_m)
        x21_b = self.b9_4(x21)
        x21_m = self.relu(x21_b)

        # CBR5_4
        x22 = self.c10_2(x21_m)
        x22_b = self.b10_2(x22)
        x22_m = self.relu(x22_b)

        # CBR5_5
        x23 = self.c9_5(x22_m)
        x23_b = self.b9_5(x23)
        x23_m = self.relu(x23_b)
        # return [x4, x4, x4]
        return x23_m, x9_m, x16_m


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class ResBlock(nn.Module):
    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            conv1 = nn.Conv2d(ch, ch, 1, 1, 0, bias=False)
            bn1 = nn.BatchNorm2d(ch)
            mish = Mish()
            conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
            bn2 = nn.BatchNorm2d(ch)
            resblock_one = nn.ModuleList([conv1, bn1, mish, conv2, bn2, mish])
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.b1 = nn.BatchNorm2d(32)
        self.mish = Mish()

        self.c2 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.b2 = nn.BatchNorm2d(64)

        self.c3 = nn.Conv2d(64, 64, 1, 1, 0, bias=False)
        self.b3 = nn.BatchNorm2d(64)

        self.c4 = nn.Conv2d(64, 64, 1, 1, 0, bias=False)
        self.b4 = nn.BatchNorm2d(64)

        self.c5 = nn.Conv2d(64, 32, 1, 1, 0, bias=False)
        self.b5 = nn.BatchNorm2d(32)

        self.c6 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.b6 = nn.BatchNorm2d(64)

        self.c7 = nn.Conv2d(64, 64, 1, 1, 0, bias=False)
        self.b7 = nn.BatchNorm2d(64)

        self.c8 = nn.Conv2d(128, 64, 1, 1, 0, bias=False)
        self.b8 = nn.BatchNorm2d(64)

    def forward(self, input: torch.Tensor):
        x1 = self.c1(input)
        x1_b = self.b1(x1)
        x1_m = self.mish(x1_b)

        x2 = self.c2(x1_m)
        x2_b = self.b2(x2)
        x2_m = self.mish(x2_b)

        x3 = self.c3(x2_m)
        x3_b = self.b3(x3)
        x3_m = self.mish(x3_b)

        x4 = self.c4(x2_m)
        x4_b = self.b4(x4)
        x4_m = self.mish(x4_b)

        x5 = self.c5(x4_m)
        x5_b = self.b5(x5)
        x5_m = self.mish(x5_b)

        x6 = self.c6(x5_m)
        x6_b = self.b6(x6)
        x6_m = self.mish(x6_b)
        x6_m = x6_m + x4_m

        x7 = self.c7(x6_m)
        x7_b = self.b7(x7)
        x7_m = self.mish(x7_b)
        x7_m = torch.cat([x7_m, x3_m], dim=1)

        x8 = self.c8(x7_m)
        x8_b = self.b8(x8)
        x8_m = self.mish(x8_b)

        return x8_m


class DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.b1 = nn.BatchNorm2d(128)
        self.mish = Mish()

        self.c2 = nn.Conv2d(128, 64, 1, 1, 0, bias=False)
        self.b2 = nn.BatchNorm2d(64)

        self.c3 = nn.Conv2d(128, 64, 1, 1, 0, bias=False)
        self.b3 = nn.BatchNorm2d(64)

        self.res = ResBlock(ch=64, nblocks=2)

        self.c4 = nn.Conv2d(64, 64, 1, 1, 0, bias=False)
        self.b4 = nn.BatchNorm2d(64)

        self.c5 = nn.Conv2d(128, 128, 1, 1, 0, bias=False)
        self.b5 = nn.BatchNorm2d(128)

    def forward(self, input: torch.Tensor):
        x1 = self.c1(input)
        x1_b = self.b1(x1)
        x1_m = self.mish(x1_b)

        x2 = self.c2(x1_m)
        x2_b = self.b2(x2)
        x2_m = self.mish(x2_b)

        x3 = self.c3(x1_m)
        x3_b = self.b3(x3)
        x3_m = self.mish(x3_b)

        r1 = self.res(x3_m)

        x4 = self.c4(r1)
        x4_b = self.b4(x4)
        x4_m = self.mish(x4_b)

        x4_m = torch.cat([x4_m, x2_m], dim=1)

        x5 = self.c5(x4_m)
        x5_b = self.b5(x5)
        x5_m = self.mish(x5_b)
        return x5_m


class DownSample3(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
        self.b1 = nn.BatchNorm2d(256)
        self.mish = Mish()

        self.c2 = nn.Conv2d(256, 128, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(128)

        self.c3 = nn.Conv2d(256, 128, 1, 1, bias=False)
        self.b3 = nn.BatchNorm2d(128)

        self.res = ResBlock(128, 8)

        self.c4 = nn.Conv2d(128, 128, 1, 1, bias=False)
        self.b4 = nn.BatchNorm2d(128)

        self.c5 = nn.Conv2d(256, 256, 1, 1, bias=False)
        self.b5 = nn.BatchNorm2d(256)

    def forward(self, input: torch.Tensor):
        x1 = self.c1(input)
        x1_b = self.b1(x1)
        x1_m = self.mish(x1_b)

        x2 = self.c2(x1_m)
        x2_b = self.b2(x2)
        x2_m = self.mish(x2_b)

        x3 = self.c3(x1_m)
        x3_b = self.b3(x3)
        x3_m = self.mish(x3_b)

        r1 = self.res(x3_m)

        x4 = self.c4(r1)
        x4_b = self.b4(x4)
        x4_m = self.mish(x4_b)

        x4_m = torch.cat([x4_m, x2_m], dim=1)

        x5 = self.c5(x4_m)
        x5_b = self.b5(x5)
        x5_m = self.mish(x5_b)
        return x5_m


class DownSample4(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)
        self.b1 = nn.BatchNorm2d(512)
        self.mish = Mish()

        self.c2 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.b2 = nn.BatchNorm2d(256)

        self.c3 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.b3 = nn.BatchNorm2d(256)

        self.res = ResBlock(256, 8)

        self.c4 = nn.Conv2d(256, 256, 1, 1, 0, bias=False)
        self.b4 = nn.BatchNorm2d(256)

        self.c5 = nn.Conv2d(512, 512, 1, 1, 0, bias=False)
        self.b5 = nn.BatchNorm2d(512)

    def forward(self, input: torch.Tensor):
        x1 = self.c1(input)
        x1_b = self.b1(x1)
        x1_m = self.mish(x1_b)

        x2 = self.c2(x1_m)
        x2_b = self.b2(x2)
        x2_m = self.mish(x2_b)

        x3 = self.c3(x1_m)
        x3_b = self.b3(x3)
        x3_m = self.mish(x3_b)

        # resblock
        r = self.res(x3_m)

        x4 = self.c4(r)
        x4_b = self.b4(x4)
        x4_m = self.mish(x4_b)

        x4_m = torch.cat([x4_m, x2_m], dim=1)

        x5 = self.c5(x4_m)
        x5_b = self.b5(x5)
        x5_m = self.mish(x5_b)

        return x5_m


class DownSample5(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(512, 1024, 3, 2, 1, bias=False)
        self.b1 = nn.BatchNorm2d(1024)
        self.mish = Mish()

        self.c2 = nn.Conv2d(1024, 512, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(512)

        self.c3 = nn.Conv2d(1024, 512, 1, 1, bias=False)
        self.b3 = nn.BatchNorm2d(512)

        self.res = ResBlock(512, 4)

        self.c4 = nn.Conv2d(512, 512, 1, 1, bias=False)
        self.b4 = nn.BatchNorm2d(512)
        self.mish = Mish()

        self.c5 = nn.Conv2d(1024, 1024, 1, 1, bias=False)
        self.b5 = nn.BatchNorm2d(1024)
        self.mish = Mish()

    def forward(self, input: torch.Tensor):
        x1 = self.c1(input)
        x1_b = self.b1(x1)
        x1_m = self.mish(x1_b)

        x2 = self.c2(x1_m)
        x2_b = self.b2(x2)
        x2_m = self.mish(x2_b)

        x3 = self.c3(x1_m)
        x3_b = self.b3(x3)
        x3_m = self.mish(x3_b)

        # resblock
        r = self.res(x3_m)

        x4 = self.c4(r)
        x4_b = self.b4(x4)
        x4_m = self.mish(x4_b)

        x4_m = torch.cat([x4_m, x2_m], dim=1)

        x5 = self.c5(x4_m)
        x5_b = self.b5(x5)
        x5_m = self.mish(x5_b)

        return x5_m


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        # left side of graph
        # in_chan, out_chan, kernel, stride,
        output_ch = 255

        self.c1 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.b1 = nn.BatchNorm2d(256)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.c2 = nn.Conv2d(256, output_ch, 1, 1, 0, bias=True)

        # R -4
        self.c3 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
        self.b3 = nn.BatchNorm2d(256)

        # R -1 -16
        self.c4 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.b4 = nn.BatchNorm2d(256)

        self.c5 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.b5 = nn.BatchNorm2d(512)

        self.c6 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.b6 = nn.BatchNorm2d(256)

        self.c7 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.b7 = nn.BatchNorm2d(512)

        self.c8 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.b8 = nn.BatchNorm2d(256)

        self.c9 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.b9 = nn.BatchNorm2d(512)

        self.c10 = nn.Conv2d(512, output_ch, 1, 1, 0, bias=True)

        # R -4
        self.c11 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)
        self.b11 = nn.BatchNorm2d(512)

        self.c12 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        self.b12 = nn.BatchNorm2d(512)

        self.c13 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.b13 = nn.BatchNorm2d(1024)

        self.c14 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        self.b14 = nn.BatchNorm2d(512)

        self.c15 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.b15 = nn.BatchNorm2d(1024)

        self.c16 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        self.b16 = nn.BatchNorm2d(512)

        self.c17 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.b17 = nn.BatchNorm2d(1024)

        self.c18 = nn.Conv2d(1024, output_ch, 1, 1, 0, bias=True)

    def forward(self, input1, input2, input3):
        x1 = self.c1(input1)
        x1 = self.b1(x1)
        x1 = self.relu(x1)

        x2 = self.c2(x1)

        x3 = self.c3(input1)
        x3 = self.b3(x3)
        x3 = self.relu(x3)

        # R -1 -16
        outfromNeck1 = input3
        x3 = torch.cat([x3, outfromNeck1], dim=1)

        x4 = self.c4(x3)
        x4 = self.b4(x4)
        x4 = self.relu(x4)

        x5 = self.c5(x4)
        x5 = self.b5(x5)
        x5 = self.relu(x5)

        x6 = self.c6(x5)
        x6 = self.b6(x6)
        x6 = self.relu(x6)

        x7 = self.c7(x6)
        x7 = self.b7(x7)
        x7 = self.relu(x7)

        x8 = self.c8(x7)
        x8 = self.b8(x8)
        x8 = self.relu(x8)

        x9 = self.c9(x8)
        x9 = self.b9(x9)
        x9 = self.relu(x9)

        x10 = self.c10(x9)

        # R -4
        x11 = self.c11(x8)
        x11 = self.b11(x11)
        x11 = self.relu(x11)

        # R -1 -37
        outfromNeck2 = input2
        x11 = torch.cat([x11, outfromNeck2], dim=1)

        x12 = self.c12(x11)
        x12 = self.b12(x12)
        x12 = self.relu(x12)

        x13 = self.c13(x12)
        x13 = self.b13(x13)
        x13 = self.relu(x13)

        x14 = self.c14(x13)
        x14 = self.b14(x14)
        x14 = self.relu(x14)

        x15 = self.c15(x14)
        x15 = self.b15(x15)
        x15 = self.relu(x15)

        x16 = self.c16(x15)
        x16 = self.b16(x16)
        x16 = self.relu(x16)

        x17 = self.c17(x16)
        x17 = self.b17(x17)
        x17 = self.relu(x17)

        x18 = self.c18(x17)
        return x2, x10, x18


class Yolov4(nn.Module):
    def __init__(self):
        super(Yolov4, self).__init__()
        self.downsample1 = DownSample1()
        self.downsample2 = DownSample2()
        self.downsample3 = DownSample3()
        self.downsample4 = DownSample4()
        self.downsample5 = DownSample5()
        self.neck = Neck()
        self.head = Head()

    def forward(self, input: torch.Tensor):
        d1 = self.downsample1(input)
        d2 = self.downsample2(d1)
        d3 = self.downsample3(d2)
        d4 = self.downsample4(d3)
        d5 = self.downsample5(d4)
        x20, x13, x6 = self.neck(d5, d4, d3)
        x4, x5, x6 = self.head(x20, x13, x6)

        return x4, x5, x6

    @staticmethod
    def from_random_weights():
        model = Yolov4()
        model.eval()

        new_state_dict = {}
        for name, parameter in model.state_dict().items():
            if isinstance(parameter, torch.FloatTensor):
                new_state_dict[name] = parameter

        model.load_state_dict(new_state_dict)
        return model


class Yolov4Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        x, y, z = self.model(image)
        # Post processing inside model casts output to float32,
        # even though raw output is aligned with image.dtype
        # Therefore we need to cast it back to image.dtype
        return x.to(image.dtype), y.to(image.dtype), z.to(image.dtype)


class ModelLoader(ForgeModel):
    """YOLOv4 model loader implementation."""

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the YOLOv4 model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YOLOv4 model instance.
        """
        model = Yolov4()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Load and return sample inputs for the YOLOv4 model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        img = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (640, 480))  # Resize to model input size
        img = img / 255.0  # Normalize to [0,1]
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW format
        img = [torch.from_numpy(img).float().unsqueeze(0)]  # Add batch dimension
        batch_tensor = torch.cat(img, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor
