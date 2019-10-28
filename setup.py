'''
    Copyright (c) Facebook, Inc. and its affiliates.

    This source code is licensed under the MIT license found in the
    LICENSE file in the root directory of this source tree.
'''

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jacobian",
    version="1.0.0",
    author="Judy Hoffman, Daniel A. Roberts, and Sho Yaida",
    author_email="judy@gatech.edu, daniel.adam.roberts@gmail.com, shoyaida@fb.com",
    description="Jacobian regularization in PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/jacobian_regularizer",
    packages=['jacobian'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
