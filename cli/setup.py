#!/usr/bin/env python3
"""
ContestTrade CLI Setup
"""
from setuptools import setup, find_packages

setup(
    name="contesttrade-cli",
    version="1.0.0",
    description="ContestTrade: 基于内部竞赛机制的Multi-Agent交易系统",
    author="ContestTrade Team",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "typer>=0.9.0",
        "rich>=13.0.0",
        "questionary>=2.0.0",
        "asyncio",
        "pathlib",
        "datetime",
        "collections",
        "json",
        "sys",
    ],
    entry_points={
        "console_scripts": [
            "contesttrade=cli.main:app",
        ],
    },
    python_requires=">=3.8",
) 