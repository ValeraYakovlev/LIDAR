from glob import glob

from setuptools import setup

PKG = "metro_obstacle"

setup(
    name=PKG,
    version="0.1.0",
    packages=[PKG],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PKG}"]),
        (f"share/{PKG}", ["package.xml"]),
        (f"share/{PKG}/launch", glob("launch/*.launch.py")),
        (f"share/{PKG}/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Обнаружение препятствия на пути поезда метро по облаку 3D-лидара",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"obstacle_detector = {PKG}.node:main",
            f"offline = {PKG}.offline:main",
        ],
    },
)
