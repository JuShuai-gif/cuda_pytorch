from setuptools import find_packages, setup

package_name = "ros2_vla_profiling"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [
        "mock_camera = ros2_vla_profiling.nodes:camera_main",
        "mock_vla = ros2_vla_profiling.nodes:vla_main",
    ]},
)
