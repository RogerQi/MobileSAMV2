from setuptools import find_packages, setup

setup(
    name="mobilesamv2",
    version="1.0",
    include_package_data=True,
    package_data={'ultralytics': ['yolo/cfg/default.yaml']},
    install_requires=["timm", "torchpack", "onnx", "onnxruntime", "onnxsim"],
    packages=find_packages(exclude="notebooks"),
    extras_require={
        "all": ["matplotlib", "pycocotools", "opencv-python", "", "onnxruntime"],
        "dev": ["flake8", "isort", "black", "mypy"],
    },
)
