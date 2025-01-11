from setuptools import setup, find_packages

setup(
    name="napari-mask-reviewer",
    version="0.1.0",
    description="A napari plugin for reviewing and correcting segmentation masks",
    author="Lucas Kühl",
    author_email="l.kuehl@uni-muenster.de",
    license="MIT",
    url="https://github.com/username/napari-mask-reviewer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "napari>=0.4.16",
        "scikit-image>=0.19.0",
        "qtpy",
        "numpy",
    ],
    entry_points={
        "napari.manifest": [
            "napari-mask-reviewer = napari_mask_reviewer:napari.yaml",
        ],
    },
)