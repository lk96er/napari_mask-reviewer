from setuptools import setup, find_packages

setup(
    name="napari-mask-reviewer",
    version="0.1.0",
    description="A napari plugin for reviewing and correcting segmentation masks",
    author="Your Name",
    author_email="l.khl@pm.me",
    license="MIT",
    url="https://github.com/lk96er/napari_mask_reviewer",
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
        "napari.plugin": [
            "napari-mask-reviewer = napari_mask_reviewer",
        ],
    },
)