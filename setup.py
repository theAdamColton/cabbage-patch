from setuptools import setup

setup(
    name="cabbage-patch",
    version="0.0.1",
    description="Highly performant dataset/dataloader for mixed resolution image batching",
    author="Adam Colton",
    url="https://github.com/theAdamColton/cabbage-patch",
    install_requires=[
        "torch>=2.0.1",
        "tensorset==0.4.2",
    ],
    py_modules=["cabbage_patch"],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
