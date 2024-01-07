from setuptools import setup  # type:ignore

from src.fluid_ai import __version__

with open("requirements.txt", "r") as f:
    install_requires = [line for line in f if not line.lstrip().startswith("#")]

setup(
    name="fluid_ai",
    version=__version__,
    url="https://github.com/kaiitunnz/fluid-ai",
    author="Noppanat Wadlom",
    author_email="noppanat.wad@gmail.com",
    package_data={"fluid_ai": ["py.typed"]},
    package_dir={"": "src"},
    install_requires=install_requires,
)
