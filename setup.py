import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="project",
    version="0.0.1",
    author="JavithNaseem-J",
    author_email="javithnaseem.j@gmail.com",
    description="project",
    long_description=long_description,
    url=f"https://github.com/JavithNaseem-J/project",
    project_urls={
        "Bug Tracker":f"https://github.com/JavithNaseem-J/project/issues",
    },
    package_dir = {"":"src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6"

)