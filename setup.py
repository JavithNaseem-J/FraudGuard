import setuptools

setuptools.setup(
    name="FraudGuard",
    version="0.0.1",
    author="JavithNaseem-J",
    author_email="javithnaseem.j@gmail.com",
    description="Fraud Detection Project",
    long_description="Fraud detection project to classify and monitor suspicious transactions.",
    url="https://github.com/JavithNaseem-J/project",
    project_urls={
        "Bug Tracker": "https://github.com/JavithNaseem-J/project/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
