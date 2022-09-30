from setuptools import setup, find_packages

package_name = 'bev-toolbox'
version = 0.0

if __name__ == "__main__":

    print(f"Building wheel {package_name}-{version}")

    with open("README.rst") as f:
        readme = f.read()

    setup(
        # Metadata
        name=package_name,
        version=version,
        author="OpenPerceptionX",
        author_email="simachonghao@pjlab.org.cn",
        url="https://github.com/OpenPerceptionX/BEVPerception-Survey-Recipe",
        description="Toolbox for BEV Perception",
        long_description=readme,
        license="Apache 2.0",
        # Package info
        packages=find_packages('bev-toolbox'),
        package_dir={'': 'bev-toolbox'},
        zip_safe=False,
        install_requires=['opencv-python', 'numpy'],
        python_requires=">=3.5",
    )
