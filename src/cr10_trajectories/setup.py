from setuptools import setup
from glob import glob

package_name = 'cr10_trajectories'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'scipy',
        'sympy',
    ],
    zip_safe=True,
    maintainer='helene',
    maintainer_email='helene@todo.todo',
    description='CR10 trajectory node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'trajectory_node = cr10_trajectories.trajectory_node:main',
            'gui_node         = cr10_trajectories.gui_node:main',
            'initial_pose     = cr10_trajectories.initial_pose:main',
        ],
    },
)
