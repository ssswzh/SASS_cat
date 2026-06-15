from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'sensors'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lain',
    maintainer_email='ad3morgado@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',

    entry_points={
        'console_scripts': [
            "camera = sensors.camera:main",
            "microphone = sensors.microphone:main",
        ],
    },
)
