#!/usr/bin/env python
#  Copyright 2008-2015 Nokia Networks
#  Copyright 2016-     Robot Framework Foundation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import sys
from os.path import abspath, join, dirname
from setuptools import setup, find_packages
from setuptools.command.install import install
from typing import Mapping

ROOT_DIR = dirname(abspath(__file__))
SOURCE_DIR = 'src'
REQUIREMENTS = ['PyPubSub',
                'Pygments',
                'psutil',
                'Pywin32; sys_platform=="win32"',
                'wxPython',
                'packaging',
                'requests>=2.32.4']

PACKAGE_DATA = {
    'robotide.preferences': ['settings.cfg'],
    'robotide.widgets': ['*.png', '*.gif', '*.ico'],
    'robotide.messages': ['*.html'],
    'robotide.application': ['*.html', '*.css'],
    'robotide.publish.htmlmessages': ['no_robot.html'],
    'robotide.postinstall': ['RIDE.app/Contents/PkgInfo', 'RIDE.app/Contents/Info.plist',
                             'RIDE.app/Contents/MacOS/RIDE', 'RIDE.app/Contents/Resources/*.icns']
}

my_list = []
for curr_dir, _, files in os.walk('src/robotide/localization'):
    for item in files:
        if '.' in item:
             my_list.append(os.path.join(curr_dir, item).replace('\\', '/').replace('src/robotide/localization/', ''))

PACKAGE_DATA['robotide.localization'] = my_list[:]

my_list = []
for curr_dir, _, files in os.walk('src/robotide/preferences/configobj/src/configobj'):
    for item in files:
        if '.' in item:
             my_list.append(os.path.join(curr_dir, item).replace('\\', '/').replace('src/robotide/preferences/configobj/src/configobj/', ''))

PACKAGE_DATA['robotide.preferences.configobj.src.configobj'] = my_list[:]

LONG_DESCRIPTION = """
Robot Framework is a generic test automation framework for acceptance
level testing. RIDE is a lightweight and intuitive editor for Robot
Framework test data.

Project: https://github.com/robotframework/RIDE#readme
""".strip()

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description_full = (this_directory / "README.md").read_text()

CLASSIFIERS = """
Development Status :: 5 - Production/Stable
License :: OSI Approved :: Apache Software License
Operating System :: OS Independent
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3.11
Programming Language :: Python :: 3.12
Programming Language :: Python :: 3.13
Topic :: Software Development :: Testing
""".strip().splitlines()


# This solution is found at http://stackoverflow.com/a/26490820/5889853
class CustomInstallCommand(install):
    """Customized setuptools install command - install RIDE desktop shortcut."""
    def run(self):
        install.run(self)
        sys.stdout.write("Creating Desktop Shortcut to RIDE...\n")
        # post_installer_file = join(ROOT_DIR, SOURCE_DIR, 'robotide', 'postinstall', '__main__.py')
        post_installer_file = join(ROOT_DIR, SOURCE_DIR, 'bin', 'ride_postinstall.py')
        command = sys.executable + " " + post_installer_file + " -install"
        os.system(command)


main_ns = dict()
version_file = join(ROOT_DIR, SOURCE_DIR, 'robotide', 'version.py')
with open(version_file) as _:
    exec(_.read(), main_ns)

setup(
    name='robotframework-ride',
    version=main_ns['VERSION'],
    description='RIDE :: Robot Framework Test Data Editor',
    long_description=long_description_full,
    long_description_content_type='text/markdown',
    license='Apache License 2.0',
    keywords='robotframework testing testautomation',
    platforms='any',
    classifiers=CLASSIFIERS,
    author='Robot Framework Developers',
    author_email='robotframework@gmail.com',
    url='https://github.com/robotframework/RIDE/',
    download_url='https://pypi.python.org/pypi/robotframework-ride',
    install_requires=REQUIREMENTS,
    include_package_data=True,
    package_dir={'': SOURCE_DIR},
    packages=find_packages(SOURCE_DIR),
    package_data=PACKAGE_DATA,
    python_requires='>=3.8, <3.15',
    # Robot Framework package data is not included, but RIDE does not need it.
    # Always install everything, since we may be switching between versions
    options={'install': {'force': True}},
    scripts=['src/bin/ride.py', 'src/bin/ride_postinstall.py'],
    cmdclass={'install':CustomInstallCommand},
)
