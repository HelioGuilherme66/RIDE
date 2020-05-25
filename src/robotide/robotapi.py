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

# import robotide.lib.robot.parsing.populators
# robotide.lib.robot.parsing.populators.PROCESS_CURDIR = False

import os

from .lib.robot.model import SuiteNamePatterns
from .lib.robot.utils import get_error_message, unic

from .lib.robot.errors import DataError, VariableError, Information
from .lib.robot.model import TagPatterns
from .lib.robot.output import LOGGER as ROBOT_LOGGER
from .lib.robot.output.loggerhelper import LEVELS as LOG_LEVELS
# REMOVED 3.2 from robotide.lib.robot.parsing.datarow import DataRow
# REMOVED 3.2 from robotide.lib.robot.parsing.model import (
#     TestDataDirectory, ResourceFile, TestCaseFile, UserKeyword,
#     Variable, Step, VariableTable, KeywordTable, TestCaseTable,
#     TestCaseFileSettingTable)
from .lib.robot.parsing.model import (TestCase, File, KeywordSection,
                                              Statement, ForLoop, VariableSection,
                                              Keyword, CommentSection, SettingSection)
from .lib.robot.running.model import (ResourceFile,  # DEBUG 3.2
                                              UserKeyword, Variable)
# REMOVED 3.2 from robotide.lib.robot.parsing.populators import FromFilePopulator
from .lib.robot.parsing import SuiteStructureBuilder, SuiteStructureVisitor
# REMOVED 3.2  from robotide.lib.robot.parsing.settings import ( Library, Resource,
#      Variables, Comment, _Import, Template, Fixture, Documentation,
#      Timeout, Tags, Return)
from .lib.robot.parsing.lexer.settings import (TestCaseFileSettings,
                                                       ResourceFileSettings, TestCaseSettings,
                                                       InitFileSettings, KeywordSettings)

# REMOVED 3.2  from robotide.lib.robot.parsing.tablepopulators import (UserKeywordPopulator,
#     TestCasePopulator)
from .lib.robot.parsing.lexer.sections import (TestCaseFileSections,
                                                       ResourceFileSections,
                                                       InitFileSections)
from .lib.robot.parsing.lexer import Token
# REMOVED 3.2 from robotide.lib.robot.parsing.txtreader import TxtReader
from .lib.robot.parsing.parser.fileparser import FileParser
from .lib.robot.running import TestLibrary, EXECUTION_CONTEXTS
from .lib.robot.libraries import STDLIBS as STDLIB_NAMES
from .lib.robot.running.usererrorhandler import UserErrorHandler
from .lib.robot.running.arguments.embedded import EmbeddedArgumentParser
from .lib.robot.utils import normpath, NormalizedDict
from .lib.robot.variables import Variables as RobotVariables
from .lib.robot.variables import is_scalar_variable, is_list_variable, is_variable, is_dict_variable
# REMOVED 3.2 from robotide.lib.robot.variables import VariableSplitter
from .lib.robot.variables.filesetter import VariableFileSetter
# REMOVED 3.2 from robotide.lib.robot.variables.tablesetter import VariableTableReader
from .lib.compat import TestDataDirectory, ResourceFile, TestCaseFile
from .lib.robot.utils import FileReader
from .lib.robot.api import TestSuite, SuiteVisitor

from .lib.robot import get_version

ROBOT_VERSION = get_version()
