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
import copy
import warnings

from robotide.lib.robot.errors import DataError
from robotide.lib.robot.variables import is_var
from robotide.lib.robot.output import LOGGER
from ..writer import DataFileWriter
from robotide.lib.robot.utils import abspath, is_string, normalize, py2to3, NormalizedDict
from robotide.lib.robot.parsing.lexer.sections import (TestCaseFileSections, ResourceFileSections, InitFileSections)
from robotide.lib.robot.parsing.lexer.settings import (TestCaseFileSettings, ResourceFileSettings, TestCaseSettings,
                                                       InitFileSettings, KeywordSettings)
# from robotide.lib.robot.parsing.model import (TestCase, File, KeywordSection,
#                                              Statement, TestCaseSection, TestCaseTable, ForLoop, VariableSection,
#                                              Keyword, CommentSection,
#                                              SettingSection)

from robotide.lib.robot.api import TestSuite
from robotide.lib.robot.model.testcase import TestCase
from .comments import Comment
from .populators import FromFilePopulator, FromDirectoryPopulator, NoTestsFound
from .settings import (Documentation, Fixture, Timeout, Tags, Metadata,
                       Library, Resource, Variables, Arguments, Return,
                       Template, MetadataList, ImportList)


def TestData(parent=None, source=None, include_suites=None,
             warn_on_skipped='DEPRECATED', extensions=None):
    """Parses a file or directory to a corresponding model object.

    :param parent: Optional parent to be used in creation of the model object.
    :param source: Path where test data is read from.
    :param warn_on_skipped: Deprecated.
    :param extensions: List/set of extensions to parse. If None, all files
        supported by Robot Framework are parsed when searching test cases.
    :returns: :class:`~.model.TestDataDirectory`  if `source` is a directory,
        :class:`~.model.TestCaseFile` otherwise.
    """
    # TODO: Remove in RF 3.2.
    if warn_on_skipped != 'DEPRECATED':
        warnings.warn("Option 'warn_on_skipped' is deprecated and has no "
                      "effect.", DeprecationWarning)
    if os.path.isdir(source):
        return TestDataDirectory(parent, source).populate(include_suites,
                                                          extensions)
    return TestCaseFile(parent, source).populate()


class _TestData(object):
    _setting_table_names = 'Setting', 'Settings'
    _variable_table_names = 'Variable', 'Variables'
    _testcase_table_names = 'Test Case', 'Test Cases', 'Task', 'Tasks'
    _keyword_table_names = 'Keyword', 'Keywords'
    _comment_table_names = 'Comment', 'Comments'

    def __init__(self, parent=None, source=None):
        self.parent = parent
        self.source = abspath(source) if source else None
        self.children = []
        self._tables = dict(self._get_tables())

    def _get_tables(self):
        for names, table in [(self._setting_table_names, self.setting_table),
                             (self._variable_table_names, self.variable_table),
                             (self._testcase_table_names, self.testcase_table),
                             (self._keyword_table_names, self.keyword_table),
                             (self._comment_table_names, None)]:
            for name in names:
                yield name, table

    def start_table(self, header_row):
        table = self._find_table(header_row)
        if table is None or not self._table_is_allowed(table):
            return None
        table.set_header(header_row)
        return table

    def _find_table(self, header_row):
        name = header_row[0] if header_row else ''
        title = name.title()
        if title not in self._tables:
            title = self._resolve_deprecated_table(name)
            if title is None:
                self._report_unrecognized_table(name)
                return None
        return self._tables[title]

    def _resolve_deprecated_table(self, used_name):
        normalized = normalize(used_name)
        for name in (self._setting_table_names + self._variable_table_names +
                     self._testcase_table_names + self._keyword_table_names +
                     self._comment_table_names):
            if normalize(name) == normalized:
                self._report_deprecated_table(used_name, name)
                return name
        return None

    def _report_deprecated_table(self, deprecated, name):
        self.report_invalid_syntax(
            "Section name '%s' is deprecated. Use '%s' instead."
            % (deprecated, name), level='WARN'
        )

    def _report_unrecognized_table(self, name):
        self.report_invalid_syntax(
            "Unrecognized table header '%s'. Available headers for data: "
            "'Setting(s)', 'Variable(s)', 'Test Case(s)', 'Task(s)' and "
            "'Keyword(s)'. Use 'Comment(s)' to embedded additional data."
            % name
        )

    def _table_is_allowed(self, table):
        return True

    @property
    def name(self):
        return self._format_name(self._get_basename()) if self.source else None

    @property
    def rawname(self):
        return self._get_basename() if self.source else None
        # To be used on resource prefixed suggestions

    def _get_basename(self):
        return os.path.splitext(os.path.basename(self.source))[0]

    def _format_name(self, name):
        name = self._strip_possible_prefix_from_name(name)
        name = name.replace('_', ' ').strip()
        return name.title() if name.islower() else name

    def _strip_possible_prefix_from_name(self, name):
        return name.split('__', 1)[-1]

    @property
    def keywords(self):
        return self.keyword_table.keywords

    @property
    def imports(self):
        return self.setting_table.imports

    def report_invalid_syntax(self, message, level='ERROR'):
        initfile = getattr(self, 'initfile', None)
        path = os.path.join(self.source, initfile) if initfile else self.source
        LOGGER.write("Error in file '%s': %s" % (path, message), level)

    def save(self, **options):
        """Writes this datafile to disk.

        :param options: Configuration for writing. These are passed to
            :py:class:`~robot.writer.datafilewriter.WritingContext` as
            keyword arguments.

        See also :py:class:`robot.writer.datafilewriter.DataFileWriter`
        """
        return DataFileWriter(**options).write(self)


class TestCaseFile(_TestData):
    """The parsed test case file object.

    :param parent: parent object to be used in creation of the model object.
    :param source: path where test data is read from.
    """

    def __init__(self, parent=None, source=None):
        self.directory = os.path.dirname(source) if source else None
        self.setting_table = TestCaseFileSettings()
        self.variable_table = VariableSection(self)
        self.testcase_table = TestCaseTable(self)
        self.keyword_table = KeywordSection(self)
        _TestData.__init__(self, parent, source)

    def populate(self):
        FromFilePopulator(self).populate(self.source)
        self._validate()
        return self

    def _validate(self):
        if not self.testcase_table.is_started():
            raise NoTestsFound('File has no tests or tasks.')

    def has_tests(self):
        return True

    def __iter__(self):
        for table in [self.setting_table, self.variable_table,
                      self.testcase_table, self.keyword_table]:
            yield table

    def __nonzero__(self):
        return any(table for table in self)


class ResourceFile(_TestData):
    """The parsed resource file object.

    :param source: path where resource file is read from.
    """

    def __init__(self, source=None):
        self.directory = os.path.dirname(source) if source else None
        self.setting_table = ResourceFileSettings()
        self.variable_table = VariableSection(self)
        self.testcase_table = TestCaseTable(self)
        self.keyword_table = KeywordSection(self)
        _TestData.__init__(self, source=source)

    def populate(self):
        FromFilePopulator(self).populate(self.source, resource=True)
        self._report_status()
        return self

    def _report_status(self):
        if self.setting_table or self.variable_table or self.keyword_table:
            LOGGER.info("Imported resource file '%s' (%d keywords)."
                        % (self.source, len(self.keyword_table.keywords)))
        else:
            LOGGER.warn("Imported resource file '%s' is empty." % self.source)

    def _table_is_allowed(self, table):
        if table is self.testcase_table:
            raise DataError("Resource file '%s' cannot contain tests or "
                            "tasks." % self.source)
        return True

    def __iter__(self):
        for table in [self.setting_table, self.variable_table, self.keyword_table]:
            yield table


class TestDataDirectory(_TestData):
    """The parsed test data directory object. Contains hiearchical structure
    of other :py:class:`.TestDataDirectory` and :py:class:`.TestCaseFile`
    objects.

    :param parent: parent object to be used in creation of the model object.
    :param source: path where test data is read from.
    """

    def __init__(self, parent=None, source=None):
        self.directory = source
        self.initfile = None
        self.setting_table = InitFileSettings()
        self.variable_table = VariableSection(self)
        self.testcase_table = TestCaseTable(self)
        self.keyword_table = KeywordSection(self)
        _TestData.__init__(self, parent, source)

    def populate(self, include_suites=None, extensions=None, recurse=True):
        FromDirectoryPopulator().populate(self.source, self, include_suites,
                                          extensions, recurse)
        self.children = [ch for ch in self.children if ch.has_tests()]
        return self

    def _get_basename(self):
        return os.path.basename(self.source)

    def _table_is_allowed(self, table):
        if table is self.testcase_table:
            LOGGER.error("Test suite initialization file in '%s' cannot "
                         "contain tests or tasks." % self.source)
            return False
        return True

    def add_child(self, path, include_suites, extensions=None):
        self.children.append(TestData(parent=self,
                                      source=path,
                                      include_suites=include_suites,
                                      extensions=extensions))

    def has_tests(self):
        return any(ch.has_tests() for ch in self.children)

    def __iter__(self):
        for table in [self.setting_table, self.variable_table, self.keyword_table]:
            yield table


@py2to3
class _Table(object):

    def __init__(self, parent):
        self.parent = parent
        self._header = None

    def set_header(self, header):
        self._header = self._prune_old_style_headers(header)

    def _prune_old_style_headers(self, header):
        if len(header) < 3:
            return header
        if self._old_header_matcher.match(header):
            return [header[0]]
        return header

    @property
    def header(self):
        return self._header or [self.type.title() + 's']

    @property
    def name(self):
        return self.header[0]

    @property
    def source(self):
        return self.parent.source

    @property
    def directory(self):
        return self.parent.directory

    def report_invalid_syntax(self, message, level='ERROR'):
        self.parent.report_invalid_syntax(message, level)

    def __nonzero__(self):
        return bool(self._header or len(self))

    def __len__(self):
        return sum(1 for item in self)


class _WithSettings(object):
    _setters = {}
    _aliases = {}

    def get_setter(self, name):
        if name[-1:] == ':':
            name = name[:-1]
        setter = self._get_setter(name)
        if setter is not None:
            return setter
        setter = self._get_deprecated_setter(name)
        if setter is not None:
            return setter
        self.report_invalid_syntax("Non-existing setting '%s'." % name)
        return None

    def _get_setter(self, name):
        title = name.title()
        if title in self._aliases:
            title = self._aliases[name]
        if title in self._setters:
            return self._setters[title](self)
        return None

    def _get_deprecated_setter(self, name):
        normalized = normalize(name)
        for setting in list(self._setters) + list(self._aliases):
            if normalize(setting) == normalized:
                self._report_deprecated_setting(name, setting)
                return self._get_setter(setting)
        return None

    def _report_deprecated_setting(self, deprecated, correct):
        self.report_invalid_syntax(
            "Setting '%s' is deprecated. Use '%s' instead."
            % (deprecated, correct), level='WARN'
        )

    def report_invalid_syntax(self, message, level='ERROR'):
        raise NotImplementedError


class _SettingTable(_Table, _WithSettings):
    type = 'setting'

    def __init__(self, parent):
        _Table.__init__(self, parent)
        self.doc = Documentation('Documentation', self)
        self.suite_setup = Fixture('Suite Setup', self)
        self.suite_teardown = Fixture('Suite Teardown', self)
        self.test_setup = Fixture('Test Setup', self)
        self.test_teardown = Fixture('Test Teardown', self)
        self.force_tags = Tags('Force Tags', self)
        self.default_tags = Tags('Default Tags', self)
        self.test_template = Template('Test Template', self)
        self.test_timeout = Timeout('Test Timeout', self)
        self.metadata = MetadataList(self)
        self.imports = ImportList(self)

    @property
    def _old_header_matcher(self):
        return OldStyleSettingAndVariableTableHeaderMatcher()

    def add_metadata(self, name, value='', comment=None):
        self.metadata.add(Metadata(self, name, value, comment))
        return self.metadata[-1]

    def add_library(self, name, args=None, comment=None):
        self.imports.add(Library(self, name, args, comment=comment))
        return self.imports[-1]

    def add_resource(self, name, invalid_args=None, comment=None):
        self.imports.add(Resource(self, name, invalid_args, comment=comment))
        return self.imports[-1]

    def add_variables(self, name, args=None, comment=None):
        self.imports.add(Variables(self, name, args, comment=comment))
        return self.imports[-1]

    def __len__(self):
        return sum(1 for setting in self if setting.is_set())

@py2to3
class Variable(object):

    def __init__(self, parent, name, value, comment=None):
        self.parent = parent
        self.name = name.rstrip('= ')
        if name.startswith('$') and value == []:
            value = ''
        if is_string(value):
            value = [value]
        self.value = value
        self.comment = Comment(comment)

    def as_list(self):
        if self.has_data():
            return [self.name] + self.value + self.comment.as_list()
        return self.comment.as_list()

    def is_set(self):
        return True

    def is_for_loop(self):
        return False

    def has_data(self):
        return bool(self.name or ''.join(self.value))

    def __nonzero__(self):
        return self.has_data()

    def report_invalid_syntax(self, message, level='ERROR'):
        self.parent.report_invalid_syntax("Setting variable '%s' failed: %s"
                                          % (self.name, message), level)


class _WithSteps(object):

    def add_step(self, content, comment=None):
        self.steps.append(Step(content, comment))
        return self.steps[-1]

    def copy(self, name):
        new = copy.deepcopy(self)
        new.name = name
        self._add_to_parent(new)
        return new

class UserKeyword(TestCase):

    def __init__(self, parent, name):
        self.parent = parent
        self.name = name
        self.doc = Documentation('[Documentation]', self)
        self.args = Arguments('[Arguments]', self)
        self.return_ = Return('[Return]', self)
        self.timeout = Timeout('[Timeout]', self)
        self.teardown = Fixture('[Teardown]', self)
        self.tags = Tags('[Tags]', self)
        self.steps = []
        if name == '...':
            self.report_invalid_syntax(
                "Using '...' as keyword name is deprecated. It will be "
                "considered line continuation in Robot Framework 3.2.",
                level='WARN'
            )

    _setters = {'Documentation': lambda s: s.doc.populate,
                'Arguments': lambda s: s.args.populate,
                'Return': lambda s: s.return_.populate,
                'Timeout': lambda s: s.timeout.populate,
                'Teardown': lambda s: s.teardown.populate,
                'Tags': lambda s: s.tags.populate}

    def _add_to_parent(self, test):
        self.parent.keywords.append(test)

    @property
    def settings(self):
        return [self.args, self.doc, self.tags, self.timeout, self.teardown, self.return_]

    def __iter__(self):
        for element in [self.args, self.doc, self.tags, self.timeout] \
                        + self.steps + [self.teardown, self.return_]:
            yield element

class Step(object):

    def __init__(self, content, comment=None):
        self.assign = self._get_assign(content)
        # print("DEBUG RFLib init Step: content %s" % content[:])
        self.name = content.pop(0) if content else None
        # print("DEBUG RFLib init Step: self.name %s" % self.name)
        self.args = content
        self.comment = Comment(comment)

    def _get_assign(self, content):
        assign = []
        while content and is_var(content[0].rstrip('= ')):
            assign.append(content.pop(0))
        return assign

    def is_comment(self):
        return not (self.assign or self.name or self.args)

    def is_for_loop(self):
        return False

    def is_set(self):
        return True

    def as_list(self, indent=False, include_comment=True):
        # print("DEBUG RFLib Model Step: self.name %s" % self.name )
        kw = [self.name] if self.name is not None else []
        comments = self.comment.as_list() if include_comment else []
        data = self.assign + kw + self.args + comments
        if indent:
            data.insert(0, '')
        # print("DEBUG RFLib Model Step: data %s" % data)
        return data


class OldStyleSettingAndVariableTableHeaderMatcher(object):

    def match(self, header):
        return all(value.lower() == 'value' for value in header[1:])


class OldStyleTestAndKeywordTableHeaderMatcher(object):

    def match(self, header):
        if header[1].lower() != 'action':
            return False
        for arg in header[2:]:
            if not arg.lower().startswith('arg'):
                return False
        return True
