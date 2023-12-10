"""
    pygments.lexers.robotframework
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Lexer for Robot Framework.

    :copyright: Copyright 2006-2023 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""
from __future__ import annotations

#  Copyright 2012 Nokia Siemens Networks Oyj
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

#  Copyright 2012-2015 Nokia Networks
#  Copyright 2023-     Robot Framework Foundation
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
#
# This is a modified copy from Pygments 2.17.0, by Hélio Guilherme 2023-12-09
# The changes in this file were done with the purpose to add colorization
# support for test suites using languages other than English and supported
# by Robot Framework 6.0 or higher (at the time, version is 6.1.1)


import re

from pygments.lexer import Lexer
from pygments.token import Token
from robotide.lib.compat.parsing.language import Language, get_settings_for, get_headers_for, get_english_label

__all__ = ['RobotFrameworkLexer']


HEADING = Token.Generic.Heading
SETTING = Token.Keyword.Namespace
IMPORT = Token.Name.Namespace
TC_KW_NAME = Token.Generic.Subheading
KEYWORD = Token.Name.Function
ARGUMENT = Token.String
VARIABLE = Token.Name.Variable
COMMENT = Token.Comment
SEPARATOR = Token.Punctuation
SYNTAX = Token.Punctuation
GHERKIN = Token.Generic.Emph
ERROR = Token.Error


def normalize(string, remove='', nolower=False):
    if not nolower:
        string = string.lower()
    if nolower:
        print(f"DEBUG: robotframework.py normalize_dict string={string} nolower={nolower}")
    for char in remove + ' ':
        if char in string:
            string = string.replace(char, '')
    return string


def normalize_dict(table: dict, nolower=False) -> dict:
    ndict = {}
    for key, value in table.items():
        if nolower:
            print(f"DEBUG: robotframework.py normalize_dict key={key} nolower={nolower}")
        if key:
            k = normalize(key, nolower=nolower)
            v = normalize(value)
            ndict[k] = v
    return ndict


def get_key_by_value(table: dict, value: str) -> str:
    for k, v in table.items():
        if v == value:
            return k
    return value  # Returns original if not in dict, deprecated/old labels


class RobotFrameworkLexer(Lexer):
    """
    For Robot Framework test data.

    Supports both space and pipe separated plain text formats.

    ... versionadded:: 1.6
    """
    name = 'RobotFramework'
    url = 'http://robotframework.org'
    aliases = ['robotframework']
    filenames = ['*.robot', '*.resource']
    mimetypes = ['text/x-robotframework']

    def __init__(self, **options):
        options['tabsize'] = 2
        options['encoding'] = 'UTF-8'
        Lexer.__init__(self, **options)
        self.language = options['language']
        if not self.language:
            self.new_lang = Language.from_name('en')
        else:
            self.new_lang = Language.from_name(self.language[0])  # DEBUG: We consider a single language
        print(f"DEBUG: robotframework.py after RobotFrameworkLexer _init_ mimetypes={self.mimetypes}\n"
              f"options['language']={options['language']}\n"
              f"self.new_lang={self.new_lang.code}")

    def get_tokens_unprocessed(self, text):
        row_tokenizer = RowTokenizer(self.new_lang)
        var_tokenizer = VariableTokenizer()
        index = 0
        for row in text.splitlines():
            for value, token in row_tokenizer.tokenize(row):
                for value, token in var_tokenizer.tokenize(value, token):
                    if value:
                        yield index, token, str(value)
                        index += len(value)


class VariableTokenizer:

    def tokenize(self, string, token):
        var = VariableSplitter(string, identifiers='$@%&')
        if var.start < 0 or token in (COMMENT, ERROR):
            yield string, token
            return
        for value, token in self._tokenize(var, string, token):
            if value:
                yield value, token

    def _tokenize(self, var, string, orig_token):
        before = string[:var.start]
        yield before, orig_token
        yield var.identifier + '{', SYNTAX
        yield from self.tokenize(var.base, VARIABLE)
        yield '}', SYNTAX
        if var.index is not None:
            yield '[', SYNTAX
            yield from self.tokenize(var.index, VARIABLE)
            yield ']', SYNTAX
        yield from self.tokenize(string[var.end:], orig_token)


class RowTokenizer:

    def __init__(self, new_lang=None):
        self.new_lang = new_lang
        if not self.new_lang:
            self.new_lang = Language.from_name('en')
        self._table = UnknownTable()
        self._splitter = RowSplitter()
        testcases = TestCaseTable(new_lang=new_lang)
        settings = SettingTable(testcases.set_default_template, new_lang=new_lang)
        variables = VariableTable()
        keywords = KeywordTable()
        self.nolower = self.new_lang.code in ('bg', 'hi', 'ru', 'th', 'uk', 'zhcn', 'zhtw')
        print(f"DEBUG: robotframework.py RowTokenizer self.new_lang.code={self.new_lang.code} self.nolower={self.nolower}")
        normalized_headers = normalize_dict(self.new_lang.headers, nolower=self.nolower)
        self._tables = {get_key_by_value(normalized_headers, 'settings'): settings,
                        get_key_by_value(normalized_headers, 'metadata'): settings,
                        get_key_by_value(normalized_headers, 'variables'): variables,
                        get_key_by_value(normalized_headers, 'testcases'): testcases,
                        get_key_by_value(normalized_headers, 'tasks'): testcases,
                        get_key_by_value(normalized_headers, 'keywords'): keywords}
        print(f"DEBUG: robotframework.py RowTokenizer self._tables={self._tables}")

    def tokenize(self, row):
        commented = False
        heading = False
        for index, value in enumerate(self._splitter.split(row)):
            # First value, and every second after that, is a separator.
            index, separator = divmod(index-1, 2)
            if value.startswith('#'):
                commented = True
            elif index == 0 and value.startswith('*'):
                self._table = self._start_table(value)
                heading = True
            yield from self._tokenize(value, index, commented,
                                      separator, heading)
        self._table.end_row()

    def _start_table(self, header):
        name = normalize(header, remove='*', nolower=self.nolower)
        return self._tables.get(name, UnknownTable())

    def _tokenize(self, value, index, commented, separator, heading):
        if commented:
            yield value, COMMENT
        elif separator:
            yield value, SEPARATOR
        elif heading:
            yield value, HEADING
        else:
            yield from self._table.tokenize(value, index)


class RowSplitter:
    _space_splitter = re.compile('( {2,})')
    _pipe_splitter = re.compile(r'((?:^| +)\|(?: +|$))')

    def split(self, row):
        splitter = (row.startswith('| ') and self._split_from_pipes
                    or self._split_from_spaces)
        yield from splitter(row)
        yield '\n'

    def _split_from_spaces(self, row):
        yield ''  # Start with (pseudo)separator similarly as with pipes
        yield from self._space_splitter.split(row)

    def _split_from_pipes(self, row):
        _, separator, rest = self._pipe_splitter.split(row, 1)
        yield separator
        while self._pipe_splitter.search(rest):
            cell, separator, rest = self._pipe_splitter.split(rest, 1)
            yield cell
            yield separator
        yield rest


class Tokenizer:
    _tokens = None

    def __init__(self):
        self._index = 0

    def tokenize(self, value):
        values_and_tokens = self._tokenize(value, self._index)
        self._index += 1
        if isinstance(values_and_tokens, type(Token)):
            values_and_tokens = [(value, values_and_tokens)]
        return values_and_tokens

    def _tokenize(self, value, index):
        index = min(index, len(self._tokens) - 1)
        return self._tokens[index]

    def _is_assign(self, value):
        if value.endswith('='):
            value = value[:-1].strip()
        var = VariableSplitter(value, identifiers='$@&')
        return var.start == 0 and var.end == len(value)


class Comment(Tokenizer):
    _tokens = (COMMENT,)


class Setting(Tokenizer):
    _tokens = (SETTING, ARGUMENT)
    _keyword_settings = ('suitesetup', 'suiteprecondition', 'suiteteardown',
                         'suitepostcondition', 'testsetup', 'tasksetup', 'testprecondition',
                         'testteardown', 'taskteardown', 'testpostcondition', 'testtemplate', 'tasktemplate')
    _import_settings = ('library', 'resource', 'variables')
    _other_settings = ('documentation', 'metadata', 'forcetags', 'defaulttags',
                       'testtimeout', 'tasktimeout')
    _custom_tokenizer = None

    def __init__(self, template_setter=None, new_lang=None):
        Tokenizer.__init__(self)
        self._template_setter = template_setter
        self.new_lang = new_lang
        if not self.new_lang:
            self.new_lang = Language.from_name('en')
        self.nolower = self.new_lang.code in ('bg', 'hi', 'ru', 'th', 'uk', 'zhcn', 'zhtw')
        self.normalized_settings = normalize_dict(self.new_lang.settings, nolower=self.nolower)
        self._keyword_settings = (get_key_by_value(self.normalized_settings,'suitesetup'),
                                  get_key_by_value(self.normalized_settings, 'suiteteardown'),
                                  get_key_by_value(self.normalized_settings,'testsetup'),
                                  get_key_by_value(self.normalized_settings,'tasksetup'),
                                  get_key_by_value(self.normalized_settings,'testteardown'),
                                  get_key_by_value(self.normalized_settings,'taskteardown'),
                                  get_key_by_value(self.normalized_settings, 'testtemplate'),
                                  get_key_by_value(self.normalized_settings,'tasktemplate'))
        self._import_settings = (get_key_by_value(self.normalized_settings,'library'),
                                 get_key_by_value(self.normalized_settings,'resource'),
                                 get_key_by_value(self.normalized_settings,'variables'))
        self._other_settings = (get_key_by_value(self.normalized_settings,'documentation'),
                                get_key_by_value(self.normalized_settings, 'metadata'),
                                get_key_by_value(self.normalized_settings, 'testtags'),
                                get_key_by_value(self.normalized_settings,'tasktags'),
                                get_key_by_value(self.normalized_settings,'forcetags'),  # Non-existing
                                get_key_by_value(self.normalized_settings,'defaulttags'),  # Non-existing
                                get_key_by_value(self.normalized_settings, 'testtimeout'),
                                get_key_by_value(self.normalized_settings, 'tasktimeout'))
        print(f"DEBUG: robotframework.py Setting self.normalized_settings={self.normalized_settings}\n"
              f"self._keyword_settings={self._keyword_settings}\n"
              f"self._import_settings={self._import_settings}\n"
              f"self._other_settings={self._other_settings}\n")

    def _tokenize(self, value, index):
        if index == 1 and self._template_setter:
            self._template_setter(value)
        if index == 0:
            normalized = normalize(value, nolower=self.nolower)
            if normalized in self._keyword_settings:
                self._custom_tokenizer = KeywordCall(support_assign=False)
            elif normalized in self._import_settings:
                self._custom_tokenizer = ImportSetting()
            elif normalized not in self._other_settings:
                return ERROR
        elif self._custom_tokenizer:
            return self._custom_tokenizer.tokenize(value)
        return Tokenizer._tokenize(self, value, index)


class ImportSetting(Tokenizer):
    _tokens = (IMPORT, ARGUMENT)


class TestCaseSetting(Setting):
    _keyword_settings = ('setup', 'precondition', 'teardown', 'postcondition',
                         'template')
    _import_settings = ()
    _other_settings = ('documentation', 'tags', 'timeout')

    def __init__(self, template_setter=None, new_lang=None):
        self.new_lang = new_lang
        if not self.new_lang:
            self.new_lang = Language.from_name('en')
        Setting.__init__(self, template_setter=template_setter, new_lang=new_lang)
        self.nolower = self.new_lang.code in ('bg', 'hi', 'ru', 'th', 'uk', 'zhcn', 'zhtw')
        self.normalized_settings = normalize_dict(self.new_lang.settings, nolower=self.nolower)
        self._keyword_settings = (get_key_by_value(self.normalized_settings, 'setup'),
                                  get_key_by_value(self.normalized_settings, 'teardown'),
                                  get_key_by_value(self.normalized_settings, 'testsetup'),
                                  get_key_by_value(self.normalized_settings, 'template'))
        self._import_settings = ()
        self._other_settings = (get_key_by_value(self.normalized_settings, 'documentation'),
                                get_key_by_value(self.normalized_settings, 'tags'),
                                get_key_by_value(self.normalized_settings, 'timeout'))
        print(f"DEBUG: robotframework.py TestCaseSetting \n"
              f"self._keyword_settings={self._keyword_settings}\n"
              f"self._import_settings={self._import_settings}\n"
              f"self._other_settings={self._other_settings}\n")

    def _tokenize(self, value, index):
        if index == 0:
            type = Setting(new_lang=self.new_lang)._tokenize(value[1:-1], index)
            return [('[', SYNTAX), (value[1:-1], type), (']', SYNTAX)]
        return Setting(new_lang=self.new_lang)._tokenize(value, index)


class KeywordSetting(TestCaseSetting):
    _keyword_settings = ('teardown',)
    _other_settings = ('documentation', 'arguments', 'return', 'timeout', 'tags')

    def __init__(self, template_setter=None, new_lang=None):
        self.new_lang = new_lang
        if not self.new_lang:
            self.new_lang = Language.from_name('en')
        TestCaseSetting.__init__(self, template_setter=template_setter, new_lang=new_lang)
        self.nolower = self.new_lang.code in ('bg', 'hi', 'ru', 'th', 'uk', 'zhcn', 'zhtw')
        self.normalized_settings = normalize_dict(self.new_lang.settings, nolower=self.nolower)
        self._keyword_settings = (get_key_by_value(self.normalized_settings, 'teardown'),)
        self._other_settings = (get_key_by_value(self.normalized_settings, 'documentation'),
                                get_key_by_value(self.normalized_settings, 'arguments'),
                                get_key_by_value(self.normalized_settings, 'return'),  # Non-existing
                                get_key_by_value(self.normalized_settings, 'tags'),
                                get_key_by_value(self.normalized_settings, 'timeout'))


class Variable(Tokenizer):
    _tokens = (SYNTAX, ARGUMENT)

    def _tokenize(self, value, index):
        if index == 0 and not self._is_assign(value):
            return ERROR
        return Tokenizer._tokenize(self, value, index)


class KeywordCall(Tokenizer):
    _tokens = (KEYWORD, ARGUMENT)

    def __init__(self, support_assign=True):
        Tokenizer.__init__(self)
        self._keyword_found = not support_assign
        self._assigns = 0

    def _tokenize(self, value, index):
        if not self._keyword_found and self._is_assign(value):
            self._assigns += 1
            return SYNTAX  # VariableTokenizer tokenizes this later.
        if self._keyword_found:
            return Tokenizer._tokenize(self, value, index - self._assigns)
        self._keyword_found = True
        return GherkinTokenizer().tokenize(value, KEYWORD)


class GherkinTokenizer:
    _gherkin_prefix = re.compile('^(Given|When|Then|And|But) ', re.IGNORECASE)

    def tokenize(self, value, token):
        match = self._gherkin_prefix.match(value)
        if not match:
            return [(value, token)]
        end = match.end()
        return [(value[:end], GHERKIN), (value[end:], token)]


class TemplatedKeywordCall(Tokenizer):
    _tokens = (ARGUMENT,)


class ForLoop(Tokenizer):

    def __init__(self):
        Tokenizer.__init__(self)
        self._in_arguments = False

    def _tokenize(self, value, index):
        token = self._in_arguments and ARGUMENT or SYNTAX
        if value.upper() in ('IN', 'IN ENUMERATE', 'IN RANGE', 'IN ZIP'):
            self._in_arguments = True
        return token


class _Table:
    _tokenizer_class = None

    def __init__(self, prev_tokenizer=None):
        self._tokenizer = self._tokenizer_class()
        self._prev_tokenizer = prev_tokenizer
        self._prev_values_on_row = []

    def tokenize(self, value, index):
        if self._continues(value, index):
            self._tokenizer = self._prev_tokenizer
            yield value, SYNTAX
        else:
            yield from self._tokenize(value, index)
        self._prev_values_on_row.append(value)

    def _continues(self, value, index):
        return value == '...' and all(self._is_empty(t)
                                      for t in self._prev_values_on_row)

    def _is_empty(self, value):
        return value in ('', '\\')

    def _tokenize(self, value, index):
        return self._tokenizer.tokenize(value)

    def end_row(self):
        self.__init__(prev_tokenizer=self._tokenizer)


class UnknownTable(_Table):
    _tokenizer_class = Comment

    def _continues(self, value, index):
        return False


class VariableTable(_Table):
    _tokenizer_class = Variable


class SettingTable(_Table):
    _tokenizer_class = Setting

    def __init__(self, template_setter, prev_tokenizer=None, new_lang=None):
        _Table.__init__(self, prev_tokenizer)
        self._template_setter = template_setter
        self.new_lang = new_lang
        if not self.new_lang:
            self.new_lang = Language.from_name('en')
        self.nolower = self.new_lang.code in ('bg', 'hi', 'ru', 'th', 'uk', 'zhcn', 'zhtw')
        self.normalized_settings = normalize_dict(self.new_lang.settings, nolower=self.nolower)
        print(f"DEBUG: robotframework.py SettingTable self.normalized_settings={self.normalized_settings}")

    def _tokenize(self, value, index):
        if index == 0 and normalize(value, nolower=self.nolower) in (
                get_key_by_value(self.normalized_settings, 'testtemplate'),
                get_key_by_value(self.normalized_settings, 'tasktemplate')):
            self._tokenizer = Setting(template_setter=self._template_setter, new_lang=self.new_lang)
        return _Table._tokenize(self, value, index)

    def end_row(self):
        self.__init__(self._template_setter, prev_tokenizer=self._tokenizer, new_lang=self.new_lang)


class TestCaseTable(_Table):
    _setting_class = TestCaseSetting
    _test_template = None
    _default_template = None

    def __init__(self, prev_tokenizer=None, new_lang=None):
        _Table.__init__(self, prev_tokenizer)
        self.new_lang = new_lang
        if not self.new_lang:
            self.new_lang = Language.from_name('en')
        self.nolower = self.new_lang.code in ('bg', 'hi', 'ru', 'th', 'uk', 'zhcn', 'zhtw')
        self.normalized_settings = normalize_dict(self.new_lang.settings, nolower=self.nolower)
        print(f"DEBUG: robotframework.py TestCaseTable self.normalized_settings={self.normalized_settings}")

    @property
    def _tokenizer_class(self):
        if self._test_template or (self._default_template and
                                   self._test_template is not False):
            return TemplatedKeywordCall
        return KeywordCall

    def _continues(self, value, index):
        return index > 0 and _Table._continues(self, value, index)

    def _tokenize(self, value, index):
        if index == 0:
            if value:
                self._test_template = None
            return GherkinTokenizer().tokenize(value, TC_KW_NAME)
        if index == 1 and self._is_setting(value):
            if self._is_template(value):
                self._test_template = False
                self._tokenizer = self._setting_class(template_setter=self.set_test_template, new_lang=self.new_lang)
            else:
                self._tokenizer = self._setting_class(new_lang=self.new_lang)
        if index == 1 and self._is_for_loop(value):
            self._tokenizer = ForLoop()
        if index == 1 and self._is_empty(value):
            return [(value, SYNTAX)]
        return _Table._tokenize(self, value, index)

    @staticmethod
    def _is_setting(value):
        return value.startswith('[') and value.endswith(']')

    def _is_template(self, value):
        return normalize(value, nolower=self.nolower) == f"[{get_key_by_value(self.normalized_settings, 'template')}]"

    def _is_for_loop(self, value):
        return normalize(value, remove=':', nolower=self.nolower) == 'for'

    def set_test_template(self, template):
        self._test_template = self._is_template_set(template)

    def set_default_template(self, template):
        self._default_template = self._is_template_set(template)

    def _is_template_set(self, template):
        return normalize(template, nolower=self.nolower) not in ('', '\\', 'none', '${empty}')


class KeywordTable(TestCaseTable):
    _tokenizer_class = KeywordCall
    _setting_class = KeywordSetting

    def _is_template(self, value):
        return False


# Following code copied directly from Robot Framework 2.7.5.

class VariableSplitter:

    def __init__(self, string, identifiers):
        self.identifier = None
        self.base = None
        self.index = None
        self.start = -1
        self.end = -1
        self._identifiers = identifiers
        self._may_have_internal_variables = False
        try:
            self._split(string)
        except ValueError:
            pass
        else:
            self._finalize()

    def get_replaced_base(self, variables):
        if self._may_have_internal_variables:
            return variables.replace_string(self.base)
        return self.base

    def _finalize(self):
        self.identifier = self._variable_chars[0]
        self.base = ''.join(self._variable_chars[2:-1])
        self.end = self.start + len(self._variable_chars)
        if self._has_list_or_dict_variable_index():
            self.index = ''.join(self._list_and_dict_variable_index_chars[1:-1])
            self.end += len(self._list_and_dict_variable_index_chars)

    def _has_list_or_dict_variable_index(self):
        return self._list_and_dict_variable_index_chars\
        and self._list_and_dict_variable_index_chars[-1] == ']'

    def _split(self, string):
        start_index, max_index = self._find_variable(string)
        self.start = start_index
        self._open_curly = 1
        self._state = self._variable_state
        self._variable_chars = [string[start_index], '{']
        self._list_and_dict_variable_index_chars = []
        self._string = string
        start_index += 2
        for index, char in enumerate(string[start_index:]):
            index += start_index  # Giving start to enumerate only in Py 2.6+
            try:
                self._state(char, index)
            except StopIteration:
                return
            if index  == max_index and not self._scanning_list_variable_index():
                return

    def _scanning_list_variable_index(self):
        return self._state in [self._waiting_list_variable_index_state,
                               self._list_variable_index_state]

    def _find_variable(self, string):
        max_end_index = string.rfind('}')
        if max_end_index == -1:
            raise ValueError('No variable end found')
        if self._is_escaped(string, max_end_index):
            return self._find_variable(string[:max_end_index])
        start_index = self._find_start_index(string, 1, max_end_index)
        if start_index == -1:
            raise ValueError('No variable start found')
        return start_index, max_end_index

    def _find_start_index(self, string, start, end):
        index = string.find('{', start, end) - 1
        if index < 0:
            return -1
        if self._start_index_is_ok(string, index):
            return index
        return self._find_start_index(string, index+2, end)

    def _start_index_is_ok(self, string, index):
        return string[index] in self._identifiers\
        and not self._is_escaped(string, index)

    def _is_escaped(self, string, index):
        escaped = False
        while index > 0 and string[index-1] == '\\':
            index -= 1
            escaped = not escaped
        return escaped

    def _variable_state(self, char, index):
        self._variable_chars.append(char)
        if char == '}' and not self._is_escaped(self._string, index):
            self._open_curly -= 1
            if self._open_curly == 0:
                if not self._is_list_or_dict_variable():
                    raise StopIteration
                self._state = self._waiting_list_variable_index_state
        elif char in self._identifiers:
            self._state = self._internal_variable_start_state

    def _is_list_or_dict_variable(self):
        return self._variable_chars[0] in ('@','&')

    def _internal_variable_start_state(self, char, index):
        self._state = self._variable_state
        if char == '{':
            self._variable_chars.append(char)
            self._open_curly += 1
            self._may_have_internal_variables = True
        else:
            self._variable_state(char, index)

    def _waiting_list_variable_index_state(self, char, index):
        if char != '[':
            raise StopIteration
        self._list_and_dict_variable_index_chars.append(char)
        self._state = self._list_variable_index_state

    def _list_variable_index_state(self, char, index):
        self._list_and_dict_variable_index_chars.append(char)
        if char == ']':
            raise StopIteration
