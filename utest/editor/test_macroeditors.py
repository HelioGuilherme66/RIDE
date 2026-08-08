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

import unittest
import os
import pytest
# DISPLAY = os.getenv('DISPLAY')
# if not DISPLAY:
#     pytest.skip("Skipped because of missing DISPLAY", allow_module_level=True) # Avoid failing unit tests in system without X11
import wx

import robotide.editor.editors
from .fakeplugin import FakePlugin
from robotide.controller.macrocontrollers import TestCaseController
from robotide.publish.messages import RideItemNameChanged
from robotide.editor.editors import FindUsagesHeader
from robotide.editor.macroeditors import TestCaseEditor, UserKeywordEditor

DELEGATED_METHODS =[('save', 'save'),
             # ('undo', 'on_undo'),  # Disabled because of double Ctrl-Z
             ('redo', 'on_redo'),
             ('cut', 'on_cut'),
             ('copy', 'on_copy'),
             ('paste', 'on_paste'),
             ('insert', 'on_insert'),
             ('insert_cells', 'on_insert_cells'),
             ('insert_rows', 'on_insert_rows'),
             ('delete_rows', 'on_delete_rows'),
             ('delete_cells', 'on_delete_cells'),
             ('delete', 'on_delete'),
             ('show_content_assist', 'show_content_assist'),
             ('on_move_rows_up', 'on_move_rows_up'),
             ('on_move_rows_down', 'on_move_rows_down'),
             ('comment_rows', 'on_comment_rows'),
             ('uncomment_rows', 'on_uncomment_rows'),
             ('sharp_comment_rows', 'on_sharp_comment_rows'),
             ('sharp_uncomment_rows', 'on_sharp_uncomment_rows'),
             ('comment_cells', 'on_comment_cells'),
             ('uncomment_cells', 'on_uncomment_cells')]

TestCaseEditor._populate = lambda self: None


class IncredibleMock(object):

    def __getattr__(self, item):
        return self

    def __call__(self, *args, **kwargs):
        return self


class MockKwEditor(object):

    _expect = None
    _called = None

    def __getattr__(self, item):
        self._active_item = item
        return self

    def __call__(self, *args, **kwargs):
        self._called = self._active_item

    def is_to_be_called(self):
        self._expect = self._active_item

    def has_been_called(self):
        return self._active_item == self._expect == self._called


class MacroEditorTest(unittest.TestCase):

    def setUp(self):
        myapp = wx.App(None)
        self.controller = TestCaseController(IncredibleMock(), IncredibleMock())
        plugin = FakePlugin({}, self.controller)
        self.tc_editor = TestCaseEditor(
            plugin, wx.Frame(None), self.controller, None)

    def test_delegation_to_kw_editor(self):
        for method, kw_method in DELEGATED_METHODS:
            kw_mock = MockKwEditor()
            self.tc_editor.kweditor = kw_mock
            getattr(kw_mock, kw_method).is_to_be_called()
            getattr(self.tc_editor, method)()
            assert getattr(kw_mock, kw_method).has_been_called(), (f"Should have called \""
                                                                   f"{kw_method}\" when calling \"{method}\"")

        self.tc_editor._populate()
        RideItemNameChanged(item=self.controller, old_name='Title of User Keyword', new_name="New Name").publish()
        # self.tc_editor._name_changed(message)
        # print(f"DEBUG: Name={self.tc_editor.header}")
        # assert self.tc_editor.title == "New Name"


class MacroEditorUserKWTest(unittest.TestCase):

    def setUp(self):
        myapp = wx.App(None)
        self.controller = TestCaseController(IncredibleMock(), IncredibleMock())
        plugin = FakePlugin({}, self.controller)
        self.tc_editor = UserKeywordEditor(
            plugin, wx.Frame(None), self.controller, None)

    def test_delegation_to_kw_editor(self):
        for method, kw_method in DELEGATED_METHODS:
            kw_mock = MockKwEditor()
            self.tc_editor.kweditor = kw_mock
            getattr(kw_mock, kw_method).is_to_be_called()
            getattr(self.tc_editor, method)()
            assert getattr(kw_mock, kw_method).has_been_called(), (f"Should have called \""
                                                                   f"{kw_method}\" when calling \"{method}\"")

    def test_header(self):
        myapp = wx.App(None)
        kw_mock = MockKwEditor()
        self.tc_editor.kweditor = kw_mock
        header = self.tc_editor._create_header(text="Title of User Keyword", readonly=False)
        assert isinstance(header, FindUsagesHeader)
        label = header._header.GetLabel()
        assert label == "Title of User Keyword"

    def test_header_read_only(self):
        myapp = wx.App(None)
        kw_mock = MockKwEditor()
        self.tc_editor.kweditor = kw_mock
        header = self.tc_editor._create_header(text="Title of User Keyword", readonly=True)
        assert isinstance(header, FindUsagesHeader)
        label = header._header.GetLabel()
        assert label == "Title of User Keyword (READ ONLY)"

    def test_misc(self):
        myapp = wx.App(None)
        kw_mock = MockKwEditor()
        self.tc_editor.kweditor = kw_mock
        self.tc_editor.undo()   # Should do nothing
        self.tc_editor.close()


if __name__ == '__main__':
    app = wx.App()
    unittest.main()

