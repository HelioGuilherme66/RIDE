import wx
from wx import stc
from StringIO import StringIO
from robot.parsing.populators import FromFilePopulator
from robot.parsing.txtreader import TxtReader
from robotide.publish.messages import RideItemNameChanged

from robotide.widgets import VerticalSizer
from robotide.pluginapi import (Plugin, ActionInfo, RideSaving,
        TreeAwarePluginMixin, RideTreeSelection, RideNotebookTabChanging)


class SourceEditorPlugin(Plugin, TreeAwarePluginMixin):
    title = 'Edit Source'

    def __init__(self, application):
        Plugin.__init__(self, application)
        self._editor_component = None

    @property
    def _editor(self):
        if not self._editor_component:
            self._editor_component = SourceEditor(self.notebook, self.title)
        return self._editor_component

    def enable(self):
        self.register_action(ActionInfo('Edit', 'Edit Source', self.OnOpen))
        self.add_self_as_tree_aware_plugin()
        self.subscribe(self.OnSaving, RideSaving)
        self.subscribe(self.OnTreeSelection, RideTreeSelection)
        self.subscribe(self.OnTreeSelection, RideItemNameChanged)
        self.subscribe(self.OnTabChange, RideNotebookTabChanging)
        self._open()

    def disable(self):
        self.unsubscribe_all()
        self.remove_self_from_tree_aware_plugins()
        self.unregister_actions()
        self._editor_component = None

    def OnOpen(self, event):
        self._open()

    def _open(self):
        self._editor.open(self.tree.get_selected_datafile_controller())
        self.show_tab(self._editor)

    def OnSaving(self, message):
        if self.is_focused():
            self._editor.save()
            self.tree.refresh_current_datafile()

    def OnTreeSelection(self, message):
        if self.is_focused():
            self._editor.open(self.tree.get_selected_datafile_controller())

    def OnTabChange(self, message):
        if message.newtab == self.title:
            self._open()
        if message.oldtab == self.title:
            if self._editor.dirty:
                self._ask_and_apply()

    def _ask_and_apply(self):
        # TODO: use widgets.Dialog
        ret = wx.MessageDialog(self._editor, 'Apply changes?', 'Source changed',
                               style=wx.YES_NO | wx.ICON_QUESTION).ShowModal()
        if ret == wx.ID_YES:
            self._editor.save()
            self.tree.refresh_current_datafile()
        self._editor.reset()

    def is_focused(self):
        return self.notebook.current_page_title == self.title


class SourceEditor(wx.Panel):

    def __init__(self, parent, title):
        wx.Panel.__init__(self, parent)
        self._parent = parent
        self.SetSizer(VerticalSizer())
        self._editor = RobotDataEditor(self)
        self.Sizer.add_expanding(self._editor)
        self._parent.AddPage(self, title)
        self._editor.Bind(wx.EVT_KEY_DOWN, self.OnEditorKey)
        self._data = None
        self._dirty = False

    @property
    def dirty(self):
        return self._dirty

    def open(self, data_controller):
        if data_controller:
            output = StringIO()
            data_controller.datafile.save(output=output, format='txt')
            self._editor.set_text(output.getvalue())
            self._data = data_controller

    def reset(self):
        self._dirty = False

    def save(self):
        if self.dirty:
            src = StringIO(self._editor.GetText().encode('UTF-8'))
            target = self._create_target()
            FromStringIOPopulator(target).populate(src)
            self._data.set_datafile(target)
            self._data.mark_dirty()
        self.reset()

    def _create_target(self):
        datafile_class = type(self._data.data)
        target = datafile_class(source=self._data.source)
        return target

    def OnEditorKey(self, event):
        self._dirty = True
        event.Skip()


class RobotDataEditor(stc.StyledTextCtrl):

    def __init__(self, parent):
        stc.StyledTextCtrl.__init__(self, parent)

    def set_text(self, text):
        self.SetText(text)


class FromStringIOPopulator(FromFilePopulator):

    def populate(self, content):
        TxtReader().read(content, self)