from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, \
    QStackedWidget, QLabel, QGridLayout, QToolBar, QLineEdit, QComboBox
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QFont


class FeatureMenu(QWidget):
    """Represents a settings menu for a specific image processing tool.
    
    Parameters
    ----------
    title : Contains the title of this settings menu.
    
    Methods
    -------
    selectImages(value)
        Called when the user clicks on the select images button.
        Instructs the image list to change to multiple selection mode.    
    """

    button_default_SS = \
        """
        font-size: 10px;
        border: 1px solid #000000;
    """
    button_selected_SS = \
        """
        font-size: 10px;
        border: 1px solid #404041;
        background: #000000;
    """
    apply_signal = pyqtSignal(tuple)
    select_signal = pyqtSignal(bool)

    def __init__(self, title):
        super().__init__()
        self.outer_layout = QHBoxLayout(self)
        self.lower_layout = QVBoxLayout()
        self.menu_layout = QHBoxLayout()
        self.button_layout = QHBoxLayout()

        self.outer_layout.setContentsMargins(0, 0, 0, 0)
        self.outer_layout.setSpacing(1)
        self.lower_layout.setContentsMargins(0, 0, 0, 0)
        self.lower_layout.setSpacing(1)
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.setSpacing(1)

        self.select_button = QLabel('Select Images')
        self.select_button.setFixedHeight(18)
        self.apply_button = QLabel('Apply')
        self.apply_button.setFixedHeight(18)
        self.select_button.setStyleSheet(self.button_default_SS)
        self.apply_button.setStyleSheet(self.button_default_SS)

        self.title = QLabel(title)
        self.title.setFixedHeight(22)
        self.title.setObjectName('settings-title')

        self.outer_layout.addWidget(self.title)
        self.outer_layout.addLayout(self.lower_layout)
        self.lower_layout.addLayout(self.menu_layout)
        self.lower_layout.addLayout(self.button_layout)

        self.select_highlight = False
        self.select_button.mousePressEvent = lambda e: \
            self.selectImages(self.select_highlight)

    def selectImages(self, value):
        if value == False:
            self.select_button.setStyleSheet(self.button_selected_SS)
            self.select_highlight = True
            self.select_signal.emit(self.select_highlight)
        else:
            self.select_button.setStyleSheet(self.button_default_SS)
            self.select_highlight = False
            self.select_signal.emit(self.select_highlight)


class ProcessingMenu(FeatureMenu):

    def __init__(self):
        super().__init__('Hand x-ray image pre-processing pipeline: ')
        self.button_layout.addWidget(self.select_button)
        self.button_layout.addWidget(self.apply_button)

        self.apply_button.mousePressEvent = lambda e: \
            self.applyProcess(())

    def applyProcess(self, settings):
        self.selectImages(self.select_highlight)
        self.apply_signal.emit(settings)


class ComparisonMenu(FeatureMenu):

    def __init__(self):
        super().__init__('Contour Comparison: ')
        self.select_button.setText('Select hand')
        self.button_layout.addWidget(self.select_button)
        self.button_layout.addWidget(self.apply_button)
        self.apply_button.mousePressEvent = lambda e: \
            self.applyProcess(())

    def applyProcess(self, settings):
        self.selectImages(self.select_highlight)
        self.apply_signal.emit(settings)


class SearchMenu(FeatureMenu):

    def __init__(self):
        super().__init__('Search for similar hands: ')
        self.select_button.setText('Select hand')
        self.button_layout.addWidget(self.select_button)
        self.button_layout.addWidget(self.apply_button)
        self.apply_button.mousePressEvent = lambda e: \
            self.applyProcess(())

    def applyProcess(self, settings):
        self.selectImages(self.select_highlight)
        self.apply_signal.emit(settings)


class GraphMenu(FeatureMenu):

    def __init__(self):
        super().__init__('K-means data cluster graph')
        self.select_button.setText('Select hand')
        self.button_layout.addWidget(self.select_button)
        self.button_layout.addWidget(self.apply_button)
        self.apply_button.mousePressEvent = lambda e: \
            self.applyProcess(())

    def applyProcess(self, settings):
        self.selectImages(self.select_highlight)
        self.apply_signal.emit(settings)


class ControlButtons(QWidget):

    button_default_SS = """
        border: 1px solid #000000;
    """
    button_selected_SS = \
        """
        border: 1px solid #404041;
        background: #000000;
    """
    undo_signal = pyqtSignal()
    redo_signal = pyqtSignal()
    save_signal = pyqtSignal()
    settings_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        undo_button = QLabel()
        redo_button = QLabel()

        undo_button.mousePressEvent = lambda e: self.undo_signal.emit()
        redo_button.mousePressEvent = lambda e: self.redo_signal.emit()

        undo_button.setContentsMargins(4, 4, 4, 4)
        redo_button.setContentsMargins(4, 4, 4, 4)

        undo_button.setStyleSheet(self.button_default_SS)
        redo_button.setStyleSheet(self.button_default_SS)

        undo_button.setScaledContents(True)
        redo_button.setScaledContents(True)

        undo_button.setFixedHeight(34)
        undo_button.setFixedWidth(34)
        redo_button.setFixedHeight(34)
        redo_button.setFixedWidth(34)

        undo_pixmap = QPixmap('grasp/grasp_gui/recourses/undo_grey.png')
        redo_pixmap = QPixmap('grasp/grasp_gui/recourses/redo_grey.png')

        undo_button.setPixmap(undo_pixmap)
        redo_button.setPixmap(redo_pixmap)

        control_layout = QHBoxLayout()
        outer_layout = QVBoxLayout()

        control_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        control_layout.setSpacing(1)
        outer_layout.setSpacing(1)

        control_layout.addWidget(undo_button)
        control_layout.addWidget(redo_button)
        control_layout.addSpacing(3)

        control_widget = QWidget()
        control_widget.setLayout(control_layout)
        outer_layout.addSpacing(4)
        outer_layout.addWidget(control_widget)
        outer_layout.addStretch()

        self.setLayout(outer_layout)


class MenuButtons(QWidget):

    button_default_SS = """
        border: 1px solid #000000;
    """
    button_selected_SS = \
        """
        border: 1px solid #404041;
        background: #000000;
    """
    menu_select_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setFixedHeight(40)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        processing_button = QLabel()
        comparison_button = QLabel()
        search_button = QLabel()
        graph_button = QLabel()

        layout.addWidget(processing_button)
        layout.addWidget(comparison_button)
        layout.addWidget(search_button)
        layout.addWidget(graph_button)

        processing_button.setFixedHeight(34)
        processing_button.setFixedWidth(34)
        comparison_button.setFixedHeight(34)
        comparison_button.setFixedWidth(34)
        search_button.setFixedHeight(34)
        search_button.setFixedWidth(34)
        graph_button.setFixedHeight(34)
        graph_button.setFixedWidth(34)

        processing_button.setStyleSheet(self.button_default_SS)
        comparison_button.setStyleSheet(self.button_default_SS)
        search_button.setStyleSheet(self.button_default_SS)
        graph_button.setStyleSheet(self.button_default_SS)

        processing_button.setScaledContents(True)
        comparison_button.setScaledContents(True)
        search_button.setScaledContents(True)
        graph_button.setScaledContents(True)

        processing_button.setContentsMargins(4, 4, 4, 4)
        comparison_button.setContentsMargins(4, 4, 4, 4)
        search_button.setContentsMargins(4, 4, 4, 4)
        graph_button.setContentsMargins(4, 4, 4, 4)

        processing_pixmap = \
            QPixmap('grasp/grasp_gui/recourses/pipeline.png')
        comparison_pixmap = \
            QPixmap('grasp/grasp_gui/recourses/contour_comparison.png')
        search_pixmap = QPixmap('grasp/grasp_gui/recourses/search.png')
        graph_pixmap = QPixmap('grasp/grasp_gui/recourses/cluster.png')

        processing_button.setPixmap(processing_pixmap)
        comparison_button.setPixmap(comparison_pixmap)
        search_button.setPixmap(search_pixmap)
        graph_button.setPixmap(graph_pixmap)

        processing_button.mousePressEvent = lambda e: \
            self.menuSelect(processing_button, 'Processing')
        comparison_button.mousePressEvent = lambda e: \
            self.menuSelect(comparison_button, 'Comparison')
        search_button.mousePressEvent = lambda e: \
            self.menuSelect(search_button, 'Search')
        graph_button.mousePressEvent = lambda e: \
            self.menuSelect(graph_button, 'Graph')

        self.current_menu = 'None'

    def menuSelect(self, menu_button, menu_name):
        if self.current_menu != 'None':
            self.current_menu.setStyleSheet(self.button_default_SS)
        self.current_menu = menu_button
        menu_button.setStyleSheet(self.button_selected_SS)
        self.menu_select_signal.emit(menu_name)
