from PyQt5.QtWidgets import QMainWindow, QTabWidget, QFileDialog, \
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QMenuBar, \
    QMenu, QDockWidget, QVBoxLayout, QWidget, QToolBar, QTabBar, \
    QHBoxLayout, QSizePolicy, QStackedWidget
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import QFileInfo, Qt, pyqtSignal

from image_list import Explorer
from image_view import ImageViewer, ItemSideBar, StepBar
from toolbar import MenuButtons, ControlButtons, ProcessingMenu, \
    ComparisonMenu, GraphMenu, SearchMenu
import time
from grasp.grasp_engine import GraspEngine


class MainWindow(QMainWindow):

    """The main window of the user interface, contains a toolbar, image view central
    widget, and an image list dock.
    
    Methods
    -------
    imageSelected(image_name)
        Called when the user selects in image in the image list, displays the image view
        tab relating to that image.
    
    importImages()
        Called when the user chooses to import images, opens a QFileDialog to retreive 
        a list of image paths, then creates their image view tabs and adds them to the
        image list.

    showSettings(tool_name)
        Called when the user selects a specific processing tool, displays the relating
        tool settings menu.
    """

    import_sig = pyqtSignal(list)
    export_sig = pyqtSignal(str)
    custom_settings_sig = pyqtSignal(tuple)

    main_window_SS = \
        """
        QMainWindow{
            background: #0e0e0e;
            background-image: url(grasp/grasp_gui/recourses/background.png);
        }

        QToolBar {
            border-bottom: 1px ridge #515151;
            background: #212121;
        }

        QToolBar::separator{
            background-color: #515151;
            width: 1px;
        }

        QComboBox{
            border: none;
            height: 15px;
        }

        #component-menu{
            border-left: 1px solid #000000;
        }

        QWidget#sidebar{
            border-left: 1px ridge #515151;
            background: #212121;
        }

        QLabel#setting-name{
            border: none;
            border-left: none;
            height: 15px;
        }

        QLabel#settings-title{
            font-size: 15px;
            color: #adadad;
            border: none;
        }

        QLabel#side-settings-title{
            font-size: 12px;
            color: #adadad;
            border-bottom: 1px ridge #515151;
        }

        QLineEdit{
            height: 15px;
        }

        QLabel{
            padding: 2px;
            color: #808080;
            border: none;
        }

        QDockWidget::title{
            text-align: left;
            background: #212121;
            border-bottom: 1px ridge #515151;
            border-right: 1px ridge #515151;
            padding: 5px;
        }

        QDockWidget{
            font-size: 12pt;
            color: #adadad;
        }

        QMenu{
            padding: 5px;
        }

        QMenuBar{
            background: #191919;
            border-bottom: 1px ridge #515151;
        }

        QMenuBar::item{
            color: #adadad;
        }

        QTabWidget{
            border: none;
            background: transparent;
        }

        QTabWidget::pane{
            background: transparent;
            border: none;
        }

        QTabWidget::tab-bar{
            left: 0;
            background: #000000;
        }

        QTabBar::tab{
            color: #808080;
            background: transparent;;
            padding: 5px;
        }

        QTabBar::tab:selected{
            color: #a8a8a8;
            border-bottom: 2px solid #a8a8a8;
            background: #000000;
            padding: 5px;
        }

        QGraphicsView#view{
            border: none;
            background: transparent;
        }

        QCheckBox{
            border: none;
            background: #212121;
        }

        QWidget#step-view{
            border-top: 1px ridge #515151;
            background: #212121;
        }
        
        QLabel#step{
            border-top: 4px solid #000000;
            border-right: 1px solid #000000;
            border-left: 1px solid #000000;
            border-bottom: 1px solid #000000;
        }

        QLabel#error-step{
            border-top: 4px solid #FF0000;
            border-right: 1px solid #FF0000;
            border-left: 1px solid #FF0000;
            border-bottom: 1px solid #FF0000;
        }

        QWidget#sidebar{
            background: #212121;
            border-left: 1px ridge #515151;
        }
        
        QWidget#dock{
            border-right: 1px ridge #515151;
            background: #212121;
        }

        QWidget#dev-title{
            border-bottom: none;
            border-right: 1px ridge #515151;
            border-left: none;
            border-top: 1px ridge #515151;
            background: #212121;
        }

        QScrollArea{
            background: none;
        }

        QLabel#title{
            border: none;
            background: none;
            color: #adadad;
            font-size: 12px;
        }
    """

    def __init__(self):
        super().__init__()
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.BottomDockWidgetArea)

        self.num_tabs = 0
        self.num_stepbars = 1
        self.item_count = 0

        self.image_indexes = []
        self.deliverable_indexes = []

        self.setStyleSheet(self.main_window_SS)
        self.item_widgets = [0 for i in range(999)]
        self.item_tab_widget = QTabWidget()

        self.tab_bar = QTabBar()
        self.tab_bar.setTabsClosable(True)
        self.item_tab_widget.setTabBar(self.tab_bar)
        self.tab_bar.tabCloseRequested.connect(lambda i: \
                self.tabClose(i))
        self.tab_bar.tabBarClicked.connect(lambda i: self.tabOpen(i))
        self.item_tab_widget.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(self.item_tab_widget)

        self.sidebar_stack = QStackedWidget()

        self.sidebar_dock = QDockWidget()
        self.sidebar_dock.setMinimumHeight(200)
        self.sidebar_dock.setFixedWidth(150)
        self.sidebar_dock.setTitleBarWidget(QWidget())
        self.sidebar_dock.setWidget(self.sidebar_stack)
        self.sidebar_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.RightDockWidgetArea, self.sidebar_dock)

        self.stepbar_stack = QStackedWidget()
        back_widget = QWidget()
        back_widget.setObjectName('step-view')
        back_layout = QVBoxLayout(back_widget)
        back_layout.setContentsMargins(0, 0, 0, 0)
        back_layout.setSpacing(0)
        self.step_dock = QDockWidget()
        self.step_dock.setVisible(False)
        back_layout.addWidget(self.stepbar_stack)
        self.step_dock.setTitleBarWidget(QWidget())
        self.step_dock.setFixedHeight(80)
        self.step_dock.setWidget(back_widget)
        self.step_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.step_dock)

        menu_bar = QMenuBar(self)
        menu_bar.setNativeMenuBar(False)
        file_menu = QMenu('&File', self)
        import_act = file_menu.addAction('&Import')
        import_act.triggered.connect(self.importImages)
        export_act = file_menu.addAction('&Export')
        export_act.triggered.connect(self.exportImages)
        menu_bar.addMenu(file_menu)
        menu_bar.adjustSize()
        self.setMenuBar(menu_bar)

        self.explorer = Explorer()
        self.explorer.item_selected.connect(lambda i: \
                self.itemSelected(i))
        dock_widget = QDockWidget('Explorer')
        dock_widget.setFixedWidth(150)
        dock_container = QWidget()
        dock_layout = QVBoxLayout(dock_container)
        dock_layout.setContentsMargins(0, 0, 0, 0)
        dock_layout.setSpacing(0)
        dock_widget.setWidget(dock_container)
        dock_layout.addWidget(self.explorer)
        dock_widget.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_widget)

        menu_buttons = MenuButtons()
        self.control_buttons = ControlButtons()
        self.processing_menu = ProcessingMenu()
        self.comparison_menu = ComparisonMenu()
        self.search_menu = SearchMenu()
        self.graph_menu = GraphMenu()

        menu_buttons.menu_select_signal.connect(lambda n: \
                self.showMenu(n))
        self.processing_menu.select_signal.connect(lambda v: \
                self.explorer.setMultipleSelection(v))
        self.processing_menu.apply_signal.connect(lambda : \
                self.explorer.setMultipleSelection(False))
        self.comparison_menu.select_signal.connect(lambda v: \
                self.explorer.setTargetSelection(v))
        self.comparison_menu.apply_signal.connect(lambda : \
                self.explorer.setTargetSelection(False))
        self.search_menu.select_signal.connect(lambda v: \
                self.explorer.setTargetSelection(v))
        self.search_menu.apply_signal.connect(lambda : \
                self.explorer.setTargetSelection(False))
        self.graph_menu.select_signal.connect(lambda v: \
                self.explorer.setTargetSelection(v))
        self.graph_menu.apply_signal.connect(lambda : \
                self.explorer.setTargetSelection(False))

        self.menu_stack = QStackedWidget()
        self.menu_stack.addWidget(QWidget())
        self.menu_stack.addWidget(self.processing_menu)
        self.menu_stack.addWidget(self.comparison_menu)
        self.menu_stack.addWidget(self.search_menu)
        self.menu_stack.addWidget(self.graph_menu)

        menu_widget = QWidget()
        menu_layout = QHBoxLayout()
        menu_layout.setContentsMargins(0, 0, 0, 0)
        menu_layout.addWidget(self.menu_stack)
        menu_layout.addStretch(1)
        menu_widget.setLayout(menu_layout)

        toolbar = QToolBar()
        toolbar.setFixedHeight(44)
        toolbar.addWidget(menu_buttons)
        toolbar.addSeparator()
        toolbar.addWidget(menu_widget)
        toolbar.addSeparator()
        toolbar.addWidget(self.control_buttons)
        toolbar.addSeparator()
        self.addToolBar(toolbar)

        self.showMaximized()

        self.visible_tabs = []

    def itemSelected(self, index):
        """Called when the user selects in image in the image list, displays the image view
        tab relating to that image.
        
        Parameters
        ----------
        image_name : Contains the name of the image which was selected by the user."""

        self.item_tab_widget.setVisible(True)
        if self.item_tab_widget.isTabVisible(index) == False:
            self.item_tab_widget.setTabVisible(index, True)
            self.visible_tabs.append(index)
        self.item_tab_widget.setCurrentIndex(index)
        if index not in self.deliverable_indexes:
            self.stepbar_stack.setCurrentIndex(index)
            self.sidebar_stack.setCurrentIndex(index)
            self.step_dock.setVisible(True)
            self.sidebar_dock.setVisible(True)
        else:
            self.sidebar_dock.setVisible(True)
            self.step_dock.setVisible(False)
            self.sidebar_stack.setCurrentIndex(index)

    def deliverableSelected(self, index):
        if self.item_tab_widget.isTabVisible(index) == False:
            self.item_tab_widget.setTabVisible(index, True)
            self.visible_tabs.append(index)
        self.item_tab_widget.setCurrentIndex(index)
        self.step_dock.setVisible(False)

    def addDeliverable(
        self,
        deliverable_image,
        deliverable_name,
        sidebar,
        ):

        self.explorer.addExplorerItem(deliverable_name,
                self.item_count, 'Deliverable')
        deliverable_view = ImageViewer(deliverable_image)
        self.item_tab_widget.insertTab(self.item_count,
                deliverable_view, deliverable_name)
        self.item_tab_widget.setTabVisible(self.item_count, False)
        self.deliverable_indexes.append(self.item_count)
        self.item_widgets[self.item_count] = (deliverable_view, sidebar)
        self.sidebar_stack.insertWidget(self.item_count, sidebar)
        self.itemSelected(self.item_count)
        self.item_count += 1

    def tabClose(self, index):
        self.item_tab_widget.setTabVisible(index, False)
        if index in self.visible_tabs:
            self.visible_tabs.pop(self.visible_tabs.index(index))

        if len(self.visible_tabs) != 0:
            self.stepbar_stack.setCurrentIndex(self.visible_tabs[0])
            self.item_tab_widget.setCurrentIndex(self.visible_tabs[0])
            self.explorer.itemSelected(self.visible_tabs[0])
        else:
            self.step_dock.setVisible(False)
            self.item_tab_widget.setCurrentIndex(-1)
            self.item_tab_widget.setVisible(False)
            self.sidebar_dock.setVisible(False)
            self.explorer.itemSelected(-1)

    def tabOpen(self, index):
        self.explorer.itemSelected(index)
        self.item_tab_widget.setCurrentIndex(index)
        self.stepbar_stack.setCurrentIndex(index)
        self.sidebar_stack.setCurrentIndex(index)

    def stepSelected(
        self,
        image_index,
        pixmap,
        step_index,
        ):

        (viewer, stepbar, sidebar) = self.item_widgets[image_index]
        sidebar.showItemSettings(step_index)
        viewer.changeCurrentImage(pixmap)

    def importImages(self):
        """Called when the user chooses to import images, opens a QFileDialog to retreive 
        a list of image paths, then creates their image view tabs and adds them to the
        image list."""

        file_names = QFileDialog.getOpenFileNames(self, 'Import Images'
                , '/Documents', 'Image Files (*.png *.jpg)')[0]
        images = []
        for name in file_names:
            f_name = QFileInfo(name).fileName()
            self.explorer.addExplorerItem(f_name, self.item_count,
                    'Image')
            image_data = GraspEngine.convertToArray(name)
            q_image = GraspEngine.convert_cv_qt(image_data)
            viewer = ImageViewer(q_image)
            self.item_tab_widget.insertTab(self.item_count, viewer,
                    f_name)
            self.item_tab_widget.setMaximumWidth(self.item_tab_widget.width())
            self.item_tab_widget.setTabVisible(self.item_count, False)
            sidebar = ItemSideBar(f_name, self.item_count)
            sidebar.setObjectName('sidebar')
            sidebar.apply_signal.connect(lambda s: \
                    self.custom_settings_sig.emit(s))
            stepbar = StepBar(self.item_count)
            self.sidebar_stack.insertWidget(self.item_count, sidebar)
            self.stepbar_stack.insertWidget(self.item_count, stepbar)
            stepbar.step_signal.connect(lambda i, p, s: \
                    self.stepSelected(i, p, s))
            self.item_widgets[self.item_count] = (viewer, stepbar,
                    sidebar)
            images.append((f_name, name, self.item_count))
            self.image_indexes.append(self.item_count)
            self.item_count = self.item_count + 1
        self.import_sig.emit(images)

    def exportImages(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.Directory)
        file_name = dialog.getSaveFileName(self, 'Save Processed Images'
                , '/Documents')
        self.export_sig.emit(file_name[0])

    def showMenu(self, menu_name):
        """Called when the user selects a specific processing tool, displays the relating
        tool settings menu.
        
        Parameters
        ----------
        tool_name : Indicates the name of the tool settings menu which should be displayed."""

        self.explorer.returnToDefaultSelection()
        if menu_name != 'Processing':
            self.processing_menu.selectImages(True)
        if menu_name != 'Comparison':
            self.comparison_menu.selectImages(True)
        if menu_name != 'Search':
            self.search_menu.selectImages(True)
        if menu_name != 'Graph':
            self.graph_menu.selectImages(True)
        if menu_name == 'Processing':
            self.menu_stack.setCurrentWidget(self.processing_menu)
        elif menu_name == 'Comparison':
            self.menu_stack.setCurrentWidget(self.comparison_menu)
        elif menu_name == 'Search':
            self.menu_stack.setCurrentWidget(self.search_menu)
        elif menu_name == 'Graph':
            self.menu_stack.setCurrentWidget(self.graph_menu)
