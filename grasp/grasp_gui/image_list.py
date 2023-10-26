from this import s
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, \
    QFrame, QScrollArea
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtCore import pyqtSignal, Qt, QSize, QRect
import copy


class Explorer(QWidget):

    """This is the left dock widget of the main window which lists
    the images imported by the user.
    
    Methods
    -------
    addImageItem(image_name)
        Creates an image item object and adds it to the image list.

    setMultipleSelection(value)
        Changes the mode of the image list, true indicates that the user
        can select multiple images. False indicates that selecting an image
        opens the image tab relating to that image.

    imageSelected(image_name)
        Called when the user clicks on an image item in the list. Opens the
        image tab relating to that image.
    """

    item_selected = pyqtSignal(int)

    class ExplorerItem(QWidget):

        selected = pyqtSignal(int)

        default_SS = \
            """
            padding: 2px;
            color: #777777;
            border: none;
        """
        highlight_SS = \
            """
            background-color: #000000;
            padding: 2px;
            color: #a8a8a8;
            border: none;
        """
        invalid_SS = \
            """
            padding: 2px;
            color: red;
            border: none;
        """
        invalid_highlight_SS = \
            """
            background-color: #000000;
            padding: 2px;
            color: red;
            border: none;
        """
        select_SS = \
            """
            background-color: #282828;
            padding: 2px;
            color: #a8a8a8;
            border: none;
        """

        def __init__(self, item_name, item_index):
            super().__init__()
            self.item_index = item_index
            self.setAttribute(Qt.WA_StyledBackground, True)
            self.setStyleSheet(self.default_SS)
            self.item_name = item_name
            self.image_label = QLabel(str(self.item_name))
            self.load_status = QLabel()
            self.load_status.setVisible(False)
            self.load_movie = \
                QMovie('grasp/grasp_gui/recourses/loading_gif.webp')
            self.load_movie.setScaledSize(QSize(20, 20))
            self.load_status.setMovie(self.load_movie)
            self.load_status.setVisible(False)
            self.item_layout = QHBoxLayout()
            self.item_layout.setContentsMargins(0, 0, 0, 0)
            self.item_layout.setSpacing(0)
            self.setLayout(self.item_layout)
            self.item_layout.addWidget(self.image_label)
            self.item_layout.addStretch()
            self.item_layout.addWidget(self.load_status)
            self.setLoadAnimation(False)
            self.item_status = 'Default'

            self.mousePressEvent = lambda e: \
                self.selected.emit(self.item_index)
            self.mouseMoveEvent = lambda e: \
                self.selected.emit(self.item_index)

        def setLoadAnimation(self, value):

            if value is True:
                self.load_status.setVisible(True)
                self.load_movie.start()
            else:
                self.load_status.setVisible(False)
                self.load_movie.stop()

        def setItemStatus(self, status):

            self.item_status = status
            if status == 'Highlight':
                self.setStyleSheet(self.highlight_SS)
            elif status == 'Invalid-Highlight':
                self.setStyleSheet(self.invalid_highlight_SS)
            elif status == 'Default':
                self.setStyleSheet(self.default_SS)
            elif status == 'Invalid':
                self.setStyleSheet(self.invalid_SS)
            elif status == 'Select':
                self.setStyleSheet(self.select_SS)

    default_SS = \
        """
        padding: 2px;
        color: #777777;
        border: 1px solid #404041;
    """
    highlight_SS = \
        """
        background-color: #030405;
        padding: 2px;
        color: #777777;
        border: 1px solid #404041;
    """

    select_default_SS = \
        """
        font-size: 10px;
        border: 1px solid #000000;
    """
    select_highlight_SS = \
        """
        font-size: 10px;
        border: 1px solid #404041;
        background: #000000;
    """

    def __init__(self):
        super().__init__()
        self.items = [0 for i in range(999)]
        self.selected_item = ''
        self.selected_images = []
        self.select_multiple = False
        self.target_selection = False
        self.select_all = False
        self.image_indexes = []
        self.deliverable_indexes = []

        layout = QVBoxLayout()
        layout.setSpacing(0)
        title_widget = QWidget()
        title_widget.setObjectName('dock')
        title_layout = QHBoxLayout()
        title_widget.setLayout(title_layout)
        title_layout.setContentsMargins(1, 1, 1, 1)
        title_label = QLabel('Images')
        title_label.setObjectName('title')
        title_label.setScaledContents(True)
        title_label.setFixedWidth(50)
        title_label.setFixedHeight(18)
        title_layout.addWidget(title_label)
        self.select_all_button = QLabel('Select All')
        self.select_all_button.hide()
        self.select_all_button.setStyleSheet(self.select_default_SS)
        self.select_all_button.setFixedWidth(60)
        self.select_all_button.setFixedHeight(20)
        self.select_all_button.mousePressEvent = lambda e: \
            self.selectAll()
        title_layout.addWidget(self.select_all_button)
        layout.addWidget(title_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area = QScrollArea()
        self.scroll_area.setFrameStyle(QFrame.NoFrame)
        self.list_viewport = QWidget()
        self.list_layout = QVBoxLayout(self.list_viewport)
        self.list_layout.setContentsMargins(2, 2, 2, 2)
        self.list_layout.setSpacing(1)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_layout = QVBoxLayout()
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.addWidget(self.list_viewport)
        self.scroll_layout.addStretch()
        self.scroll_widget = QWidget()
        self.scroll_widget.setObjectName('dock')
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_widget)
        self.valid_images = []
        self.scroll_area.setObjectName('image-scroll')
        layout.addWidget(self.scroll_area)

        deliverable_layout = QVBoxLayout()
        deliverable_layout.setObjectName('composition')
        deliverable_layout.setSpacing(0)
        deliverable_title_widget = QWidget()
        deliverable_title_widget.setObjectName('dev-title')
        deliverable_title_layout = QHBoxLayout()
        deliverable_title_widget.setLayout(deliverable_title_layout)
        deliverable_title_layout.setContentsMargins(1, 1, 1, 1)
        deliverable_title_label = QLabel('Things')
        deliverable_title_label.setObjectName('title')
        deliverable_title_label.setScaledContents(True)
        deliverable_title_label.setFixedWidth(50)
        deliverable_title_label.setFixedHeight(18)
        deliverable_title_layout.addWidget(deliverable_title_label)
        deliverable_layout.addWidget(deliverable_title_widget)
        deliverable_layout.setContentsMargins(0, 0, 0, 0)
        self.deliverable_scroll_area = QScrollArea()
        self.deliverable_scroll_area.setFrameStyle(QFrame.NoFrame)
        self.deliverable_list_viewport = QWidget()
        self.deliverable_list_layout = \
            QVBoxLayout(self.deliverable_list_viewport)
        self.deliverable_list_layout.setContentsMargins(2, 2, 2, 2)
        self.deliverable_list_layout.setSpacing(1)
        self.deliverable_scroll_area.setWidgetResizable(True)
        self.deliverable_scroll_layout = QVBoxLayout()
        self.deliverable_scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.deliverable_scroll_layout.addWidget(self.deliverable_list_viewport)
        self.deliverable_scroll_layout.addStretch()
        self.deliverable_scroll_widget = QWidget()
        self.deliverable_scroll_widget.setObjectName('dock')
        self.deliverable_scroll_widget.setLayout(self.deliverable_scroll_layout)
        self.deliverable_scroll_area.setWidget(self.deliverable_scroll_widget)
        deliverable_layout.addWidget(self.deliverable_scroll_area)

        explorer_layout = QVBoxLayout(self)
        explorer_layout.setContentsMargins(0, 0, 0, 0)
        explorer_layout.setSpacing(0)
        explorer_layout.addLayout(layout)
        explorer_layout.addLayout(deliverable_layout)
        self.invalid_items = []
        self.processed_items = []

        self.item_count = 0

    def addExplorerItem(
        self,
        item_name,
        index,
        item_type,
        ):

        explorer_item = self.ExplorerItem(item_name, index)
        self.items[index] = explorer_item
        explorer_item.selected.connect(lambda i: self.itemSelected(i))
        if item_type == 'Image':
            self.image_indexes.append(index)
            self.list_layout.addWidget(explorer_item)
        else:
            self.deliverable_indexes.append(index)
            self.deliverable_list_layout.addWidget(explorer_item)
        self.item_count += 1

    def removeExplorerItem(self, index):
        if index in self.deliverable_indexes:
            self.list_layout.removeWidget(self.items[index])
            self.items[index].setVisible(False)
        self.item_count -= 1

    def setInvalidItems(self, invalid_items):
        self.invalid_items = invalid_items
        for item in self.invalid_items:
            if self.items[item].item_status == 'Default':
                self.items[item].setItemStatus('Invalid')
            else:
                self.items[item].setItemStatus('Invalid-Highlight')

    def setMultipleSelection(self, value):

        self.select_multiple = value
        if self.selected_item != '':
            if self.selected_item not in self.invalid_items:
                self.items[self.selected_item].setItemStatus('Default')
            else:
                self.items[self.selected_item].setItemStatus('Invalid')
            self.selected_item = ''
        if value == True:
            self.selected_images = []
            self.select_all_button.show()
            for index in self.processed_items:
                self.items[index].setItemStatus('Invalid')
        elif value == False:
            self.select_all_button.hide()
            self.selectAll()
            for index in self.processed_items:
                self.items[index].setItemStatus('Default')
            for i in range(self.item_count):
                if i not in self.invalid_items:
                    self.items[i].setItemStatus('Default')

    def setTargetSelection(self, value):
        if self.select_multiple == True:
            self.setMultipleSelection(False)
        self.target_selection = value
        if self.selected_item != '':
            if self.selected_item not in self.invalid_items:
                self.items[self.selected_item].setItemStatus('Default')

    def addInvalidItem(self, invalid_index):
        if invalid_index not in self.invalid_items:
            self.invalid_items.append(invalid_index)
            if self.items[invalid_index].item_status == 'Default':
                self.items[invalid_index].setItemStatus('Invalid')
            else:
                self.items[invalid_index].setItemStatus('Invalid-Highlight'
                        )

    def addProcessedItem(self, processed_index):
        if processed_index not in self.processed_items:
            self.processed_items.append(processed_index)

    def removedProcessedItem(self, processed_index):
        if processed_index in self.processed_items:
            self.processed_items.pop(self.processed_items.index(processed_index))

    def removeInvalidItem(self, invalid_index):
        self.invalid_items.pop(self.invalid_items.index(invalid_index))
        if self.items[invalid_index].item_status == 'Invalid':
            self.items[invalid_index].setItemStatus('Default')
        else:
            self.items[invalid_index].setItemStatus('Highlight')

    def returnToDefaultSelection(self):
        self.setMultipleSelection(False)
        self.setTargetSelection(False)

    def itemSelected(self, index):

        if self.selected_item != '' and self.selected_item \
            not in self.invalid_items:
            self.items[self.selected_item].setItemStatus('Default')
        elif self.selected_item != '' and self.selected_item \
            in self.invalid_items:
            self.items[self.selected_item].setItemStatus('Invalid')
        if self.select_multiple is True:
            if index not in self.selected_images and index \
                not in self.deliverable_indexes and index \
                not in self.invalid_items and index \
                not in self.processed_items:
                self.selected_images.append(index)
                self.items[index].setItemStatus('Highlight')
        elif self.target_selection is True:
            if index not in self.invalid_items:
                self.items[index].setItemStatus('Select')
                self.selected_item = index
        elif index != -1:
            self.item_selected.emit(index)
            self.selected_item = index
            if index in self.invalid_items:
                self.items[index].setItemStatus('Invalid-Highlight')
            else:
                self.items[index].setItemStatus('Highlight')

    def selectAll(self):
        self.selected_images = []
        if self.select_all == True:
            self.select_all = False
            self.select_all_button.setStyleSheet(self.select_default_SS)
        else:
            self.select_all = True
            self.select_all_button.setStyleSheet(self.select_highlight_SS)

        for i in range(self.item_count):
            if self.select_all == True and i \
                not in self.deliverable_indexes and i \
                not in self.invalid_items and i \
                not in self.processed_items:
                self.selected_images.append(self.items[i].item_index)
                self.items[i].setItemStatus('Select')
            elif i not in self.processed_items:
                if i not in self.invalid_items:
                    self.items[i].setItemStatus('Default')
                else:
                    self.items[i].setItemStatus('Invalid')


class SidebarList(QWidget):

    class ListItem(QWidget):

        selected = pyqtSignal(int)
        default_SS = \
            """
            padding: 2px;
            color: #777777;
            border: none;
        """
        highlight_SS = \
            """
            background-color: #000000;
            padding: 2px;
            color: #a8a8a8;
            border: none;
        """
        invalid_SS = """
            color: red;
        """
        percentage_SS = \
            """
            color: #a8a8a8;
            background-color: none;
            border: none;
        """

        def __init__(
            self,
            item_name,
            item_index,
            percentage,
            ):

            super().__init__()
            self.setFixedHeight(22)
            self.item_index = item_index
            self.setAttribute(Qt.WA_StyledBackground, True)
            self.setStyleSheet(self.default_SS)
            self.item_name = item_name
            self.image_label = QLabel(str(self.item_name))
            self.item_layout = QHBoxLayout()
            self.item_layout.setContentsMargins(0, 0, 0, 0)
            self.item_layout.setSpacing(0)
            self.setLayout(self.item_layout)
            self.item_layout.addWidget(self.image_label)
            if percentage != '':
                percentage_label = QLabel(percentage)
                percentage_label.setStyleSheet(self.percentage_SS)
                self.item_layout.addStretch(1)
                self.item_layout.addWidget(percentage_label)
                self.item_layout.addSpacing(5)
            self.mousePressEvent = lambda e: \
                self.selected.emit(self.item_index)
            self.mouseMoveEvent = lambda e: \
                self.selected.emit(self.item_index)

        def setItemStatus(self, status):

            if status == 'Highlight':
                self.setStyleSheet(self.highlight_SS)
            elif status == 'Default':
                self.setStyleSheet(self.default_SS)
            elif status == 'Invalid':
                self.setStyleSheet(self.invalid_SS)

    default_SS = \
        """
        padding: 2px;
        color: #404040;
        border: 1px solid #404041;
    """
    highlight_SS = \
        """
        background-color: #030405;
        padding: 2px;
        color: #404041;
        border: 1px solid #404041;
    """
    select_default_SS = \
        """
        font-size: 10px;
        border: 1px solid #000000;
    """
    select_highlight_SS = \
        """
        font-size: 10px;
        border: 1px solid #404041;
        background: #000000;
    """
    list_SS = \
        """
        border-left: 1px ridge #515151;
        border-right: none;
        border-top: 1px solid #000000;
        border-bottom: 1px solid #000000;
    """
    item_selected = pyqtSignal(int)

    def __init__(self, list_items, search):
        super().__init__()
        self.list_items = list_items
        self.items = [0 for i in range(99999)]
        self.selected_item = ''
        self.selected_images = []
        self.select_multiple = False
        self.select_all = False
        self.image_indexes = []

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        title_widget = QWidget()
        title_widget.setObjectName('dock')
        title_layout = QHBoxLayout()
        title_widget.setLayout(title_layout)
        title_layout.setContentsMargins(1, 1, 1, 1)
        self.select_all_button = QLabel('Select All')
        self.select_all_button.hide()
        self.select_all_button.setStyleSheet(self.select_default_SS)
        self.select_all_button.setFixedWidth(140)
        self.select_all_button.setFixedHeight(20)
        self.select_all_button.mousePressEvent = lambda e: \
            self.selectAll()
        title_layout.addStretch(1)
        title_layout.addWidget(self.select_all_button)
        title_layout.addStretch(1)
        layout.addWidget(title_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area = QScrollArea()
        self.scroll_area.setFrameStyle(QFrame.NoFrame)
        self.list_viewport = QWidget()
        self.list_viewport.setStyleSheet(self.list_SS)
        self.list_layout = QVBoxLayout(self.list_viewport)
        self.list_layout.setContentsMargins(2, 2, 2, 2)
        self.list_layout.setSpacing(1)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_layout = QVBoxLayout()
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.addWidget(self.list_viewport)
        self.scroll_widget = QWidget()
        self.scroll_widget.setObjectName('dock')
        self.scroll_widget.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_widget)
        self.valid_images = []
        layout.addWidget(self.scroll_area)
        self.item_count = 0
        if search is True:
            for item in self.list_items:
                (item_name, item_index, perc) = item
                self.addListItem(item_name, item_index, perc)
        else:
            for item in self.list_items:
                (item_name, item_index) = item
                self.addListItem(item_name, item_index, '')
        self.list_layout.addStretch(1)

    def addListItem(
        self,
        item_name,
        index,
        perc,
        ):

        if perc != '':
            list_item = self.ListItem(item_name, index, perc)
        else:
            list_item = self.ListItem(item_name, index, '')
        list_item.setItemStatus('Default')
        self.items[index] = list_item
        list_item.selected.connect(lambda i: self.itemSelected(index))
        self.image_indexes.append(index)
        self.list_layout.addWidget(list_item)
        self.item_count += 1

    def itemSelected(self, index):
        if self.selected_item != '':
            self.items[self.selected_item].setItemStatus('Default')
        if self.select_multiple is True:
            if index not in self.selected_images:
                self.selected_images.append(index)
                self.items[index].setItemStatus('Highlight')
        elif index != -1:
            self.item_selected.emit(self.items[index].item_index)
            self.selected_item = index
            self.items[index].setItemStatus('Highlight')

    def setMultipleSelection(self, value):
        self.select_multiple = value
        if self.selected_item != '':
            self.items[self.selected_item].setItemStatus('Default')
            self.selected_item = ''
        if value == True:
            self.selected_images = []
            self.select_all_button.show()
        elif value == False:
            self.select_all_button.hide()
            for i in self.image_indexes:
                self.items[i].setItemStatus('Default')

    def selectAll(self):
        self.selected_images = []
        if self.select_all == True:
            self.select_all = False
            self.select_all_button.setStyleSheet(self.select_default_SS)
        else:
            self.select_all = True
            self.select_all_button.setStyleSheet(self.select_highlight_SS)

        for i in self.image_indexes:
            if self.select_all == True:
                self.selected_images.append(i)
                self.items[i].setItemStatus('Highlight')
            else:
                self.items[i].setItemStatus('Default')
