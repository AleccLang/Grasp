from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QWidget, \
    QVBoxLayout, QHBoxLayout, QLabel, QGraphicsPixmapItem, \
    QStackedWidget, QGridLayout, QLineEdit, QComboBox, QCheckBox
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import QByteArray, Qt, QFileInfo, pyqtSignal

from image_list import SidebarList

import cv2
import copy


class ItemSideBar(QWidget):

    class ItemSettings(QVBoxLayout):

        button_default_SS = \
            """
            font-size: 10px;
            border: 1px solid #000000;
        """
        setting_name_SS = \
            """
            border: none;
            border-left: none;
            height: 15px;
        """
        apply_signal = pyqtSignal(tuple)

        def __init__(self, settings_type):
            super().__init__()
            self.setContentsMargins(0, 0, 0, 0)
            self.setSpacing(3)
            self.settings_type = settings_type
            title = QLabel(settings_type)
            title.setFixedHeight(25)
            title.setObjectName('side-settings-title')
            self.button_layout = QHBoxLayout()
            self.button_layout.setContentsMargins(0, 0, 0, 0)
            self.button_layout.setSpacing(2)
            self.button_layout.addStretch(0)
            self.addWidget(title)
            self.apply_button = QLabel('Apply')
            self.apply_button.setFixedHeight(20)
            self.apply_button.setFixedWidth(50)
            self.apply_button.setStyleSheet(self.button_default_SS)

    class NoItemSettings(ItemSettings):

        def __init__(self, settings_type):
            super().__init__(settings_type)
            no_settings = QLabel('No custom settings.')
            no_settings.setFixedHeight(20)
            no_settings.setObjectName('setting-name')
            self.addWidget(no_settings)
            self.addStretch(1)

    class StupidLineSettings(ItemSettings):

        def __init__(self, settings_type, settings):
            super().__init__(settings_type)
            if settings == ():
                self.lines_removed = ''
            else:
                (self.lines_removed, ) = settings
            lines_removed_label = QLabel('Lines removed: ')
            lines_removed_label.setFixedHeight(20)
            lines_removed_label.setObjectName('setting-name')
            self.lines_text_edit = QLineEdit(self.lines_removed)
            self.lines_text_edit.setFixedHeight(20)
            self.lines_text_edit.setFixedWidth(140)
            lines_layout = QHBoxLayout()
            lines_layout.addStretch(1)
            lines_layout.addWidget(self.lines_text_edit)
            lines_layout.addStretch(1)
            self.addWidget(lines_removed_label)
            self.addLayout(lines_layout)
            self.button_layout.addWidget(self.apply_button)
            self.button_layout.addSpacing(5)
            self.addLayout(self.button_layout)
            self.apply_button.mousePressEvent = lambda e: \
                self.apply_signal.emit((self.lines_text_edit.text(), ))
            self.addStretch(1)

    class CLAHESettings(ItemSettings):

        def __init__(self, settings_type, settings):
            super().__init__('Histogram Equalization')
            if settings == ():
                self.clip_limit = ''
                self.gx = ''
                self.gy = ''
            else:
                (self.clip_limit, self.gx, self.gy) = settings
            clip_label = QLabel('Clip limit: ')
            clip_label.setFixedHeight(20)
            clip_label.setObjectName('setting-name')
            self.clip_text_edit = QLineEdit(self.clip_limit)
            self.clip_text_edit.setFixedHeight(20)
            self.clip_text_edit.setFixedWidth(140)
            grid_label = QLabel('Grid size: ')
            grid_label.setFixedHeight(20)
            grid_label.setObjectName('setting-name')
            self.grid_x_edit = QLineEdit(self.gx)
            self.grid_x_edit.setFixedHeight(20)
            self.grid_x_edit.setFixedWidth(30)
            self.grid_y_edit = QLineEdit(self.gy)
            self.grid_y_edit.setFixedHeight(20)
            self.grid_y_edit.setFixedWidth(30)
            clip_layout = QHBoxLayout()
            clip_layout.addStretch(1)
            clip_layout.addWidget(self.clip_text_edit)
            clip_layout.addStretch(1)
            grid_layout = QHBoxLayout()
            b1 = QLabel('(')
            b1.setStyleSheet(self.setting_name_SS)
            grid_layout.addStretch(1)
            grid_layout.addWidget(b1)
            grid_layout.addWidget(self.grid_x_edit)
            c = QLabel(',')
            c.setStyleSheet(self.setting_name_SS)
            grid_layout.addWidget(c)
            grid_layout.addWidget(self.grid_y_edit)
            b2 = QLabel(')')
            b2.setStyleSheet(self.setting_name_SS)
            grid_layout.addWidget(b2)
            grid_layout.addStretch(1)
            self.addWidget(clip_label)
            self.addLayout(clip_layout)
            self.addWidget(grid_label)
            self.addLayout(grid_layout)
            self.button_layout.addWidget(self.apply_button)
            self.button_layout.addSpacing(5)
            self.addLayout(self.button_layout)
            self.apply_button.mousePressEvent = lambda e: \
                self.apply_signal.emit((self.clip_text_edit.text(),
                    self.grid_x_edit.text(), self.grid_y_edit.text()))
            self.addStretch(1)

    class NormalizationSettings(ItemSettings):

        def __init__(self, settings_type, settings):
            super().__init__('Min-Max Normalization')
            if settings == ():
                self.alpha = ''
                self.beta = ''
            else:
                (self.alpha, self.beta) = settings
            alpha_label = QLabel('Alpha: ')
            alpha_label.setFixedHeight(20)
            alpha_label.setObjectName('setting-name')
            self.alpha_text_edit = QLineEdit(self.alpha)
            self.alpha_text_edit.setFixedHeight(20)
            self.alpha_text_edit.setFixedWidth(140)
            beta_label = QLabel('Beta: ')
            beta_label.setFixedHeight(20)
            beta_label.setObjectName('setting-name')
            self.beta_text_edit = QLineEdit(self.beta)
            self.beta_text_edit.setFixedHeight(20)
            self.beta_text_edit.setFixedWidth(140)
            alpha_layout = QHBoxLayout()
            alpha_layout.addStretch(1)
            alpha_layout.addWidget(self.alpha_text_edit)
            alpha_layout.addStretch(1)
            beta_layout = QHBoxLayout()
            beta_layout.addStretch(1)
            beta_layout.addWidget(self.beta_text_edit)
            beta_layout.addStretch(1)
            self.addWidget(alpha_label)
            self.addLayout(alpha_layout)
            self.addWidget(beta_label)
            self.addLayout(beta_layout)
            self.button_layout.addWidget(self.apply_button)
            self.button_layout.addSpacing(5)
            self.addLayout(self.button_layout)
            self.apply_button.mousePressEvent = lambda e: \
                self.apply_signal.emit((self.alpha_text_edit.text(),
                    self.beta_text_edit.text()))
            self.addStretch(1)

    class ThresholdSettings(ItemSettings):

        def __init__(self, settings_type, settings):
            super().__init__('Binary Threshold')
            if settings == ():
                self.threshold = ''
            else:
                (self.threshold, ) = settings
            thresh_label = QLabel('Threshold: ')
            thresh_label.setFixedHeight(20)
            thresh_label.setObjectName('setting-name')
            self.thresh_edit = QLineEdit(self.threshold)
            self.thresh_edit.setFixedHeight(20)
            self.thresh_edit.setFixedWidth(140)
            thresh_layout = QHBoxLayout()
            thresh_layout.addStretch(1)
            thresh_layout.addWidget(self.thresh_edit)
            thresh_layout.addStretch(1)
            self.addWidget(thresh_label)
            self.addLayout(thresh_layout)
            self.button_layout.addWidget(self.apply_button)
            self.button_layout.addSpacing(5)
            self.addLayout(self.button_layout)
            self.apply_button.mousePressEvent = lambda e: \
                self.apply_signal.emit((self.thresh_edit.text(), ))
            self.addStretch(1)

    class SmoothSettings(ItemSettings):

        def __init__(self, settings_type, settings):
            super().__init__('Boundary Smoothing')
            if settings == ():
                self.ox = ''
                self.oy = ''
                self.cx = ''
                self.cy = ''
                self.gx = ''
                self.gy = ''
            else:

                (
                    self.ox,
                    self.oy,
                    self.cx,
                    self.cy,
                    self.gx,
                    self.gy,
                    ) = settings
            open_label = QLabel('Open: ')
            open_label.setFixedHeight(20)
            open_label.setObjectName('setting-name')
            close_label = QLabel('Close: ')
            close_label.setFixedHeight(20)
            close_label.setObjectName('setting-name')
            gaus_label = QLabel('Gaussian: ')
            gaus_label.setFixedHeight(20)
            gaus_label.setObjectName('setting-name')
            self.open_x = QLineEdit(self.ox)
            self.open_x.setFixedHeight(20)
            self.open_x.setFixedWidth(30)
            self.open_y = QLineEdit(self.oy)
            self.open_y.setFixedHeight(20)
            self.open_y.setFixedWidth(30)
            self.close_x = QLineEdit(self.cx)
            self.close_x.setFixedHeight(20)
            self.close_x.setFixedWidth(30)
            self.close_y = QLineEdit(self.cy)
            self.close_y.setFixedHeight(20)
            self.close_y.setFixedWidth(30)
            self.gaus_x = QLineEdit(self.gx)
            self.gaus_x.setFixedHeight(20)
            self.gaus_x.setFixedWidth(30)
            self.gaus_y = QLineEdit(self.gy)
            self.gaus_y.setFixedHeight(20)
            self.gaus_y.setFixedWidth(30)
            open_layout = QHBoxLayout()
            open_layout.addStretch(1)
            b1 = QLabel('(')
            b1.setStyleSheet(self.setting_name_SS)
            open_layout.addWidget(b1)
            open_layout.addWidget(self.open_x)
            c1 = QLabel(',')
            c1.setStyleSheet(self.setting_name_SS)
            open_layout.addWidget(c1)
            open_layout.addWidget(self.open_y)
            b2 = QLabel(')')
            b2.setStyleSheet(self.setting_name_SS)
            open_layout.addWidget(b2)
            open_layout.addStretch(1)
            close_layout = QHBoxLayout()
            close_layout.addStretch(1)
            b3 = QLabel('(')
            b3.setStyleSheet(self.setting_name_SS)
            close_layout.addWidget(b3)
            close_layout.addWidget(self.close_x)
            c2 = QLabel(',')
            c2.setStyleSheet(self.setting_name_SS)
            close_layout.addWidget(c2)
            close_layout.addWidget(self.close_y)
            b4 = QLabel(')')
            b4.setStyleSheet(self.setting_name_SS)
            close_layout.addWidget(b4)
            close_layout.addStretch(1)
            gaus_layout = QHBoxLayout()
            gaus_layout.addStretch(1)
            b5 = QLabel('(')
            b5.setStyleSheet(self.setting_name_SS)
            gaus_layout.addWidget(b5)
            gaus_layout.addWidget(self.gaus_x)
            c3 = QLabel(',')
            c3.setStyleSheet(self.setting_name_SS)
            gaus_layout.addWidget(c3)
            gaus_layout.addWidget(self.gaus_y)
            b6 = QLabel(')')
            b6.setStyleSheet(self.setting_name_SS)
            gaus_layout.addWidget(b6)
            gaus_layout.addStretch(1)
            self.addWidget(open_label)
            self.addLayout(open_layout)
            self.addWidget(close_label)
            self.addLayout(close_layout)
            self.addWidget(gaus_label)
            self.addLayout(gaus_layout)
            self.button_layout.addWidget(self.apply_button)
            self.button_layout.addSpacing(5)
            self.addLayout(self.button_layout)
            self.apply_button.mousePressEvent = lambda e: \
                self.apply_signal.emit((
                    self.open_x.text(),
                    self.open_y.text(),
                    self.close_x.text(),
                    self.close_y.text(),
                    self.gaus_x.text(),
                    self.gaus_y.text(),
                    ))
            self.addStretch(1)

    class ComparisonSettings(ItemSettings):

        def __init__(self, settings_type, settings):
            super().__init__('Contour Comparison')

            (self.target_index, self.target_name,
             self.included_items) = settings
            type_label = QLabel('Compare by: ')
            type_label.setFixedHeight(20)
            type_label.setObjectName('setting-name')
            self.comparison_type = QComboBox()
            self.comparison_type.addItems(['Corresponding points',
                    'Closest points'])
            self.comparison_type.setFixedHeight(20)
            self.comparison_type.setFixedWidth(140)
            lines_layout = QHBoxLayout()
            lines_layout.setContentsMargins(0, 0, 0, 0)
            self.lines_check = QCheckBox()
            self.lines_check.setCheckState(False)
            lines_label = QLabel('Show lines: ')
            lines_label.setObjectName('setting-name')
            lines_layout.addStretch(1)
            lines_layout.addWidget(lines_label)
            lines_layout.addStretch(1)
            lines_layout.addWidget(self.lines_check)
            lines_layout.add
            compare_label = QLabel('Compare to:')
            compare_label.setFixedHeight(20)
            compare_label.setObjectName('setting-name')
            self.item_list = SidebarList(self.included_items, False)
            self.addWidget(type_label)
            self.addWidget(self.comparison_type)
            self.addLayout(lines_layout)
            self.addWidget(compare_label)
            self.addWidget(self.item_list)
            self.button_layout.addWidget(self.apply_button)
            self.button_layout.addSpacing(5)
            self.addLayout(self.button_layout)
            self.apply_button.mousePressEvent = lambda e: \
                self.applyComparison()
            self.addStretch(1)

        def applyComparison(self):
            if self.item_list.selected_item != '':
                self.apply_signal.emit((self.item_list.selected_item,
                        self.comparison_type.currentText(),
                        self.lines_check.isChecked()))

    class SearchSettings(ItemSettings):

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
        default_SS = \
            """
            padding: 2px;
            color: #777777;
            border: none;
            border-left: 1px ridge #515151;
        """
        highlight_SS = \
            """
            background-color: #000000;
            padding: 2px;
            color: #a8a8a8;
            border: none;
            border-left: 1px ridge #515151;
        """

        def __init__(self, settings_type, settings):
            super().__init__('Hand Search')
            (self.hand_name, self.hand_index, self.included_items) = \
                settings
            self.hand_label = QLabel(self.hand_name)
            self.hand_label.setFixedHeight(20)
            self.hand_label.setStyleSheet(self.default_SS)
            self.search_hand_select = True
            self.hand_label.mousePressEvent = lambda e: \
                self.searchHandSelect()
            similar_label = QLabel('Similar hands: ')
            similar_label.setFixedHeight(20)
            similar_label.setObjectName('setting-name')
            self.item_list = SidebarList(self.included_items, True)
            self.item_list.item_selected.connect(lambda i: \
                    self.listItemSelect(i))
            self.addWidget(self.hand_label)
            self.addWidget(similar_label)
            self.addWidget(self.item_list)
            self.addStretch(1)

        def listItemSelect(self, index):
            if self.search_hand_select is True:
                self.searchHandSelect()
            self.apply_signal.emit((index, ))

        def searchHandSelect(self):
            if self.search_hand_select == True:
                self.search_hand_select = False
                self.hand_label.setStyleSheet(self.default_SS)
            else:
                self.search_hand_select = True
                self.hand_label.setStyleSheet(self.highlight_SS)
                self.apply_signal.emit((self.hand_index, ))
                self.item_list.itemSelected(-1)

    class GraphSettings(ItemSettings):

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

        def __init__(self, settings_type, settings):
            super().__init__('Cluster Graph')
            (self.hand_name, self.hand_index, self.included_items) = \
                settings
            self.select_hands_button = QLabel('Select hands')
            self.select_hands_button.setStyleSheet(self.button_default_SS)
            self.select_hands_button.setFixedHeight(20)
            self.select_hands_button.setFixedWidth(80)
            self.select_highlight = False
            cluster_label = QLabel('Cluster hands:')
            cluster_label.setFixedHeight(20)
            cluster_label.setObjectName('setting-name')
            self.item_list = SidebarList(self.included_items, False)
            self.select_hands_button.mousePressEvent = lambda e: \
                self.multipleSelection()
            self.apply_button.mousePressEvent = lambda e: \
                self.applyGraph()
            self.addWidget(cluster_label)
            self.addWidget(self.item_list)
            self.button_layout.addStretch(1)
            self.button_layout.addWidget(self.select_hands_button)
            self.button_layout.addWidget(self.apply_button)
            self.button_layout.addSpacing(5)
            self.addLayout(self.button_layout)
            self.addStretch(1)

        def applyGraph(self):
            if self.item_list.selected_images != []:
                self.multipleSelection()
                self.apply_signal.emit((self.item_list.selected_images,
                        ))

        def multipleSelection(self):
            if self.select_highlight == False:
                self.select_hands_button.setStyleSheet(self.button_selected_SS)
                self.select_highlight = True
                self.item_list.setMultipleSelection(True)
            else:
                self.select_hands_button.setStyleSheet(self.button_default_SS)
                self.select_highlight = False
                self.item_list.setMultipleSelection(False)

    apply_signal = pyqtSignal(tuple)
    sidebar_SS = \
        """
        border-left: 1px ridge #515151;
        background: #212121;
    """

    def __init__(self, item_name, item_index):
        super().__init__()
        self.setStyleSheet(self.sidebar_SS)
        self.item_index = item_index
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.settings_stack = QStackedWidget()
        self.layout.addWidget(self.settings_stack)
        self.num_settings = 0

    def addItemSettings(self, settings_type, settings):
        settings_widget = QWidget()
        layout = ''
        if settings_type == 'Line':
            layout = self.StupidLineSettings(settings_type, settings)
        elif settings_type == 'CLAHE':
            layout = self.CLAHESettings(settings_type, settings)
        elif settings_type == 'Normalize':
            layout = self.NormalizationSettings(settings_type, settings)
        elif settings_type == 'Binary':
            layout = self.ThresholdSettings(settings_type, settings)
        elif settings_type == 'Smooth':
            layout = self.SmoothSettings(settings_type, settings)
        elif settings_type == 'Comparison':
            layout = self.ComparisonSettings(settings_type, settings)
        elif settings_type == 'Search':
            layout = self.SearchSettings(settings_type, settings)
        elif settings_type == 'Graph':
            layout = self.GraphSettings(settings_type, settings)
        else:
            layout = self.NoItemSettings(settings_type)
        layout.apply_signal.connect(lambda s: \
                                    self.apply_signal.emit((settings_type,
                                    self.item_index, s)))
        settings_widget.setLayout(layout)
        self.settings_stack.insertWidget(self.num_settings,
                settings_widget)
        self.settings_stack.setCurrentWidget(settings_widget)
        self.num_settings += 1

    def showItemSettings(self, index):
        self.settings_stack.setCurrentIndex(index)


class StepBar(QWidget):

    step_signal = pyqtSignal(int, QPixmap, int)

    def __init__(self, image_index):
        super().__init__()
        self.setObjectName('step-view')

        self.image_steps = [0 for i in range(99)]
        self.num_steps = 0
        self.image_index = image_index
        self.step_layout = QHBoxLayout()
        self.step_layout.setContentsMargins(5, 5, 5, 5)
        self.step_layout.setSpacing(2)
        self.step_view_layout = QHBoxLayout()
        self.step_view_layout.setContentsMargins(0, 0, 0, 0)
        self.step_view_layout.setSpacing(0)
        self.setLayout(self.step_view_layout)
        self.step_widget = QWidget()
        self.step_widget.setLayout(self.step_layout)
        self.step_view_layout.addWidget(self.step_widget)
        self.step_view_layout.addStretch()
        self.error_step_index = ''
        self.error_step = ''

    def addStep(self, image_data):
        step_label = QLabel()
        step_label.setObjectName('step')
        image = image_data.toImage().scaledToHeight(75)
        step_pixmap = QPixmap.fromImage(image)
        step_label.setPixmap(step_pixmap)
        self.step_layout.addWidget(step_label)
        step_index = self.num_steps + 0
        step_label.mousePressEvent = lambda e: \
            self.stepSelected(step_index, image_data, step_label)
        self.image_steps[self.num_steps] = step_label
        self.num_steps += 1

    def addErrorStep(self, image_data):
        step_label = QLabel()
        step_label.setObjectName('error-step')
        image = image_data.toImage().scaledToHeight(75)
        step_pixmap = QPixmap.fromImage(image)
        step_label.setPixmap(step_pixmap)
        self.step_layout.addWidget(step_label)
        step_index = self.num_steps + 0
        self.error_step_index = step_index
        self.error_step = step_label
        step_label.mousePressEvent = lambda e: \
            self.stepSelected(step_index, image_data, step_label)

    def stepSelected(
        self,
        step_index,
        image_data,
        step,
        ):

        if step_index != self.error_step_index:

            # step.setStyleSheet(self.step_selected_SS)

            pass
        self.step_signal.emit(self.image_index, image_data, step_index)

    def undoSteps(self, step_count, all):
        if self.error_step_index != '':
            self.error_step_index = ''
            self.step_layout.removeWidget(self.error_step)
            self.error_step.setVisible(False)
            self.error_step = ''
        if all == False:
            index = self.num_steps - 1
            for i in range(step_count):
                step = self.image_steps[index]
                if step != 0:
                    self.step_layout.removeWidget(step)
                    step.setVisible(False)
                    self.num_steps -= 1
                index -= 1
        else:
            for i in range(self.num_steps):
                step = self.image_steps[i]
                self.step_layout.removeWidget(step)
                step.setVisible(False)
            self.num_steps = 0


class ImageViewer(QGraphicsView):

    button_default_SS = \
        """
            border: 1px solid #404041;
        """

    def __init__(self, pixmap):
        super().__init__()
        self.setObjectName('view')
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.image_scene = QGraphicsScene()
        self.current_image_item = QGraphicsPixmapItem()
        self.current_image_pixmap = QPixmap()
        self.image_scene.addItem(self.current_image_item)
        self.setScene(self.image_scene)
        self.changeCurrentImage(pixmap)

    def changeCurrentImage(self, image_pixmap):
        self.image_scene = QGraphicsScene()
        self.setScene(self.image_scene)
        self.current_image_pixmap = image_pixmap
        self.current_image_item = QGraphicsPixmapItem(image_pixmap)
        self.current_image_item.setTransformationMode(Qt.SmoothTransformation)
        self.image_scene.addItem(self.current_image_item)
        self.centerOn(self.current_image_item)
        self.fitInView(self.current_image_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.scale(1.25, 1.25)
        else:
            self.scale(0.8, 0.8)
