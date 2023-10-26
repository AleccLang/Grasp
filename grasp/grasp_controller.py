from grasp.grasp_engine import GraspEngine
from grasp.grasp_gui.image_view import ItemSideBar
from grasp.image_item import ImageItem
from grasp.grasp_gui.main_window import MainWindow
from grasp.action_controller import ActionController
from PyQt5.QtCore import QRunnable, QThread, pyqtSignal, pyqtSlot, \
    QThreadPool, QObject
import copy
import numpy as np
import cv2
import math


class PipelineWorkerSignals(QObject):

    """
    Provides a method of communication between the controller and worker threads.
    """

    pipeline_finished = pyqtSignal(list, int, str)


class UpdatePipelineWorker(QRunnable):

    """
    A worker thread used to execute a pipeline with custom settings
    
    ...
    
    Methods
    -------
    run():
        Executes the pipeline with the custom settings.
    """

    def __init__(
        self,
        previous_data,
        item_index,
        step_index,
        custom_settings,
        ):

        super().__init__()
        self.previous_data = previous_data
        self.signals = PipelineWorkerSignals()
        self.step_index = step_index
        self.custom_settings = custom_settings
        self.item_index = item_index
        self.current_step_index = 0

    def run(self):
        data = []
        try:
            if self.step_index == 1:
                original_image = self.previous_data.original_image
                (no_stupid_line, settings) = \
                    GraspEngine.removeStupidLine(original_image,
                        self.custom_settings)
                no_stupid_line = \
                    GraspEngine.convertToUniformSize(no_stupid_line)
                data.append(('Line', no_stupid_line, settings, ()))
                self.current_step_index = 2
            if self.step_index == 2 or self.current_step_index == 2:
                settings = ()
                if self.current_step_index != 2:
                    settings = self.custom_settings
                    no_stupid_line = \
                        self.previous_data.processing_steps['Line'
                            ].image_data
                (normalized, settings) = \
                    GraspEngine.justNormalization(no_stupid_line,
                        settings)
                data.append(('Normalize', normalized, settings, ()))
                self.current_step_index = 3
            if self.step_index == 3 or self.current_step_index == 3:
                settings = ()
                if self.current_step_index != 3:
                    settings = self.custom_settings
                    normalized = \
                        self.previous_data.processing_steps['Normalize'
                            ].image_data
                (binary, settings) = \
                    GraspEngine.imageThreshold(normalized, settings)
                data.append(('Binary', binary, settings, ()))

                (components, analysis, area, hand_component,
                 hand_component_centroid) = \
                    GraspEngine.connectedComponents(binary)
                data.append(('Components', components, (), analysis))
                data.append(('Component', hand_component, (), (area,
                            hand_component_centroid)))
                self.current_step_index = 4
            if self.step_index == 4 or self.current_step_index == 4:
                settings = ()
                if self.current_step_index != 4:
                    settings = self.custom_settings
                    hand_component = \
                        self.previous_data.processing_steps['Component'
                            ].image_data
                (smooth, settings) = \
                    GraspEngine.smoothEdges(hand_component, settings)
                data.append(('Smooth', smooth, settings, ()))
                self.current_step_index = 5
            if self.step_index == 5 or self.current_step_index == 5:
                settings = ()
                if 1 < self.step_index < 5:
                    no_stupid_line = \
                        self.previous_data.processing_steps['Line'
                            ].image_data
                if self.step_index == 5:
                    smooth = \
                        self.previous_data.processing_steps['Smooth'
                            ].image_data
                    no_stupid_line = \
                        self.previous_data.processing_steps['Line'
                            ].image_data
                if self.current_step_index != 5:
                    settings = self.custom_settings
                contours = GraspEngine.extractContours(smooth)
                (masked_clahe, settings) = \
                    GraspEngine.maskedCLAHE(no_stupid_line, contours,
                        settings)
                data.append(('CLAHE', masked_clahe, settings, ()))
                contour_clahe = \
                    GraspEngine.drawContourOverImage(masked_clahe.copy(),
                        contours)
                data.append(('Contour', contour_clahe, (), contours))
                (convex_image, defect_coordinates, defects) = \
                    GraspEngine.convexityDefects(contour_clahe,
                        contours)
                data.append(('Defect', convex_image, (),
                            (defect_coordinates, defects)))
                (finger_joins, wrist_defects) = \
                    GraspEngine.identifyDefects(defect_coordinates)
                (tip_image, finger_tips) = \
                    GraspEngine.identifyFingerTips(convex_image.copy(),
                        finger_joins, wrist_defects, contours)
                (wrist_image, wrist_coord) = \
                    GraspEngine.identifyAccurateWristPoint(tip_image.copy(),
                        contours, wrist_defects, finger_tips)
                wrist_defects['wrist-left'] = wrist_coord
                (centroid_image, palm_centroid) = \
                    GraspEngine.identifyPalmCentroid(wrist_image.copy(),
                        finger_joins, wrist_defects)
                vector_image = \
                    GraspEngine.identifyAlignmentVector(centroid_image.copy(),
                        finger_tips, finger_joins)
                data.append(('Features', vector_image, (),
                            (finger_joins, wrist_defects,
                            palm_centroid, finger_tips)))
                center_hand = \
                    GraspEngine.centerHand(vector_image.copy(),
                        palm_centroid)
                data.append(('Center', center_hand, (), ()))
                rotated_hand = GraspEngine.rotateHand(center_hand,
                        finger_joins, finger_tips)
                data.append(('Rotated', rotated_hand, (), ()))
                (vector_hand, similarity_vector) = \
                    GraspEngine.spikyHand(masked_clahe.copy(),
                        finger_joins, finger_tips, wrist_defects)
                centered_vector = \
                    GraspEngine.centerHand(vector_hand.copy(),
                        palm_centroid)
                rotated_vector = \
                    GraspEngine.rotateHand(centered_vector.copy(),
                        finger_joins, finger_tips)
                data.append(('Vector', rotated_vector, (),
                            similarity_vector))
                centered_clahe = \
                    GraspEngine.centerHand(masked_clahe.copy(),
                        palm_centroid)
                rotated_clahe = GraspEngine.rotateHand(centered_clahe,
                        finger_joins, finger_tips)
                data.append(('Final', rotated_clahe, (), ()))
            self.current_step_index = 1
            self.signals.pipeline_finished.emit(data, self.item_index,
                    'Success')
        except Exception as e:
            print(e)
            self.signals.pipeline_finished.emit(data, self.item_index,
                    'Failure')


class PipelineWorker(QRunnable):

    """
    A worker thread which execute the pipeline for a given image.
    
    ...
    
    Methods
    -------
    run():
        Executes the pipeline for the given image.
    """

    def __init__(
        self,
        original_image,
        item_index,
        name,
        ):

        super().__init__()
        self.item_index = item_index
        self.original_image = original_image
        self.signals = PipelineWorkerSignals()
        self.name = name

    def run(self):
        data = []
        try:
            uniform_original = \
                GraspEngine.convertToUniformSize(self.original_image)
            data.append(('Original', uniform_original, (), ()))
            (no_stupid_line, settings) = \
                GraspEngine.removeStupidLine(self.original_image, ())
            no_stupid_line = \
                GraspEngine.convertToUniformSize(no_stupid_line)
            data.append(('Line', no_stupid_line, settings, ()))
            (normalized, settings) = \
                GraspEngine.justNormalization(no_stupid_line, ())
            data.append(('Normalize', normalized, settings, ()))
            (binary, settings) = GraspEngine.imageThreshold(normalized,
                    ())
            data.append(('Binary', binary, settings, ()))

            (components, analysis, area, hand_component,
             hand_component_centroid) = \
                GraspEngine.connectedComponents(binary)
            data.append(('Components', components, (), analysis))
            data.append(('Component', hand_component, (), (area,
                        hand_component_centroid)))
            (smooth, settings) = \
                GraspEngine.smoothEdges(hand_component, ())
            data.append(('Smooth', smooth, settings, ()))
            contours = GraspEngine.extractContours(smooth)
            (masked_clahe, settings) = \
                GraspEngine.maskedCLAHE(no_stupid_line, contours, ())
            data.append(('CLAHE', masked_clahe, settings, ()))
            contour_clahe = \
                GraspEngine.drawContourOverImage(masked_clahe.copy(),
                    contours)
            data.append(('Contour', contour_clahe, (), contours))
            (convex_image, defect_coordinates, defects) = \
                GraspEngine.convexityDefects(contour_clahe, contours)
            data.append(('Defect', convex_image, (),
                        (defect_coordinates, defects)))
            (finger_joins, wrist_defects) = \
                GraspEngine.identifyDefects(defect_coordinates)
            (tip_image, finger_tips) = \
                GraspEngine.identifyFingerTips(convex_image.copy(),
                    finger_joins, wrist_defects, contours)
            (wrist_image, wrist_coord) = \
                GraspEngine.identifyAccurateWristPoint(tip_image.copy(),
                    contours, wrist_defects, finger_tips)
            wrist_defects['wrist-left'] = wrist_coord
            (centroid_image, palm_centroid) = \
                GraspEngine.identifyPalmCentroid(wrist_image.copy(),
                    finger_joins, wrist_defects)
            vector_image = \
                GraspEngine.identifyAlignmentVector(centroid_image.copy(),
                    finger_tips, finger_joins)
            data.append(('Features', vector_image, (), (finger_joins,
                        wrist_defects, palm_centroid, finger_tips)))
            center_hand = GraspEngine.centerHand(vector_image.copy(),
                    palm_centroid)
            data.append(('Center', center_hand, (), ()))
            rotated_hand = GraspEngine.rotateHand(center_hand,
                    finger_joins, finger_tips)
            data.append(('Rotated', rotated_hand, (), ()))
            (vector_hand, similarity_vector) = \
                GraspEngine.spikyHand(masked_clahe.copy(),
                    finger_joins, finger_tips, wrist_defects)
            centered_vector = \
                GraspEngine.centerHand(vector_hand.copy(),
                    palm_centroid)
            rotated_vector = \
                GraspEngine.rotateHand(centered_vector.copy(),
                    finger_joins, finger_tips)
            data.append(('Vector', rotated_vector, (),
                        similarity_vector))
            centered_clahe = \
                GraspEngine.centerHand(masked_clahe.copy(),
                    palm_centroid)
            rotated_clahe = GraspEngine.rotateHand(centered_clahe,
                    finger_joins, finger_tips)
            data.append(('Final', rotated_clahe, (), ()))
            self.signals.pipeline_finished.emit(data, self.item_index,
                    'Success')
        except Exception as e:
            print(e)
            self.signals.pipeline_finished.emit(data, self.item_index,
                    'Failure')


class GraspController:

    """
    The main controller class for the system, acts as the link between all other components.
    Receives signals from the user interface and acts accordingly.
    Executes pipelines using worker threads and engine.
    
    ...
    
    Methods
    -------
    applyProcessingPipeline(included_items, settings):
        Creates and starts worker threads to apply the pipeline to the given list of images using the given settings.

    onPipelineComplete(data, item_index, outcome, action):
        Called when a worker thread has finished executing a pipeline, stores the resulting data and sends it to the ui.

    onAllPipelinesComplete():
        Called when a group of worker threads have all finished executing their pipelines, stores resulting data, sends to ui, and makes thread pool available.

    applyCustomSettings(type, item_index, custom_settings):
        Creates a worker thread to execute a pipeline with the users given custom settings.

    onCustomPipelineComplete(data, item_index, outcome, type, action):
        Called when a worker thread has finished executing a pipeline with users custom settings, stores results and sends to ui.

    createGraphItem():
        Creates a graph item.

    createClusterGraph(item_index, base_index, graph_indexes):
        Creates a cluster graph using the given item indexes.

    createSearchItem():
        Computes comparison data for selected hand using engine, sends data to ui.

    createPipelineDeliverables():
        Called when a group of pipelines has finished executing, creates alignment accuracy image and send to ui.

    createComparisonItem():
        Creates a contour comparison item for the selected hand.

    generateContourComparison(comparison_index, comparison_type, item_index, lines):
        Generates and draws comparison contour for given index and sends to ui.

    onProcessedStep(item_index, process, processed_data, settings):
        Called when a processing step has been executed, sends data to ui.

    addImages(images):
        Creates and stores image items for given images.

    exportProcessedImages():
        Exports all images that have been processed.

    undoLastAction():
        Undoes the last processing action made by the user.

    redoLastUndo():
        Redoes the last action the user has undone.

    """

    def __init__(self):
        self.num_items = 0
        self.items = [0 for i in range(999)]
        self.image_indexes = []
        self.deliverable_indexes = []
        self.error_indexes = []
        self.processed_indexes = []

        self.action_history = []
        self.current_action = -1
        self.main_window = MainWindow()
        self.main_window.export_sig.connect(lambda n: \
                self.exportProcessedImages(n))

        self.explorer = self.main_window.explorer
        self.item_tab = self.main_window.item_tab_widget
        self.item_widgets = self.main_window.item_widgets
        self.stepbar_stack = self.main_window.stepbar_stack

        self.processing_menu = self.main_window.processing_menu
        self.comparison_menu = self.main_window.comparison_menu
        self.search_menu = self.main_window.search_menu
        self.graph_menu = self.main_window.graph_menu

        self.control_buttons = self.main_window.control_buttons
        self.main_window.import_sig.connect(lambda i: self.addImages(i))
        self.main_window.custom_settings_sig.connect(lambda s: \
                self.applyCustomSettings(s[0], s[1], s[2]))

        self.processing_menu.apply_signal.connect(lambda s: \
                self.applyProcessingPipeline(self.explorer.selected_images,
                s))
        self.comparison_menu.apply_signal.connect(lambda s: \
                self.createComparisonItem())
        self.search_menu.apply_signal.connect(lambda s: \
                self.createSearchItem())
        self.graph_menu.apply_signal.connect(lambda s: \
                self.createGraphItem())

        self.control_buttons.undo_signal.connect(lambda : \
                self.undoLastAction())
        self.control_buttons.redo_signal.connect(lambda : \
                self.redoLastUndo())

        global_pool = QThreadPool().globalInstance()
        global_max = global_pool.maxThreadCount()
        worker_max = global_max / 2
        self.worker_pool = QThreadPool()
        self.worker_pool.setMaxThreadCount(math.ceil(worker_max))
        self.active_pipelines = 0
        self.pipeline_group_results = []
        self.pipeline_group_indexes = []
        self.success_indexes = []
        self.success_results = []
        self.pipeline_data = []

        self.num_pipelines = 1

        self.action_controller = ActionController()

    def applyProcessingPipeline(self, included_items, settings):
        """Creates and starts worker threads to apply the pipeline to the given list of images using the given settings.
        
        Parameters
        ----------
        included_items: List of items to which the processing should be applied to.
        settings: Contains the settings which should be applied."""

        self.pipeline_group_results = []
        self.pipeline_group_indexes = []
        self.success_indexes = []
        self.pipeline_data = []
        for index in included_items:
            self.explorer.items[index].setLoadAnimation(True)
            pipeline_worker = \
                PipelineWorker(copy.deepcopy(self.items[index].original_image),
                               index, self.items[index].image_name)
            pipeline_worker.signals.pipeline_finished.connect(lambda d, \
                    i, o: self.onPipelineComplete(d, i, o, False))
            self.worker_pool.start(pipeline_worker)
            self.active_pipelines = self.active_pipelines + 1

    def onPipelineComplete(
        self,
        data,
        item_index,
        outcome,
        action,
        ):
        """Called when a worker thread has finished executing a pipeline, stores the resulting data and sends it to the ui.
        
        Parameters
        ----------
        data: Contains the resulting data of the pipeline.
        item_index: Identifies which item the pipeline was applied to.
        outcome: Indicates if the pipeline was applied successfully or not."""

        self.pipeline_data.append((data, item_index, outcome, action))
        if action is False:
            self.active_pipelines = self.active_pipelines - 1
        for step in data:
            (process, result, settings, additional) = step
            self.items[item_index].addProcessingStep(process, result,
                    settings, additional)
            self.onProcessedStep(item_index, process, result, settings)
        self.pipeline_group_results.append(self.items[item_index])
        self.pipeline_group_indexes.append(item_index)
        self.explorer.items[item_index].setLoadAnimation(False)
        if outcome == 'Success':
            self.success_indexes.append(item_index)
            self.success_results.append(self.items[item_index])
            if item_index in self.error_indexes:
                self.error_indexes.pop(self.error_indexes.index(item_index))
            if item_index not in self.processed_indexes:
                self.processed_indexes.append(item_index)
                self.explorer.addProcessedItem(item_index)
        else:
            if item_index not in self.error_indexes:
                self.error_indexes.append(item_index)
            self.explorer.addInvalidItem(item_index)
        if self.active_pipelines == 0 and action is False:
            self.onAllPipelinesComplete()

    def onAllPipelinesComplete(self):
        """Called when a group of worker threads have all finished executing their pipelines, stores resulting data, sends to ui, and makes thread pool available."""

        indexes = []
        items = []
        for (i, index) in enumerate(self.pipeline_group_indexes):
            indexes.append(('Hand', index))
            items.append(self.pipeline_data[i])

        self.createPipelineDeliverables(self.success_results, indexes,
                items, False)
        self.pipeline_group_indexes = []
        self.pipeline_group_results = []
        self.success_indexes = []
        self.success_results = []

    def applyCustomSettings(
        self,
        type,
        item_index,
        custom_settings,
        ):
        """Creates a worker thread to execute a pipeline with the users given custom settings.
        
        Parameters
        ----------
        type : Identifies which custom settings were applied.
        item_index: Identifies when item the settings were applied to.
        custom_settings: Contains the settings which were applied."""

        if type == 'Line':
            step_index = 1
        elif type == 'Normalize':
            step_index = 2
        elif type == 'Binary':
            step_index = 3
        elif type == 'Smooth':
            step_index = 4
        elif type == 'CLAHE':
            step_index = 4
        self.explorer.items[item_index].setLoadAnimation(True)
        previous_data = self.items[item_index]
        pipeline_worker = UpdatePipelineWorker(previous_data,
                item_index, step_index, custom_settings)
        pipeline_worker.signals.pipeline_finished.connect(lambda d, i, \
                o: self.onCustomPipelineComplete(d, i, o, type, False))
        self.worker_pool.start(pipeline_worker)

    def onCustomPipelineComplete(
        self,
        data,
        item_index,
        outcome,
        type,
        action,
        ):
        """Called when a worker thread has finished executing a pipeline with users custom settings, stores results and sends to ui.
        
        Parameters
        ----------
        data: Contains the resulting data of the pipelines.
        item_index: contains the index of the alignment accuracy item.
        outcome: Indicates if the pipeline was sucessfully executed or not."""

        if action is False:
            self.action_controller.recordNewAction('Custom', [('Hand',
                    item_index)], [(data, outcome, type)])
        step_count = 0
        if type == 'Line':
            step_count = 14
        elif type == 'Normalize':
            step_count = 13
        elif type == 'Binary':
            step_count = 12
        elif type == 'Smooth':
            step_count = 9
        elif type == 'CLAHE':
            step_count = 8
        (viewer, stepbar, sidebar) = self.item_widgets[item_index]
        process_names = [
            'Original',
            'Line',
            'Uniform',
            'Normalize',
            'Binary',
            'Components',
            'Component',
            'Smooth',
            'CLAHE',
            'Contour',
            'Hull',
            'Defect',
            'Features',
            'Align',
            'Center',
            'Roted',
            'Final',
            ]
        stepbar.undoSteps(0, False)
        if outcome == 'Success':
            if item_index in self.processed_indexes:
                stepbar.undoSteps(step_count, False)
            else:
                stepbar.undoSteps(stepbar.num_steps - (15
                                  - step_count), False)
            if item_index not in self.processed_indexes:
                self.processed_indexes.append(item_index)
                self.explorer.addProcessedItem(item_index)
            if item_index in self.error_indexes:
                self.error_indexes.pop(self.error_indexes.index(item_index))
                self.explorer.removeInvalidItem(item_index)
        else:
            if item_index in self.processed_indexes:
                stepbar.undoSteps(step_count, False)
                self.processed_indexes.pop(self.processed_indexes.index(item_index))
                self.explorer.removedProcessedItem(item_index)
            else:
                stepbar.undoSteps(len(data), False)
            if item_index not in self.error_indexes:
                self.error_indexes.append(item_index)
                self.explorer.addInvalidItem(item_index)
        last_process = type
        for step in data:
            (process, result, settings, additional) = step
            self.items[item_index].addProcessingStep(process, result,
                    settings, additional)
            self.onProcessedStep(item_index, process, result, settings)
            last_process = process
        self.explorer.items[item_index].setLoadAnimation(False)
        if outcome == 'Failure':
            stepbar.addErrorStep(GraspEngine.convert_cv_qt(np.zeros((2500,
                                 2500, 3)).astype(np.uint8)))
            error_process = \
                process_names[process_names.index(last_process) + 1]
            sidebar.addItemSettings(error_process, ())

    def createGraphItem(self):
        """Creates a graph item."""

        base_index = self.explorer.selected_item
        base_name = self.items[base_index].image_name
        base_image = self.items[base_index].processing_steps['Final'
                ].image_data
        item_index = self.num_items + 0
        graph_sidebar = ItemSideBar(base_name, item_index)
        items = []
        for index in self.image_indexes:
            if index != base_index and index not in self.error_indexes:
                items.append((self.items[index].image_name, index))
        graph_sidebar.addItemSettings('Graph', (base_name, base_index,
                items))
        self.main_window.addDeliverable(GraspEngine.convert_cv_qt(base_image),
                base_name + ' cluster', graph_sidebar)
        graph_sidebar.apply_signal.connect(lambda s: \
                self.createClusterGraph(item_index, base_index,
                s[2][0]))
        self.num_items += 1

    def createClusterGraph(
        self,
        item_index,
        base_index,
        graph_indexes,
        ):
        """Creates a cluster graph using the given item indexes.
        
        Parameters
        ----------
        item_index: Identifies the graph item to show the graph in.
        base_index: Identifies the index of the base hand for the graph.
        graph_indexes: Identiefies the other hands to be included in the graph."""

        hands = []
        for index in graph_indexes:
            if index != base_index and index not in self.error_indexes:
                hand_component = \
                    self.items[index].processing_steps['Smooth'
                        ].image_data
                palm_centroid = \
                    self.items[index].processing_steps['Features'
                        ].palm_centroid
                finger_joins = \
                    self.items[index].processing_steps['Features'
                        ].finger_joins
                finger_tips = \
                    self.items[index].processing_steps['Features'
                        ].finger_tips
                center_hand = GraspEngine.centerHand(hand_component,
                        palm_centroid)
                component = GraspEngine.rotateHand(center_hand,
                        finger_joins, finger_tips)
                spiky = self.items[index].processing_steps['Vector'
                        ].similarity_vector
                wrist_defects = \
                    self.items[index].processing_steps['Features'
                        ].wrist_defects
                hands.append((index, (component.astype(np.uint8),
                             spiky, wrist_defects)))
        ideal_hand_component = \
            self.items[base_index].processing_steps['Smooth'].image_data
        ideal_palm_centroid = \
            self.items[base_index].processing_steps['Features'
                ].palm_centroid
        ideal_finger_joins = \
            self.items[base_index].processing_steps['Features'
                ].finger_joins
        ideal_finger_tips = \
            self.items[base_index].processing_steps['Features'
                ].finger_tips
        ideal_center_hand = \
            GraspEngine.centerHand(ideal_hand_component,
                                   ideal_palm_centroid)
        ideal_component = GraspEngine.rotateHand(ideal_center_hand,
                ideal_finger_joins, ideal_finger_tips)
        ideal_spiky = self.items[base_index].processing_steps['Vector'
                ].similarity_vector
        ideal_wrist_defects = \
            self.items[base_index].processing_steps['Features'
                ].wrist_defects
        datapoints = GraspEngine.handComp(hands,
                (ideal_component.astype(np.uint8), ideal_spiky,
                ideal_wrist_defects))
        items = []
        for point in datapoints:
            items.append((point[1][0], point[1][1]))
        graph = GraspEngine.kMeans(items)
        self.main_window.item_widgets[item_index][0].changeCurrentImage(GraspEngine.convert_cv_qt(graph))

    def createSearchItem(self):
        """Computes comparison data for selected hand using engine, sends data to ui."""

        search_index = self.explorer.selected_item
        search_name = self.items[search_index].image_name
        search_image = self.items[search_index].processing_steps['Final'
                ].image_data
        item_index = self.num_items + 0
        search_sidebar = ItemSideBar(search_name, item_index)
        hands = []
        for index in self.image_indexes:
            if index != search_index and index \
                not in self.error_indexes:
                hand_component = \
                    self.items[index].processing_steps['Smooth'
                        ].image_data
                palm_centroid = \
                    self.items[index].processing_steps['Features'
                        ].palm_centroid
                finger_joins = \
                    self.items[index].processing_steps['Features'
                        ].finger_joins
                finger_tips = \
                    self.items[index].processing_steps['Features'
                        ].finger_tips
                center_hand = GraspEngine.centerHand(hand_component,
                        palm_centroid)
                component = GraspEngine.rotateHand(center_hand,
                        finger_joins, finger_tips)
                spiky = self.items[index].processing_steps['Vector'
                        ].similarity_vector
                wrist_defects = \
                    self.items[index].processing_steps['Features'
                        ].wrist_defects
                hands.append((index, (component.astype(np.uint8),
                             spiky, wrist_defects)))
        ideal_hand_component = \
            self.items[search_index].processing_steps['Smooth'
                ].image_data
        ideal_palm_centroid = \
            self.items[search_index].processing_steps['Features'
                ].palm_centroid
        ideal_finger_joins = \
            self.items[search_index].processing_steps['Features'
                ].finger_joins
        ideal_finger_tips = \
            self.items[search_index].processing_steps['Features'
                ].finger_tips
        ideal_center_hand = \
            GraspEngine.centerHand(ideal_hand_component,
                                   ideal_palm_centroid)
        ideal_component = GraspEngine.rotateHand(ideal_center_hand,
                ideal_finger_joins, ideal_finger_tips)
        ideal_spiky = self.items[search_index].processing_steps['Vector'
                ].similarity_vector
        ideal_wrist_defects = \
            self.items[search_index].processing_steps['Features'
                ].wrist_defects
        datapoints = GraspEngine.handComp(hands,
                (ideal_component.astype(np.uint8), ideal_spiky,
                ideal_wrist_defects))
        items = []
        for point in datapoints:
            perc = '{:.0%}'.format(abs(2 - (point[1][1] + point[1][0])))
            items.append((self.items[point[0]].image_name, point[0],
                         str(perc)))
        search_sidebar.addItemSettings('Search', (search_name,
                search_index, items))
        self.main_window.addDeliverable(GraspEngine.convert_cv_qt(search_image),
                search_name + ' search', search_sidebar)
        search_sidebar.apply_signal.connect(lambda s: \
                self.main_window.item_widgets[item_index][0].changeCurrentImage(GraspEngine.convert_cv_qt(self.items[s[2][0]].processing_steps['Final'
                ].image_data)))
        self.num_items += 1

    def createPipelineDeliverables(
        self,
        results,
        indexes,
        items,
        action,
        ):
        """Called when a group of pipelines has finished executing, creates alignment accuracy image and send to ui.
        
        Parameters
        ----------
        results: Contains the resulting data of all processed pipelines.
        indexes: Contains the item indexes for which the processing was applied to.
        items: Contains the items for the hands being processed.
        action: identifies if this is being called as a result of a user action or not."""

        components = []
        for result in results:
            hand_component = result.processing_steps['Smooth'
                    ].image_data
            palm_centroid = result.processing_steps['Features'
                    ].palm_centroid
            finger_joins = result.processing_steps['Features'
                    ].finger_joins
            finger_tips = result.processing_steps['Features'
                    ].finger_tips
            center_hand = GraspEngine.centerHand(hand_component,
                    palm_centroid)
            rotated_hand = GraspEngine.rotateHand(center_hand,
                    finger_joins, finger_tips)
            components.append(rotated_hand)
        new_indexes = indexes
        new_items = items
        new_indexes.append(('Accuracy', self.num_items))
        alignment_accuracy = \
            GraspEngine.showAlignmentAccuracy(components)
        new_items.append((results, indexes, items, action))
        if action is False:
            self.action_controller.recordNewAction('Pipeline',
                    new_indexes, new_items)
        self.items[self.num_items] = alignment_accuracy
        sidebar = ItemSideBar('weeee', self.num_items)
        self.main_window.addDeliverable(GraspEngine.convert_cv_qt(alignment_accuracy),
                'Pipeline ' + str(self.num_pipelines) + ' accuracy',
                sidebar)
        self.num_pipelines += 1
        self.deliverable_indexes.append(self.num_items)
        self.num_items += 1

    def createComparisonItem(self):
        """Creates a contour comparison item for the selected hand."""

        target_index = self.explorer.selected_item
        target_name = self.items[target_index].image_name
        temp_image = np.zeros((2500, 2500, 3)).astype(np.uint8)
        hand_component = \
            self.items[target_index].processing_steps['Smooth'
                ].image_data
        palm_centroid = \
            self.items[target_index].processing_steps['Features'
                ].palm_centroid
        finger_joins = \
            self.items[target_index].processing_steps['Features'
                ].finger_joins
        finger_tips = \
            self.items[target_index].processing_steps['Features'
                ].finger_tips
        center_hand = GraspEngine.centerHand(hand_component,
                palm_centroid)
        rotated_hand = GraspEngine.rotateHand(center_hand,
                finger_joins, finger_tips)
        hand_contour = GraspEngine.extractContours(rotated_hand)
        (convex_image, defect_coordinates, defects) = \
            GraspEngine.convexityDefects(temp_image, hand_contour)
        (finger_joins, wrist_defects) = \
            GraspEngine.identifyDefects(defect_coordinates)
        (tip_image, finger_tips) = \
            GraspEngine.identifyFingerTips(convex_image.copy(),
                finger_joins, wrist_defects, hand_contour)
        (wrist_image, wrist_coord) = \
            GraspEngine.identifyAccurateWristPoint(tip_image.copy(),
                hand_contour, wrist_defects, finger_tips)
        wrist_defects['wrist-left'] = wrist_coord
        ordered_contour = \
            GraspEngine.arrangeContourPoints(hand_contour,
                wrist_defects)
        comparison_image = cv2.cvtColor(rotated_hand.astype(np.uint8),
                cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(comparison_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 1), (255, 255, 255))
        comparison_image1 = comparison_image.copy()
        comparison_image1[mask > 0] = (214, 194, 194)
        comparison_image = cv2.addWeighted(comparison_image, 0.1,
                comparison_image1, 0.1, 0)
        self.items[self.num_items] = (comparison_image.copy(),
                ordered_contour, finger_joins, wrist_defects,
                finger_tips)
        self.deliverable_indexes.append(self.num_items)
        items = []
        for index in self.image_indexes:
            if index != target_index and index \
                not in self.error_indexes:
                items.append((self.items[index].image_name, index))
        comparison_sidebar = ItemSideBar('huehue', self.num_items)
        comparison_sidebar.addItemSettings('Comparison',
                (self.items[target_index].image_name, target_index,
                items))
        index = self.num_items + 0
        comparison_sidebar.apply_signal.connect(lambda s: \
                self.generateContourComparison(s[2][0], s[2][1], index,
                s[2][2]))
        self.main_window.addDeliverable(GraspEngine.convert_cv_qt(comparison_image),
                target_name + ' contour', comparison_sidebar)
        self.num_items += 1

    def generateContourComparison(
        self,
        comparison_index,
        comparison_type,
        item_index,
        lines,
        ):
        """Generates and draws comparison contour for given index and sends to ui.
        
        Parameters
        ----------
        comparison_index: Contains the index of the hand being compared to.
        comparison_type: identifies the type of comparison to be made.
        item_index: Identifies the comparison item.
        lines: Boolean specifying if the engine should draw comparison lines or not."""

        (base_image, base_contour, base_ordered_finger_joins,
         base_wrist_defects, base_ordered_finger_tips) = \
            self.items[item_index]
        (viewer, sidebar) = self.item_widgets[item_index]
        base_image = base_image.copy()
        base_contour = base_contour.copy()
        hand_component = \
            self.items[comparison_index].processing_steps['Smooth'
                ].image_data
        palm_centroid = \
            self.items[comparison_index].processing_steps['Features'
                ].palm_centroid
        finger_joins = \
            self.items[comparison_index].processing_steps['Features'
                ].finger_joins
        finger_tips = \
            self.items[comparison_index].processing_steps['Features'
                ].finger_tips
        center_hand = GraspEngine.centerHand(hand_component,
                palm_centroid)
        rotated_hand = GraspEngine.rotateHand(center_hand,
                finger_joins, finger_tips)
        hand_contour = GraspEngine.extractContours(rotated_hand)
        temp_image = np.zeros((2500, 2500, 3)).astype(np.uint8)
        (convex_image, defect_coordinates, defects) = \
            GraspEngine.convexityDefects(temp_image, hand_contour)
        (finger_joins, wrist_defects) = \
            GraspEngine.identifyDefects(defect_coordinates)
        (tip_image, finger_tips) = \
            GraspEngine.identifyFingerTips(convex_image.copy(),
                finger_joins, wrist_defects, hand_contour)
        (wrist_image, wrist_coord) = \
            GraspEngine.identifyAccurateWristPoint(tip_image.copy(),
                hand_contour, wrist_defects, finger_tips)
        wrist_defects['wrist-left'] = wrist_coord
        comparison_contour = \
            GraspEngine.arrangeContourPoints(hand_contour,
                wrist_defects)
        contour_image = GraspEngine.drawBaseContours(base_image,
                base_contour)
        segmented_base_contour = \
            GraspEngine.separateContourIntoSegments(base_contour,
                base_ordered_finger_joins, base_wrist_defects,
                base_ordered_finger_tips)
        segmented_comparison_contour = \
            GraspEngine.separateContourIntoSegments(comparison_contour,
                finger_joins, wrist_defects, finger_tips)
        allignment_error_array = \
            GraspEngine.generateAllignmentErrors(segmented_base_contour,
                segmented_comparison_contour)
        edge_errors = \
            GraspEngine.generateContrastArray(segmented_base_contour,
                segmented_comparison_contour)
        if comparison_type == 'Corresponding points':
            comparison_image = \
                GraspEngine.drawColouredContour(contour_image,
                    edge_errors, lines)
        elif comparison_type == 'Closest points':
            comparison_image = \
                GraspEngine.drawColouredContour(contour_image,
                    allignment_error_array, lines)
        viewer.changeCurrentImage(GraspEngine.convert_cv_qt(comparison_image))

    def onProcessedStep(
        self,
        item_index,
        process,
        processed_data,
        settings,
        ):
        """Called when a processing step has been executed, sends data to ui.
        
        Parameters
        ----------
        item_index: Contains the index of the item which has been processed.
        process: Identifies the process which was applied.
        processed_data: Contains the resulting data of the process applied.
        settings: Contains the settings used to apply the process."""

        (viewer, stepbar, sidebar) = self.item_widgets[item_index]
        viewer.changeCurrentImage(GraspEngine.convert_cv_qt(processed_data))
        stepbar.addStep(GraspEngine.convert_cv_qt(processed_data))
        sidebar.addItemSettings(process, settings)

    def addImages(self, images):
        """Creates and stores image items for given images.
        
        Parameters
        ----------
        images : Contains the list of images to be added"""

        import_indexes = []
        import_images = []
        for image in images:
            (image_name, image_path, image_index) = image
            image_item = ImageItem(image_name, image_path,
                                   GraspEngine.convertToArray(image[1]),
                                   image_index)
            import_indexes.append(('Hand', image_index))
            import_images.append(image_item)
            self.items[image_index] = image_item
            self.image_indexes.append(image_index)
            self.num_items += 1
        self.action_controller.recordNewAction('Import',
                import_indexes, import_images)

    def exportProcessedImages(self, name):
        """Exports all images that have been processed."""

        images = []
        for i in self.processed_indexes:
            images.append((self.items[i].image_name,
                          self.items[i].processing_steps['Final'
                          ].image_data))
        GraspEngine.saveImages(images, name)

    def undoLastAction(self):
        """Undoes the last processing action made by the user."""

        if self.action_controller.current_action_index != 0:

            (
                previous_action_type,
                current_action_type,
                previous_action_indexes,
                current_action_indexes,
                previous_items,
                current_items,
                ) = self.action_controller.undoLastAction()
            if previous_action_type == 'Import':
                for (i, item_index) in \
                    enumerate(current_action_indexes):
                    (item_type, index) = item_index
                    if item_type == 'Hand':
                        (viewer, stepbar, sidebar) = \
                            self.item_widgets[index]
                        sidebar.showItemSettings(0)
                        stepbar.undoSteps(0, True)
                        original = previous_items[i].original_image
                        viewer.changeCurrentImage(GraspEngine.convert_cv_qt(original))
                        if index in self.processed_indexes:
                            self.processed_indexes.pop(self.processed_indexes.index(index))
                            self.explorer.removedProcessedItem(index)
                    if item_type == 'Accuracy':
                        self.explorer.removeExplorerItem(index)
                        self.main_window.tabClose(index)
                        self.item_tab.removeTab(index)
                        self.num_pipelines -= 1
                        self.num_items -= 1
                        self.main_window.item_count -= 1
            elif previous_action_type == 'Custom':
                (item_type, item_index) = previous_action_indexes[0]
                (viewer, stepbar, sidebar) = \
                    self.item_widgets[item_index]
                (previous_d, previous_outcome, previous_type) = \
                    previous_items[0]
                self.onCustomPipelineComplete(previous_d, item_index,
                        previous_outcome, previous_type, True)
            elif previous_action_type == 'Pipeline':
                for (i, item_index) in \
                    enumerate(current_action_indexes):
                    (item_type, index) = item_index
                    self.explorer.removeInvalidItem(index)
                    if item_type == 'Hand':
                        (viewer, stepbar, sidebar) = \
                            self.item_widgets[index]
                        stepbar.undoSteps(0, True)
                        previous_index = \
                            previous_action_indexes.index(('Hand',
                                index))

                        (previous_d, previous_index, previous_outcome,
                         previous_action) = \
                            previous_items[previous_index]
                        self.onPipelineComplete(previous_d,
                                previous_index, previous_outcome,
                                previous_action)

    def redoLastUndo(self):
        """Redoes the last action the user has undone."""

        if self.action_controller.current_action_index \
            != len(self.action_controller.action_history) - 1:

            (
                undone_action_type,
                current_action_type,
                undone_indexes,
                current_indexes,
                undone_items,
                current_items,
                ) = self.action_controller.redoLastUndo()
            if undone_action_type == 'Pipeline':
                for (i, item_index) in enumerate(undone_indexes):
                    (item_type, index) = item_index
                    if item_type == 'Hand':
                        (viewer, stepbar, sidebar) = \
                            self.item_widgets[index]
                        stepbar.undoSteps(0, True)

                        (undone_d, undone_index, undone_outcome,
                         undone_action) = undone_items[i]
                        self.onPipelineComplete(undone_d, undone_index,
                                undone_outcome, undone_action)
                    if item_type == 'Accuracy':
                        (results, indexes, items, action) = \
                            undone_items[i]
                        self.createPipelineDeliverables(results,
                                indexes, items, True)
            if undone_action_type == 'Custom':
                (item_type, item_index) = undone_indexes[0]
                (viewer, stepbar, sidebar) = \
                    self.item_widgets[item_index]
                (undone_d, undone_outcome, undone_type) = \
                    undone_items[0]
                self.onCustomPipelineComplete(undone_d, item_index,
                        undone_outcome, undone_type, True)
