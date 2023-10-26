import sys
sys.path.append('../grasp')
from PyQt5.QtWidgets import QApplication

from grasp.grasp_controller import GraspController
from main_window import MainWindow

grasp = QApplication(sys.argv)
grasp_controller = GraspController()

sys.exit(grasp.exec_())
