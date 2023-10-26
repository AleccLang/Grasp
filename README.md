Grasp is an image processing program used for x-ray hand images.

----------------------------------------------------------------------------------------------------------------------------------------------------

Grasp requires that the packages in the requirements.txt file are installed. A Makefile is included to create a virtual environment and install the packages to it, from which the Grasp can be run.
To run the gui the command "python3 grasp/grasp_gui" must be typed from the directory containing the "grasp" file.
The program requires that there are x-ray hand images downloaded to your machine.
When processing images in Grasp, the tools must be used in the correct sequence (histogram -> binary -> connected components -> contour) as we are yet to include exeption handling in the current demo version.
The latest fully implemented tool is the hand contouring, the other tools are yet to be implemented in the current demo version and so their gui buttons will have no effect on the image.

----------------------------------------------------------------------------------------------------------------------------------------------------

Files included:

Makefile: Used to build the virtual environment and install packages with "make" and clean up the ./venv file with "make clean".

requirements.txt: Contains the packages used to run Grasp, and is installed using the Makefile.

test2.py: File is used to test and experiment with image processing functions before implementing them into grasp_engine.

grasp_engine.py: Contains all the methods necessary for the proccessing and evaluation of images.

grasp_controller.py: The controller class which manages the entire software.

grasp_gui/: Contains the classes that are used to build the gui as well as the main class of the program.

----------------------------------------------------------------------------------------------------------------------------------------------------
