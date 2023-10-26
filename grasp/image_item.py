class ImageItem:
    """
    Contains the model for an image item.
    """
    class Image:

        def __init__(self, image_data, image_name):
            self.image_data = image_data
            self.image_name = image_name

    class OriginalImage(Image):

        def __init__(
            self,
            image_data,
            image_name,
            image_path,
            ):
            super().__init__(image_data, image_name)
            self.image_path = image_path

    class NoLineImage(Image):

        def __init__(
            self,
            image_data,
            image_name,
            settings,
            ):
            super().__init__(image_data, image_name)
            (self.rows_removed, ) = settings

    class UniformImage(Image):

        def __init__(
            self,
            image_data,
            image_name,
            settings,
            ):
            super().__init__(image_data, image_name)

    class NormalizedImage(Image):

        def __init__(
            self,
            image_data,
            image_name,
            settings,
            ):
            super().__init__(image_data, image_name)
            (self.alpha, self.beta) = settings

    class BinaryImage(Image):

        def __init__(
            self,
            image_data,
            image_name,
            settings,
            ):
            super().__init__(image_data, image_name)
            (self.threshold, ) = settings

    class ConnectedComponentsImage(Image):

        def __init__(
            self,
            image_data,
            image_name,
            settings,
            additional,
            ):
            super().__init__(image_data, image_name)
            self.settings = settings
            self.analysis = additional

    class LargestComponentImage(Image):

        def __init__(
            self,
            image_data,
            image_name,
            settings,
            additional,
            ):
            super().__init__(image_data, image_name)
            (self.area, self.hand_component_centroid) = additional

    class SmoothedImage(Image):

        def __init__(
            self,
            image_data,
            image_name,
            settings,
            additional,
            ):
            super().__init__(image_data, image_name)
            (
                self.ox,
                self.oy,
                self.cx,
                self.cy,
                self.gx,
                self.gy,
                ) = settings

    class CLAHEImage(Image):

        def __init__(
            self,
            image_data,
            image_name,
            settings,
            additional,
            ):
            super().__init__(image_data, image_name)
            (self.clip_limit, self.x, self.y) = settings

    class ContourImage(Image):

        def __init__(
            self,
            image_data,
            image_name,
            settings,
            additional,
            ):
            super().__init__(image_data, image_name)
            self.contours = additional

    class ConvexHullImage(Image):

        def __init__(
            self,
            image_data,
            image_name,
            settings,
            additional,
            ):
            super().__init__(image_data, image_name)

    class DefectImage(Image):

        def __init__(
            self,
            image_data,
            image_name,
            settings,
            additional,
            ):
            super().__init__(image_data, image_name)
            (self.defect_coordinates, self.defects) = additional

    class FeaturesImage(Image):

        def __init__(
            self,
            image_data,
            image_name,
            settings,
            additional,
            ):
            super().__init__(image_data, image_name)
            (self.finger_joins, self.wrist_defects, self.palm_centroid,
             self.finger_tips) = additional

    class AlignImage(Image):

        def __init__(
            self,
            image_data,
            image_name,
            settings,
            additional,
            ):
            super().__init__(image_data, image_name)

    class CenterImage(Image):

        def __init__(
            self,
            image_data,
            image_name,
            settings,
            additional,
            ):
            super().__init__(image_data, image_name)

    class RotatedImage(Image):

        def __init__(
            self,
            image_data,
            image_name,
            settings,
            additional,
            ):
            super().__init__(image_data, image_name)

    class FinalImage(Image):

        def __init__(
            self,
            image_data,
            image_name,
            settings,
            additional,
            ):
            super().__init__(image_data, image_name)

    class CenterContourImage(Image):

        def __init__(
            self,
            image_data,
            image_name,
            settings,
            additional,
            ):
            super().__init__(image_data, image_name)

    class VectorImage(Image):

        def __init__(
            self,
            image_data,
            image_name,
            settings,
            additional,
            ):
            super().__init__(image_data, image_name)
            self.similarity_vector = additional

    def __init__(
        self,
        image_name,
        image_path,
        image_data,
        list_index,
        ):
        self.list_index = list_index
        self.processing_steps = {}
        self.current_step = 0
        self.image_name = image_name
        self.image_path = image_path
        self.original_image = image_data

    def addProcessingStep(
        self,
        process,
        image_data,
        settings,
        additional,
        ):
        if process == 'Original':
            image = self.OriginalImage(image_data, self.image_name,
                    self.image_path)
        elif process == 'Line':
            image = self.NoLineImage(image_data, 'Line_'
                    + self.image_name, settings)
        elif process == 'Uniform':
            image = self.UniformImage(image_data, 'Uniform_'
                    + self.image_name, settings)
        elif process == 'Normalize':
            image = self.NormalizedImage(image_data, 'Contrast_'
                    + self.image_name, settings)
        elif process == 'Binary':
            image = self.BinaryImage(image_data, 'Binary_'
                    + self.image_name, settings)
        elif process == 'Components':
            image = self.ConnectedComponentsImage(image_data, 'Connect_'
                     + self.image_name, settings, additional)
        elif process == 'Component':
            image = self.LargestComponentImage(image_data, 'Component_'
                    + self.image_name, settings, additional)
        elif process == 'Smooth':
            image = self.SmoothedImage(image_data, 'Smooth_'
                    + self.image_name, settings, additional)
        elif process == 'CLAHE':
            image = self.CLAHEImage(image_data, 'CLAHE_'
                                    + self.image_name, settings,
                                    additional)
        elif process == 'Contour':
            image = self.ContourImage(image_data, 'Contour_'
                    + self.image_name, settings, additional)
        elif process == 'Hull':
            image = self.ConvexHullImage(image_data, 'ConvexHull_'
                    + self.image_name, settings, additional)
        elif process == 'Defect':
            image = self.DefectImage(image_data, 'Defect_'
                    + self.image_name, settings, additional)
        elif process == 'Features':
            image = self.FeaturesImage(image_data, 'Features_'
                    + self.image_name, settings, additional)
        elif process == 'Align':
            image = self.AlignImage(image_data, 'Align_'
                                    + self.image_name, settings,
                                    additional)
        elif process == 'Center':
            image = self.CenterImage(image_data, 'Center_'
                    + self.image_name, settings, additional)
        elif process == 'Rotated':
            image = self.RotatedImage(image_data, 'Rotated_'
                    + self.image_name, settings, additional)
        elif process == 'Final':
            image = self.FinalImage(image_data, 'Final_'
                                    + self.image_name, settings,
                                    additional)
        elif process == 'Center-Contour':
            image = self.CenterContourImage(image_data, 'Center-Contour'
                     + self.image_name, settings, additional)
        elif process == 'Vector':
            image = self.VectorImage(image_data, 'Vector_'
                    + self.image_name, settings, additional)

        self.processing_steps[process] = image
