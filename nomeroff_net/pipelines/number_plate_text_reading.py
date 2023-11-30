import cv2
from torch import no_grad
from typing import Any, Dict, Optional, Union
from nomeroff_net.image_loaders import BaseImageLoader
from nomeroff_net.pipelines.base import Pipeline
from nomeroff_net.tools import unzip
from nomeroff_net.pipes.number_plate_text_readers.text_detector import TextDetector

DEFAULT_PRESETS = {
    # "eu_ua_2004_2015": {
    #     "for_regions": ["eu_ua_2015", "eu_ua_2004"],
    #     "model_path": "latest"
    # },
    # "eu_ua_1995": {
    #     "for_regions": ["eu_ua_1995"],
    #     "model_path": "latest"
    # },
    # "eu": {
    #     "for_regions": ["eu", "xx_transit", "xx_unknown"],
    #     "model_path": "latest"
    # },
    "ru": {
        "for_regions": ["ru", "eu_ua_ordlo_lpr", "eu_ua_ordlo_dpr"],
        "model_path": "latest"
    },
    # "kz": {
    #     "for_regions": ["kz"],
    #     "model_path": "latest"
    # },
    # "kg": {  # "kg_shufflenet_v2_x2_0"
    #     "for_regions": ["kg"],
    #     "model_path": "latest"
    # },
    # "ge": {
    #     "for_regions": ["ge"],
    #     "model_path": "latest"
    # },
    # "su": {
    #     "for_regions": ["su"],
    #     "model_path": "latest"
    # },
    # "am": {
    #     "for_regions": ["am"],
    #     "model_path": "latest"
    # },
    # "by": {
    #     "for_regions": ["by"],
    #     "model_path": "latest"
    # },
}


class NumberPlateTextReading(Pipeline):
    """
    Number Plate Text Reading Pipeline
    """

    def __init__(self,
                 task,
                 image_loader: Optional[Union[str, BaseImageLoader]],
                 presets: Dict = None,
                 default_label: str = "ru",
                 default_lines_count: int = 1,
                 class_detector=TextDetector,
                 need_preprocess=False,
                 **kwargs):
        if presets is None:
            presets = DEFAULT_PRESETS
        super().__init__(task, image_loader, **kwargs)
        self.detector = class_detector(presets, default_label, default_lines_count)
        # print("$#$#$#$detector:", self.detector)
        self.need_preprocess = need_preprocess
        print("$#$#$#$need_preprocess:", self.need_preprocess)

    def sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def __call__(self, images: Any, **kwargs):
        return super().__call__(images, **kwargs)

    def preprocess(self, inputs: Any, **preprocess_parameters: Dict) -> Any:
        print(">>>NumberPlateTextReading PREPROCESS called")
        images, labels, lines, preprocessed_np = unzip(inputs)
        images = [self.image_loader.load(item) for item in images]
        return unzip([images, labels, lines, preprocessed_np])

    @no_grad()
    def forward(self, inputs: Any, **forward_parameters: Dict) -> Any:
        print(">>>NumberPlateTextReading forward is called")
        print(type(inputs[0]), len(inputs[0]))
        for input in inputs[0]:
            print(type(input))
        images, labels, lines, preprocessed_np = unzip(inputs)
        print(">>>labels:", labels, '>>>lines\n', lines, type(images), len(images))
        print(images)
        # for image in images:
        #     cv2.imshow('image_asd123', image)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # print(">>> processed_np:", preprocessed_np)
        # if self.need_preprocess or all([np is None for np in preprocessed_np]):
        # print(">>> if statement is called")
        model_inputs = self.detector.preprocess(images, labels, lines)
        # else:
        #     print(">>> else statement is called")
        #     model_inputs = self.detector.preprocess(preprocessed_np, labels, lines, need_preprocess=False)
        print(">>> models inputs:", list(model_inputs[0].keys()))
        model_outputs = self.detector.forward(model_inputs)
        model_outputs = self.detector.postprocess(model_outputs)
        print(model_outputs, labels, sep="asd")
        return unzip([images, model_outputs, labels])

    def postprocess(self, inputs: Any, **postprocess_parameters: Dict) -> Any:
        print(">>>NumberPlateTextReading postprocess", inputs[0][2])
        images, model_outputs, labels = unzip(inputs)
        return unzip([model_outputs, images])
