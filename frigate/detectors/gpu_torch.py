import logging
import numpy as np

from frigate.detectors.detection_api import DetectionApi
import torch

import sys

sys.path.insert(1, '/opt/frigate/frigate/detectors/yolov6')

# if os.getcwd()=="/frigate":
#     os.chdir("detectors/yolov6")

from yolov6.layers.common import DetectBackend
from yolov6.utils.nms import non_max_suppression
from yolov6.utils.events import load_yaml
from yolov6.data.data_augment import letterbox

logger = logging.getLogger(__name__)


class GpuTorch(DetectionApi):
    def __init__(self, det_device=None, model_config=None):
        try:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                logging.info("GPU device found.")

                self.img_size:int = 640#@param {type:"integer"}

                self.conf_thres: float =.25 #@param {type:"number"}
                self.iou_thres: float =.45 #@param {type:"number"}
                self.max_det:int =  1000#@param {type:"integer"}
                self.agnostic_nms: bool = False #@param {type:"boolean"}

                # default to /yolov6n.pt
                self.model = DetectBackend(f"{model_config.path}", device=self.device)
                self.stride = self.model.stride
                self.class_names = load_yaml("/opt/frigate/frigate/detectors/yolov6/data/coco.yaml")['names']

                self.model(torch.zeros(3, 3, 640, 480).to(self.device).type_as(next(self.model.model.parameters())))  # warmup


        except Exception:
            logging.error("GPU or Cuda device not detected. Verify Cuda drivers and torch installation.")
            raise


        # device_config = {"device": "usb"}
        # if not det_device is None:
        #     device_config = {"device": det_device}

        # edge_tpu_delegate = None

        # try:
        #     logger.info(f"Attempting to load TPU as {device_config['device']}")
        #     edge_tpu_delegate = load_delegate("libedgetpu.so.1.0", device_config)
        #     logger.info("TPU found")
        #     self.interpreter = tflite.Interpreter(
        #         model_path=model_config.path or "/edgetpu_model.tflite",
        #         experimental_delegates=[edge_tpu_delegate],
        #     )

        # self.interpreter.allocate_tensors()

        # self.tensor_input_details = self.interpreter.get_input_details()
        # self.tensor_output_details = self.interpreter.get_output_details()

    def _check_img_size(img_size, s=32, floor=0):
        def make_divisible( x, divisor):
            # Upward revision the value x to make it evenly divisible by the divisor.
            return math.ceil(x / divisor) * divisor
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
            new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size,list) else [new_size]*2

    def detect_raw(self, tensor_input):
        self.img_size = _check_img_size(tensor_input, s=self.stride)

        # img, img_src = precess_image(url, img_size, stride, half)

        ##### process_image start
        image = letterbox(tensor_input, self.img_size, stride=self.stride)[0]

        # Convert
        #image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB

        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        ##### process image end

        image = image.to(self.device)
        if len(image.shape) == 3:
            image = image[None]
            # expand for batch dim
        pred_results = self.model(image)
        classes:Optional[List[int]] = None # the classes to keep
        det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, self.agnostic_nms, max_det=self.max_det)[0]

        # gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        # img_ori = img_src.copy()
        if len(det):
            det[:, :4] = Inferer.rescale(image.shape[2:], det[:, :4], tensor_input.shape).round()
        # for *xyxy, conf, cls in reversed(det):
        #     class_num = int(cls)
        #     label = None if hide_labels else (class_names[class_num] if hide_conf else f'{class_names[class_num]} {conf:.2f}')
        #     Inferer.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=Inferer.generate_colors(class_num, True))

        #####################################################################################

        # self.interpreter.set_tensor(self.tensor_input_details[0]["index"], tensor_input)
        # self.interpreter.invoke()

        # boxes = self.interpreter.tensor(self.tensor_output_details[0]["index"])()[0]
        # class_ids = self.interpreter.tensor(self.tensor_output_details[1]["index"])()[0]
        # scores = self.interpreter.tensor(self.tensor_output_details[2]["index"])()[0]
        # count = int(
        #     self.interpreter.tensor(self.tensor_output_details[3]["index"])()[0]
        # )

        #######################################################################################3

        detections = np.zeros((20, 6), np.float32)
        i = 1
        for *xyxy, conf, class_ids in reversed(det):
            if conf < 0.4 or i > 20:
                break
            detections[i] = [
                class_ids,
                float(conf),
                xyxy[0],
                xyxy[1],
                xyxy[2],
                xyxy[3],
            ]

            i = i + 1

        return detections