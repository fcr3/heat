from openvino.inference_engine import IENetwork, IECore
import numpy as np
import cv2

def preprocess(frame, shape):
    n, c, h, w = shape
    scale_x = scale_y = min(h / frame.shape[0], w / frame.shape[1])
    input_image = cv2.resize(frame, None, fx=scale_x, fy=scale_y)
    
    input_image_size = input_image.shape[:2]
    input_image = np.pad(input_image, ((0, h - input_image_size[0]),
                                       (0, w - input_image_size[1]),
                                       (0, 0)),
                         mode='constant', constant_values=0)
    # Change data layout from HWC to CHW.
    input_image = input_image.transpose((2, 0, 1))
    input_image = input_image.reshape((n, c, h, w)).astype(np.float32)
    input_image_info = np.asarray(
        [[input_image_size[0], input_image_size[1], 1]], dtype=np.float32
    )
    
    return {'im_data': input_image, 'im_info': input_image_info}, frame, scale_x, scale_y

def segm_postprocess(box, raw_cls_mask, im_h, im_w):
    # Add zero border to prevent upsampling artifacts on segment borders.
    raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), 'constant', constant_values=0)
    extended_box = expand_box(box, raw_cls_mask.shape[0] / (raw_cls_mask.shape[0] - 2.0)).astype(int)
    w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
    x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
    x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

    raw_cls_mask = cv2.resize(raw_cls_mask, (w, h)) > 0.5
    mask = raw_cls_mask.astype(np.uint8)
    # Put an object mask in an image mask.
    im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
    im_mask[y0:y1, x0:x1] = mask[(y0 - extended_box[1]):(y1 - extended_box[1]),
                            (x0 - extended_box[0]):(x1 - extended_box[0])]
    return im_mask

def expand_box(box, scale):
    w_half = (box[2] - box[0]) * .5
    h_half = (box[3] - box[1]) * .5
    x_c = (box[2] + box[0]) * .5
    y_c = (box[3] + box[1]) * .5
    w_half *= scale
    h_half *= scale
    box_exp = np.zeros(box.shape)
    box_exp[0] = x_c - w_half
    box_exp[2] = x_c + w_half
    box_exp[1] = y_c - h_half
    box_exp[3] = y_c + h_half
    return box_exp

def post_process(frame, outputs, scale_x, scale_y):
    boxes = outputs['boxes']
    boxes[:, 0::2] /= scale_x
    boxes[:, 1::2] /= scale_y
    scores = outputs['scores']
    classes = outputs['classes'].astype(np.uint32)
    
    masks = []
    for box, cls, raw_mask in zip(boxes, classes, outputs['raw_masks']):
        raw_cls_mask = raw_mask[cls, ...]
        mask = segm_postprocess(
            box, raw_cls_mask, frame.shape[0], frame.shape[1]
        )
        masks.append(mask)

    # Filter out detections with low confidence.
    detections_filter = scores > 0
    scores = scores[detections_filter]
    classes = classes[detections_filter]
    boxes = boxes[detections_filter]
    masks = list(
        segm for segm, is_valid in zip(masks, detections_filter) if is_valid
    )
    picker = np.copy(frame)
    picker[masks[0] == 0] = 0
    return picker, masks[0] == 0

class person_segmenter(object):
    def __init__(self):
        self.ie = IECore()
        self.net = self.ie.read_network('instance_segmentation_demo/instance-segmentation-security-1025.xml',
                                   'instance_segmentation_demo/instance-segmentation-security-1025.bin')
        self.shape = self.net.input_info['im_data'].input_data.shape
        self.exec_net = self.ie.load_network(network=self.net, device_name='CPU')
        
    def infer(self, depth_image, color_image):
        input_dict, color_image, scale_x, scale_y = preprocess(color_image, self.shape)
        outputs = self.exec_net.infer(input_dict)
        picker, picker_mask = post_process(color_image, outputs, scale_x, scale_y)
        depth_ppl = np.copy(depth_image)
        depth_ppl[picker_mask] = 0
        return depth_ppl
        
        
        
        
        
        
        
        
        