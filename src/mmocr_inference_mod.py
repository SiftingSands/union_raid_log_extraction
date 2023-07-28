from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from mmocr.utils import bbox2poly, crop_img, poly2bbox
from mmocr.apis.inferencers import MMOCRInferencer
from mmocr.apis.inferencers.base_mmocr_inferencer import InputsType, PredType, ConfigType
from mmengine.structures import InstanceData

class MMOCRInferencer_merged_dets(MMOCRInferencer):
    """ Inherit from mmocr.apis.inferencers.mmocr_inferencer.MMOCRInferencer
        and modify the forward() method to merge overlapping quads before
        passing them to the text recognition model.
    """
    def __init__(self,
                 det: Optional[Union[ConfigType, str]] = None,
                 det_weights: Optional[str] = None,
                 rec: Optional[Union[ConfigType, str]] = None,
                 rec_weights: Optional[str] = None,
                 kie: Optional[Union[ConfigType, str]] = None,
                 kie_weights: Optional[str] = None,
                 device: Optional[str] = None,
                 intersection_threshold: float = 0.01,
                 min_area: int = 100,
                 det_score_threshold: float = 0.4
                 ) -> None:
        super().__init__(det, det_weights, rec, rec_weights, kie, kie_weights, device)
        self.intersection_threshold = intersection_threshold
        self.min_area = min_area
        self.det_score_threshold = det_score_threshold

    def _intersection(self, box_1: np.ndarray, box_2: np.ndarray) -> float:
        """
        Calculate the intersection area of two bounding boxes.
        """
        x1 = max(box_1[0], box_2[0])
        y1 = max(box_1[1], box_2[1])
        x2 = min(box_1[2], box_2[2])
        y2 = min(box_1[3], box_2[3])
        return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    def forward(self,
                inputs: InputsType,
                batch_size: int = 1,
                det_batch_size: Optional[int] = None,
                rec_batch_size: Optional[int] = None,
                kie_batch_size: Optional[int] = None,
                **forward_kwargs) -> PredType:
        """Forward the inputs to the model.

        Args:
            inputs (InputsType): The inputs to be forwarded.
            batch_size (int): Batch size. Defaults to 1.
            det_batch_size (Optional[int]): Batch size for text detection
                model. Overwrite batch_size if it is not None.
                Defaults to None.
            rec_batch_size (Optional[int]): Batch size for text recognition
                model. Overwrite batch_size if it is not None.
                Defaults to None.
            kie_batch_size (Optional[int]): Batch size for KIE model.
                Overwrite batch_size if it is not None.
                Defaults to None.

        Returns:
            Dict: The prediction results. Possibly with keys "det", "rec", and
            "kie"..
        """
        result = {}
        forward_kwargs['progress_bar'] = False
        if det_batch_size is None:
            det_batch_size = batch_size
        if rec_batch_size is None:
                rec_batch_size = batch_size
        if kie_batch_size is None:
            kie_batch_size = batch_size
        if self.mode == 'rec':
            # The extra list wrapper here is for the ease of postprocessing
            self.rec_inputs = inputs
            predictions = self.textrec_inferencer(
                self.rec_inputs,
                return_datasamples=True,
                batch_size=rec_batch_size,
                **forward_kwargs)['predictions']
            result['rec'] = [[p] for p in predictions]
        elif self.mode.startswith('det'):  # 'det'/'det_rec'/'det_rec_kie'
            result['det'] = self.textdet_inferencer(
                inputs,
                return_datasamples=True,
                batch_size=det_batch_size,
                **forward_kwargs)['predictions']
            if self.mode.startswith('det_rec'):  # 'det_rec'/'det_rec_kie'
                result['rec'] = []
                for sample_idx, (img, det_data_sample) in enumerate(zip(
                        self._inputs2ndarrray(inputs), result['det'])):
                    det_pred = det_data_sample.pred_instances

                    self.rec_rects = []
                    # Convert polygons to rectangles
                    for polygon in det_pred['polygons']:
                        # xyxy format
                        rect = poly2bbox(polygon)
                        self.rec_rects.append(rect)
                        
                    # Merge overlapping quads
                    merged_rectangles = []
                    for box_idx, box_1 in enumerate(self.rec_rects):
                        boxes_to_remove = []
                        for j, box_2_dict in enumerate(merged_rectangles):
                            # scale threshold by the size of the bounding box
                            box_2 = box_2_dict['xyxy']
                            intersection_threshold_scaled = self.intersection_threshold * (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
                            if self._intersection(box_1, box_2) > intersection_threshold_scaled:
                                # resize the box to include the other box and move on to the next box
                                box_1 = (min(box_1[0], box_2[0]), min(box_1[1], box_2[1]), max(box_1[2], box_2[2]), max(box_1[3], box_2[3]))
                                boxes_to_remove.append(j)
                        # remove the boxes that were merged into box_1
                        for j in sorted(boxes_to_remove, reverse=True):
                            del merged_rectangles[j]
                        merged_rectangles.append({'xyxy': box_1, 'score': det_pred['scores'][box_idx]})

                    # could replace the next two steps with a list comprehension
                    # check for minimum area
                    area_filtered_rectangles = []
                    for box_dict in merged_rectangles:
                        box = box_dict['xyxy']
                        if (box[2] - box[0]) * (box[3] - box[1]) >= self.min_area:
                            area_filtered_rectangles.append(box_dict)

                    # check for minimum score
                    final_filtered_rectangles = []
                    for box_dict in area_filtered_rectangles:
                        if box_dict['score'] >= self.det_score_threshold:
                            final_filtered_rectangles.append(box_dict)
                    
                    # crop the image with the merged rectangles
                    self.rec_inputs = []
                    scores = []
                    polygons = []
                    for box_dict in final_filtered_rectangles:
                        # Roughly convert the polygon to a quadangle with
                        # 4 points
                        box = box_dict['xyxy']
                        quad = bbox2poly(box).tolist()
                        self.rec_inputs.append(crop_img(img, quad))
                        scores.append(box_dict['score'])
                        polygons.append(np.array(quad))

                    # modify the InstanceData object with the merged rectangles and scores
                    # https://github.com/open-mmlab/mmocr/blob/main/mmocr/structures/textdet_data_sample.py
                    # https://github.com/open-mmlab/mmengine/blob/main/mmengine/structures/instance_data.py
                    temp = InstanceData()
                    scores = torch.tensor(scores)
                    temp.scores = scores
                    temp.polygons = polygons
                    det_data_sample.pred_instances = temp
                    
                    result['det'][sample_idx] = det_data_sample
                    result['rec'].append(
                        self.textrec_inferencer(
                            self.rec_inputs,
                            return_datasamples=True,
                            batch_size=rec_batch_size,
                            **forward_kwargs)['predictions'])
                if self.mode == 'det_rec_kie':
                    self.kie_inputs = []
                    # TODO: when the det output is empty, kie will fail
                    # as no gt-instances can be provided. It's a known
                    # issue but cannot be solved elegantly since we support
                    # batch inference.
                    for img, det_data_sample, rec_data_samples in zip(
                            inputs, result['det'], result['rec']):
                        det_pred = det_data_sample.pred_instances
                        kie_input = dict(img=img)
                        kie_input['instances'] = []
                        for polygon, rec_data_sample in zip(
                                det_pred['polygons'], rec_data_samples):
                            kie_input['instances'].append(
                                dict(
                                    bbox=poly2bbox(polygon),
                                    text=rec_data_sample.pred_text.item))
                        self.kie_inputs.append(kie_input)
                    result['kie'] = self.kie_inferencer(
                        self.kie_inputs,
                        return_datasamples=True,
                        batch_size=kie_batch_size,
                        **forward_kwargs)['predictions']
        return result