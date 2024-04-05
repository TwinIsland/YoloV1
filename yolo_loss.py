import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# device to use for torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        # Your code here
        center_x, center_y, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        # Calculate the top-left corner coordinates
        top_left_x = center_x / self.S - 0.5 * width
        top_left_y = center_y / self.S - 0.5 * height

        # Calculate the bottom-right corner coordinates
        bottom_right_x = center_x / self.S + 0.5 * width
        bottom_right_y = center_y / self.S + 0.5 * height

        # Combine the corner coordinates into the final format
        corner_boxes = torch.hstack((top_left_x.reshape(-1, 1), top_left_y.reshape(-1, 1),
                                     bottom_right_x.reshape(-1, 1), bottom_right_y.reshape(-1, 1)))

        return corner_boxes

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters:
        box_pred_list : (list) [(tensor) size (-1, 5)]
        box_target : (tensor)  size (-1, 4)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """

        ### CODE ###
        # Your code here

        best_iou = torch.zeros((box_target.size(0), 1), device=box_target.device)
        best_boxes = torch.zeros((box_target.size(0), 5), device=box_target.device)

        for i in range(box_target.size(0)):
            # update best_iou/boxes wrt current target box

            # step 1: get the xyxy coordinate for predict boxes and target
            preds_xyxy = self.xywh2xyxy(torch.stack([pred[i, :4] for pred in pred_box_list]).to(device))
            target_xyxy = self.xywh2xyxy(
                box_target[i].unsqueeze(dim=0))  # add missing dimension to target to match preds

            ious = compute_iou(preds_xyxy, target_xyxy)
            best_iou_ele, best_iou_idx = torch.max(ious, dim=0)
            best_iou[i], best_boxes[i] = best_iou_ele, pred_box_list[best_iou_idx][i]

        return best_iou, best_boxes

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        ### CODE ###
        # Your code here

        # lec10_detection.pdf   page 48
        # add all mse for bonding box
        # seems like simple MSE
        classes_pred_in_classes, classes_target_in_class = classes_pred[has_object_map], classes_target[has_object_map]

        return torch.nn.functional.mse_loss(classes_pred_in_classes, classes_target_in_class, reduction='sum')

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """F
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        ### CODE ###

        # https://stats.stackexchange.com/questions/287486/yolo-loss-function-explanation

        # calculate the loss for each bonding box
        # for i in range(self.B):
        #     pred_boxes = pred_boxes_list[i]
        #     total_loss += torch.sum((pred_boxes[:, :, :, -1][~has_object_map] ** 2))
        # return self.l_noobj * total_loss

        total_loss = 0

        # hint 1: compute loss for all predictions in the pred_boxes_list list
        for cur_pred_box in pred_boxes_list:
            pred_conf = cur_pred_box[..., -1]

            # hint 2: Only compute loss for cell which doesn't contain object
            # yolo 1 paper: if no obj in box, confident should be 0, no need to mse, just sqrt
            total_loss += torch.sum(torch.square(pred_conf[~has_object_map]))

        return self.l_noobj * total_loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """
        ### CODE
        # your code here

        # lec10_detection: page 48
        # just simply add the mse of no_obj and obj, since both use mse and mse are linear

        return F.mse_loss(box_pred_conf, box_target_conf, reduction='sum')

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        ### CODE
        # your code here

        # https://stats.stackexchange.com/questions/287486/yolo-loss-function-explanation

        # sum of mse of x and y
        xy_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], reduction='sum')

        # sum of mse for sqrt root of w and h
        wh_loss = F.mse_loss(torch.sqrt(box_pred_response[:, 2:]), torch.sqrt(box_target_response[:, 2:]),
                             reduction='sum')

        return self.l_coord * (xy_loss + wh_loss)

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) where:
                            N - batch_size
                            S - width/height of network output grid
                            B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """

        # Split the pred_tensor into bounding box predictions and class predictions using tensor slicing

        # B's bounding box in each grid box
        bbox_predictions = [pred_tensor[..., 5 * i:5 * (i + 1)] for i in range(self.B)]

        # classes prediction wrt the grid
        class_predictions = pred_tensor[..., -20:]

        # Compute losses
        batch_size = pred_tensor.size(0)
        classification_loss = self.get_class_prediction_loss(class_predictions, target_cls,
                                                             has_object_map) / batch_size
        no_object_detection_loss = self.get_no_object_loss(bbox_predictions, has_object_map) / batch_size

        # Filter out predictions and target_boxes for cells that contain objects
        object_bbox_predictions = [box[has_object_map] for box in bbox_predictions]
        object_target_boxes = target_boxes[has_object_map]

        # Determine the best boxes among the predictions and their IoUs with the target
        best_iou, best_box = self.find_best_iou_boxes(object_bbox_predictions, object_target_boxes)

        # Compute regression and object confidence loss

        # yolo paper: if contain object, confident = iou
        bbox_regression_loss = self.get_regression_loss(best_box[:, :4], object_target_boxes) / batch_size

        object_confidence_loss = self.get_contain_conf_loss(best_box[:, -1:], best_iou) / batch_size

        # Aggregate total loss
        total_loss = bbox_regression_loss + object_confidence_loss + no_object_detection_loss + classification_loss

        # Construct and return the loss dictionary
        return {
            "total_loss": total_loss,
            "reg_loss": bbox_regression_loss,
            "containing_obj_loss": object_confidence_loss,
            "no_obj_loss": no_object_detection_loss,
            "cls_loss": classification_loss,
        }
