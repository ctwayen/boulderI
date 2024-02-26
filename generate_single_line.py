import numpy as np
import cv2
import copy
import os

def process_image_filter(image, hue_range, saturation_range, value_range):
    # def find_dominant_hue_range(image, bounding_boxes, range_width=40):
    #     hues = []
    #     for box in bounding_boxes:
    #         x, y, w, h = box
    #         roi = image[y:h, x:w]
    #         hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    #         hues.extend(hsv_roi[:, :, 0].flatten())
        
    #     hues = np.array(hues)
    #     hist, bins = np.histogram(hues, bins=180, range=[0, 180])
    #     dominant_hue = bins[np.argmax(hist)]

    #     lower_hue = max(0, dominant_hue - range_width)
    #     upper_hue = min(180, dominant_hue + range_width)
        
    #     return (lower_hue, upper_hue)
        
    def calculate_weighted_mask(h, s, v, hue_range, saturation_range, value_range):
        # Adjust logic to include checks for saturation and value
        # median_hue = (hue_range[1] + hue_range[0]) / 2
        hue_mask = np.logical_and(h >= hue_range[0], h <= hue_range[1])
        saturation_mask = np.logical_and(s >= saturation_range[0], s <= saturation_range[1])
        value_mask = np.logical_and(v >= value_range[0], v <= value_range[1])
        
        combined_mask = hue_mask & saturation_mask & value_mask
        
        # Initialize weights based on combined mask
        weights = np.zeros_like(h, dtype=np.float32)
        weights[combined_mask] = 1
        
        # # Further refine weights based on proximity to median hue within the core range
        # core_lower_hue = max(hue_range[0], median_hue - core_range / 2)
        # core_upper_hue = min(hue_range[1], median_hue + core_range / 2)
        # core_hue_mask = np.logical_and(h >= core_lower_hue, h <= core_upper_hue)
        # weights[core_hue_mask] = 1  # Maximize weight within core hue range
        
        # Apply Gaussian blur to smooth the weights
        weights_blurred = cv2.GaussianBlur(weights, (13, 13), 0)
        return weights_blurred
    
    
    def enhance_selected_colors(image, hue_range, saturation_range, value_range, saturation_increase=0.2, value_increase=0.2):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        # Create a mask for pixels within the target hue range
        weights = calculate_weighted_mask(h, s, v, hue_range, saturation_range, value_range)
        
        # weights = weights[:, :, np.newaxis]  # Make weights 3D for broadcasting
        
        # s = s * (1 + saturation_increase) * (weights+1)
        # v = v * (1 + value_increase) * (weights+1)
        
        # # Clip values to stay within valid range
        # s = np.clip(s, 0, 255).astype(np.uint8)
        # v = np.clip(v, 0, 255).astype(np.uint8)
        
        # Merge channels back and convert to BGR
        enhanced_hsv = cv2.merge([h, s, v])
        enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
        # Convert the original image to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Convert the grayscale image back to BGR format so it can be combined with the original
        grayscale_bgr_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
        
        # Blend the original image with the grayscale image based on weights
        weights = weights[:, :, np.newaxis]  # Make weights 3D for broadcasting
        output_image = (enhanced_image * weights + grayscale_bgr_image * (1 - weights)).astype(np.uint8)
        return output_image

    # output_image = preserve_color_convert_others_to_grayscale(image, dominant_hue_range)
    output_image = enhance_selected_colors(image, hue_range, saturation_range, value_range)
    
    return output_image

def rotate_bbox(xmin, ymin, xmax, ymax, angle, img_width, img_height):
    if angle == 90:
        return ymin, img_width - xmax, ymax, img_width - xmin
    elif angle == 180:
        return img_width - xmax, img_height - ymax, img_width - xmin, img_height - ymin
    elif angle == 270 or angle == -90:
        return img_height - ymax, xmin, img_height - ymin, xmax
    else:
        return xmin, ymin, xmax, ymax  # No rotation
    
class Process(object):
    suffix = '.jpg'
    def __init__(self):
        self.output_images = {
            'crop': None,
            'start': None,
            'end': None,
            'filter': None
        }
    
    def process(self, image_file, shapes, degrees, huerange, satrurationRange, valueRange):
        start = shapes['起步点']
        print(start)
        end = shapes['结束点']
        crop = shapes['裁剪边框']
        
        src_img = cv2.imread(image_file)
        print(src_img.shape)
        # Get the image dimensions
        (h, w) = src_img.shape[:2]
        # Calculate the center of the image
        center = (w // 2, h // 2)
        if degrees == 90:
            # For a 90 degree rotation, the width and height are swapped
            M = cv2.getRotationMatrix2D(center, degrees, 1.0)
            src_img = cv2.warpAffine(src_img, M, (h, w))  # Note the swapped dimensions
        elif degrees == 270:
            # For a -90 (or 270) degree rotation, similarly swap dimensions
            M = cv2.getRotationMatrix2D(center, degrees, 1.0)
            src_img = cv2.warpAffine(src_img, M, (h, w))  # Note the swapped dimensions
        elif degrees == 180:
            # For other angles, you might need to adjust this accordingly
            M = cv2.getRotationMatrix2D(center, degrees, 1.0)
            src_img = cv2.warpAffine(src_img, M, (w, h))
        # src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

        print(src_img.shape)
        height, width = src_img.shape[:2]
        bounding_boxes = []
        start_img = copy.deepcopy(src_img)
        for start_shapes in start:
            points = start_shapes['points']
            xmin, ymin, xmax, ymax = Process.convertPoints2BndBox(points)
            xmin, ymin, xmax, ymax = rotate_bbox(xmin, ymin, xmax, ymax, degrees, width, height)
            bounding_boxes.append((xmin, ymin, xmax, ymax))
            color = (0, 0, 255)
            thickness = 10
            cv2.rectangle(start_img, (xmin, ymin), (xmax, ymax), color, thickness)
        
        end_img = copy.deepcopy(src_img)
        points = end['points']
        xmin, ymin, xmax, ymax = Process.convertPoints2BndBox(points)
        xmin, ymin, xmax, ymax = rotate_bbox(xmin, ymin, xmax, ymax, degrees, width, height)
        bounding_boxes.append((xmin, ymin, xmax, ymax))
        color = (0, 0, 255)
        thickness = 10
        cv2.rectangle(end_img, (xmin, ymin), (xmax, ymax), color, thickness)
        
        filter_image = process_image_filter(src_img, hue_range=huerange, saturation_range=satrurationRange, value_range=valueRange)
        if crop:
            points = crop['points']
            xmin, ymin, xmax, ymax = Process.convertPoints2BndBox(points)
            xmin, ymin, xmax, ymax = rotate_bbox(xmin, ymin, xmax, ymax, degrees, width, height)
            src_img = src_img[ymin:ymax, xmin:xmax]
            start_img = start_img[ymin:ymax, xmin:xmax]
            end_img = end_img[ymin:ymax, xmin:xmax]
            filter_image = filter_image[ymin:ymax, xmin:xmax]
        self.output_images['crop'] = src_img
        self.output_images['start'] = start_img
        self.output_images['end'] = end_img
        self.output_images['filter'] = filter_image
        return self.output_images
        
    @staticmethod
    def convertPoints2BndBox(points):
        xmin = float('inf')
        ymin = float('inf')
        xmax = float('-inf')
        ymax = float('-inf')
        for p in points:
            x = p[0]
            y = p[1]
            xmin = min(x, xmin)
            ymin = min(y, ymin)
            xmax = max(x, xmax)
            ymax = max(y, ymax)

        # Martin Kersner, 2015/11/12
        # 0-valued coordinates of BB caused an error while
        # training faster-rcnn object detector.
        if xmin < 1:
            xmin = 1

        if ymin < 1:
            ymin = 1

        return (int(xmin), int(ymin), int(xmax), int(ymax))

    @staticmethod
    def resultSave(save_path, output_images):
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(f'{save_path}/full_image.png', output_images['crop'])
        cv2.imwrite(f'{save_path}/start.png', output_images['start'])
        cv2.imwrite(f'{save_path}/end.png', output_images['end'])
        cv2.imwrite(f'{save_path}/filter.png', output_images['filter'])
        