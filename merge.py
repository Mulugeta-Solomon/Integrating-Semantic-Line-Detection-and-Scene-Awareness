import json 
import numpy as np 
import argparse
import os
import os.path as osp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from logger import Logger
from collections import defaultdict
from tqdm import tqdm
from matplotlib.colors import ListedColormap

def get_data(segPred, linePred, filename, threshold):
    
    for idx in range(len(segPred)):
        if segPred[idx]['filename'] == filename:
            seg_pred = segPred[idx]['seg_pred']
            break
    
    if seg_pred is None:
        raise ValueError(f"No segmentation prediction found for filename: {filename}")


    for idx in range(len(linePred)):
        if linePred[idx]['filename'].split('/')[-1] == filename:
            line_predictions = linePred[idx]['lines_pred']
            score = linePred[idx]['lines_score']
            break

    if line_predictions is None or score is None:
        raise ValueError(f"No line predictions found for filename: {filename}")
    
    score = linePred[idx]['lines_score']
    lines_pred = []
    for i in range(len(score)):
        if score[i] >= threshold:
            lines_pred.append(line_predictions[i])

    return seg_pred, lines_pred

def visualize(image, lines=None, cmap=None, line_annotations=None):
    # Define the custom colors in the order specified by the annotation map
    custom_colors = ['#B91C1C',  # horizontalUpperEdge (0) - bg-red-400
                    '#34D399',  # wallEdge (1) - bg-green-400
                    '#60A5FA',  # horizontalLowerEdge (2) - bg-blue-400
                    '#FBBF24',  # doorEdge (3) - bg-yellow-400
                    '#C084FC',  # windowEdge (4) - bg-magenta-400
                    '#F97316']  # miscellaneousObjects (5) - bg-orange-600
    custom_cmap = ListedColormap(custom_colors)

    if lines is not None:
        lines_np = np.array(lines)
    if line_annotations is not None:
        line_annotations_np = np.array(line_annotations)

    if image is not None:
        plt.imshow(image, cmap=custom_cmap, interpolation='bilinear')
        
    # Visualize the annotated lines
    if lines is not None and line_annotations is not None:
        for i in range(lines_np.shape[0]):
            line = lines_np[i]
            annotation_class = int(line_annotations_np[i])
            color = custom_colors[annotation_class]
            plt.plot([line[0], line[2]], [line[1], line[3]], color=color, linewidth=2)
            plt.scatter([line[0], line[2]], [line[1], line[3]], color='b', s=7, edgecolors='none', zorder=5)
    
    # Create a legend
    legend_labels = ['UpperEdge', 'wallEdge', 'LowerEdge', 'doorEdge', 'windowEdge', 'miscellaneous']
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in custom_colors]
    plt.legend(handles, legend_labels, loc='lower right', fontsize='xx-small')
    
    plt.axis('off')
    plt.show()

    if image is not None:
        plt.imshow(image, cmap=custom_cmap, interpolation='bilinear')

    if lines is not None and line_annotations is not None:
        for i in range(lines_np.shape[0]):
            line = lines_np[i]
            plt.plot([line[0], line[2]], [line[1], line[3]], color='green', linewidth=2)
            plt.scatter([line[0], line[2]], [line[1], line[3]], color='b', s=8, edgecolors='none', zorder=5)
    
    plt.axis('off')
    plt.show()

def merging_algorithm(x1, y1, x2, y2, seg_pred):
    seg_pred = np.array(seg_pred)

    # Compute the Bresenham points
    bresenham_points_ceil = bresenhams_line(np.ceil(x1), np.ceil(y1), np.ceil(x2), np.ceil(y2))
    bresenham_points_floor = bresenhams_line(np.floor(x1), np.floor(y1), np.floor(x2), np.floor(y2))
    
    # Select the best points
    best_points = select_best_points(bresenham_points_ceil, bresenham_points_floor, x1, y1, x2, y2)
    
    annotation = defaultdict(int)
    
    for (x, y) in best_points:
        annotation[seg_pred[int(y), int(x)]] += 1

    ann = max(annotation, key=annotation.get)
    
    return ann

def select_best_points(bresenham_points_ceil, bresenham_points_floor, x1, y1, x2, y2):
    best_points = []
    distances = []
    min_length = min(len(bresenham_points_ceil), len(bresenham_points_floor))
    threshold = 0
    
    for i in range(min_length):
        (xc, yc), (xf, yf) = bresenham_points_ceil[i], bresenham_points_floor[i]
        if (xc, yc) == (xf, yf):
            best_points.append((xc, yc))
            distances.append(compute_distance_to_line(xc, yc, x1, y1, x2, y2))
        else:
            dist_ceil = compute_distance_to_line(xc, yc, x1, y1, x2, y2)
            dist_floor = compute_distance_to_line(xf, yf, x1, y1, x2, y2)
            if dist_ceil < dist_floor:
                best_points.append((xc, yc))
                distances.append(dist_ceil)
                if dist_floor <= threshold:
                    best_points.append((xf, yf))
            else:
                best_points.append((xf, yf))
                distances.append(dist_floor)
                if dist_ceil <= threshold:
                    best_points.append((xc, yc))

    # Compute the average minimum distance
    avg_distance = np.mean(distances)

    # Append remaining points from the longer list if within the threshold
    if len(bresenham_points_ceil) > min_length:
        for (xc, yc) in bresenham_points_ceil[min_length:]:
            dist_ceil = compute_distance_to_line(xc, yc, x1, y1, x2, y2)
            if dist_ceil <= avg_distance:
                best_points.append((xc, yc))
                
    if len(bresenham_points_floor) > min_length:
        for (xf, yf) in bresenham_points_floor[min_length:]:
            dist_floor = compute_distance_to_line(xf, yf, x1, y1, x2, y2)
            if dist_floor <= avg_distance:
                best_points.append((xf, yf))
    
    return best_points

def compute_distance_to_line(x, y, x1, y1, x2, y2):
    # Project the point onto the line
    t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2)
    x_proj = x1 + t * (x2 - x1)
    y_proj = y1 + t * (y2 - y1)
    # Compute the Euclidean distance
    return np.sqrt((x - x_proj) ** 2 + (y - y_proj) ** 2)

def bresenhams_line(x1, y1, x2, y2):
    points = []
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))
        
        if x1 == x2 and y1 == y2:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    
    return points

if __name__ == "__main__":
    logger = Logger("Merge")

    args = argparse.ArgumentParser(description='Merging Algorithm')
    args.add_argument('--segPredPath', type=str, required=True, help='path to segmentation prediction file')
    args.add_argument('--linePredPath', type=str, required=True, help='path to line prediction file')
    args.add_argument('--image_path', type=str, required=True, help='path to image file')
    args.add_argument('--outputPath', type=str, required=True, help='path to save the output file')
    args.add_argument("--threshold", default=0.5, type=float)

    args = args.parse_args()

    logger.info("Reading segmentation prediction file")
    try:
        with open(args.segPredPath, 'r') as file:
            segPred = json.load(file)
    except Exception as e:
        logger.error(f"Failed to read the segmentation prediction file: {e}")
    
    try:
        with open(args.linePredPath, 'r') as file:
            linePred = json.load(file)
    except Exception as e:
        logger.error(f"Failed to read the line prediction file: {e}")
    
    logger.success("Successfully read the prediction files.")

    logger.info("Reading image file")
    try:
        image_paths = [osp.join(args.image_path, fname) for fname in os.listdir(args.image_path) if fname.endswith(('png', 'jpg'))]
        logger.success("Successfully read the image file.")
    except Exception as e:
        logger.error(f"Failed to read the image file: {e}")
    
    threshold = args.threshold
    merged_pred = []
    for image_path in tqdm(image_paths, desc="Merging Algorithm", total=len(image_paths), unit="image"):
        curr = {}
        image = Image.open(image_path)
        filename = osp.basename(image_path)
        seg_pred, line_pred = get_data(segPred, linePred, filename, threshold)
        #logger.success(f"Successfully read the prediction data for {filename}")

        annotations = []
        # Merging Algorithm
        for line in line_pred:
            x1, y1, x2, y2 = line
            annotation = merging_algorithm(x1, y1, x2, y2, seg_pred)
            annotations.append(annotation)

        curr['filename'] = filename
        curr['line_annotations'] = annotations
        curr['lines_pred'] = line_pred
        merged_pred.append(curr)
    
    logger.success("Successfully merged the predictions.")

    logger.info("Saving the output file")
    try:
        with open(args.outputPath, 'w') as file:
            json.dump(merged_pred, file)
            logger.success("Successfully saved the output file.")
    except Exception as e:
        logger.error(f"Failed to save the output file: {e}")
    
    

    for data in merged_pred:
        image_path = osp.join(args.image_path, data['filename'])
        image = Image.open(image_path)

        visualize(image, lines=data['lines_pred'], line_annotations=data['line_annotations'])

    logger.success("Successfully visualized the merged predictions.")





# python -m merge --segPredPath prediction/seg_pred.json --linePredPath prediction/hawp.json --image_path im --outputPath prediction/prediction.json