import os
import pandas as pd
import numpy as np
import cv2
import skimage.exposure

def midpoint(node_list):
    node_len = len(node_list)
    x_coords = [node[0] for node in node_list]
    y_coords = [node[1] for node in node_list]
    return [sum(x_coords) / node_len, sum(y_coords) / node_len]


def remove_close_nodes(nodes, min_dist=0.5):
    new_nodes = nodes

    for node in nodes:
        distance = np.sum((nodes - node) ** 2, axis=1)
        close_nodes_ind = np.where(distance < min_dist)[0]

        if len(close_nodes_ind) > 1:
            new_node = midpoint(nodes[close_nodes_ind])
            new_nodes = np.delete(nodes, close_nodes_ind, axis=0)
            new_nodes = np.append(new_nodes, [new_node], axis=0)
            new_nodes = remove_close_nodes(new_nodes, min_dist)
            break

    return new_nodes

def generate_image(img_list, stack=0, slide=0, channel=1, bg_thresh=10, adaptive_thresh=171,
                   erosion_it=3, dilation_it=3, min_dist=300, gamma=0.7, gain=1, connectivity=10, circle_radius=4):

    frame = np.array(img_list[stack].get_frame(z=slide, t=0, c=channel))
    resized = cv2.resize(frame, (2048, 2048), interpolation=cv2.INTER_AREA)
    ret, th1 = cv2.threshold(resized, bg_thresh, 255, cv2.THRESH_TOZERO)
    th2 = cv2.adaptiveThreshold(th1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY, adaptive_thresh, 0)
    th3 = th2
    kernel = np.ones((3, 3), np.uint8)
    th3 = cv2.erode(th3, kernel, iterations=erosion_it)
    th3 = cv2.dilate(th3, kernel, iterations=dilation_it)
    components = cv2.connectedComponentsWithStats(th3, connectivity, cv2.CV_32S)
    centers = components[3]

    cimg = cv2.merge((resized, resized, resized))
    gimg = skimage.exposure.adjust_gamma(cimg, gamma=gamma, gain=gain)

    new_centers = remove_close_nodes(centers, min_dist=min_dist)

    for center in new_centers:
        cv2.circle(gimg, (int(center[0]), int(center[1])), circle_radius, (255, 0, 0), thickness=2)

    gimg = cv2.cvtColor(gimg, cv2.COLOR_RGB2BGR)

    return gimg, len(new_centers)

def download_image(option, export_path, img_list, stack_list, stack_dict_list,
                   stack, zframe, channel,
                   bg_thresh, adaptive_thresh, erosion_it, dilation_it,
                   min_dist, gamma, gain, connectivity, circle_radius):

    all_data = []

    if option == "image":
        export_file_path = os.path.join(export_path, f"STACK_{stack}_SLIDE_{zframe}_CHANNEL_{channel}.png")
        img, centers_no = generate_image(img_list,stack, zframe, channel,
                   bg_thresh, adaptive_thresh, erosion_it, dilation_it,
                   min_dist, gamma, gain, connectivity, circle_radius)
        cv2.imwrite(export_file_path, img)
        all_data = [{"STACK" : stack, "SLIDE": zframe, "CHANNEL": channel, "NUM_CELL": centers_no}]

    if option == "stack":
        z_list = stack_dict_list[stack]['Z_LIST']
        for zframe in z_list:
            data = download_image("image", export_path, img_list, stack_list, stack_dict_list,
                           stack, zframe, channel,
                           bg_thresh, adaptive_thresh, erosion_it, dilation_it,
                           min_dist, gamma, gain, connectivity, circle_radius)
            all_data.extend(data)

    if option == "all":
        for stack in stack_list:
            data = download_image("stack", export_path, img_list, stack_list, stack_dict_list,
                           stack, zframe, channel,
                           bg_thresh, adaptive_thresh, erosion_it, dilation_it,
                           min_dist, gamma, gain, connectivity, circle_radius)
            all_data.extend(data)

    return all_data