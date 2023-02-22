import cv2
import numpy as np
import logging
import math

from sklearn.linear_model import LinearRegression

_SHOW_IMAGE = True

'''
Hough Line Function Parameters
'''
rho = 2                  # Resolution of accumulator buckets in pixels, Larger -> more lines
angle = 1                # Angle resolution of the accumulator in degrees, converted to radians. Larger -> fewer lines found
min_threshold = 15       # minimal of votes to determine line exists. Larger -> fewer lines
minLineLength = 15       # minimum length of segment to be considered a line. In pixels
maxLineGap = 40          # maximum distance between 2 segments to be considered a continuous line

class HandCodedLaneFollower(object):

    def __init__(self, car=None):
        logging.info('Creating a HandCodedLaneFollower...')
        self.car = car
        self.curr_steering_angle = 90

    def follow_lane(self, frame):

        frame = np.array(frame)

        # Main entry point of the lane follower
        show_image("orig", frame)

        lane_lines, frame = detect_lane(frame)

        final_frame = self.steer(frame, lane_lines)

        return final_frame

    def steer(self, frame, lane_lines):
        logging.debug('steering...')
        if len(lane_lines) == 0:
            logging.error('No lane lines detected, nothing to do.')
            return frame

        new_steering_angle = compute_steering_angle(frame, lane_lines)
        self.curr_steering_angle = stabilize_steering_angle(self.curr_steering_angle, new_steering_angle, len(lane_lines))

        if self.car is not None:
            self.car.front_wheels.turn(self.curr_steering_angle)
        curr_heading_image = display_heading_line(frame, self.curr_steering_angle)
        show_image("heading", curr_heading_image)

        return curr_heading_image

############################
# Frame processing steps
############################
def detect_lane(frame):
    logging.debug('detecting lane lines...')

    edges = detect_edges(frame)
    show_image('edges', edges)

    cropped_edges = region_of_interest(edges)
    show_image('edges cropped', cropped_edges)

    line_segments = detect_line_segments(cropped_edges)
    line_segment_image = display_lines(frame, line_segments)
    show_image("hough line segments", line_segment_image)

    lane_lines = average_slope_intercept(frame, line_segments)
    lane_lines_image = display_lines(frame, lane_lines)
    show_image("lane lines", lane_lines_image)

    return lane_lines, lane_lines_image

# detects blue lines formed by edge detected algo
def detect_edges(frame):
    edges = cv2.Canny(np.array(frame), 50, 200)
    return edges

# Used to remove segmentations that are too far away and are irrelevant to the current state of the bot.
def region_of_interest(img):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    mask = np.zeros_like(img)
    height, width = img.shape

    scale_w = 7 / 16
    scale_h = 11 / 18

    left_bottom = [0, height - 1]
    right_bottom = [width - 1, height - 1]
    left_up = [scale_w * width, scale_h * height]
    right_up = [(1 - scale_w) * width, scale_h * height]
    vertices = np.array([[left_bottom, left_up, right_up, right_bottom]], dtype=np.int32)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    show_image("ROI", mask)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def detect_line_segments(cropped_edges):
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle *  np.pi / 180, min_threshold, 
                                    np.array([]), minLineLength=minLineLength, maxLineGap=maxLineGap)
    return line_segments

def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        logging.info('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary # right lane line segment should be on left 2/3 of the screen

    threshold_angle = 25  # if the line angle is between -25 to 25 degrees, lines are discarded
    threshold_slope = math.tan(threshold_angle / 180 * math.pi)

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]

            if x1 == x2: # line is vertical, skip
                continue
            if abs(slope) < threshold_slope:  # remove horizontal lines
                continue
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(left_fit_average[0], left_fit_average[1], 'l', height))
    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(right_fit_average[0], right_fit_average[1], 'r', height))

    return lane_lines

# TODO: This is fked
def compute_steering_angle(frame, lane_lines):
    """ Find the steering angle based on lane line coordinate
        We assume that camera is calibrated to point to dead center
    """
    if len(lane_lines) == 0:
        return -90

    height, width, _ = frame.shape
    if len(lane_lines) == 1:
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
    else:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        camera_mid_offset_percent = 0.0 # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
        mid = int(width / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

    # find the steering angle, which is angle between navigation direction to end of center line
    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
    steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel

    return steering_angle

def stabilize_steering_angle(curr_steering_angle, new_steering_angle, num_of_lane_lines, max_angle_deviation_two_lines=5, max_angle_deviation_one_lane=1):
    if num_of_lane_lines == 2 :
        # if both lane lines detected, then we can deviate more
        max_angle_deviation = max_angle_deviation_two_lines
    else :
        # if only one lane detected, don't deviate too much
        max_angle_deviation = max_angle_deviation_one_lane
    
    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle
                                        + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle
    return stabilized_steering_angle

############################
# Utility Functions
############################
def display_lines(frame, lines, line_color=(0, 255, 0), line_width=10):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

def outlierCleaner(predictions, x, y, inlier_percent=0.9):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual y values).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (x, y, error).
    """
    threshold = 10  # if the biggest error is greater than 5 pixels, we perform outliers remove
    errs = np.fabs(y - predictions)
    max_err = max(errs)[0]
    if max_err > threshold:  # if the biggest error is greater than 5 pixels, we remove the outliers
        k = math.ceil(errs.size * inlier_percent)
        survived_idx = np.argsort(errs, axis=0)[:k]  # find the number of k min errs, and return their index
        if survived_idx.size > 0:
            x = np.take(x, survived_idx)
            y = np.take(y, survived_idx)
    return (x, y)

# Unused
def regress_a_line(img, x, y):
    """ regress a line from x, y and add it to img
    (1) use a linear regressor to fit the data (x,y)
    (2) remove outlier, and then fit the cleaned data again to get slope and intercept
    (3) find the two ends of the desired line by using slope and intercept
    
    :param img: input image
    :param x: x coordinate
    :param y: y coordinate
    :param color: line color
    :param thickness: thickness of the line  
    """
    reg = LinearRegression()
    reg.fit(x, y)

    # identify and remove outliers
    cleaned_data = []
    try:
        predictions = reg.predict(x)
        cleaned_data = outlierCleaner(predictions, x, y)
    except NameError:
        print("err in regression prediction")

    if len(cleaned_data) > 0:
        x, y = cleaned_data   
        # refit cleaned data!
        try:
            reg.fit(x, y)
        except NameError:
            print("err in reg.fit for cleaned data")
    else:
        print("outlierCleaner() is returning an empty list, no refitting to be done")

    height = img.shape[0]
    slope = reg.coef_
    inter = reg.intercept_

    # find the two end points of the line by using slope and iter, and then visulize the line
    if slope < 0:  # left lane
        p1_x, p1_y, p2_x, p2_y = make_points(slope, inter, 'l', height)
    else:  # right lane
        p1_x, p1_y, p2_x, p2_y = make_points(slope, inter, 'r', height)
    return [[p1_x, p1_y, p2_x, p2_y]]

def display_heading_line(frame, steering_angle, line_color=(255, 0, 0), line_width=5, ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # Note: the steering angle of:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right 
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)
    print("Steering Angle: "+ str(steering_angle))
    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
    return heading_image

def length_of_line_segment(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)
        cv2.waitKey()

pre_l_slopes = []
pre_l_inters = []  
pre_r_slopes = []
pre_r_inters = []

def make_points(slope, inter, side, height):
    number_buffer_frames = 3
    scale_y = 0.65
    top_y = int(float(height) * scale_y)  # fix the y coordinates of the top point, so that the line is more stable

    if side == 'l':
        if len(pre_l_slopes) == number_buffer_frames:  # reach the max
            pre_l_slopes.pop(0)  # remove the oldest frame
            pre_l_inters.pop(0)

        pre_l_slopes.append(slope)
        pre_l_inters.append(inter)
        slope = sum(pre_l_slopes) / len(pre_l_slopes)
        inter = sum(pre_l_inters) / len(pre_l_inters)

        p1_y = height-1
        p1_x = int((float(p1_y)-inter)/slope)
        p2_y = top_y
        p2_x = int((float(p2_y)-inter)/slope)
    else: 
        if len(pre_r_slopes) == number_buffer_frames:  # reach the max
            pre_r_slopes.pop(0)  # remove the oldest frame
            pre_r_inters.pop(0)

        pre_r_slopes.append(slope)
        pre_r_inters.append(inter)
        slope = sum(pre_r_slopes) / len(pre_r_slopes)
        inter = sum(pre_r_inters) / len(pre_r_inters)

        p1_y = top_y
        p1_x = int((float(p1_y)-inter)/slope)
        p2_y = height-1
        p2_x = int((float(p2_y)-inter)/slope)

    return [[p1_x, p1_y, p2_x, p2_y]]

############################
# Test Functions
############################
def test_photo(test_image):
    land_follower = HandCodedLaneFollower()
    #frame = cv2.imread(test_image)
    combo_image = land_follower.follow_lane(test_image)
    show_image('final', combo_image, True)