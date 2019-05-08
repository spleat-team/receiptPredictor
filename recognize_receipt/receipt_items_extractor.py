import cv2
import base64
import tensorflow as tf
import numpy as np

def prep_points(boxes):
    all_points_boxes = []
    for box in boxes:
        curr_box = []
        box_list =  list(box.items());
        for i in range(4):
            label_x, point_x = box_list[i * 2]
            label_y, point_y = box_list[(i * 2) + 1]
            point = (int(point_x), int(point_y))
            curr_box.append(point)

        all_points_boxes.append(curr_box)

    return (all_points_boxes)

def bb_intersection_over_union(boxA, boxB):
    _, boxA_top_left_y = boxA[0];
    _, boxA_bottom_right_y = boxA[1];

    _, boxB_top_left_y = boxB[0];
    _, boxB_bottom_right_y = boxB[1];

    inter_top_left_y = max(boxA_top_left_y, boxB_top_left_y)
    inter_bottom_right_y = min(boxA_bottom_right_y, boxB_bottom_right_y)
    interArea = max(0, inter_bottom_right_y - inter_top_left_y)

    boxAArea = (boxA_bottom_right_y - boxA_top_left_y)
    boxBArea = (boxB_bottom_right_y - boxB_top_left_y)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def foramt_4_points_bb_to_2_points_bb(bounding_box):
    aX, aY = bounding_box[0];
    bX, bY = bounding_box[1];
    cX, cY = bounding_box[2];
    dX, dY = bounding_box[3];

    top_left_x = min(aX, bX, cX, dX)
    top_left_y = min(aY, bY, cY, dY)

    bottom_right_x = max(aX, bX, cX, dX)
    bottom_right_y = max(aY, bY, cY, dY)

    return ([(top_left_x, top_left_y), (bottom_right_x, bottom_right_y)])

def IoU_by_y(dishes_list, prices_list):
    l = list();
    for dish in dishes_list:
        best_iou_value = 0;
        best_iou_price = ();

        dish_box = foramt_4_points_bb_to_2_points_bb(dish)

        for price in prices_list:
            price_box = foramt_4_points_bb_to_2_points_bb(price)

            curr_iou = bb_intersection_over_union(dish_box, price_box)

            if (curr_iou > best_iou_value):
                best_iou_value = curr_iou
                best_iou_price = price

        if (best_iou_value > 0):
            l.append({"price": best_iou_price, "dish": dish})

    return l

def IoU_by_y_with_threshold(dishes_list, prices_list):
    l = list();
    for dish in dishes_list:
        all_prices_in_same_row = list()
        best_iou_value = 0;
        best_iou_price = ();

        dish_box = foramt_4_points_bb_to_2_points_bb(dish)

        for price in prices_list:
            price_box = foramt_4_points_bb_to_2_points_bb(price)

            curr_iou = bb_intersection_over_union(dish_box, price_box)

            if (curr_iou >= 0.5):
                all_prices_in_same_row.append(price)

        if (len(all_prices_in_same_row) > 0):
            all_prices_in_same_row = sorted(all_prices_in_same_row, key=lambda curr_price: curr_price[0][0])

            l.append({"price": all_prices_in_same_row[0], "dish": dish})

    return l

def crop_image(img, bounding_box):
    bb_2_points = foramt_4_points_bb_to_2_points_bb(bounding_box)
    Ax, Ay = bb_2_points[0]
    Bx, By = bb_2_points[1]
    cropped_img = img[Ay:By, Ax:Bx]
    return (cropped_img)


def bb_2_points_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def find_best_contours(img):
    (original_h, original_w, original_z) = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)

    if (original_h <= 35):
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 2)
    else:
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)

    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, original_h))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, rect_kernel)

    contours, hierarchy = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [];

    for i, cnt in enumerate(contours):
        is_found_bigger = False
        [x_i, y_i, w_i, h_i] = cv2.boundingRect(cnt)

        for j, _ in enumerate(contours):
            [x_j, y_j, w_j, h_j] = cv2.boundingRect(contours[j])
            iou = bb_2_points_intersection_over_union([x_i, y_i, x_i + w_i, y_i + h_i],
                                                      [x_j, y_j, x_j + w_j, h_j + y_j])

            if (iou > 0):
                if (w_i * h_i < w_j * h_j):
                    is_found_bigger = True
                    break

        if not (is_found_bigger):
            filtered_contours.append(contours[i])

    filtered_contours = sorted(filtered_contours, key=lambda cnt: cv2.boundingRect(cnt)[0])
    return filtered_contours


def predict_price_contours(filtered_contours, img, digits_model, graph):
    (original_h, original_w, original_z) = img.shape
    predicted_price = ""
    for cnt in filtered_contours:
        [x, y, w, h] = cv2.boundingRect(cnt)

        if (h < 0.2 * original_h and w < 0.2 * original_w):
            h = int(0.2 * original_h)
            w = int(0.2 * original_w)

        current_digit = img[y:y + h, x:x + w]
        current_digit = cv2.cvtColor(current_digit, cv2.COLOR_BGR2GRAY)
        row, col = current_digit.shape[:2]
        bottom = current_digit[row - 2:row, 0:col]
        mean = cv2.mean(bottom)[0]

        if (original_h <= 35):
            bordersize = 5
        else:
            bordersize = 10

        th = cv2.resize(current_digit, None, fx=2, fy=2)
        th = cv2.adaptiveThreshold(th, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 29, 15)

        th = cv2.resize(th, None, fx=0.5, fy=0.5)
        th = cv2.copyMakeBorder(th, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                                borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        th = cv2.resize(th, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)

        colors, counts = np.unique(th.reshape(-1), axis=0, return_counts=True)

        black_ratio = counts[0] / float(28 * 28)
        if (black_ratio >= 0.99):
            continue

        if (h / float(original_h) < 0.5 or black_ratio > 0.90):
            if (predicted_price == "" or "." in predicted_price):
                continue

            predicted_price = predicted_price + "."
            continue

        th = np.reshape(th, newshape=(28, 28, 1))
        img_for_prediction = np.expand_dims(th, axis=0)

        with graph.as_default():
            pred = digits_model.predict(img_for_prediction)[0]

        maximum = np.max(pred)
        predicted_digit = np.where(pred == maximum)[0][0]
        predicted_price = predicted_price + str(predicted_digit)

    return predicted_price

def convert_points_to_images(img, prices_dishes_matches, digits_model, graph):
    ret = list()
    for x in prices_dishes_matches:
        cropped_dish = crop_image(img, x["dish"])
        cropped_price = crop_image(img, x["price"])
        price_contours = find_best_contours(cropped_price)
        price_string = predict_price_contours(price_contours, cropped_price, digits_model, graph)
        _, binframe = cv2.imencode('.jpg', cropped_dish)
        dish_base64 = base64.b64encode(binframe).decode('UTF-8')

        try:
            price_float = float(price_string)
        except ValueError:
            continue

        ret.append({"dish": dish_base64, "price": price_float})
    return ret;

def extract_items(img, dishes, prices, digits_model, graph):
    dishes_points = prep_points(dishes)
    prices_points = prep_points(prices)

    prices_dishes_matches = IoU_by_y_with_threshold(dishes_points, prices_points)
    prices_dishes_matches = sorted(prices_dishes_matches, key=lambda pair: pair["dish"][0][1])

    final_res = convert_points_to_images(img, prices_dishes_matches, digits_model, graph)
    return final_res

