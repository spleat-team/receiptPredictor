import cv2
import pytesseract
import base64


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

            if (curr_iou >= 0.6):
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

def convert_points_to_images(img, prices_dishes_matches):
    ret = list()
    for x in prices_dishes_matches:
        cropped_dish = crop_image(img, x["dish"])
        cropped_price = crop_image(img, x["price"])
        price_string = pytesseract.image_to_string(cropped_price, config='--psm 7')
        price_float = -1
        price_string = price_string.replace(" ", "")

        _, binframe = cv2.imencode('.jpg', cropped_dish)
        dish_base64 = base64.b64encode(binframe).decode('UTF-8')

        try:
            price_float = float(price_string)
        except ValueError:
            continue

        ret.append({"dish": dish_base64, "price": price_float})
    return ret;


def extract_items(img, dishes, prices):
    dishes_points = prep_points(dishes)
    prices_points = prep_points(prices)

    prices_dishes_matches = IoU_by_y_with_threshold(dishes_points, prices_points)
    prices_dishes_matches = sorted(prices_dishes_matches, key=lambda pair: pair["dish"][0][1])

    final_res = convert_points_to_images(img, prices_dishes_matches)
    print(final_res)
    return final_res

