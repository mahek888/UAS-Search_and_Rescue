import cv2 as cv
import numpy as np

def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

# ---------------- HSV ranges ------------------

# Land 
lower_land = np.array([0, 93, 0])
upper_land = np.array([72, 255, 255])

# Ocean
lower_ocean = np.array([90, 50, 0])
upper_ocean = np.array([130, 255, 255])

# Rescue pads
lower_blue_pad = np.array([99, 50, 202]) 
upper_blue_pad = np.array([129, 255, 255])

lower_pink_pad = np.array([118, 18, 176]) 
upper_pink_pad = np.array([170, 110, 255]) 

lower_grey_pad = np.array([0, 0, 137])
upper_grey_pad = np.array([28, 18, 255])

# Casualty severity colors
lower_red = np.array([0, 14, 184])
upper_red = np.array([72, 99, 255])

lower_yellow = np.array([20, 100, 162])
upper_yellow = np.array([36, 255, 255])

lower_green = np.array([39, 92, 15])
upper_green = np.array([50, 244, 255])

# --- Priority & weight -----
pad_names = ["blue", "pink", "grey"]
pad_capacity = {"blue": 4, "pink": 3, "grey": 2}
shape_priority = {"star": 3, "triangle": 2, "square": 1}
emergency_priority = {"red": 3, "yellow": 2, "green": 1}
weight_priority = 10
weight_distance = 1

# --- declaring and initializing output lists ---
all_rescue_ratios = []
all_image_names = []

# --- for processing all task imgs ---
for i in range(1, 11):
    path = f"/Users/mahek/Downloads/task_images/{i}.png"
    img = cv.imread(path)
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # -- making masks --
    mask_land = cv.inRange(imgHSV, lower_land, upper_land)
    mask_ocean = cv.inRange(imgHSV, lower_ocean, upper_ocean)

    mask_blue = cv.inRange(imgHSV, lower_blue_pad, upper_blue_pad)
    mask_pink = cv.inRange(imgHSV, lower_pink_pad, upper_pink_pad)
    mask_grey = cv.inRange(imgHSV, lower_grey_pad, upper_grey_pad)

    mask_red = cv.inRange(imgHSV, lower_red, upper_red)
    mask_yellow = cv.inRange(imgHSV, lower_yellow, upper_yellow)
    mask_green = cv.inRange(imgHSV, lower_green, upper_green)

    # -- unique overlays --
    overlay = img.copy()
    overlay[mask_land > 0] = (0, 255, 255)
    overlay[mask_ocean > 0] = (255, 0, 0)

    overlay[mask_red > 0] = (158, 163, 241)
    overlay[mask_yellow > 0] = (79, 213, 250)
    overlay[mask_green > 0] = (124, 243, 193)

    overlay[mask_blue > 0] = (253, 176, 137)
    overlay[mask_pink > 0] = (254, 161, 230)
    overlay[mask_grey > 0] = (215, 218, 222)

    # -- pad centres --
    centresofpads = {}
    for color_label, mask in [("blue", mask_blue), ("pink", mask_pink), ("grey", mask_grey)]:
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv.contourArea)
            M = cv.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centresofpads[color_label] = (cx, cy)

    # -- casualties --
    casualties = []
    color_ranges = {
        "red": (lower_red, upper_red),
        "yellow": (lower_yellow, upper_yellow),
        "green": (lower_green, upper_green)
                    }

    for color_label, (lower, upper) in color_ranges.items():
        mask = cv.inRange(imgHSV, lower, upper)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv.contourArea(cnt) < 100:  
                continue

            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            corners = len(approx)

            # shape
            if corners == 3:
                shape = "triangle"
            elif 4 <= corners <= 6:
                x, y, w, h = cv.boundingRect(approx)
                aspect_ratio = float(w) / h
                shape = "square" if 0.85 <= aspect_ratio <= 1.15 else "star"
            else:
                shape = "star"

            # -- centroid --
            M = cv.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            casualties.append({
                "shape": shape,
                "emergency": color_label,
                "center": (cx, cy)
            })

    # -- distance matrix --
    casualty_points = np.array([c["center"] for c in casualties])
    pad_points = np.array([centresofpads[p] for p in pad_names])
    dists = np.linalg.norm(casualty_points[:, None, :] - pad_points[None, :, :], axis=2)

    # -- priority score --
    for c in casualties:
        c["priority_score"] = shape_priority[c["shape"]] * emergency_priority[c["emergency"]]

    casualties_sorted = sorted(
        enumerate(casualties),
        key=lambda x: (x[1]["priority_score"], emergency_priority[x[1]["emergency"]]),
        reverse=True
    )

    # -- assigning pads --
    pad_assignments = {p: [] for p in pad_names}
    pad_remaining = pad_capacity.copy()

    for idx, c in casualties_sorted:
        scores = []
        for j, pad in enumerate(pad_names):
            if pad_remaining[pad] > 0:
                score = c["priority_score"] * weight_priority - dists[idx, j] * weight_distance
            else:
                score = -np.inf
            scores.append(score)

        best_pad_idx = np.argmax(scores)
        best_pad = pad_names[best_pad_idx]

        if scores[best_pad_idx] > -np.inf:
            pad_assignments[best_pad].append(c)
            pad_remaining[best_pad] -= 1

    # -- final pad-casualty combinations --
    Image_n = []
    for pad in pad_names:
        pad_list = [[shape_priority[c["shape"]], emergency_priority[c["emergency"]]] for c in pad_assignments[pad]]
        Image_n.append(pad_list)

    # -- pad priority and rescue ratio --
    pad_priority = [sum(shape_priority[c["shape"]] * emergency_priority[c["emergency"]] for c in pad_assignments[p]) for p in pad_names]
    rescue_ratio = sum(pad_priority) / len(casualties)

    image_name = f"Image{i}"
    all_image_names.append(image_name)
    all_rescue_ratios.append(rescue_ratio)

    # -- final output prints --
    print(f"{image_name} = {Image_n}")
    print(f"pad_priority = {pad_priority}")
    print(f"Priority_ratio = {rescue_ratio}")

    
    cv.imshow("Overlay", overlay)
    cv.waitKey(0)


#-- sorting acc to rescue ratio --

sorted_indices = sorted(range(len(all_rescue_ratios)), key=lambda k: all_rescue_ratios[k], reverse=True)

image_by_rescue_ratio = [all_image_names[i] for i in sorted_indices]
print("Images sorted by rescue ratio:", image_by_rescue_ratio)
