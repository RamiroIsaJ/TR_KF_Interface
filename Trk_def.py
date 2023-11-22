import numpy as np
import cv2
import glob
import math
import os
import pandas as pd
from skimage import morphology
from skimage.metrics import structural_similarity
# print(cv2.__version__)


def f_sorted(files_, id_sys):
    symbol = '\\' if id_sys == 0 else '/'
    ids = []
    for f in files_:
        parts = f.split(symbol)
        name_i = parts[len(parts) - 1]
        ids.append(name_i.split('.')[0].split('_')[-1])
    ids_1 = list(map(int, ids))
    idx_1 = int(np.where(np.array(ids_1) == 1)[0])
    if len(ids[idx_1]) >= 2:
        ids = list(map(str, ids))
        ids.sort(key=str)
    else:
        ids = list(map(int, ids))
        ids.sort(key=int)
    file_r = []
    for i in range(len(files_)):
        parts = files_[i].split(symbol)
        name = parts[len(parts) - 1].split('.')
        exp = name[0].split('_')
        if len(exp) >= 2:
            n_exp = exp[0]
            for j in range(1, len(exp)-1):
                n_exp += '_' + exp[j]
            n_name = n_exp + '_' + str(ids[i]) + '.' + name[1]
        else:
            n_name = str(ids[i]) + '.' + name[1]

        if id_sys == 0:
            n_file = (parts[0] + symbol)
        else:
            n_file = (symbol + parts[0])
        for j in range(1, len(parts)-1):
            n_file += (parts[j] + symbol)
        n_file += n_name
        file_r.append(n_file)
    return file_r


def load_image_i(orig, i, type_, filenames, exp, id_sys):
    symbol = '\\' if id_sys == 0 else '/'
    if len(filenames) == 0:
        filenames = [img for img in glob.glob(orig+type_)]
        # filenames.sort()
        filenames = f_sorted(filenames, id_sys)
    if i < len(filenames):
        name = filenames[i]
        parts = name.split(symbol)
        exp, name_i = parts[len(parts)-2], parts[len(parts)-1]
        # read image
        image_ = cv2.imread(name)
    else:
        image_, name_i = [], []
    return filenames, image_, exp, name_i


def update_dir(path):
    path_s = path.split('/')
    cad, path_f = len(path_s), path_s[0]
    for p in range(1, cad):
        path_f += '\\' + path_s[p]
    return path_f


def bytes_(img, m, n):
    ima = cv2.resize(img, (m, n))
    return cv2.imencode('.png', ima)[1].tobytes()


def preprocessing(img):
    image_gray_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clh = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(10, 10))
    clh_img = clh.apply(image_gray_)
    # blurred = cv2.GaussianBlur(clh_img, (3, 5), 0)
    return clh_img


def show_features(img, features_):
    for i in features_:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    return img


def features_img(img, v_th, ide, difference, relation, ima_res):
    ima_gray = preprocessing(img)
    m, n = ima_gray.shape
    if ide == 0:
        ima_res, diff = np.copy(ima_gray), np.copy(ima_gray)
    else:
        (score, diff) = structural_similarity(ima_res, ima_gray, full=True)
        difference.append(score)
        ima_res = np.copy(ima_gray)
        diff = (diff * 255).astype(np.uint8)
        thresh_diff = cv2.threshold(diff, v_th, 255, cv2.THRESH_TOZERO_INV)[1]
        values = np.sum(thresh_diff > np.min(thresh_diff.ravel()))
        relation.append(values / (m*n))
    thresh = cv2.threshold(ima_gray, v_th, 255, cv2.THRESH_TOZERO_INV)[1]
    arr = thresh > 0
    thresh1 = morphology.remove_small_objects(arr, min_size=180, connectivity=1).astype(np.uint8)
    thresh1 = thresh1.astype(np.uint8)
    thresh2 = morphology.remove_small_objects(arr, min_size=10, connectivity=1).astype(np.uint8)
    thresh2 = thresh2.astype(np.uint8)
    thresh_f = thresh2 - thresh1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(thresh_f, cv2.MORPH_CLOSE, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=1)
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    features_ = []
    for c in contours:
        mts = cv2.moments(c)
        cx = int(mts["m10"] / mts["m00"])
        cy = int(mts["m01"] / mts["m00"])
        features_.append((cx, cy))
    features_ = np.asarray(sorted(features_, key=lambda k: [k[0], k[1]]))
    frame = show_features(img, features_)
    return features_, frame, difference, relation, ima_res


def distance(x, y):
    return math.hypot(y[0] - x[0], y[1] - x[1])


def sort_features(last_f, curr_f, max_v, min_v):
    for i in range(len(last_f)):
        xy = last_f[i, :2]
        idx = None
        for kd in range(curr_f.shape[0]):
            dist = distance(xy, curr_f[int(kd), :])
            if max_v > dist > min_v:
                idx = kd
                break
        if idx is not None:
            last_f[i, 2:4] = curr_f[idx, :]
            last_f[i, 4], last_f[i, 5] = 1, 0
        else:
            last_f[i, 4], last_f[i, 5] = 0, last_f[i, 5] + 1
    return np.array(last_f)


def find_seq_feat(k, features_, tab_feat, max_v, min_v):
    if k == 0:
        tab_feat = np.append(features_, features_, axis=1)
        tab_feat = np.append(tab_feat, np.ones((len(features_), 2)), axis=1)
    else:
        tab_feat = sort_features(tab_feat, features_, max_v, min_v)

    idx = np.where((tab_feat[:, 4] == 1) & (tab_feat[:, 5] < 5))
    f_track = np.array(tab_feat[idx, :])
    f_track = f_track.reshape((f_track.shape[0] * f_track.shape[1]), f_track.shape[2])
    return tab_feat, f_track


def find_track_feat(k, features_, tab_feat, max_v, min_v):
    if k == 1:
        tab_feat = np.append(features_, features_, axis=1)
        tab_feat = np.append(tab_feat, np.ones((len(features_), 2)), axis=1)
    elif k == 9:
        tab_feat = sort_features(tab_feat, features_, max_v, min_v)
        idx = np.where((tab_feat[:, 4] == 0) & (tab_feat[:, 5] > 5))
        tab_feat = np.delete(tab_feat, idx, 0)
    else:
        tab_feat = sort_features(tab_feat, features_, max_v, min_v)
    return tab_feat


def tracking_feat(frame, tracker, f_track, delta):
    # update tracker
    tracker.update(f_track, delta)
    errors, move_dist = [], [0]
    n = len(tracker.tracks)
    for j in range(n):
        if len(tracker.tracks[j].trace) > 1:
            x = int(tracker.tracks[j].trace[-1][0, 0])
            y = int(tracker.tracks[j].trace[-1][0, 1])
            # compute error
            errors.append(distance(f_track[j, :], tracker.tracks[j].trace[-1][0, :]))
            # compute distances
            n1 = len(tracker.tracks[j].trace)
            move_dist.append(distance(tracker.tracks[j].trace[n1 - 2][0, :], tracker.tracks[j].trace[n1 - 1][0, :]))
            # graphics
            cv2.circle(frame, (x, y), 6, (255, 20, 25), -1)
            cv2.rectangle(frame, (x - 15, y - 15), (x + 15, y + 15), (0, 255, 0), 2)
            cv2.putText(frame, str(tracker.tracks[j].track_id), (x - 10, y - 20), 0, 0.5, (0, 55, 255), 2)
        cv2.circle(frame, (int(f_track[j, 0]), int(f_track[j, 1])), 6, (0, 0, 0), -1)
    r_mse = np.round(np.sqrt(np.sum(np.array(errors)) / n), 4)
    r_mse = 100 if r_mse == 0 else r_mse
    mean_dist = np.round(np.mean(np.array(move_dist)), 4)
    std_dist = np.round(np.std(np.array(move_dist)), 4)
    mean_vel = np.round(np.mean(np.array(move_dist) / delta), 4)
    std_vel = np.round(np.std(np.array(move_dist) / delta), 4)
    return frame, r_mse, move_dist, mean_dist, std_dist, mean_vel, std_vel


def outliers(data):
    # Median imputation
    data = pd.DataFrame(data, columns=['Values'])
    q1, q3 = data['Values'].quantile(0.25), data['Values'].quantile(0.75)
    iqr = q3 - q1
    lower_tail, upper_tail = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    for i in data['Values']:
        if i > upper_tail or i < lower_tail:
            data['Values'] = data['Values'].replace(i, np.median(data['Values']))
    return np.round(np.mean(data['Values']), 4), np.round(np.std(data['Values']), 4)


def save_image_out(ima_out_, path_des_, name_ima):
    root_ima = os.path.join(path_des_, name_ima+'.jpg')
    cv2.imwrite(root_ima, ima_out_)
    print('-------------------------------------')
    print('..... Image saved successfully .....')
    print('-------------------------------------')


def save_csv_file(results_, path_des_, id_):
    root_file = os.path.join(path_des_, 'Results_'+str(id_)+'.csv')
    results_.to_csv(root_file, index=False)
    print('----------------------------------------------')
    print('..... Save data in CSV file successfully .....')
    print('----------------------------------------------')
    
