# Ramiro Isa-Jara, ramiro.isaj@gmail.com

import PySimpleGUI as sg
import numpy as np
import Trk_def as Chg
from datetime import datetime
from Tracker_def import Tracker
import matplotlib.pyplot as plt

# -------------------------------
# Adjust size screen
# -------------------------------
Screen_size = 10
# -------------------------------
sg.theme('LightGrey1')
m1, n1 = 450, 400
img = np.ones((m1, n1, 1), np.uint8)*255

layout1 = [[sg.Radio('Windows', "RADIO1", enable_events=True, default=True, key='_SYS_')],
           [sg.Radio('Linux', "RADIO1", enable_events=True, key='_LIN_')], [sg.Text('')]]

layout2 = [[sg.Checkbox('*.jpg', default=True, key="_IN1_")], [sg.Checkbox('*.png', default=False, key="_IN2_")],
           [sg.Checkbox('*.tiff', default=False, key="_IN3_")]]

layout3 = [[sg.Text('Min Thresh:', size=(12, 1)), sg.InputText('90', key='_ITH_', size=(5, 1)),
            sg.Text('', size=(2, 1)),
            sg.Text('Max-Distance:', size=(12, 1)), sg.InputText('20', key='_MAD_', size=(5, 1))],
           [sg.Text('Ini-Feature:', size=(12, 1)), sg.InputText('50', key='_INF_', size=(5, 1)),
            sg.Text('', size=(2, 1)),
            sg.Text('Min-Distance:', size=(12, 1)), sg.InputText('5', key='_MID_', size=(5, 1))],
           [sg.Text('Fin-Feature:', size=(12, 1)), sg.InputText('100', key='_FNF_', size=(5, 1)),
            sg.Text('', size=(2, 1)),
            sg.Text('Delta t:', size=(12, 1)), sg.InputText('0.70', key='_DET_', size=(5, 1))]]

layout4 = [[sg.Text('Source : ', size=(10, 1)), sg.InputText(size=(30, 1), key='_ORI_'), sg.FolderBrowse()],
           [sg.Text('Destiny: ', size=(10, 1)), sg.InputText(size=(30, 1), key='_DES_'), sg.FolderBrowse()]]

layout5 = [[sg.Text('Dist thresh:', size=(11, 1)), sg.InputText('150', key='_DTH_', size=(5, 1))],
           [sg.Text('Frames skip:', size=(11, 1)), sg.InputText('20', key='_FSK_', size=(5, 1))],
           [sg.Text('Max trace  :', size=(11, 1)), sg.InputText('5', key='_MTR_', size=(5, 1))]]

layout6 = [[sg.T("", size=(30, 1)), sg.Text('NO PROCESS', size=(42, 1), key='_MES_', text_color='DarkRed')]]

layout7 = [[sg.Text('Current time: ', size=(10, 1)), sg.Text('', size=(12, 1), key='_TAC_'), sg.T("", size=(2, 1)),
            sg.Text('Start time:', size=(10, 1)), sg.Text('-- : -- : --', size=(11, 1), key='_TIN_', text_color='blue'),
            sg.Text('Finish time: ', size=(9, 1)), sg.Text('-- : -- : --', size=(9, 1), key='_TFI_', text_color='red')],
           [sg.Text('Experiment:', size=(10, 1)), sg.InputText('', key='_NEX_', size=(16, 1)),
            sg.Text('Image:', size=(5, 1)), sg.InputText('', key='_NIM_', size=(17, 1)),
            sg.Text('N. image:', size=(7, 1)), sg.InputText('', key='_CIM_', size=(7, 1))]]

layout8 = [[sg.Text('N_Feat_D:', size=(12, 1)), sg.InputText('', key='_NFD_', size=(6, 1)),
            sg.Text('N_Feat_T:', size=(12, 1)), sg.InputText('', key='_NFT_', size=(6, 1))],
           [sg.Text('%Repetition:', size=(12, 1)), sg.InputText('', key='_RPC_', size=(6, 1)),
            sg.Text('%Mean_Rep: ', size=(12, 1)), sg.InputText('', key='_RPM_', size=(6, 1))],
           [sg.Text('Max_Feat_D:', size=(12, 1)), sg.InputText('', key='_MFD_', size=(6, 1)),
            sg.Text('Max_Feat_T:', size=(12, 1)), sg.InputText('', key='_MFT_', size=(6, 1))],
           [sg.Text('Mean_Feat_D:', size=(12, 1)), sg.InputText('', key='_PFD_', size=(6, 1)),
            sg.Text('Mean_Feat_F:', size=(12, 1)), sg.InputText('', key='_PFT_', size=(6, 1))]]

particles = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
layout9 = [[sg.Text('Graphics: ', size=(13, 1)),
            sg.Combo(values=particles, size=(10, 1), enable_events=True, key='_GRA_')],
           [sg.Text('Mean_RMSE:', size=(13, 1)), sg.InputText('', key='_MER_', size=(12, 1))],
           [sg.Text('Mean_Distance:', size=(13, 1)), sg.InputText('', key='_MDI_', size=(12, 1))],
           [sg.Text('Mean_Velocity:', size=(13, 1)), sg.InputText('', key='_MVL_', size=(12, 1))],
           [sg.Text('Max_Distance:', size=(13, 1)), sg.InputText('', key='_MMD_', size=(12, 1))]]

v_image = [sg.Image(filename='', key="_IMA_")]
# columns
col_1 = [[sg.Frame('', [v_image])]]
col_2 = [[sg.Frame('Operative System: ', layout1, title_color='Blue'),
          sg.Frame('Type image: ', layout2, title_color='Blue'), sg.Frame('Settings: ', layout3, title_color='Blue')],
         [sg.Frame('Directories: ', layout4, title_color='Blue'), sg.Frame('Tracker: ', layout5, title_color='Blue')],
         [sg.T(" ", size=(15, 1)), sg.Button('Evaluate', size=(8, 1)), sg.Button('Tracking', size=(8, 1)),
          sg.Button('Pause', size=(8, 1)), sg.Button('Finish', size=(8, 1))],
         [sg.Frame('', layout6)], [sg.Frame('', layout7)],
         [sg.Frame('Eval Results:', layout8, title_color='Blue'), sg.Text(''),
          sg.Frame('Tracking Results:', layout9, title_color='Blue')]]

layout = [[sg.Column(col_1), sg.Column(col_2)]]

# Create the Window
window = sg.Window('TR_KF Interface', layout, font="Helvetica "+str(Screen_size), finalize=True)
window['_IMA_'].update(data=Chg.bytes_(img, m1, n1))
# ---------------------------------------------------------------------
eval_c, finish_, eval_press, track_c, track_press = False, False, False, False, False
filenames, exp, path_org, type_i, tab_features, n_features, tr_features, rms_errors = [], [], [], [], [], [], [], []
tot_dist, mean_dist = [], []
i, id_sys, tracker, delta = -1, 0, None, 0

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read(timeout=10)
    window.Refresh()
    now = datetime.now()
    now_time = now.strftime("%H : %M : %S")
    window['_TAC_'].update(now_time)

    if event is None or event == sg.WIN_CLOSED:
        break

    if event == 'Finish':
        print('FINISH')
        if finish_ or track_c or eval_c:
            window['_IMA_'].update(data=Chg.bytes_(img, m1, n1))
            i, filenames = -1, []
            finish_, track_c, eval_c = False, False, False
            tab_features, n_features, tr_features, rms_errors, tot_dist, mean_dist, mean_vel = [], [], [], [], [], [], []
            window['_TIN_'].update('-- : -- : --')
            window['_TFI_'].update('-- : -- : --')
            window['_NEX_'].update('')
            window['_NIM_'].update('')
            window['_CIM_'].update('')
            window['_MES_'].update('Process is completed')

    if event == 'Pause':
        eval_c, track_c, eval_pres, track_press = False, False, False, False

    if event == 'Tracking':
        print('SELECT TRACKING')
        if values['_SYS_'] is True:
            id_sys = 0
            path_org = Chg.update_dir(values['_ORI_']) + "\\"
            path_org = r'{}'.format(path_org)
            path_des = Chg.update_dir(values['_DES_']) + "\\"
            path_des = r'{}'.format(path_des)

        else:
            id_sys = 1
            path_org, path_des = values['_ORI_'] + '/', values['_DES_'] + '/'

            # -----------------------------------------------------------------
        if values['_IN2_']:
            type_i = "*.png"
        elif values['_IN3_']:
            type_i = "*.tiff"
        else:
            type_i = "*.jpg"
            # ------------------------------------------------------------------
        if len(path_org) > 1 and finish_ is False:
            now_time = now.strftime("%H : %M : %S")
            window['_TIN_'].update(now_time)
            window['_MES_'].update('Evaluate is running')
            d_thresh = int(values['_DTH_'])
            frame_skip = int(values['_FSK_'])
            max_trace = int(values['_MTR_'])
            tracker = Tracker(d_thresh, frame_skip, max_trace)
            track_c = True
        else:
            sg.Popup('Error', ['Information not valid or Finish process...'])

    if event == 'Evaluate':
        print('SELECT EVALUATE')
        if values['_SYS_'] is True:
            id_sys = 0
            path_org = Chg.update_dir(values['_ORI_']) + "\\"
            path_org = r'{}'.format(path_org)
            path_des = Chg.update_dir(values['_DES_']) + "\\"
            path_des = r'{}'.format(path_des)
        else:
            id_sys = 1
            path_org, path_des = values['_ORI_']+'/', values['_DES_']+'/'
            # path_org = os.path.join(values['_ORI_']) + '/'

        # -----------------------------------------------------------------
        if values['_IN2_']:
            type_i = "*.png"
        elif values['_IN3_']:
            type_i = "*.tiff"
        else:
            type_i = "*.jpg"
        # ------------------------------------------------------------------
        if len(path_org) > 1 and finish_ is False:
            now_time = now.strftime("%H : %M : %S")
            window['_TIN_'].update(now_time)
            window['_MES_'].update('Evaluate is running')
            eval_c = True
        else:
            sg.Popup('Error', ['Information not valid or Finish process...'])

    if eval_c:
        print('EVALUATE PROCESS')
        v_thresh = int(values['_ITH_'])
        d_max, d_min = int(values['_MAD_']), int(values['_MID_'])
        i += 1
        filenames, image, exp, name = Chg.load_image_i(path_org, i, type_i, filenames, exp, id_sys)
        if len(image) == 0 and i > 0:
            eval_c, eval_press = False, True
            continue
        elif len(image) == 0 and i == 0:
            eval_c = False
            continue

        window['_NEX_'].update(exp)
        window['_NIM_'].update(name)
        window['_CIM_'].update(i)

        features_, ima_out = Chg.features_img(image, v_thresh)
        window['_IMA_'].update(data=Chg.bytes_(ima_out, m1, n1))
        n_features.append(features_.shape[0])

        tab_features, feat_track = Chg.find_seq_feat(i, features_, tab_features, d_max, d_min)
        tr_features.append(feat_track.shape[0])
        # Compute % repeatability
        val_min, val_max = min(feat_track.shape[0], features_.shape[0]), max(feat_track.shape[0], features_.shape[0])
        rep = np.round((val_min / val_max) * 100, 1)
        window['_NFD_'].update(features_.shape[0])
        window['_NFT_'].update(feat_track.shape[0])
        window['_RPC_'].update(rep)

    if eval_press:
        print('EVALUATE RESULTS')
        window['_MES_'].update('Evaluate successfully')
        n_feat, tr_feat = np.array(n_features), np.array(tr_features)
        rep_ = (tr_feat / n_feat) * 100
        max_n, max_tr = np.max(n_feat[1:]), np.max(tr_feat[1:])
        mean_n, mean_tr = np.round(np.mean(n_feat[1:]), 1), np.round(np.mean(tr_feat[1:]), 1)
        rep_p = np.round(np.mean(tr_feat / n_feat) * 100, 1)
        window['_MFD_'].update(max_n)
        window['_MFT_'].update(max_tr)
        window['_PFD_'].update(mean_n)
        window['_PFT_'].update(mean_tr)
        window['_RPM_'].update(rep_p)

        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(n_feat, 'o-r', label='Feat-detected')
        ax1.plot(tr_feat, '*-b', label='Feat-tracking')
        ax1.set_title('Feat detected vs. tracked')
        ax1.set_xlabel('Number of frames')
        ax1.set_ylabel('Number of features')
        ax1.legend(loc='upper right')
        ax1.grid()

        ax2.plot(rep_, 'o-g', label='rep_factor')
        ax2.set_title('Repeatability factor in frames')
        ax2.set_xlabel('Number of frames')
        ax2.set_ylabel('% Repeatability factor')
        ax2.legend(loc='upper right')
        ax2.grid()
        plt.show()
        eval_press, finish_ = False, True

    if track_c:
        print('TRACK PROCESS')
        v_thresh = int(values['_ITH_'])
        d_max, d_min = int(values['_MAD_']), int(values['_MID_'])
        ini_feat, fin_feat = int(values['_INF_']), int(values['_FNF_'])
        delta = float(values['_DET_'])
        i += 1
        filenames, image, exp, name = Chg.load_image_i(path_org, i, type_i, filenames, exp, id_sys)
        if len(image) == 0 and i > 0:
            track_c, track_press = False, True
            continue
        elif len(image) == 0 and i == 0:
            eval_c = False
            continue

        window['_NEX_'].update(exp)
        window['_NIM_'].update(name)
        window['_CIM_'].update(i)
        window['_MES_'].update('Tracking is running')

        features_, ima_out = Chg.features_img(image, v_thresh)

        tab_features = Chg.find_track_feat(i, features_, tab_features, d_max, d_min)
        if i > 10:
            print('this......' + str(tab_features.shape[0]))
            feat_tracking = tab_features[ini_feat:fin_feat, 2:4]
            ima_out, error, dists, mean_d = Chg.tracking_feat(image, tracker, feat_tracking, delta)
            rms_errors.append(error)
            tot_dist.append(np.array(dists))
            mean_dist.append(mean_d)
            window['_MER_'].update(np.round(error, 4))
            window['_MDI_'].update(mean_d)
            mean_vel = np.round(mean_d / delta, 4)
            window['_MVL_'].update(mean_vel)

        window['_IMA_'].update(data=Chg.bytes_(ima_out, m1, n1))

    if track_press:
        print('TRACK RESULTS')
        n_errors = np.array(rms_errors)
        m_dist = np.cumsum(np.array(mean_dist[2:]))
        m_velocity = (np.array(mean_dist[2:])) / delta
        window['_MER_'].update(np.round(np.mean(n_errors), 4))
        window['_MES_'].update('Tracking successfully')

        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.plot(n_errors, 'o-r', label='Error_track')
        ax1.set_title('Mean of Error tracking')
        ax1.set_xlabel('Number of frames')
        ax1.set_ylabel('Error by Frame')
        ax1.legend(loc='upper right')
        ax1.grid()

        ax2.plot(m_dist, 'o-g', label='Mean_distance')
        ax2.set_title('Mean of Distance tracking')
        ax2.set_xlabel('Number of frames')
        ax2.set_ylabel('Distance by frame (px.)')
        ax2.legend(loc='upper right')
        ax2.grid()

        ax3.plot(m_velocity, 'o-b', label='Mean_velocity')
        ax3.set_title('Mean of Velocity tracking')
        ax3.set_xlabel('Number of frames')
        ax3.set_ylabel('Velocity by frame (px./ s.)')
        ax3.legend(loc='upper right')
        ax3.grid()
        fin_time = now.strftime("%H : %M : %S")
        window['_TFI_'].update(fin_time)
        plt.show()
        track_press, finish_ = False, True

    if event == '_GRA_':
        print('GRAPHICS')
        if len(tot_dist) > 1:
            tot_dist = np.array(tot_dist[1:])
            max_dist = np.round(np.max(tot_dist[:]), 4)
            window['_MMD_'].update(max_dist)
            list_val = list(map(str, np.arange(1, tot_dist.shape[1])))
            window.Element('_GRA_').update(values=list_val)
            part_g = int(values['_GRA_'])
            dist_par = tot_dist[:, part_g]
            dist_par = np.insert(dist_par, 0, 0)
            dist_cum = np.cumsum(dist_par)

            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(dist_cum, 'o-g', label='Distance_track')
            ax1.set_title('Distance of particle: ' + str(part_g))
            ax1.set_xlabel('Number of frames')
            ax1.set_ylabel('Distance by particle (px.)')
            ax1.legend(loc='upper left')
            ax1.grid()

            ax2.plot(dist_par/delta, 'o-b', label='Velocity_track')
            ax2.set_title('Velocity of particle: ' + str(part_g))
            ax2.set_xlabel('Number of frames')
            ax2.set_ylabel('Velocity by particle (px./ s.)')
            ax2.legend(loc='upper right')
            ax2.grid()
            plt.show()
        else:
            sg.Popup('Message', ['Tracking not performed..'])

print('CLOSE WINDOW')
window.close()
