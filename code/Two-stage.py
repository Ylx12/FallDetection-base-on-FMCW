import os
from collections import deque
from Network import RDTNet
from Function import *

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

barycenter_detecting = 0
model_working = 0
sleep_mode = 2

if __name__ == '__main__':
    device = torch.device("cuda")
    debug = False

    print('------------------initialization------------------')
    dete_length = 16
    RDM_list = deque(maxlen=dete_length)  # A cache of dozens of frames of RDM input to the network

    # initialize RDM_list
    for i in range(16):
        zero_array = np.zeros((62, 50))
        RDM_list.append(zero_array)

    barycenter_buffer = np.zeros(dete_length, dtype=int)  # A cache of several frames of barycenter in a row
    fall_times = 0  # record how many falls have occurred
    fall_frames = []
    num_pos_pred = 0  # temporarily save the number of times of positive result, used to vote when the final judgment is whether a fall has occurred
    dida = 0
    sleep_time = 0
    total_call_times = 0  # record the total number of times the neural network executes
    model_work_times = 0  # When the network is activated, it works 5 times in a row. This value is used to record times
    state = barycenter_detecting
    frame = 0
    minimum_index = 0
    maximum_index = 0

    print('------------------geting parameters------------------')
    para = get_para()
    print('frames:', para['frames'], '    Tframe:', para['Tframe'], 'ms', '    sweepslope:',
          para['sweepslope'] / 10 ** 12, 'MHz/us')
    print('chirps(fft_Vel):', para['chirps'], '      Tchirp:', para['Tchirp'] * 10 ** 6, 'us')
    print('samples(fft_Range):', para['samples'], '   sample_rate:', para['sample_rate'] / 1000, 'kps')

    # This is the folder where the real-time radar data is stored.Set this path based on the actual situation
    top_data_dir = '/home/linxin/radar_data/txt/15min0_framing/'

    #  para setting
    Rx_num = 4
    range_num_bin = 62
    vel_num_bin = 25
    threshold_T = 70  # the barycenter variation threshold.A change greater than this value is considered a suspected fall
    score_th = 0.80

    print('------------------Loading models------------------')
    print(torch.__version__)
    print(torch.cuda.is_available())

    model = RDTNet([[1, 16, 32, 64, 128, 256], [256, 256, 256]])
    pth_path = 'model/RDTNet.pth'
    model_pth = torch.load(pth_path)
    model.load_state_dict(model_pth)
    model.eval()
    model.to(device=device)

    print('--------------------begin working---------------------')
    while (1):
        # This is the path where the real-time radar data is stored.
        # Set this path based on the actual situation.
        frame = frame + 1
        frame_data_path = top_data_dir + '/frame' + str(frame) + '.txt'

        RDM_amp = data_to_RDM(frame_data_path, para, Rx_num)
        RDM_dB = torch.tensor(20 * np.log10(RDM_amp + 1))
        RDM_dB_norm = normalize_to_0_1(RDM_dB)
        data_buffer = np.array(RDM_dB_norm[6:range_num_bin + 6, 64 - vel_num_bin: 64 + vel_num_bin])
        RDM_list.append(data_buffer)

        RDM_amp_norm = normalize_to_0_1(torch.tensor(RDM_amp))
        RDM_amp_norm_fil = amplitude_filtering(RDM_amp_norm[6: 6 + 62, 64 - 25: 64 + 25], amplitude_threshold=0.3)
        barycenter, count = calculate_barycenter(RDM_amp_norm_fil, count_threshold=1, debug=0)
        barycenter = barycenter * 11.2
        if barycenter > 0:
            barycenter = barycenter + 6 * 11.2

        if state == barycenter_detecting:
            # Update barycenter queue
            for i in range(dete_length - 1):
                barycenter_buffer[i] = barycenter_buffer[i + 1]
            barycenter_buffer[dete_length - 1] = barycenter

            point = sum(i > 0 for i in barycenter_buffer)
            veri_height = 0
            if point >= 2:
                max_height = np.max(barycenter_buffer)

                # Find the non-zero minimum in the list
                j = 0
                while barycenter_buffer[j] == 0:
                    j = j + 1
                min_height = barycenter_buffer[j]
                while j < dete_length:
                    if barycenter_buffer[j] < min_height and barycenter_buffer[j] != 0:
                        min_height = barycenter_buffer[j]
                    j = j + 1

                temp_list = list(barycenter_buffer)
                maximum_index = find_last_index(temp_list, max_height)
                minimum_index = temp_list.index(min_height)
                if min_height != 0 and minimum_index < maximum_index:
                    veri_height = max_height - min_height
            else:
                model_working = model_working

            print('frame', frame, 'count', count, 'barycenter', int(barycenter),
                  'minimum_index:', minimum_index, 'maximum_index:', maximum_index, 'vari:', veri_height,
                  '               model_working', model_working, ' total_call_times', total_call_times)

            if veri_height >= threshold_T:
                if model_working == 0:
                    print('------------------fall may happen ------------------')
                    print('fall may happen around frame', str(frame), 'veri_height', veri_height)
                    print('call neural network')
                    model_working = 1
            else:
                model_working = model_working

        if model_working == 1:
            total_call_times = total_call_times + 1
            model_work_times = model_work_times + 1
            data_buffer = RDM_prepare(RDM_list, device)
            output = model(data_buffer)

            if output[0] > score_th:
                pred = 1
                num_pos_pred = num_pos_pred + 1
            else:
                pred = 0

            print('frame', frame, 'output', round(output[0].item(), 3), 'preds', pred, 'num_pos_pred=', num_pos_pred)

            # Determine whether three of the five consecutive test results are positive
            if model_work_times == 5:
                if num_pos_pred >= 3:
                    fall_times = fall_times + 1
                    fall_frames.append(frame - 4)
                    print('------------------neural calculation finish ------------------')
                    print('--------There is a high probability of falling --------!')
                    print('*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('--------------------------------------------------------------')
                    print('fall_times', fall_times, 'fall_frames', fall_frames)
                    print('\n')
                    sleep_time = 29  # When a fall is detected, the network sleeps for 3s
                    state = sleep_mode
                model_work_times = 0
                model_working = 0
                num_pos_pred = 0

        if state == sleep_mode:
            if dida < sleep_time:
                dida = dida + 1

                # The system still updates the list of barycenter though it is asleep
                for i in range(dete_length - 1):
                    barycenter_buffer[i] = barycenter_buffer[i + 1]
                barycenter_buffer[dete_length - 1] = barycenter
                point = sum(i > 0 for i in barycenter_buffer)
                print(dida)
            else:
                dida = 0
                state = barycenter_detecting
