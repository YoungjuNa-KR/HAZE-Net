import torch
from functools import partial
import numpy as np


_loss_fn = {
            "mse": partial(torch.nn.MSELoss, reduction="sum")
        }
_param_num = {
    "mse": 2
}

def computeGazeLoss(angular_out, gaze_batch_label):
    _criterion = _loss_fn.get("mse")()
    gaze_loss = _criterion(angular_out, gaze_batch_label).cuda()
    
    pi_angular_error = 0
    theta_angular_error = 0
    detected_number = angular_out.shape[0]
    for i in range(detected_number):

        # COMPUTE PI ANGULAR ERROR
        EST_pi = angular_out[i][0].cpu().detach().numpy()
        GT_pi = gaze_batch_label[i][0].cpu().detach().numpy()


        if EST_pi > GT_pi:
            pi_angular_error += np.abs(EST_pi - GT_pi)
        else:
            pi_angular_error += np.abs(GT_pi - EST_pi)
    
         # COMPUTE THETA ANGULAR ERROR
        EST_theta = angular_out[i][1].cpu().detach().numpy()
        GT_theta = gaze_batch_label[i][1].cpu().detach().numpy()

        if EST_theta > GT_theta:
            theta_angular_error += np.abs(EST_theta - GT_theta)
        else:
            theta_angular_error += np.abs(GT_theta - EST_theta)
    
    # 추후에 인식된 모든 사람 수를 고려하여 평균을 내주어야 한다.
    # angular_error는 Pi 와 Theta각도만을 고려하여 계산되었다.
    angular_error = (pi_angular_error + theta_angular_error)*45
        
    return gaze_loss, angular_error





def loadLabel(lines, names):
    gaze_batch_label = []
    head_batch_label = []
    flag = False

    # image_name : P02_000008.png,82,91,134,88
    # line : P02_000008.png,82,91,134,88

    # Batch_Image_Person_number : int(image_name[i].split("_")[1:])
    # Batch_Image_number : int(image_name[i].split("_")[:5])

    # Person_number : int(line.split("_")[1:])
    # Image_number : int(line.split("_")[:5])

    for name in names:
        # print("load label ...")

        batch_image_person_number = int(name.split("_")[0][1:])
        batch_image_number = int(name.split(".")[0].split("_")[1])  


        if batch_image_person_number == 0:
            min = 0
            max = 2999 + 3000 * 0
            mid = int((min+max)/2)

        elif batch_image_person_number == 1:
            min = 3000 * 1
            max = 2999 + 3000 * 1
            mid = int((min+max)/2)

        elif batch_image_person_number == 2:
            min = 3000 * 2
            max = 2999 + 3000 * 2
            mid = int((min+max)/2)
        
        elif batch_image_person_number == 3:
            min = 3000 * 3
            max = 2999 + 3000 * 3
            mid = int((min+max)/2)

        elif batch_image_person_number == 4:
            min = 3000 * 4
            max = 2999 + 3000 * 4
            mid = int((min+max)/2)
        
        elif batch_image_person_number == 5:
            min = 3000 * 5
            max = 2999 + 3000 * 5
            mid = int((min+max)/2)
        
        elif batch_image_person_number == 6:
            min = 3000 * 6
            max = 2999 + 3000 * 6
            mid = int((min+max)/2)
        
        elif batch_image_person_number == 7:
            min = 3000 * 7
            max = 2999 + 3000 * 7
            mid = int((min+max)/2)

        elif batch_image_person_number == 8:
            min = 3000 * 8
            max = 2999 + 3000 * 8
            mid = int((min+max)/2)

        elif batch_image_person_number == 9:
            min = 3000 * 9
            max = 2999 + 3000 * 9
            mid = int((min+max)/2)

        elif batch_image_person_number == 10:
            min = 3000 * 10
            max = 2999 + 3000 * 10
            mid = int((min+max)/2)

        elif batch_image_person_number == 11:
            min = 3000 * 11
            max = 2999 + 3000 * 11
            mid = int((min+max)/2)
        
        elif batch_image_person_number == 12:
            min = 3000 * 12
            max = 2999 + 3000 * 12
            mid = int((min+max)/2)
        
        elif batch_image_person_number == 13:
            min = 3000 * 13
            max = 2999 + 3000 * 13
            mid = int((min+max)/2)

        elif batch_image_person_number == 14:
            min = 3000 * 14
            max = 2999 + 3000 * 14
            mid = int((min+max)/2)

        image_number = int(lines[mid].split(" ")[0].split("_")[1])  

        while True:
            batch_image_number = int(name.split(" ")[0].split("_")[1])
            image_number = int(lines[mid].split(" ")[0].split("_")[1])  

            if batch_image_number == image_number:
                label = lines[mid].split(" ")
                head_batch_label.append([float(label[1]),float(label[2])])
                gaze_batch_label.append([float(label[3]),float(label[4])])
                break

            elif batch_image_number < image_number:
                max = mid-1

            elif batch_image_number > image_number:
                min = mid + 1

            mid = int((min+max)/2)  


    head_batch_label= torch.FloatTensor(head_batch_label)
    gaze_batch_label = torch.FloatTensor(gaze_batch_label)

    return head_batch_label, gaze_batch_label


def generateEyePatches_fast(sr_batch_size, image_names, type='train'):
    le_c_list = None
    re_c_list = None
    flag = False
    detected_list = []

    if type == "train":
        labels = open("./dataset/Training_eye_coordinate.txt", "r")
    else:
        labels = open("./dataset/Valdation_eye_coordinate.txt", "r")

    lines = labels.readlines()

    # image_name : P02_000008.png,82,91,134,88
    # line : P02_000008.png,82,91,134,88

    # Batch_Image_Person_number : int(image_name[i].split("_")[1:])
    # Batch_Image_number : int(image_name[i].split("_")[:5])

    # Person_number : int(line.split("_")[1:])
    # Image_number : int(line.split("_")[:5])

    for i in range(len(sr_batch_size)):
        try_num = 0
        # print("load patch ...", i)

        if type == "validation":
            image_names = [image_names]
        
        batch_image_person_number = int(image_names[i].split("_")[0][1:])
        batch_image_number = int(image_names[i].split(".")[0].split("_")[1][:6])  

        if type == "train":
            if batch_image_person_number == 0:
                min = 0
                max = 2399
                mid = int((min+max)/2)

            elif batch_image_person_number == 1:
                min = 2400
                max = 4792
                mid = int((min+max)/2)

            elif batch_image_person_number == 2:
                min = 4793
                max = 7192
                mid = int((min+max)/2)

            elif batch_image_person_number == 3:
                min = 7193
                max = 9592
                mid = int((min+max)/2)

            elif batch_image_person_number == 4:
                min = 9593
                max = 11992
                mid = int((min+max)/2)

            elif batch_image_person_number == 5:
                min = 11993
                max = 14392
                mid = int((min+max)/2)

            elif batch_image_person_number == 6:
                min = 14393
                max = 16792
                mid = int((min+max)/2)

            elif batch_image_person_number == 7:
                min = 16793
                max = 19192
                mid = int((min+max)/2)

            elif batch_image_person_number == 8:
                min = 19193
                max = 21592
                mid = int((min+max)/2)

            elif batch_image_person_number == 9:
                min = 21593
                max = 23992
                mid = int((min+max)/2)

            elif batch_image_person_number == 10:
                min = 23993
                max = 26392
                mid = int((min+max)/2)

            elif batch_image_person_number == 11:
                min = 26393
                max = 28792
                mid = int((min+max)/2)

            elif batch_image_person_number == 12:
                min = 28793
                max = 31192
                mid = int((min+max)/2)

            elif batch_image_person_number == 13:
                min = 31193
                max = 33592
                mid = int((min+max)/2)

            elif batch_image_person_number == 14:
                min = 33593
                max = 35992
                mid = int((min+max)/2)
    
        if type == "validation":
            if batch_image_person_number == 0:
                min = 0
                max = 599
                mid = int((min+max)/2)

            elif batch_image_person_number == 1:
                min = 600 * 1
                max = 599 + 600 * 1
                mid = int((min+max)/2)

            elif batch_image_person_number == 2:
                min = 600 * 2
                max = 599 + 600 * 2
                mid = int((min+max)/2)

            elif batch_image_person_number == 3:
                min = 600 * 3
                max = 599 + 600 * 3
                mid = int((min+max)/2)

            elif batch_image_person_number == 4:
                min = 600 * 4
                max = 599 + 600 * 4
                mid = int((min+max)/2)

            elif batch_image_person_number == 5:
                min = 600 * 5
                max = 599 + 600 * 5
                mid = int((min+max)/2)

            elif batch_image_person_number == 6:
                min = 600 * 6
                max = 599 + 600 * 6
                mid = int((min+max)/2)

            elif batch_image_person_number == 7:
                min = 600 * 7
                max = 599 + 600 * 7
                mid = int((min+max)/2)

            elif batch_image_person_number == 8:
                min = 600 * 8
                max = 599 + 600 * 8
                mid = int((min+max)/2)

            elif batch_image_person_number == 9:
                min = 600 * 9
                max = 599 + 600 * 9
                mid = int((min+max)/2)

            elif batch_image_person_number == 10:
                min = 600 * 10
                max = 599 + 600 * 10
                mid = int((min+max)/2)

            elif batch_image_person_number == 11:
                min = 600 * 11
                max = 599 + 600 * 11
                mid = int((min+max)/2)

            elif batch_image_person_number == 12:
                min = 600 * 12
                max = 599 + 600 * 12
                mid = int((min+max)/2)

            elif batch_image_person_number == 13:
                min = 600 * 13
                max = 599 + 600 * 13
                mid = int((min+max)/2)

            elif batch_image_person_number == 14:
                min = 600 * 14
                max = 599 + 600 * 14
                mid = int((min+max)/2)

        person_number = int(lines[mid].split("_")[0][1:])
        image_number = int(lines[mid].split(".")[0].split("_")[1])  

        while True:
            batch_image_number = int(image_names[i].split(".")[0].split("_")[1])  
            image_number = int(lines[mid].split(".")[0].split("_")[1])  

            # print("min : ",min,"\tmid : ",mid,"\tmax : ",max)
            # print("batch_image_number : ",batch_image_number)
            # print("image_number : ",image_number)

            if batch_image_number == image_number:
                break

            elif batch_image_number < image_number:
                max = mid-1
                try_num += 1

            elif batch_image_number > image_number:
                min = mid + 1
                try_num += 1

            mid = int((min+max)/2)  

            if try_num == 20:
                flag == False
                continue


        if image_names[i] in lines[mid]:
            left_y = int(lines[mid].split(",")[2])
            left_x = int(lines[mid].split(",")[4])
            right_y =int(lines[mid].split(",")[1])
            right_x =int(lines[mid].split(",")[3])
            flag = True
    
        if flag == False:
            print("I Couldn't find eye coordinate")
            continue

        # r,g,b 채널에 대하여 각각 crop한 후에, 병합한다.
        # print(left_y, left_x, right_y, right_x)
        detected_list.append(i)
        # print(sr_batch_size.shape)
        left_r = sr_batch_size[i][0][left_y-18:left_y+18, left_x-30:left_x + 30]
        left_g = sr_batch_size[i][1][left_y-18:left_y+18, left_x-30:left_x + 30]
        left_b = sr_batch_size[i][2][left_y-18:left_y+18, left_x-30:left_x + 30]

        right_r = sr_batch_size[i][0][right_y-18:right_y+18, right_x-30:right_x + 30]
        right_g = sr_batch_size[i][1][right_y-18:right_y+18, right_x-30:right_x + 30]
        right_b = sr_batch_size[i][2][right_y-18:right_y+18, right_x-30:right_x + 30]

        if left_r.size() != (36, 60):
            shift = 60 - left_r.size()[1]
            left_r = sr_batch_size[i][0][left_y-18:left_y+18, (left_x-shift)-30:(left_x-shift) + 30]
            left_g = sr_batch_size[i][0][left_y-18:left_y+18, (left_x-shift)-30:(left_x-shift) + 30]
            left_b = sr_batch_size[i][0][left_y-18:left_y+18, (left_x-shift)-30:(left_x-shift) + 30]
        
        if right_r.size() != (36, 60):
            shift = 60 - right_r.size()[1]
            right_r = sr_batch_size[i][0][right_y-18:right_y+18, (right_x-shift)-30:(right_x-shift) + 30]
            right_g = sr_batch_size[i][0][right_y-18:right_y+18, (right_x-shift)-30:(right_x-shift) + 30]
            right_b = sr_batch_size[i][0][right_y-18:right_y+18, (right_x-shift)-30:(right_x-shift) + 30]

        le_c = torch.stack([left_r,left_g,left_b])
        re_c = torch.stack([right_r,right_g,right_b])
        # print(le_c.shape)


        if i ==0 or le_c_list == None:
            le_c_list = le_c
            le_c_list = torch.reshape(le_c_list, (1, 3, 36, 60))
            re_c_list = re_c
            re_c_list = torch.reshape(re_c_list, (1, 3, 36, 60))
        else:
            le_c = torch.reshape(le_c, (1, 3, 36, 60))
            re_c = torch.reshape(re_c, (1, 3, 36, 60))
            le_c_list = torch.cat([le_c_list, le_c], dim=0)
            re_c_list = torch.cat([re_c_list, re_c], dim=0)
    
    # print("Completely Generate Eye Patches")
    return le_c_list, re_c_list, detected_list