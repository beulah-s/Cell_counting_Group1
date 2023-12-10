import albumentations as A
import albumentations.augmentations.functional as F
import torch
from PIL import Image
import cv2
from torchvision import transforms
import pandas as pd
from model.CellQuant import CellQuant
from random import choice
import os
import numpy as np
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


model = CellQuant.load_from_checkpoint(r'lightning_logs\version_9\checkpoints\epoch=199-step=1200.ckpt') #add your respective model

test_samples=['220812_GFP-AHPC_A_GFAP_F6_DAPI_ND1_20x.tiff', '220812_GFP-AHPC_C_GFAP_F1_DAPI_ND1_20x.tiff', '220812_GFP-AHPC_C_GFAP_F4_DAPI_ND1_20x.tiff', '220812_GFP-AHPC_C_GFAP_F6_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_A_Ki67_F2_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_A_Ki67_F8_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_A_MAP2ab_F8_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_A_Nestin_F10_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_A_Nestin_F1_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_A_Nestin_F4_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_A_Nestin_F5_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_A_Nestin_F6_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_A_Nestin_F7_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_A_TuJ1_F1_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_A_TuJ1_F2_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_A_TuJ1_F5_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_A_TuJ1_F7_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_A_TuJ1_F9_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_B_GFAP_F3_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_B_GFAP_F4_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_B_GFAP_F5_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_B_GFAP_F8_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_B_GFAP_F9_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_B_MAP2ab_F4_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_B_MAP2ab_F7_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_B_MAP2ab_F8_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_B_MAP2ab_F9_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_B_Nestin_F10_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_B_Nestin_F1_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_B_Nestin_F8_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_B_TuJ1_F5_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_B_TuJ1_F6_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_C_Ki67_F10_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_C_Ki67_F1_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_C_Ki67_F2_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_C_Ki67_F6_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_C_MAP2ab_F10_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_C_MAP2ab_F3_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_C_MAP2ab_F4_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_C_MAP2ab_F6_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_C_MAP2ab_F8_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_C_Nestin_F7_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_C_Nestin_F8_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_C_Nestin_F9_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_C_RIP_F10_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_C_RIP_F6_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_C_RIP_F9_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_C_TuJ1_F3_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_C_TuJ1_F4_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_C_TuJ1_F5_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_C_TuJ1_F8_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_D_GFAP_F10_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_D_GFAP_F5_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_D_GFAP_F8_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_D_GFAP_F9_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_D_Ki67_F5_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_D_Ki67_F9_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_D_MAP2ab_F3_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_D_MAP2ab_F5_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_D_MAP2ab_F9_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_D_Nestin_F4_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_D_TuJ1_F1_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_D_TuJ1_F3_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_D_TuJ1_F6_DAPI_ND1_20x.tiff', '220815_GFP-AHPC_D_TuJ1_F9_DAPI_ND1_20x.tiff', '220816_GFP-AHPC_A_RIP_F3_DAPI_ND1_20x.tiff', '220816_GFP-AHPC_A_RIP_F4_DAPI_ND1_20x.tiff', '220816_GFP-AHPC_B_Ki67_F7_DAPI_ND1_20x.tiff', '220816_GFP-AHPC_B_RIP_F6_DAPI_ND1_20x.tiff', '220816_GFP-AHPC_B_RIP_F7_DAPI_ND1_20x.tiff', '220816_GFP-AHPC_B_RIP_F8_DAPI_ND1_20x.tiff', '220816_GFP-AHPC_D_RIP_F1_DAPI_ND1_20x.tiff', '220816_GFP-AHPC_D_RIP_F3_DAPI_ND1_20x.tiff', '220816_GFP-AHPC_D_RIP_F6_DAPI_ND1_20x.tiff', '220909_GFP-AHPC_A_GFAP_F4_DAPI_ND1_20x.tiff', '220909_GFP-AHPC_A_GFAP_F6_DAPI_ND1_20x.tiff', '220909_GFP-AHPC_A_Ki67_F7_DAPI_ND1_20x.tiff', '220909_GFP-AHPC_B_GFAP_F6_DAPI_ND1_20x.tiff', '220909_GFP-AHPC_B_GFAP_F8_DAPI_ND1_20x.tiff', '220909_GFP-AHPC_B_Ki67_F6_DAPI_ND1_20x.tiff', '220909_GFP-AHPC_B_Ki67_F8_DAPI_ND1_20x.tiff', '220909_GFP-AHPC_C_GFAP_F7_DAPI_ND1_20x.tiff', '220909_GFP-AHPC_C_Ki67_F8_DAPI_ND1_20x.tiff', '220909_GFP-AHPC_C_Nestin_F6_DAPI_ND1_20x.tiff', '220909_GFP-AHPC_D_GFAP_F6_DAPI_ND1_20x.tiff', '220909_GFP-AHPC_D_Ki67_F2_DAPI_ND1_20x.tiff', '220909_GFP-AHPC_D_Ki67_F8_DAPI_ND1_20x.tiff', '220909_GFP-AHPC_D_MAP2ab_F5_DAPI_ND1_20x.tiff', '220909_GFP-AHPC_D_MAP2ab_F7_DAPI_ND1_20x.tiff', '220909_GFP-AHPC_D_MAP2ab_F9_DAPI_ND1_20x.tiff', '220909_GFP-AHPC_D_Nestin_F6_DAPI_ND1_20x.tiff', '220909_GFP-AHPC_D_Nestin_F7_DAPI_ND1_20x.tiff', '220909_GFP-AHPC_D_Nestin_F9_DAPI_ND1_20x.tiff', '220909_GFP-AHPC_D_TuJ1_F3_DAPI_ND1_20x.tiff', '220912_GFP-AHPC_A_Map2AB_F1_DAPI_ND1_20x.tiff', '220912_GFP-AHPC_A_Map2AB_F4_DAPI_ND1_20x.tiff', '220912_GFP-AHPC_A_Nestin_F8_DAPI_ND1_20x.tiff', '220912_GFP-AHPC_A_Nestin_F9_DAPI_ND1_20x.tiff', '220912_GFP-AHPC_A_TuJ1_F3_DAPI_ND1_20x.tiff', '220912_GFP-AHPC_A_TuJ1_F5_DAPI_ND1_20x.tiff', '220912_GFP-AHPC_B_Map2AB_F9_DAPI_ND1_20x.tiff', '220912_GFP-AHPC_B_Nestin_F10_DAPI_ND1_20x.tiff', '220912_GFP-AHPC_B_Nestin_F3_DAPI_ND1_20x.tiff', '220912_GFP-AHPC_B_TuJ1_F10_DAPI_ND1_20x.tiff', '220912_GFP-AHPC_B_TuJ1_F1_DAPI_ND1_20x.tiff', '220912_GFP-AHPC_C_Map2AB_F9_DAPI_ND1_20x.tiff', '220912_GFP-AHPC_C_TuJ1_F10_DAPI_ND1_20x.tiff', '220912_GFP-AHPC_C_TuJ1_F3_DAPI_ND1_20x.tiff']
'''
for i in range(20):
    ch = choice(test_samples)

    print(ch)

    img = Image.open('data/IDCIA/images/' + ch).convert("RGB")

    test_transform = A.Compose(
    [
        A.PadIfNeeded(min_height=256, min_width=256,
                      always_apply=True, border_mode=0),
        A.Resize(height=256, width=256, always_apply=True),
        A.Normalize(),
        ToTensorV2()
    ]
    )
    model.eval()
    with torch.no_grad():
        prediction = model(test_transform(image=np.asarray(img))['image'].unsqueeze(0))

    # Window name in which image is displayed
    window_name = 'Image'

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (20, 20)

    # fontScale
    fontScale = .5

    # Blue color in BGR
    color = (0, 0, 240)

    # Line thickness of 2 px
    thickness = 1

    # Using cv2.putText() method
    image = cv2.putText(np.asarray(img), str(round(prediction.item())), org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    # Using cv2.putText() method
    image = cv2.putText(image, str(pd.read_csv("data/IDCIA/ground_truth/" + ch.replace('tiff', 'csv')).shape[0]), (20, 40),
                        font,
                        fontScale, (0, 240, 0), thickness, cv2.LINE_AA)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()
    print(pd.read_csv("data/IDCIA/ground_truth/" + ch.replace('tiff', 'csv')).shape[0])
'''


report={}
for i in tqdm(test_samples):
    
    if not os.path.exists(os.path.join("..","IDCIA Clone","test_images", "images" , i)):
        #add your respective directory
        print(f"Skipping")
        continue

    img = Image.open(os.path.join("..","IDCIA Clone","test_images", "images",i)).convert("RGB")
    #add your respective directory

    test_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=256, min_width=256,
                          always_apply=True, border_mode=0),
            A.Resize(height=256, width=256, always_apply=True),
            A.Normalize(),
            ToTensorV2()
        ]
    )
    model.eval()
    with torch.no_grad():
        prediction = model(test_transform(image=np.asarray(img))['image'].unsqueeze(0))

    #report[i] = (pd.read_csv(os.path.join("..","IDCIA Clone","ground_truth",i.replace('tiff', 'csv'))).shape[0], round(prediction.item()))
    #add your respective directory training_test set
    report[i]= (round(prediction.item())) 
pd.DataFrame(report,index=[0]).transpose().to_csv("report_CNN.csv", index=True, header=True)

