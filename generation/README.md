# Deepfake Generation in DeepfakesAdvTrack - Spring 2025
This is the offical code for deepfake generation in DeepfakesAdvTrack, the practice session of the course Artificial Intelligence Security, Attacks and Defenses (UCAS, spring 2025).

## ⚡ How to start
1. Download the dataset

Please access to CelebDF-v2-training from [access require form](https://github.com/yuezunli/celeb-deepfakeforensics).

2. Prepare your algorithm

Find a face-swaping algorithm that satisfies you, then swap faces according to `image_list.txt`.

3. Submit your results

Please pack the images into a file named `YOUR_TEAM_NAME.tar.gz` and send it to TA.

## Format of image names
Take `id0_id1_0000_00060.png` as an example. It stands for the `60th` frame of the target video `id0_0000.mp4` with its face swapped to `id1`. 

## Rating Calculations
We evaluate faces from the perspectives of SSIM, Noise, ID_score, and Anti-Detectors (1 - AUC). The overall rating score is their average. For fairness, we do not provide the evaluation script.

## ⚠️ Caution
1. **DO NOT** modify the file names, otherwise it will cause evaluation error and affects your final rating.
2. Note the face-swapped image should have exactly the same image size (width and height) as the original target frame.
3. Submitted images must be named exactly as specified in this file, and the format is '.png'. 

## Acknowledgement
This code is based on [DFGC_startkit](https://github.com/bomb2peng/DFGC_starterkit/tree/master).
The source data is [CelebDF](https://github.com/yuezunli/celeb-deepfakeforensics).