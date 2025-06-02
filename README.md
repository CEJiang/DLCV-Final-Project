# NYCU Computer Vision 2025 Spring Final Project
- StudentID: 313553044, 313553046, 313553037, 313553139
- Name: 江仲恩, 馬國維, 黃瑜明, 陳冠豪

## Introduction
Neurological diseases, such as Alzheimer's disease, are significant causes of death and disability. Traditionally, these diseases have been detected through manual observation using light microscopes, which offers the advantages of accessibility and non-invasiveness. However, segmenting individual neuronal cells in microscopic images is both challenging and time-intensive, particularly when cells exhibit complex morphological variations and overlapping distributions.

In cell segmentation tasks, traditional semantic segmentation can only identify which pixels belong to which category (such as cell or background) but cannot distinguish between different individual cells. In contrast, instance segmentation not only identifies cell regions but, more importantly, can distinguish each individual cell, assigning a unique identifier to each cell. This capability is crucial for cell counting, tracking, and morphological analysis.

The appearance of neuronal cells is varied and irregular. Among eight different cancer cell types tested, SH-SY5Y human neuroblastoma cells consistently exhibit the lowest precision scores, highlighting the particular difficulty in segmenting this cell type.

Computer vision technology, especially deep learning-driven instance segmentation, provides new possibilities for addressing these challenges.
In this competition, we obtained images containing multiple neuronal cells, including SH-SY5Y and two other types of neuronal cells. Our goal is to use these images to train and test models to achieve higher accuracy in neuronal cell instance segmentation, providing technical support for automated diagnosis and research of neurological diseases.

## How to install

1. Clone the repository
```
git clone https://github.com/CEJiang/NYCU-Computer-Vision-2025-Spring-HW4.git
cd NYCU-Computer-Vision-2025-Spring-HW4
```

2. Create and activate conda environment
```
conda env create -f environment.yml
conda activate cv
```

3. Download the dataset 
- You can download the dataset from the provided [LINK](https://drive.google.com/file/d/1bEIU9TZVQa-AF_z6JkOKaGp4wYGnqQ8w)
- Place it in the following structure
```
NYCU-Computer-Vision-2025-Spring-HW4
├── hw4_release_dataset
│   ├── train
│   └── test
├── environment.yml
├── main.py
├── train.py
├── test.py
.
.
.
```

4. Run for Train
    1. Train Model 
    ```
    python main.py DATAPATH [--epochs EPOCH] [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE] [--decay DECAY] [--eta_min ETA_MIN] [--saved_path SAVE_FOLDER] [--mode train]
    ```
    Example
    ```
    python main.py ./hw4_realse_dataset --epochs 80 --batch_size 1 --learning_rate 1e-4 --decay 5e-3 --saved_path saved_models
    ```
    2. Test Model
    ```
    python main.py DATAPATH --mode test
    ```
    Example
    ```
    python main.py ./hw4_realse_dataset --mode test
    ```

## Performance snapshot
### Training Parameter Configuration

| Parameter        | Value                                                                                                   |
|------------------|---------------------------------------------------------------------------------------------------------|
| Pretrained Weight| None                                                                                                    |
| Learning Rate    | 0.0001                                                                                                  |
| Batch Size       | 1                                                                                                       |
| Epochs           | 80                                                                                                      |
| decay            | 0.005                                                                                                   |
| Optimizer        | AdamW                                                                                                   |
| Eta_min          | 0.000001                                                                                                |
| T_max            | 80                                                                                                      |
| Scheduler        | `CosineAnnealingLR`                                                                                     |
| ratio            | `0.3` -> `0.7`                                                                                          |
| Criterion        | `(1 - ratio) * L1 Loss` + `ratio * SSIM Loss`                                                           |

### Training Curve
![Image](https://github.com/CEJiang/NYCU-Computer-Vision-2025-Spring-HW4/blob/main/Image/training_curve.png)
### PSNR Curve
![Image](https://github.com/CEJiang/NYCU-Computer-Vision-2025-Spring-HW4/blob/main/Image/psnr_curve.png)

### Performance
|                  | mAP                      |
|------------------|--------------------------|
| Validation       | 30.32                    |
| Public Test      | 31.17                    |
