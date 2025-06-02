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
git clone https://github.com/CEJiang/DLCV-Final-Project.git
cd DLCV-Final-Project
```

2. Create and activate conda environment
```
conda create -n cv python=3.11.11 -y
conda activate cv
pip install -r requirements.txt
```

3. Download the dataset 
- You can download the dataset from the competitions
- Place it in the following structure
```
DLCV-Final-Project
├── sartorius-cell-instance-segmentation/
│   ├── LIVECell_dataset_2021/
│   ├── train/
│   ├── test/
│   .
│   .
│   .
├── requirements.txt
├── Submission.ipynb
├── Cellpose/
├── Detectron2/
.
.
.
```
4. Preprocessing
    1. Download the K-fold Cross Validation [LINK](https://www.kaggle.com/code/ammarnassanalhajali/k-fold-crossvalidation-coco-dataset-generator)

    2. Modify handle_cross_validation_to_npy.py to adjust some parameters for your training dataset

    3. Convert image to .pny format
    ```
    python handle_cross_validation_to_npy.py
    ```

5. Run for Train
    1. Train Cellpose Model 
        Step 1: Open Cellpose main.py to adjust datapath 
    
        Example: 
            ```
            train_loader, valid_loader = load_data(
            train_data_dir='Cellpose_5fold_train/fold1', # modify it to your dataset path
            valid_data_dir='Cellpose_5fold_valid/fold1', # modify it to your dataset path
            args=args)
            ```

        Step 2: Run the program
        ```
        python Cellpose/main.py
        ```
    2. Train Detectron2 Model
        Step 1: Open Cellpose dataset.py to adjust datapath 

        Example:
            - In the dataset.py

                register_coco_instances(
                    "sartorius_Cell_train",
                    {},
                    "sartorius_patched_5fold/annotations_train_patched_fold1.json", # modify it to your dataset path
                    "sartorius_patched_5fold/train_images_fold1" # modify it to your dataset path
                )

                register_coco_instances(
                    "sartorius_Cell_valid",
                    {},
                    "crossvalidationfold5/coco_cell_valid_fold1.json", # modify it to your dataset path
                    str(data_dir) # modify it to your dataset path
                )

6. Upload the result
    Step 1: You need to upload Submission.ipynb notebook to Kaggle

    Step 2: Upload your model to kaggle datasets

    Step 3: modify the model path and run the test


## Performance snapshot
### Training Parameter Configuration

- Cellpose

| Parameter        | Value                                                                                                   |
|------------------|---------------------------------------------------------------------------------------------------------|
| Pretrained Weight| cpsam (Train LIVECell)                                                                                  |
| Learning Rate    | 0.00005                                                                                                 |
| Batch Size       | 1                                                                                                       |
| Epochs           | 100                                                                                                     |
| decay            | 0.1                                                                                                     |
| Optimizer        | AdamW                                                                                                   |
| T_max            | 80                                                                                                      |
| Scheduler        | `Linear warmup`                                                                                         |
| Criterion        | `Flow Loss` + `Cellprob Loss`                                                                           |

- Detectron2

| Parameter        | Value                                                                                                   |
|------------------|---------------------------------------------------------------------------------------------------------|
| Pretrained Weight| model_zoo pretrained model(Train LIVECell)                                                              |
| Learning Rate    | 0.0005                                                                                                  |
| Batch Size       | 2                                                                                                       |
| Epochs           | 100                                                                                                     |
| decay            | 0.1                                                                                                     |

### Individual Losses
![Image](https://github.com/CEJiang/DLCV-Final-Project/blob/main/Image/individual_losses.png)
### Total Loss
![Image](https://github.com/CEJiang/DLCV-Final-Project/blob/main/Image/total_loss.png)
### Performance
![Image](https://github.com/CEJiang/DLCV-Final-Project/blob/main/Image/result.jpg)