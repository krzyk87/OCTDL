# Optical coherence tomography dataset classification

Framework for 2D classification of the OCT dataset.

Optical coherence tomography (OCT) is an emerging technology for performing high-resolution cross-sectional imaging. 
It utilizes the interferometry concept to create a cross-sectional map of the retina. 
OCT images are two-dimensional data sets that represent the optical backscattering in a cross-sectional plane 
through the tissue. 
OCT allows non-secondary visualization of various structures of the eye, including the retina, vitreous body,
and choroid, and to detect pathological changes in them. 
The study of these images is essential for the diagnosis, treatment, and monitoring of various eye diseases.

The database consists of the following categories and images:

- Age-related Macular Degeneration (AMD): 1231
- Diabetic Macular Edema (DME): 147
- Epiretinal Membrane (ERM): 155
- Normal (NO): 332
- Retinal Artery Occlusion (RAO): 22
- Retinal Vein Occlusion (RVO): 101
- Vitreomacular Interface Disease (VID): 76

For more information and details about the dataset see:
https://rdcu.be/dELrE
https://arxiv.org/abs/2312.08255


Requirements:
- pytorch
- torchvision
- torcheval
- timm
- tqdm
- munch
- packaging
- tensorboard
- omegaconf
- opencv-python
- hydra-core
- scikit-learn



## How to use
In this work we use folder-form dataset structure:
```
├── dataset
    ├── train
        ├── class1
            ├── image1.jpg
            ├── ...
        ├── class2
        ├── class3
        ├── ...
    ├── val
    ├── test
```
Unzip the archive to the directory and use `preprocessing.py` for image preprocessing and dataset preparation:
```
python preprocessing.py
```
Optional arguments:
```
--dataset_folder', type=str, default='./OCT_dataset', help='path to dataset folder')
--labels_path', type=str, default='./OCTDL_dataset/labels.csv', help='path to labels.csv'
--output_folder', type=str, default='./dataset', help='path to output folder')
--crop_ratio', type=int, default=1, help='central crop ratio of image')
--image_dim', type=int, default=512, help='final dimensions of image')
--val_ratio', type=float, default=0.15, help='validation size')
--test_ratio', type=float, default=0.25, help='test size')
--padding', type=bool, default=False, help='padding to square')
```

Training:

```shell
python main.py
```

Optional arguments:
```
-c yaml_file      Specify the config file (default: configs/default.yaml)
-p                Print configs before training
```

## Recent Enhancements (made by krzyk87)
This repository has been extended with the following features (contributed by krzyk87):

### 1. Batch Training and Testing Scripts
Automated training and testing pipeline for multiple CNN architectures with configurable networks and config files, with intelligent checkpoint discovery and network-specific evaluation. 

Train multiple networks:
```shell
python train_all_networks.py
```

This script automates training for multiple networks (VGG16, ResNet18, ResNet50, InceptionV3, MobileNetV3, EfficientNetV2, DenseNet121).

Optional arguments:
```
--config          Config file name (default: OCTDL.yaml)
--networks        Space-separated list of networks to train (default: all networks)
```

Examples:
```
# Train all networks with default config
python train_all_networks.py

# Train specific networks
python train_all_networks.py --networks resnet50 inception_v3

# Use custom config file
python train_all_networks.py --config CAVRI-H5_cleaned.yaml
```

Test multiple networks:
```shell
python test_all_networks.py
```

This script automates testing for multiple trained networks, automatically finding and loading the appropriate checkpoints.

Optional arguments:
```
--config          Config file name (default: OCTDL.yaml)
--networks        Space-separated list of networks to test (default: all networks)
```

Examples:
``` shell
# Test all networks with default config
python test_all_networks.py

# Test specific networks
python test_all_networks.py --networks resnet50 vgg16
```

### 2. Enhanced Visualization and Analysis

- **Grad-CAM Integration**: Added comprehensive Grad-CAM (Gradient-weighted Class Activation Mapping) visualization to interpret model decisions
  - Integrated into test.py for automatic generation during testing
  - Standalone script (paper_docs/grad_cam_visualization.py) for detailed analysis across datasets
  - Device-aware tensor handling for CPU/GPU compatibility
  - Support for multiple architectures (ResNet, VGG, InceptionV3, MobileNetV3)
- **Training Curves Visualization**: Automatic extraction and plotting of training/validation metrics from TensorBoard logs
  - Generates plots for accuracy, loss, F1-score, precision, recall, and AUC
  - Saved alongside test results for comprehensive model evaluation

### 3. Improved Training Process
- **Validation Loss Tracking**: Enhanced train.py with validation loss computation and logging
- **Flexible Loss Function Handling**: Support for both string-based and callable loss criteria
- **Enhanced Logging**: Improved logging with config and network details for better traceability

### 4. Multi-Dataset Support
- Added configuration files for multiple OCT datasets:
  - OCT2017, OCT2018, OCT2023_Ravelli
  - CAVRI-H5_cleaned dataset
- Network-specific save paths for organized experiment management
- Standardized dataset directory structure across configurations

### 5. Code Quality Improvements
- Added .gitignore for better repository management
- Enhanced error handling for missing config files and checkpoints
- Improved documentation with example images and usage guides
- Better device handling and tensor management for cross-platform compatibility

These enhancements enable comprehensive experiments across multiple datasets (OCT2017, OCT2018, OCT2023_Ravelli) and architectures (VGG16, ResNet50, InceptionV3, MobileNetV3 Small/Large, EfficientNetV2, DenseNet121, ResNet18), supporting reproducible research in OCT image classification.

## Resources

> Huang, Y., Lin, L., Cheng, P., Lyu, J., Tam, R. and Tang, X., 2023. Identifying the key components in ResNet-50 for diabetic retinopathy grading from fundus images: a systematic investigation. Diagnostics, 13(10), p.1664. [[link](https://www.mdpi.com/2075-4418/13/10/1664)]


## Citation
```
@article{kulyabin2024octdl,
  title={OCTDL: Optical Coherence Tomography Dataset for Image-Based Deep Learning Methods},
  author={Kulyabin, Mikhail and Zhdanov, Aleksei and Nikiforova, Anastasia and Stepichev, Andrey 
          and Kuznetsova, Anna and Ronkin, Mikhail and Borisov, Vasilii and Bogachev, Alexander 
          and Korotkich, Sergey and Constable, Paul A and Maier, Andreas},
  journal={Scientific Data},
  volume={11},
  number={1},
  pages={365},
  year={2024},
  publisher={Nature Publishing Group UK London},
  doi={https://doi.org/10.1038/s41597-024-03182-7}
}
```
