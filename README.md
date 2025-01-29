# Real-Time Target Detection and UAV Control for Enhanced Battlefield Situational Awareness

## Overview
This project focuses on leveraging deep learning and Unmanned Aerial Vehicles (UAVs) to enhance real-time battlefield situational awareness. By integrating advanced object detection models, the system provides precise target identification and tracking in complex environments.

## Features
- **Deep Learning Models:** Uses YOLOv8, YOLOv5, ResNet, VGG16, and VGG19 for object detection and classification.
- **Real-Time Target Detection:** Achieves high accuracy in detecting battlefield targets.
- **Multi-Sensor Fusion:** Integrates data from RGB cameras, LiDAR, and infrared sensors.
- **Optimized Training Pipeline:** Utilizes transfer learning, data augmentation, and efficient model architectures.
- **Edge Deployment:** Designed for real-time inference on UAV hardware with GPU acceleration.

## Technologies Used
- **Frameworks:** TensorFlow, PyTorch
- **Models:** YOLOv8, YOLOv5, ResNet, VGG16, VGG19
- **Programming Languages:** Python
- **Hardware:** UAV platforms with onboard GPUs

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AnitPaul112/RealTime-UAV-Target-Detection.git
   cd RealTime-UAV-Target-Detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset and organize it as follows:
   ```
   dataset/
     ├── train/
     │   ├── images/
     │   ├── labels/
     ├── valid/
     │   ├── images/
     │   ├── labels/
   ```

## Training the Model
To train the YOLOv8 model:
```bash
python train.py --model yolov8 --data dataset/config.yaml --epochs 50 --batch-size 16
```

## Running Inference
Run inference on test images:
```bash
python detect.py --model yolov8 --source test_images/
```

## Evaluation Metrics
- **Mean Average Precision (mAP)**: Evaluates the object detection performance.
- **Accuracy & F1-score**: Measures classification performance.
- **Inference Latency**: Assesses real-time capability on UAV hardware.

## Contributors
- **Anit Paul** (BRAC University)
- **Istiak Zaman Shuvo** (BRAC University)
- **Tasin Ahsan Rhidy** (BRAC University)
- **Md. Ashiqur Rahman Abir** (BRAC University)

## Future Work
- Improve multi-UAV collaboration for battlefield surveillance.
- Optimize lightweight models for real-time performance.
- Enhance adversarial robustness for deployment in challenging environments.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
We acknowledge BRAC University for supporting this research and providing computational resources.
