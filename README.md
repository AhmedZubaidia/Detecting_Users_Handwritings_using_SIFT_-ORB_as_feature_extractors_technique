# Detecting Users' Handwriting Using SIFT and ORB Feature Extraction Techniques  

![image](https://github.com/user-attachments/assets/4dda30e1-796d-4feb-a6e4-d8909681c94e)

![image](https://github.com/user-attachments/assets/996df93e-169e-4959-9b3c-2ae01322462c)
![image](https://github.com/user-attachments/assets/3be912f8-084f-4036-9285-1220a315e258)

![image](https://github.com/user-attachments/assets/8641db64-29ab-44df-b3ea-c9aaa1bfbd83)


## Overview  
This project focuses on building a system to detect and classify users' handwriting using advanced feature extraction techniques: **SIFT (Scale-Invariant Feature Transform)** and **ORB (Oriented FAST and Rotated BRIEF)**. The goal is to evaluate and compare the robustness and efficiency of these techniques under various data variations, leveraging concepts such as **Bag of Words (BoW)** and **Support Vector Machines (SVM)** for classification.  

## Objectives  
- Extract features from handwriting images using **SIFT** and **ORB**.  
- Train classifiers to distinguish handwriting styles between different users.  
- Evaluate the robustness of SIFT and ORB under data augmentation, such as noise, rotation, scaling, affine transformations, and illumination changes.  
- Compare the computational efficiency and accuracy of SIFT and ORB.

---

## Methodology  

### 1. **Environment**  
- Hosted on **Google Colab** with extended resources (T4-GPU, 51GB RAM).  
- Utilized Python libraries including `OpenCV`, `scikit-learn`, and `concurrent.futures` for parallel processing.  

### 2. **Data Preprocessing**  
- **Dataset:** AHAWP dataset (82 users, 10 trials per user for 10 unique words).  
- Steps:  
  - Convert images to grayscale to reduce computational complexity.  
  - Normalize brightness and contrast across images.  
  - Resize all images to a fixed resolution of 256x256 pixels.  
  - Handle missing data (duplicating samples for missing data or removing invalid entries).  

### 3. **Data Augmentation**  
To simulate real-world variability, the following augmentations were applied:  
- **Noise:** Low, medium, and high intensity.  
- **Rotation:** Angles of ±30° and ±45°.  
- **Scaling:** 50%, 80%, 120%, and 150%.  
- **Illumination Changes:** Gamma adjustments (0.5, 2, and 3).  
- **Affine Transformations.**

### 4. **Feature Extraction**  
- **SIFT (Scale-Invariant Feature Transform):**  
  - Extracts keypoints invariant to scale, rotation, and affine transformations.  
  - Configured with parameters for enhanced robustness (`contrastThreshold=0.02`, `edgeThreshold=5`).  

- **ORB (Oriented FAST and Rotated BRIEF):**  
  - A faster alternative to SIFT, ideal for real-time applications.  
  - Configured to detect up to 1000 keypoints per image.  

### 5. **Classification**  
- **Bag of Words (BoW):**  
  - Histograms of visual features were created by clustering keypoints using KMeans.  

- **Support Vector Machines (SVM):**  
  - Used as the primary classifier.  
  - Hyperparameter tuning via Grid Search to optimize performance.  

### 6. **Testing Robustness**  
- Evaluated the performance of SIFT and ORB under various augmentations by:  
  - Comparing the number of detected keypoints.  
  - Measuring the execution time.  
  - Testing SVM accuracy with both raw and augmented datasets.

---

## Results  

### 1. **Feature Extraction**  
| Metric                | SIFT          | ORB          | Observation                                |
|-----------------------|---------------|--------------|--------------------------------------------|
| Total Keypoints       | 3.46M         | 23.43M       | ORB detected 6.77x more keypoints.         |
| Total Execution Time  | 4762.16 sec   | 268.90 sec   | ORB was 17.7x faster than SIFT.            |
| Avg. Keypoints/User   | 42,769.17     | 289,289.90   | ORB detected more features per user.       |

### 2. **SVM Classifier Accuracy**  
| Test Data Type         | SVM on SIFT (Raw Data) | SVM on SIFT (Augmented Data) | SVM on ORB (Raw Data) | SVM on ORB (Augmented Data) |
|------------------------|------------------------|------------------------------|-----------------------|-----------------------------|
| Raw Test Data          | 34.81%                | 42.72%                      | 37.28%               | 44.88%                     |
| Rotation Test Data     | 31.46%                | 42.43%                      | 33.33%               | 43.98%                     |
| Noise Test Data        | 5.86%                 | 12.47%                      | 6.54%                | 20.00%                     |
| Scaling Test Data      | 26.69%                | 36.85%                      | 16.91%               | 29.48%                     |
| Illumination Test Data | 23.15%                | 32.65%                      | 9.14%                | 16.73%                     |
| Affine Test Data       | 21.37%                | 31.05%                      | 23.89%               | 33.70%                     |

### 3. **Key Observations**  
- **Efficiency:** ORB was significantly faster and detected more keypoints than SIFT.  
- **Accuracy:** SVM trained on augmented data consistently performed better than models trained on raw data.  
- **Robustness:**  
  - **SIFT:** Robust to rotation and scaling.  
  - **ORB:** More efficient and robust to affine transformations.  

---

## Conclusion  
- **SIFT vs ORB:** While SIFT is traditionally more accurate, ORB outperformed SIFT in this dataset due to its speed and efficiency.  
- **Data Augmentation:** Improved model robustness and accuracy significantly.  
- **Future Work:** Expanding the dataset is necessary to achieve higher classification accuracies. Additionally, exploring deep learning-based feature extraction methods could further enhance the system's performance.  

---

## Authors  
- **Jana Herzallah**  
- **Ahmed Zubaidia**  

## Instructor  
- **Dr. Aziz Qaroush**  
