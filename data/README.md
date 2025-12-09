Source:
The dataset used for this project is the Face Mask Detection Dataset, publicly available on Kaggle. It was originally created to 
support computer vision projects aimed at identifying mask compliance during the COVID-19 pandemic.

Size:
The dataset contains approximately 7,000 labeled images of human faces divided into three categories:

With Mask 😷

Without Mask 🙅‍♂️

Mask Worn Incorrectly 😐

Images vary in resolution, lighting conditions, and background environments, providing a diverse and realistic sample for training 
and evaluation.

Usage:
This dataset will be used to train and evaluate a YOLOv8 object detection model implemented in PyTorch via Google Colab.
The model’s goal is to automatically detect faces and classify whether a mask is worn properly, worn incorrectly, or not worn at all.

Data preprocessing steps will include:

Resizing images to YOLOv8’s expected input dimensions

Normalizing pixel values

Splitting data into training (70%), validation (20%), and testing (10%) subsets

Purpose:
The dataset serves as the foundation for developing a real-time face mask detection system that could be applied in public safety monitoring, workplace compliance, and health screening systems — all at zero cost using Google Colab’s free tier.

In raw.md its possible to find the dataset used, conveniently uploaded in Google Drive in order to make it more accessible
