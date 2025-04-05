# Mood-Based Music Recommendation System  

## Overview  
The "Mood-Based Music Recommendation System" leverages emotional intelligence to enhance music recommendations by aligning them with the user's current emotional state. Utilizing real-time facial expression analysis, this project offers a personalized listening experience that transcends traditional historical data methods.  

## Features  
- Real-time emotion detection via a Convolutional Neural Network (CNN) trained on the FER2013 dataset.  
- Mood-based song clustering using Principal Component Analysis (PCA) and K-means clustering.  
- Personalized music recommendations based on detected emotions, using the YouTube API for streaming.  

## Key Components  
1. **Emotion Detection**:  
   - Captures live video using OpenCV.  
   - Identifies facial expressions with a pre-trained Haarcascade Classifier and CNN model.  
   - Displays real-time emotion labels on detected faces.  

2. **Song Recommendation**:  
   - Organizes songs into mood-based categories from a CSV dataset (muse_v3.csv).  
   - Generates playlists based on dominant emotions using a `fun()` function.  
   - Applies cosine similarity for enhanced music selections.  

3. **User Interface**:  
   - Developed using Streamlit for a seamless user experience.  
   - Includes a “SCAN EMOTION” button to initiate emotion detection and display song recommendations.  

