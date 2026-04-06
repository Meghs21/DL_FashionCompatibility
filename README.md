# DL_FashionCompatibility

This project is all about checking if different pieces of clothing, when put together as an outfit, actually look good. Think of it like an AI stylist! It uses advanced computer vision techniques to "see" and "understand" how well clothes match.

## What's Happening

The main idea is to figure out if an outfit is stylish and compatible. To do this, the project first looks at each individual clothing item in an outfit. It uses a smart image recognition system to extract important details (features) from each item's picture. Once it has these details for all items in an outfit, it then feeds them into another intelligent system that learns to decide if they work well together. Finally, it gives you a digital "signature" or "embedding" that represents how compatible the whole outfit is.

## Major Technologies Used

* *ResNet-18:* This is a powerful, pre-trained image recognition model. It acts like the "eyes" of the system, extracting detailed visual information from clothing images.
* *BiLSTM (Bidirectional Long Short-Term Memory):* This is a type of neural network that's great at understanding sequences. After ResNet extracts features from each item, the BiLSTM looks at these features in a sequence (like a list of items in an outfit) to understand their overall relationship and compatibility.
* *PyTorch:* A popular machine learning framework that makes it easier to build and train these complex neural networks.
* *Python:* The main programming language used for the entire project.

## Important Features

* *Outfit Analysis:* Takes a group of clothing item images as input.
* *Intelligent Feature Extraction:* Uses a pre-trained ResNet-18 model to automatically pull out key visual features from each clothing item.
* *Compatibility Learning:* Employs a BiLSTM model to learn the patterns and relationships that make an outfit compatible or not.
* *Outfit Embedding:* Generates a unique digital representation (a vector) for each outfit, summarizing its overall style and compatibility.
* *Ongoing Development:* Actively working on combining the feature extraction and compatibility learning into one system, then training and improving its accuracy.

## 🧾 Dataset

We use the *Polyvore Outfits (Disjoint)* dataset for training and testing fashion compatibility.

- 📦 *Download*: [Polyvore Dataset on Kaggle](https://www.kaggle.com/datasets/enisteper1/polyvore-outfit-dataset)
- 💡 The "disjoint" split ensures that no outfit items overlap between train and test sets.

After downloading:
- Place train.json, test.json, and valid.json inside data/polyvore_outfits/disjoint/
- Place all outfit images inside data/polyvore_outfits/images/

---
