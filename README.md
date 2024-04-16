# Multimodal Recommender System using Neural Networks

## Overview
In this project, I aim to develop a recommender application tailored specifically for women's apparel. I found this dataset appealing since It leverages a combination of numerical and textual data, making it a quintessential example of multimodal machine learning.

## Dataset
The dataset used for this project is the [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews/download?datasetVersionNumber=1) dataset. The dataset contains the following relevant features:

- **Age**: Positive integer representing the reviewer's age.
- **Review Text**: String variable for the review body.
- **Rating**: Positive ordinal integer variable for the product score (1 to 5).
- **Positive Feedback Count**: Positive integer documenting the number of other customers who found this review positive.
- **Class Name**: Categorical name of the product class.

The target variable is **Recommended IND**, a binary variable indicating whether the customer recommends the product (1 for recommended, 0 for not recommended).

## Method

1. **Data Loading and Preprocessing**:
   - Load the dataset and split it into 80% training and 20% testing data.
   - Handle duplicate rows and missing cell values.

2. **Feature Engineering**:
   - Scale all numerical features using MinMax scaler.
   - Convert categorical features using OneHotEncoder.
   - Convert the "Review Text" feature into N-hot vectors using CountVectorizer.
   - Concatenate the features for model input.

3. **Neural Network Model Building**:
   - Construct a feedforward neural network model using Keras.
   - The model architecture consists of an input layer (128 neurons), two hidden layers (64 and 32 neurons with ReLU activation), a dropout layer, and an output layer (1 neuron with Sigmoid activation).
   - Train the model using Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.01 and binary cross-entropy loss.

4. **Model Evaluation and Ablation Study**:
   - Evaluate the model's performance on the test set, measuring its accuracy in predicting product recommendations.
   - Conduct an ablation study to investigate the impact of the "Review Text" feature on the model's performance.

## Results

After evaluating the model's performance on the test split, the model achieves an accuracy of 93.24%. The training accuracy is 99.97%, indicating a possible overfitting issue that can be addressed through further regularization or model complexity adjustments. Furthermore, I found from the ablation study that the "Review Text" feature does not significantly influence the model's accuracy, as the test accuracy remains around 94% even after removing this feature (similar accuracy with and without "Review Text").

## Conclusion

This project demonstrates the feasibility of a women's apparel recommender system that combines numerical and textual data. The model's performance on the test set is promising, and the ablation study provides insights into the importance of textual features in the Neural Network. Future work may involve exploring more advanced techniques, such as multimodal deep learning architectures, to further enhance the recommender's accuracy and robustness.

## Future Work
The future work for this project involves several key areas. Firstly, addressing the observed overfitting through techniques like regularization and dropout could improve the model's generalization. In addition, I think exploring more advanced neural network architectures, such as RNNs and transformers, may better capture the nuances of the textual data. Incorporating additional features, like product images and user demographics, could also further enhance the recommender's performance. Finally, I think deploying the system in a real-world setting and gathering user feedback would enable continuous improvement and a better understanding of user preferences, leading to a more personalized and effective shopping experience.
