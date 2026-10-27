Lab 1: Environment and First System Measurement

THE GOAL

The main goal of this lab was to set up the required Python environment and compare two machine learning models from both a model performance and system performance point of view. Another goal was to use these measurements to decide whether the models would be suitable for Cloud, Edge, Mobile, or TinyML deployment.
I used Logistic Regression and Random Forest on the Breast Cancer Wisconsin dataset. I compared the models using test accuracy, training time, inference latency, model size, and memory usage.

METHOD
For this experiment, I used Python 3.11.8 and created a virtual environment with all the packages needed for the project. This helped me keep everything organised and made the experiment easier to reproduce.

I used the Breast Cancer Wisconsin dataset from scikit-learn and split it into 70% training data and 30% testing data. I used a stratified split with random_state=42 to maintain the class distribution and make sure I could get consistent results when running the experiment again.

I trained two models: Logistic Regression with max_iter=1000 and Random Forest with 100 estimators. Both models used random_state=42.

After training the models, I tested them using the test data and recorded their accuracy. I also compared their system performance by measuring training time, inference latency, model size, and memory usage.

For the training time, I did one warm-up run and then repeated the training five times, using the median as the final result. For inference latency, I selected one test sample and repeated the prediction 100 times before calculating the median. I also saved both models using Joblib so I could measure and compare their file sizes.

These measurements connect to the benchmarking concepts discussed in Chapter 12 of Reddi's Introduction to Machine Learning Systems. The reading helped me understand that evaluating a machine learning model is not only about its accuracy. Factors such as speed, memory usage, and model size are also important, especially when deciding where a model can be deployed.

Results

Baseline Accuracy

Logistic Regression had an accuracy of 94.74%, while Random Forest had an accuracy of 93.57%. Logistic Regression performed slightly better.

System Measurements

Logistic Regression had a median training time of 325.1666 ms and an inference latency of 0.0697 ms. Its model size was 1.03 KB.

Random Forest had a median training time of 163.2993 ms and an inference latency of 3.3557 ms. Its model size was 284.07 KB.

Random Forest trained faster, but Logistic Regression was much smaller and had faster inference.

Both models had different system requirements. Logistic Regression had a much lower inference latency at 0.0697 ms compared with 3.3557 ms for Random Forest. However, Random Forest trained faster in this experiment, with a median training time of 163.2993 ms compared with 325.1666 ms for Logistic Regression.

There was also a large difference in model size. Logistic Regression was only 1.03 KB, while Random Forest was 284.07 KB. This shows that models with similar accuracy can still have very different system costs.

Based on the measured latency and model size, Random Forest is suitable for Cloud, Edge, and Mobile environments. Its median inference latency was 3.3557 ms and its model size was 284.07 KB.

Random Forest is not suitable for TinyML because its model size is 284.07 KB, which is higher than the 100 KB limit. Although its inference latency of 3.3557 ms is within the 10 ms limit, the model size prevents it from meeting the TinyML requirements.

Logistic Regression had a much smaller model size of 1.03 KB and a median inference latency of 0.0697 ms. This makes it less demanding in terms of storage and inference speed.


Deployment Suitability

Cloud supports at least 1 GB of memory, a latency of up to 100 ms, and models up to 500 MB. Edge supports 256–1024 MB of memory, up to 50 ms latency, and models up to 50 MB.

Mobile supports 64–256 MB of memory, up to 20 ms latency, and models up to 10 MB. TinyML is more limited, with up to 256 KB of memory, 10 ms latency, and a maximum model size of 100 KB.

Random Forest can be used on Cloud, Edge, and Mobile. Its model size was 284.07 KB and its inference latency was 3.3557 ms. It is not suitable for TinyML because its model size is above the 100 KB limit.

Logistic Regression was much smaller at 1.03 KB and had a faster inference time of 0.0697 ms.

This relates to Chapter 2 of Reddi's textbook, which explains how Cloud, Edge, Mobile, and TinyML have different resource limits.

 Discussion

The experiment showed me that accuracy is not the only important factor when choosing a machine learning model.

Logistic Regression had slightly better accuracy and was much smaller than Random Forest. Random Forest trained faster, but it required more storage and had a higher inference latency.

This connects with Reddi's discussion of ML systems engineering and benchmarking. Model size, speed, memory, and the deployment environment should also be considered when choosing a model.

Conclusions

1. Logistic Regression had the highest accuracy at 94.74%, while Random Forest achieved 93.57%.

2. Logistic Regression was smaller and had faster inference, while Random Forest had a faster training time.

3. I learned that choosing a model is not only about accuracy. System requirements and the deployment environment are also important.

