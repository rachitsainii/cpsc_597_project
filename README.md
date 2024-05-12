# Tumor Hunter : A Machine-Learning system designed to detect the presence of different types of Brain Tumors in MRI Images.

## Developers : Rachit Saini, Xuan Bui

About: Tumor Hunter is a web-based application developed to detect the presence of brain tumors in MRI images and classify them into different tumor types. Leveraging machine learning techniques, particularly Convolutional Neural Networks (CNNs) and transfer learning with EfficientNetB0, Tumor Hunter offers a non-invasive and efficient solution for brain tumor diagnosis.

Installation: To install and run Tumor Hunter locally, follow these steps:

1. Clone the repository from GitHub: Tumor Hunter Repository
2. Install Visual Studio Code (VSCode) and Anaconda for your operating system.
3. Install the required libraries using conda: numpy, pandas, matplotlib, seaborn, OpenCV, TensorFlow, sklearn, Keras, tqdm, PIL, flask.
4. Open the project in VSCode, navigate to the Flask folder, and open app.py.
5. Right-click in app.py and choose "run Python file in command prompt/terminal".
6. Wait for the program to execute. When execution is finished, you will see a local host link for the form: https://localhost:[port].
7. Click on the link to launch the application in a web browser.

Usage:  Once the application is running, users can upload MRI images through the web interface. The application will process the images using the trained CNN model and provide predictions on the presence of brain tumors and their types. Results will be displayed on the web page for users to review.

Implementation Details: The project utilizes Python 3.9 and various libraries and frameworks, including Flask for the backend, HTML/CSS for the frontend, and TensorFlow/Keras for machine learning tasks. Detailed implementation details, including UML diagrams, architecture diagrams, and data flow diagrams, are provided in the project report.

Results: The final model achieved a peak validation accuracy of 97.86% and demonstrated robust performance in various testing scenarios. Results of the application, including accuracy metrics and sample predictions, are included in the project report.

Future Work: While the application satisfies the intended functionality, there are areas for improvement and future work, including enhanced error handling for non-MRI images, integration of image segmentation for better visualization, implementation of user authentication for data security, and expansion of the dataset for training.

Contributing: Contributions to Tumor Hunter are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request on GitHub.



