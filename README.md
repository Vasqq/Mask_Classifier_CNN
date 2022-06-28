# COMP 472

### Mask classifier using a Convolutional Neural Network

   ## Goal of the project:

Develop a system using AI that can analyze images of faces and detect whether a person is wearing a face mask or not, as well as the type of mask that is being worn.

Detect and remove a bias (if any) the system exhibits.

Improve the system's performance using k-fold cross validation

   ## Description of the project:
   
The project was developed in Python using primarily the Pytorch and Skorch libraries. The system relies on an architecture of Deep Neural Networks known as Convolutional Neural Networks (CNN) which are a category of neural nets that are very performant at classifying images. The network uses supervised learning to train the CNN, training the AI with labeled images from four different categories: people wearing no masks, cloth masks, surgical masks or N95 masks. Once the network is trained, it can now receive new images of people wearing different type of masks or no mask at all. The network should now correctly classify new images into the correct categories.

The second phase of the project required finding and removing a bias the AI had amongst a common characteristic between images. For example, if the accuracy of images of men is less than that of women by a significant amount we would say that the AI is displaying a bias in favor of women. This could become problematic in the real world. We then refined the dataset so that the network could generalize better and retrained it so the system could classifiy images of men and women with the same accuracy. We also used k-fold cross validation in order to improve the systems overall performance across all metrics.

Some other libraries used include NumPy, Matplotlib and SkLearn.
   
   
   ## Developed by:
   
Liam Pereira 40111656
Marieme Kourouss 40095022
Mobina Zavvar 40109148
Mouhamed Diane 40089430
