import os
from PIL import Image
from eigenface import EigenfaceModel, ImageLabel
import numpy as np
from matplotlib import pyplot as plt

imageClasses = os.listdir("ORL-DATABASE")
model = EigenfaceModel()
# Select first 5 image from each class
trainImages = []
testImages = []
for imageClass in imageClasses:
    if not imageClass.endswith(".TXT"):
        for img in os.listdir(os.path.join("ORL-DATABASE",imageClass))[:5]:
            trainImages.append(ImageLabel(Image.open(os.path.join("ORL-DATABASE",imageClass,img)), imageClass))
        for img in os.listdir(os.path.join("ORL-DATABASE",imageClass))[5:]:
            testImages.append(ImageLabel(Image.open(os.path.join("ORL-DATABASE",imageClass,img)), imageClass))

## Training
model.train(trainImages)
## Plot Eigenfaces
model.plotEigenFaces()
## Reconstruct
model.reconstructImage(ImageLabel(Image.open(os.path.join("ORL-DATABASE",imageClass,img)),"any"))
dogImage = Image.open(os.path.join("misc","dog.jpg")).convert('L')
dogImage = dogImage.resize((92,112))
model.reconstructImage(ImageLabel(dogImage,"any"))

## Testing
accuracies = []
variances = []
for dimensionality in range(1,200,1):
    accuracies.append(model.test(testImages, dimensionality))
    variances.append(np.sum(model.eigenvalues[:dimensionality]) / np.sum(model.eigenvalues))
plt.plot(variances, accuracies)
plt.show()
plt.title("Variance - Accuracy Performance")
plt.xlabel("Explained Variance")
plt.ylabel("Accuracy")
plt.grid("on")
plt.figure()
plt.plot(list(range(1,200,1)), accuracies)
plt.show()
plt.title("Dimensionality - Accuracy Performance")
plt.xlabel("Dimensionality")
plt.ylabel("Accuracy")
plt.grid("on")