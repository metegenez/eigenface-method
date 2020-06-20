import numpy as np
from PIL import Image
class EigenfaceModel():
    """

    """
    def __init__(self):
        pass
    def train(self,images):
        """
        Calculates eigenfaces
        :param images: List of ImageLabel obj
        :return:
        """
        self.trainImages = images
        Lmatrice = self.__formTrainingMatrice(images)
        self.eigenvalues, eigenvectors = self.__computeEigenvectors(Lmatrice)
        self.eigenfaces = self.__premultiplyEigenvectors(eigenvectors)
        pass
    def test(self, images, dimensionality):
        """
        Calculates train and test eigenspace features with given dimensionality.
        :param images: List of ImageLabel objects
        :return: Accuracy
        """
        # For
        trueLabel = 0
        falseLabel = 0
        eigenfacesSelected = self.__selectEigenfaces(dimensionality)
        weight, label = self.__trainWeigth(self.trainImages, eigenfacesSelected)
        for image in images:
            # Project onto space
            featureSet = self.__projectOntoEigenspace(eigenfacesSelected, image.image)
            # Check closest
            tempLabel = label[np.argmin(np.sum((weight - featureSet)**2,1))]
            if tempLabel == image.label:
                trueLabel = trueLabel + 1
            else:
                falseLabel = falseLabel +1

        accuracy = trueLabel / len(images)
        return accuracy

    def reconstructImage(self,image):
        eigenfaceNumber = [1, 2, 3, 5, 10, 20, 40, 60, 80, 100, 120, 150, 180, 200]
        imageArray = image.image
        faceThumbnails = []
        for N in eigenfaceNumber:
            eigenfacesSelected = self.__selectEigenfaces(N)
            projected = self.__projectOntoEigenspace(eigenfacesSelected, imageArray)
            reconst = self.averageFace
            for i in range(len(eigenfacesSelected)):
                reconst = reconst + projected[i] * eigenfacesSelected[i]
            reconstArray = reconst.reshape(112,92)
            faceThumbnails.append(self.__normalizeImage(reconstArray)*255)
        faceThumbnails.append(imageArray.reshape(112,92))
        self.__generateGrid(faceThumbnails,5,3)
        pass

    def plotEigenFaces(self):
        selectedFaces = self.__selectEigenfaces(20)
        faceThumbnails = []
        for i in range(20):
            faceThumbnails.append(self.__normalizeImage(selectedFaces[i].reshape(112,92))*255)
        self.__generateGrid(faceThumbnails, 5,4)
        pass

    def __trainWeigth(self,images, eigenfacesSelected):
        weight = []
        label = []
        for img in images:
            weight.append(self.__projectOntoEigenspace(eigenfacesSelected, img.image))
            label.append(img.label)
        return weight, label

    def __normalizeImage(self,image):
        pixels = np.asarray(image)
        mean, std = pixels.mean(), pixels.std()
        # global standardization of pixels
        pixels = (pixels - np.min(pixels))/(np.max(pixels) - np.min(pixels))
        return pixels

    def __calculateDistance(self,projection):
        pass

    def __projectOntoEigenspace(self,eigenfacesSelected, imageArray):
        return np.matmul(eigenfacesSelected, imageArray-self.averageFace)

    def __generateGrid(self, faceThumbnails, n,m):
        """

        :param faceThumbnails: List of image arrays
        :return: One picture with grid. nxm for N=20
        """

        if n*m != len(faceThumbnails):
            raise("Dikkat")

        horizontalStacks = []
        for i in range(m):
            horizontalStacks.append(np.hstack(faceThumbnails[i*n:i*n+n]))
        gridPicture = np.vstack(horizontalStacks)

        img = Image.fromarray(gridPicture)
        img.show()

    def __concentrateRows(self,images):
        rows = []
        for img in images:
            rows.append(img.image)
        return np.vstack(rows)
    def __computeAverageFace(self, rawTrainMatrice):
        return np.mean(rawTrainMatrice,0)
    def __computeDifferenceFaces(self, rawTrainMatrice):
        self.averageFace = self.__computeAverageFace(rawTrainMatrice)
        return np.transpose(rawTrainMatrice - self.averageFace)
    def __formTrainingMatrice(self,images):
        rawTrainMatrice = self.__concentrateRows(images)
        self.Amatrice = self.__computeDifferenceFaces(rawTrainMatrice)
        return self.__matriceL(self.Amatrice)

    def __matriceL(self, Amatrice):
        return np.matmul(np.transpose(Amatrice), Amatrice)
    def __computeEigenvectors(self, Lmatrice):
        eigenvalues, eigenvectors = np.linalg.eig(Lmatrice)
        idx = eigenvalues.argsort()[::-1]
        eigenValues = eigenvalues[idx]
        eigenVectors = eigenvectors[:, idx]
        return eigenValues, eigenVectors
    def __premultiplyEigenvectors(self,eigenvectors):
        eigenfaces = []
        for eig in eigenvectors:
            eigenfaces.append(np.matmul(self.Amatrice, eig))
        return np.vstack(eigenfaces)
    def __selectEigenfaces(self,N):
        return self.eigenfaces[:N,:]


class ImageLabel():
    def __init__(self, image, label):
        self.image = image
        self.label = label
        self.__vectorize()
    def __vectorize(self):
        self.image = np.array(self.image).reshape(-1)



