import random
from skimage.io import imread, imshow
import matplotlib.pyplot as plt

def check_input(num,trainOriginals, trainLabels):

  if(len(trainOriginals) != len(trainLabels)): 
    print('ERRORE: CT E LABEL NON CORRISPONDONO PER QUANTITÁ')

  for x in range(num):
        ix = random.randint(0,len(trainOriginals))
        image1= trainOriginals[ix]
        image2= trainLabels[ix]
        

        # Create a figure with two subplots
        fig, ax = plt.subplots(1, 2,figsize=(10,5))

        # Display the first image in the first subplot
        ax[0].imshow(image1)
        ax[0].set_title("original layer: " + str(ix) )

        # Display the second image in the second subplot
        ax[1].imshow(image2)
        ax[1].set_title("train layer: " + str(ix) )

        # Show the figure
        plt.show()

def check_all_input(trainOriginals, trainLabels):

  if(len(trainOriginals) != len(trainLabels)): 
    print('ERRORE: CT E LABEL NON CORRISPONDONO PER QUANTITÁ')


  for ix in range(0,len(trainOriginals)):
    image1= trainOriginals[ix]
    image2= trainLabels[ix]
    

    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2,figsize=(10,5))

    # Display the first image in the first subplot
    ax[0].imshow(image1)
    ax[0].set_title("original layer: " + str(ix) )

    # Display the second image in the second subplot
    ax[1].imshow(image2)
    ax[1].set_title("train layer: " + str(ix) )

    # Show the figure
    plt.show()



def check_all_output(testOriginals1, testMasks1, preds_train_t_1):

    #num indica quanti set di immagini vedere in modo randomico

    for ix in range(len(testOriginals)):
      
      image1= testOriginals1[ix]
      image2 = testMasks1[ix]
      image3= preds_train_t_1[ix][:, :, 0]

      # Create a figure with two subplots
      fig, ax = plt.subplots(1, 3,figsize=(15,5))

      # Display the first image in the first subplot
      ax[0].imshow(image1)
      ax[0].set_title("original test layer: " + str(ix) )

      # Display the third image in the third subplot
      ax[1].imshow(image2)
      ax[1].set_title("mask layer: " + str(ix) )


      # Display the third image in the third subplot
      ax[2].imshow(image3)
      ax[2].set_title("preds layer: " + str(ix) )

      # Show the figure
      plt.show()


def check_output(num,testOriginals2, testMasks2, preds_train_t2):

    #num indica quanti set di immagini vedere in modo randomico

    for ix in range(num):
      ix = random.randint(0,len(testOriginals))
      image1= trainOriginals2[ix]
      image2 = testMasks2[ix]
      image3= preds_train_t2[ix][:, :, 0]

      # Create a figure with two subplots
      fig, ax = plt.subplots(1, 3,figsize=(15,5))

      # Display the first image in the first subplot
      ax[0].imshow(image1)
      ax[0].set_title("original test layer: " + str(ix) )

      # Display the third image in the third subplot
      ax[1].imshow(image2)
      ax[1].set_title("mask layer: " + str(ix) )


      # Display the third image in the third subplot
      ax[2].imshow(image3)
      ax[2].set_title("preds layer: " + str(ix) )

      # Show the figure
      plt.show()
