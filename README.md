# Fashion-MNIST Convolutional Neural Net model 
The code shows how a CNN model is trained to perfrom classification on FashionMNIST dataset. Project from 'PyTorch for Deep Learning' by Daniel Bourke

## Sample dataset

The following is the preview of images that have been feed to model for classification.  

    import torch
    fig = plt.figure(figsize=(9,9))
    row, col = 4,4
    for i in range(1, row*col+1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(row, col, i)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(train_data.classes[label])
    plt.axis(False);

![image](https://github.com/jsainiML/Fashion-MNIST_CNN/assets/135480841/a2e01479-ba31-4c99-ac52-271f6575fd75)


## Model Prediction

The Model has performed well within the desired prediction expectations with only 4 epoch runs and these results were achived from ADAM optimizer faster compare to others.

![image](https://github.com/jsainiML/Fashion-MNIST_CNN/assets/135480841/28bf9013-af64-4ccb-b12e-cdb75678a9f6)
