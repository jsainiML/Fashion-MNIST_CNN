# Loading Train/Test sets from torchvision datasets.
train_data = datasets.FashionMNIST(root="data", # Which folder to download data to
                                   train=True,  # Do we want training data
                                   download=True, # should we download yes/no
                                   transform= torchvision.transforms.ToTensor(), # to what format transform the data
                                   target_transform= None  # should we transform target label/data
                                   )
test_data = datasets.FashionMNIST(root="data", 
                                   train=False,  # Here we are extracting test data
                                   download=True,
                                   transform= torchvision.transforms.ToTensor(), 
                                   target_transform= None 
                                   )



# Since the datasets are huge for ease of training, following will create mini batches with 32 objects in each.
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

# Will display shapes after conversion
len(train_loader), len(test_loader), train_loader.batch_size


=========================================================================================================================================================

# Following is just an example to view your data loading and spliting 

# next(iter(train_loader)) is used to fetch the next batch of data from the train_loader DataLoader object.
# It allows you to iterate over the dataset in batches in the DataLoader.
features, labels = next(iter(train_loader))
features.shape, labels.shape

fig = plt.figure(figsize=(3,3))
random_it = torch.randint(0, len(features), size=[1]).item()
img, label = features[random_it], labels[random_it]
plt.imshow(img.squeeze())
plt.axis(False)
