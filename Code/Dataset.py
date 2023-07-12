# Loading Train/Test sets from torchvision datasets.
train_data = datasets.FashionMNIST(root="data", # Which folder to download data to
                                   train=True,  # Do we want training data
                                   download=True, # should we download true/yes false/no
                                   transform= torchvision.transforms.ToTensor(), # to what format transform the data
                                   target_transform= None  # should we transform target label/data
                                   )
test_data = datasets.FashionMNIST(root="data", # Which folder to download data to
                                   train=False,  # Do we want training data / here we are extracting test data
                                   download=True, # should we download true/yes false/no
                                   transform= torchvision.transforms.ToTensor(), # to what format transform the data
                                   target_transform= None  # should we transform target label/data
                                   )



# Since the datasets are huge for ease of training, following will create mini batches with 32 objects in each.
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

# Will display shapes after conversion
len(train_loader), len(test_loader), train_loader.batch_size


