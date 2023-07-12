# Setting up loss function and optimizer.
lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model1.parameters(), lr=0.01)


# The Training and testing code block

start_time = timer() # Is helpful noting the time difference when the hyperparameters are tweaked and CPUs/GPUs output is needed to be tracked.

Acc = Accuracy(task="multiclass", num_classes=784)
epochs = 3
trainbatchloss = 0
testbatchloss = 0

for epoch in tqdm(range(epochs)): #Helps to keep track of each loop in time-bar format

  # Model in training mode.
  for batch, (image, labels) in enumerate(train_loader):
    model1.train()
    train_preds = model1(image)
    train_loss = lossfn(train_preds, labels)
    trainbatchloss += train_loss
    train_acc = Acc(labels, train_preds.argmax(dim=1))
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    if batch % 500 == 0:
      print(f"The Batch passed through so far {batch*len(image)}/{len(train_loader.dataset)}")

  trainbatchloss /= len(train_loader)

  # Model in tess/evaluation mode.
  model1.eval()
  with torch.inference_mode():
    for batch, (test_image, test_labels) in enumerate(test_loader):
      test_preds = model1(test_image)
      test_loss = lossfn(test_preds, test_labels)
      testbatchloss += test_loss
      test_acc = Acc(test_labels, test_preds.argmax(dim=1))

    testbatchloss /= len(test_loader)

  print(f"The Training loss per batch is: {trainbatchloss} & Accuracy {train_acc:.2f} | The testing loss per batch is: {testbatchloss} & Accuracy {test_acc:.2f}")


stop_time = timer()
print(f'The time operation lasted: {start_time - stop_time}')


