# Creating a CNN layer with fully connected classifier to train the data 

class CompVision(nn.Module):
  def __init__(self, input:int, hiden:int, output:int):
    super().__init__()

    self.conv_block1 = nn.Sequential(
        nn.Conv2d(in_channels=input, out_channels=hiden, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hiden, out_channels=hiden, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.conv_block2 = nn.Sequential(
        nn.Conv2d(in_channels=hiden, out_channels=hiden, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hiden, out_channels=hiden, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    


    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.LazyLinear(out_features=output)
    )

  def cal_inlayear(self):
    dummy = torch.rand([1,1,28,28])
    data =  self.conv_block1(dummy)
    data = self.conv_block2(data)
    return int(np.prod(data.size()))

  def forward(self, x):
    step = self.conv_block1(x)
    #print(step.shape)
    step2 = self.conv_block2(step)
    #print(step2.shape)
    step3 = self.classifier(step2)
    return step3


====================================================================================================================================================

# On side note: In above structure I have used nn.Lazylinear() module to inherit shape from previous function which remove the manual calculations troubles.
# But I have noticied some discrepencies results while running model in same data on different days, Since this function is still in development in case of error, try using following fuction does the same job. 



indims = self.cal_inlayear()

def cal_inlayear(self):
    dummy = torch.rand([1,1,28,28])
    data =  self.conv_block1(dummy)
    data = self.conv_block2(data)
    return int(np.prod(data.size()))

