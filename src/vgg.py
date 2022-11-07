from torchvision.models import vgg11, VGG11_Weights

# Step 1: Initialize model with the best available weights
#weights = VGG11_Weights.DEFAULT
model = vgg11()

print(model)


## convolution layers
#        self._body = nn.Sequential(
#            #0
#            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
#            #nn.BatchNorm2d(64),
#            nn.ReLU(inplace=True),
#            nn.MaxPool2d(kernel_size=2),
#            ##3
#            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#            #nn.BatchNorm2d(192),
#            nn.ReLU(inplace=True),
#            nn.MaxPool2d(kernel_size=2),
#            #6
#            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
#            #nn.BatchNorm2d(384),
#            nn.ReLU(inplace=True),
#            #8
#            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
#            #nn.BatchNorm2d(256),
#            nn.ReLU(inplace=True),
#            nn.MaxPool2d(kernel_size=2),
#            #11
#            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
#            nn.ReLU(inplace=True),
#            #13
#            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#            nn.ReLU(inplace=True),
#            nn.MaxPool2d(kernel_size=2),
#            #16
#            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#            nn.ReLU(inplace=True),
#            #18
#            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
#            nn.ReLU(inplace=True),
#            nn.MaxPool2d(kernel_size=2)
#            
#        )
#        
#        # Fully connected layers
#        self._head = nn.Sequential(
#            nn.Linear(in_features=512, out_features=256), 
#            
#            # ReLU activation
#            nn.ReLU(inplace=True),
#            
#            nn.Dropout(),
#            nn.Linear(in_features=256, out_features=128), 
#            
#            # ReLU activation
#            nn.ReLU(inplace=True),
#            nn.Dropout(),
#            
#            nn.Linear(in_features=128, out_features=10),
#        )


#    def forward(self, x):
#        #print(x.shape)
#        x = self._body(x)
#        # flatten the output of conv layers
#        # dimension should be batch_size * number_of weights_in_last conv_layer
#        x = x.view(x.size()[0], -1)
#        # apply classification head
#        #print(x.shape)
#        x = self._head(x)
#        
#        return x
