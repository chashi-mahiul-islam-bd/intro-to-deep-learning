import torch 
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

'''' DATASETS AND DATALOADERS '''

BATCH_SIZE = 16

# a transform for: Tensor and Normalizing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# training data
train_data = datasets.FashionMNIST('~/.pytorch/F_MNIST_data',
                            train=True,
                            download=True,
                            transform=transform)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

# testing data
test_data = datasets.FashionMNIST('~/.pytorch/F_MNIST_data',
                            train=False,
                            download=True,
                            transform=transform)
test_loader = torch.utils.data.DataLoader(test_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

label_names = ['T-shirt/top',
               'Trouser',
               'Pullover',
               'Dress',
               'Coat',
               'Sandal',
               'Shirt',
               'Sneaker',
               'Bag',
               'Ankle Boot']

'''' VISUALIZING DATA '''

# visualize data

# batch view
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

fig = plt.figure(figsize=(25,4))
for i in np.arange(16):
    ax = fig.add_subplot(2,16/2, i+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[i]), cmap='gray')
    ax.set_title(str(label_names[labels[i].item()]))
    
# detailed view 
img = np.squeeze(images[1]) 
fig = plt.figure(figsize = (12,12))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height =img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val= round(img[x][y],2) if img[x][y] !=0 else 0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment = 'center',
                    verticalalignment = 'center',
                    color = 'white' if img[x][y]<thresh else 'black')
        
'''' NETWORK ARCHITECTURE '''

# architecture
class Classifier(nn.Module):
    def __init__(self, hidden_1, hidden_2, hidden_3, dropout_prob):
        super(Classifier, self).__init__()
        
        self.fc1 = nn.Linear(28*28, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        self.fc4 = nn.Linear(hidden_3, 10)
        
        self.dropout = nn.Dropout(p=dropout_prob)
        
    def forward(self, x):
        # flattening the input tensor
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x

'''' MODEL, LOSS FUNCTION, AND OPTIMIZER '''

model = Classifier(hidden_1=256, hidden_2=128, hidden_3=64, dropout_prob=0.2)
print(model)

# =============================================================================
# if train_on_gpu:
#     model.cuda()
# =============================================================================

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
#optimizer = optim.SGD(model.parameters(), lr=0.003)


'''' TRAINING AND VALIDATION '''
# training
epochs = 10

# keeps minimum validation loss
valid_loss_min = np.Inf

train_losses, test_losses = [], []
for epoch in range(epochs):
    
    train_loss = 0
    
    # training the model
    
    model.train()
    for images, labels in train_loader:
# =============================================================================
#         if train_on_gpu:
#             images, labels = images.cuda(), labels.cuda()
# =============================================================================
            
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item()
    
    else:
        valid_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
           for images, labels in test_loader:
               # =============================================================================
               #         if train_on_gpu:
               #             images, labels = images.cuda(), labels.cuda()
               # =============================================================================
               output = model(images)
               valid_loss += criterion(output, labels)
               
               ps = torch.exp(output)
               top_p, top_class = ps.topk(1, dim=1)
               equals = top_class == labels.view(*top_class.shape)
               accuracy += torch.mean(equals.type(torch.FloatTensor))
       
        train_losses.append(train_loss/len(train_loader))
        test_losses.append(valid_loss/len(test_loader))
        
        print("Epoch: {}/{}.. ".format(epoch+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_loss/len(train_loader)),
              "Validation Loss: {:.3f}.. ".format(valid_loss/len(test_loader)),
              "Validation Accuracy: {:.3f}".format(accuracy/len(test_loader)))
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving Model ...'.format(
                    valid_loss_min,
                    valid_loss))
            torch.save(model.state_dict(), 'mlp_model_Adam_fasion_mnist.pt')
            valid_loss_min = valid_loss
        model.train()
        
            
'''' TESTING '''
           
# testing
        
# testing on the test set 
# load last saved model with minimum valid_loss
model.load_state_dict(torch.load('mlp_model_Adam_fasion_mnist.pt'))

test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
   for images, labels in test_loader:
       # =============================================================================
       # if train_on_gpu:
       #     images, labels = images.cuda(), labels.cuda()
       # =============================================================================
       output = model(images)
       test_loss += criterion(output, labels)
       
       ps = torch.exp(output)
       top_p, top_class = ps.topk(1, dim=1)
       equals = top_class == labels.view(*top_class.shape)
       accuracy += torch.mean(equals.type(torch.FloatTensor))

print("Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
      "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))  

# inference

images, labels = next(iter(test_loader))  

# turn off gradients to speed up this part
with torch.no_grad():
    logps = model(images)
    
# as output of the network is log probabity  we need to take the exp of the probabilities
ps = torch.exp(logps)
top_p, top_class = ps.topk(1, dim=1)
equals = top_class == labels.view(*top_class.shape)

top_class =top_class.numpy().tolist()

equals = equals.numpy().tolist()

# idx = 0 -> (BATCH_SIZE-1)
idx = 10
if idx < BATCH_SIZE:
    plt.imshow(images[idx].view(28,28))
    plt.title('Predicted: ' + str(label_names[top_class[idx][0]]) + ' -> ' + str(equals[idx][0]))
else:
    print('Output of BATCH_SIZE index')   