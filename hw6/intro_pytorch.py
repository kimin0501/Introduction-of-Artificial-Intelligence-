import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    
    # Transformations for input preprocessing
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Download dataset of train and test with FashionMNIST instance
    train_set = datasets.FashionMNIST('./data', train = True, download = True,transform = custom_transform)
    test_set = datasets.FashionMNIST('./data', train = False, transform = custom_transform)
    
    # Determine whether to select train dataset or test dataset based on training parameter.
    if (training == True):
        data_set = train_set
        loader = torch.utils.data.DataLoader(data_set, batch_size=64, shuffle = True)
    else:
        data_set = test_set
        loader = torch.utils.data.DataLoader(data_set, batch_size=64, shuffle = False)
        
    return loader


def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """

    # the number of each layer's nodes
    input_size = 784
    hidden_size1 = 128
    hidden_size2 = 64
    output_size = 10
    
    # building a model with a Sequential container to hold these layers
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, hidden_size1),
        nn.ReLU(),
        nn.Linear(hidden_size1, hidden_size2),
        nn.ReLU(),
        nn.Linear(hidden_size2, output_size)
    )
    
    return model



def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    # initialize the Stochastic Gradient Descent (SGD) optimizer
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # set model to train mode
    model.train()
    # total number of samples
    total_size = len(train_loader.dataset)
    
     # outer loop, initialize accumulated loss and number of correctly predicted labels for each epoch
    for epoch in range(T):
        accumulated_loss = 0.0
        correct_prediction = 0
        
        # inner loop, iterates over batches of (images, labels) pairs
        for (images, labels) in train_loader:
            # gradient initialization
            opt.zero_grad() 
            
            # compute gradient of loss
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # update the model's parameters with newly computed gradients
            opt.step()
            
            # compute the accumulated loss
            batch_size = images.size(0)
            accumulated_loss += (loss.item()) * batch_size
            
            # count the number of correct predictions
            _, predicted_labels = torch.max(outputs.data, 1) # we only need the indice of maximum value 
            sum_correct = (predicted_labels == labels).sum().item()
            correct_prediction += sum_correct
        
        # compute the accuracy and average loss of the epoch        
        epoch_accuracy = (correct_prediction / total_size) * 100 
        epoch_loss =  accumulated_loss / total_size
            
        print(f"Train Epoch: {epoch} Accuracy: {epoch_accuracy:.2f}% Loss: {epoch_loss:.3f}")                 
            
            

def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    # turn the model into evaluation mode 
    model.eval()
    
    # initialize accumulated loss and number of correctly predicted labels
    accumulated_loss = 0.0
    correct_prediction = 0
    # total number of samples
    total_size = len(test_loader.dataset)
    
    #  very similar to that of training, except that the model is turned into evaluation mode
    with torch.no_grad():
        
        # during testing there is no need to track gradients, which can cause the preoblem with the context manager
        for (images, labels) in test_loader:
            
            # compute the accumulated loss
            outputs = model(images)
            loss = criterion(outputs, labels)
            accumulated_loss += loss.item() * images.size(0)
            
            # count the number of correct predictions
            _, predicted_labels = torch.max(outputs, 1)
            correct_prediction += (predicted_labels == labels).sum().item()
            
    # compute average loss and accuracy of the test set
    test_accuracy = (correct_prediction / total_size) * 100
    test_loss = accumulated_loss / total_size
    
    
    # print both the test Loss and the test Accuracy or only Accuracy depends on boolean of show_loss 
    if (show_loss == False):
        print(f"Accuracy: {test_accuracy:.2f}%")
    else:
        print(f"Average loss: {test_loss:.4f}")
        print(f"Accuracy: {test_accuracy:.2f}%")
                 

def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    # pass test images for prediction
    logits = model(test_images)
    # convert logits to probabilities using softmax function, and again convert to a list with index
    prob = F.softmax(logits, dim = 1).tolist()[index]
    
    # list of class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    
    # initialize an empty list and pair class names with probabilities
    pair = []  
    classNum = len(class_names)
    for i in range(classNum):
         pair.append((class_names[i], prob[i]))
    
    # sort the pair list in the order of the probabilities
    sorted_pair = sorted(pair, key = lambda x: x[1], reverse = True)
    
    # print the top three most class predictions
    for i in range(3):
        print(f"{sorted_pair[i][0]}: {sorted_pair[i][1] * 100:.2f}%")




# if __name__ == '__main__':
#     '''
#     Feel free to write your own test code here to exaime the correctness of your functions. 
#     Note that this part will not be graded.
#     '''
#     criterion = nn.CrossEntropyLoss()
