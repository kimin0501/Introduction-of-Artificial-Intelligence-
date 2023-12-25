import sys
import matplotlib.pyplot as plt
import numpy as np

if __name__ =="__main__":
    
    #Q1
    #Read data from csv file 
    years = []
    frozen_days = []

    # check if the argument is valid
    if len(sys.argv) < 2:
        print("Please provide a filename to read.")
        sys.exit()
        
    filename = sys.argv[1]

    # read data and convert into arrays
    with open(filename, 'r') as file:
        next(file)
    
        for line in file:
            year, days = line.split(',')
            years.append(int(year))
            frozen_days.append(int(days))

    #Q2
    #Produce the plot from read data
    plt.plot(years, frozen_days)
    plt.title('Year vs. Number of Frozen Days')
    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    plt.savefig("plot.jpg")
    # plt.show()
    
    file.close()

    #Q3a
    #Compute the matrix X from given data    
    n1 = len(years) 
    
    # initialize the X matrix 
    X = np.zeros((n1, 2), dtype = np.int64)
    
    # allocate x values into the matrix
    for i in range(n1):
        X[i] = [1, years[i]]
    
    print("Q3a:")
    print(X)

    #Q3b
    #Compute the corresponding y values into a vector Y
    n2 = len(frozen_days)
    
    # initialize the Y matrix 
    Y = np.zeros(n2, dtype = np.int64)
    
    # allocate y values into the matrix
    for i in range(n2):
        Y[i] = frozen_days[i]
    
    print("Q3b:")
    print(Y)
    
    #Q3c
    #Compute the matrix product X^TX (we will call Z)  
    Z = np.dot(X.T ,X)
    
    print("Q3c:")
    print(Z)

    #Q3d
    #Compute the inverse of Z (we will call I)
    I = np.linalg.inv(Z)
    
    print("Q3d:")
    print(I)

    #Q3e
    #Compute the pseudo-inverse of X (we will call PI)
    PI = np.dot(I, X.T)
    
    print("Q3e:")
    print(PI)

    #Q3f
    #Compute hat beta from results we previouly calculated (hat beta = PI * Y)
    hat_beta = np.dot(PI, Y)
    
    print("Q3f:")
    print(hat_beta)

    #Q4
    #Predit the number of ice days for 2022 - 2023 winter
    x_test = 2022
    y_hat_test =  hat_beta[0] + np.dot(hat_beta[1], x_test) 
    
    print("Q4: " + str(y_hat_test))

    #Q5
    #Interpret the model by the sign of hat_beta[1]
    
    #Print the symbol by checking whether hat_beta[1] is positive, negative or zero
    if hat_beta[1] > 0:
        symbol = '>'
    elif hat_beta[1] < 0:
        symbol = '<'
    elif hat_beta[1] == 0:
        symbol = '='
    
    print("Q5a: " + symbol)
    
    # Print the interpretation
    print("Q5b: " + "'>' sign indicates that the number of frozen days increased, '<' sign indicates that " + 
          "the number of frozen days decreased, and '=' sign indicates that the number of frozen days remains same on Lake Mendota.")
    
    #Q6
    #Predict the year x_stat with given MLE(Maximum Likelihood Estimation) hat_beta  
    # 0 = beta_hat[0] + beta_hat[1] * x_star
    x_star = -(hat_beta[0] / hat_beta[1])

    print("Q6a: " + str(x_star))
    print("Q6b: The prediction is pretty compelling,because if data keeps up with this declining trend" 
      + "there may be no frozen day in the distant future.")