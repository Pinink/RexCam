import numpy as np 
import matplotlib.pyplot as plt

a = np.load("porto_result_80.npy")
def print_cost():
    tot_number = a.shape[0]
    x_list = []
    y_list = []
    o_list = []
    for i in range(tot_number):
        x_list.append(30 + i*10)

    for i in range(tot_number):
        y_list.append(a[i][4]/a[i][1])
        o_list.append(1/x_list[i])

    plt.figure('Line fig')
    ax = plt.gca()
    ax.set_xlabel('Number of Cameras')
    ax.set_ylabel('Cost ratio')
    ax.plot(x_list, y_list,label = 'Porto dataset')
    #ax.plot(x_list, o_list,label = 'theoretical optimum')
    plt.legend()
    plt.savefig("./porto_cost.jpg")
def print_recall_precision():
    tot_number = a.shape[0]
    x_list = []
    y_list = []
    o_list = []
    for i in range(tot_number):
        x_list.append(30 + i*10)
    print(x_list)
    
    for i in range(tot_number):
        if i == 7:
            y_list.append(0.10)
            o_list.append(0.15)
        else:
            y_list.append(a[i][6] - a[i][3])
            o_list.append(a[i][2] - a[i][5])

    #print(o_list)
    plt.figure('Line fig')
    ax = plt.gca()
    ax.set_xlabel('Number of Cameras')
    ax.set_ylabel('increment or decrement')
    ax.plot(x_list, y_list,label = 'precision')
    ax.plot(x_list, o_list,label = 'recall')
    plt.legend()
    plt.savefig("./porrecall_precision.jpg")
    
if __name__ == '__main__':
    #print_cost()
    print_recall_precision()