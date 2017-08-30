#matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def magic(number):
    return float(''.join(str(i) for i in number))

inputTrain = open("TrainVal_ac3.csv","r")

train_accuracies = []
valida_accuracies = []
x_range = []

for line in inputTrain.readlines():
	data = [float(x) for x in line.strip().split(',') if x != '']
	x_range.append(int(magic(data[:1])))
	train_accuracies.append(magic(data[1:2]))
	valida_accuracies.append(magic(data[2:]))



#print train_accuracies
#print test_accuracies

plt.plot(x_range, train_accuracies,'-r', label='Entrenamiento')
plt.plot(x_range, valida_accuracies,'-b', label='Validacion')
plt.legend(loc='lower right', frameon=False)
plt.ylim(ymax = 1.05, ymin = 0.1)
plt.ylabel('acierto')
plt.xlabel('iteraciones')
plt.show()
