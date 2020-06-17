import pickle
import matplotlib.pyplot as plt

n = 10
path = "Results_{}_indiv/".format(n)
# path = "Results_100_2_indiv/"

losses = pickle.load(open(path + "losses.p", "rb"))
train_acc = pickle.load(open(path + "train_acc.p", "rb"))
print(train_acc)
val_losses = pickle.load(open(path + "val_losses.p", "rb"))
val_acc = pickle.load(open(path + "val_acc.p", "rb"))

# print(losses[0])
# print(len(losses))
x = losses
y = range(0, len(losses)*50, 50)
# print(x)
plt.plot(y, x)
plt.title("Training losses of {} Individuals".format(n))
plt.xlabel("Number of iterations")
plt.ylabel("Losses")
plt.show()

x = range(len(val_losses))
plt.plot(val_losses, ".")
plt.title("Validation losses of {} Individuals".format(n))
plt.xlabel("Number of iterations")
plt.ylabel("Losses")
plt.show()
x = range(0, len(train_acc)*50, 50)


# y = [i * 100 for i in train_acc]
y = train_acc
plt.plot(x, y)
# plt.ylim(0, 100)
plt.title("Training Accuracy of {} Individuals".format(n))
plt.xlabel("Number of Iterations")
plt.ylabel("Accuracy")
plt.show()
x = range(0, len(val_acc)*50, 50)
y = [i * 100 for i in val_acc]
# y = val_acc
plt.plot(x, y)
# plt.ylim(0, 100)
plt.title("Validation Accuracy of {} Individuals".format(n))
plt.xlabel("Number of Iterations")
plt.ylabel("Accuracy")
plt.show()