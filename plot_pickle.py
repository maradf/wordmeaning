import pickle
import matplotlib.pyplot as plt

path = "Results_100_indiv/"

losses = pickle.load(open(path + "losses.p", "rb"))
train_acc = pickle.load(open(path + "train_acc.p", "rb"))

val_losses = pickle.load(open(path + "val_losses.p", "rb"))
val_acc = pickle.load(open(path + "val_acc.p", "rb"))


# print(losses[0])
# print(len(losses))
# x = losses
# y = range(0, len(losses)*50, 50)
# # print(x)
# plt.plot(y, x)
# plt.show()

# x = range(len(val_losses))
# plt.plot(val_losses, ".")
# plt.show()

plt.plot(train_acc)
plt.title("Training Accuracy of 10 Individuals")
plt.xlabel("Number of Iterations")
plt.ylabel("Accuracy")
plt.show()

plt.plot(val_acc)
plt.title("Validation Accuracy of 10 Individuals")
plt.xlabel("Number of Iterations")
plt.ylabel("Accuracy")
plt.show()