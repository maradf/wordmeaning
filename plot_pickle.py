import pickle
import matplotlib.pyplot as plt

losses = pickle.load(open("Results_10_indiv/losses.p", "rb"))
val_losses = pickle.load(open("Results_10_indiv/val_losses.p", "rb"))


print(losses[0])
print(len(losses))
x = losses
y = range(0, len(losses)*50, 50)
# print(x)
plt.plot(y, x)
plt.show()

# x = range(len(val_losses))
# plt.plot(x, val_losses)
# plt.show()

