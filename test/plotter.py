import matplotlib.pyplot as plt

x = list()
y = list()

with open("libfranka_example.txt") as file_handle:
    counter = 0
    for line in file_handle:
        buffer =  line.split()
        x.append(buffer[0])   # step
        # x.append(buffer[1])   # time
        # y.append(buffer[2])   # delta_x
        # y.append(buffer[3])   # angle
        y.append(buffer[4])   # action


plt.plot(x, y)
# for a, b in zip(x, y):
#     plt.text(a, b, str(b))
plt.show()