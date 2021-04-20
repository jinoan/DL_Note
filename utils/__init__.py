from matplotlib import pyplot as plt

def view_data_scale(x, y):
    plt.figure(figsize=(10, 10))
    plt.bar(x, y, width=0.9)
    plt.xticks(x, rotation=90)
    plt.show()