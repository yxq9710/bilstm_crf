import matplotlib.pyplot as plt


class plot_loss:
    def __init__(self, history):
        plot_loss_and_accuracy(history)


def plot_loss_and_accuracy(history):
    historys = {}
    history_dict = history.history
    print(history_dict.keys())

    # =======根据history_dict.keys()的顺序来调整a,b,c,d的顺序========
    (c, d, a, b) = history_dict.keys()
    historys['val_loss'] = history_dict[a]
    historys['val_accuracy'] = history_dict[b]
    historys['loss'] = history_dict[c]
    historys['accuracy'] = history_dict[d]
    history_dict = historys
    # ============================================================

    train_loss = history_dict['loss']
    train_acc = history_dict['accuracy']
    test_loss = history_dict['val_loss']
    test_acc = history_dict['val_accuracy']

    Epochs = range(1, 1 + len(train_acc))
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(Epochs, train_loss, 'r', label='train_loss')
    plt.plot(Epochs, test_loss, 'b', label='test_loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(Epochs, train_acc, 'r', label='train_acc')
    plt.plot(Epochs, test_acc, 'b', label='test_acc')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.show()
