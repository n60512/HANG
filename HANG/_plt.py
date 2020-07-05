import matplotlib.pyplot as plt

def _load_file(path, _target='TestingLoss'):

    if(_target == 'TestingLoss'):
        _eval_metric = 'RMSE:'
    elif(_target == 'Accuracy'):
        _eval_metric = 'Accuracy:'

    _ep_val = list()
    path = ('{}/Loss/{}.txt'.format(path, _target))
    with open(path, 'r') as _file:
        content = _file.readlines()

    for _index, _row in enumerate(content):
        _ep_val.append(
            float(_row.split(_eval_metric)[1])
        )

    return _ep_val

def _draw_acc_mse(_accuracy, _mse, ax1, _type = 'share_x'):
    _epoch = [index for index in range(len(_accuracy))]
    if(_type == 'normal'):
        
        plt.plot(_epoch, _accuracy, label='accuracy')
        plt.plot(_epoch, _mse, label= 'mse')
        plt.legend(loc='upper right')
        pass
    elif(_type == 'share_x'):
        
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('RMSE', color=color)
        ax1.plot(_epoch, _mse, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
        ax2.plot(_epoch, _accuracy, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        ax1.set_xlabel('Epoch')

        # fig.tight_layout()

        pass



if __name__ == "__main__":
    path = 'HANG/log/origin/20200615_11_28_add_1.2pts_320.400'

    # Load eval. result of each epoch from file.
    _accuracy = _load_file(path, 'Accuracy')
    _mse = _load_file(path, 'TestingLoss')

    _save_name = '_acc_mse'
    plt.title("Generative model (initial HANN) (ign_nll)")
    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=288)
    plt.grid(True)
    
    _draw_acc_mse(_accuracy, _mse, ax1)
    # plt.show()
    fig.savefig('{}/Loss/{}.png'.format(path, _save_name), facecolor='w')
    plt.clf()


    pass

