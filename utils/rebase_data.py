"""
Modify the train and validation files for remote server
"""

def rebase_tesla():
    remote_data_root = '/mnt/data/home/aag106/'
    local_train_csv = '../train.csv'
    remote_train_csv = '../train-tesla.csv'

    local_val_csv = '../val.csv'
    remote_val_csv = '../val-tesla.csv'

    local_train = open(local_train_csv, 'r')
    remote_train = open(remote_train_csv, 'w')

    for line in local_train.readlines():
        new_line = remote_data_root + ''.join(line[12:])
        newer_line = new_line.replace('RawDataBIDS/', '')
        remote_train.write(newer_line)
    local_train.close()
    remote_train.close()

    local_val = open(local_val_csv, 'r')
    remote_val = open(remote_val_csv, 'w')

    for line in local_val.readlines():
        new_line = remote_data_root + ''.join(line[12:])
        newer_line = new_line.replace('RawDataBIDS/', '')
        remote_val.write(newer_line)

    local_val.close()
    remote_val.close()


def rebase_office():
    local_train_csv = 'D:/Repos/MI-DQA/train.csv'
    remote_train_csv = 'D:/Repos/MI-DQA/train-office.csv'

    local_val_csv = 'D:/Repos/MI-DQA/val.csv'
    remote_val_csv = 'D:/Repos/MI-DQA/val-office.csv'

    local_train = open(local_train_csv, 'r')
    remote_train = open(remote_train_csv, 'w')

    for line in local_train.readlines():
        new_line = line.replace('E:/', 'D:/')
        newer_line = new_line.replace('RawDataBIDS/', '')
        remote_train.write(newer_line)

    local_train.close()
    remote_train.close()

    local_val = open(local_val_csv, 'r')
    remote_val = open(remote_val_csv, 'w')

    for line in local_val.readlines():
        new_line = line.replace('E:/', 'D:/')
        newer_line = new_line.replace('RawDataBIDS/', '')
        remote_val.write(newer_line)

    local_val.close()
    remote_val.close()


if __name__=='__main__':
    rebase_office()
