import argparse
import numpy as np
from time import time
from models import BaGFN


def str2list(v):
    v = v.split(',')
    v = [int(_.strip('[]')) for _ in v]

    return v


def str2list2(v):
    v = v.split(',')
    v = [float(_.strip('[]')) for _ in v]

    return v


def str2bool(v):
    if v.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif v.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--embedding_size', type=int, default=20, help='20')
    parser.add_argument('--num_gnn', type=int, default=3, help='3')
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    parser.add_argument('--optimizer_type', type=str, default='adam', help='adam')
    parser.add_argument('--l2_reg', type=float, default=0)
    parser.add_argument('--random_seed', type=int, default=2018)

    parser.add_argument('--field_size', type=int, default=39, help='#fields 23  for Avazu| 39 for Criteo')
    parser.add_argument('--dropout_keep_prob', type=str2list2, default=[1, 1, 1])
    parser.add_argument('--deep_sizes', type=str2list, default=[20, 20, 20], help='deep layers')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/Criteo')
    parser.add_argument('--log_dir', type=str, default='./logs/Criteo')
    parser.add_argument('--data', type=str, default="Criteo", help='Avazu | Criteo')
    parser.add_argument('--data_path', type=str, default="./data/Criteo/", help='root path for all the data')

    parser.add_argument('--run_times', type=int, default=3, help='run multiple times to eliminate error')

    return parser.parse_args()


def __run__(args, file_name, run_cnt):
    path_prefix = args.data_path
    feature_size = np.load(path_prefix + '/feature_size.npy')[0]

    model = BaGFN(args, feature_size=feature_size, run_cnt=run_cnt)

    # test: file1, valid: file2, train: file3-10
    Xi_valid = np.load(path_prefix + '/part2/' + file_name[0])
    Xv_valid = np.load(path_prefix + '/part2/' + file_name[1])
    y_valid = np.load(path_prefix + '/part2/' + file_name[2])

    # -----------------------------training model-----------------------------#
    print("*****-------Start training-----******")
    for k in range(args.epoch):
        file_count, time_epoch = 0, 0

        for j in range(3, 11):
            file_count += 1
            Xi_train = np.load(path_prefix + '/part' + str(j) + '/' + file_name[0])
            Xv_train = np.load(path_prefix + '/part' + str(j) + '/' + file_name[1])
            y_train = np.load(path_prefix + '/part' + str(j) + '/' + file_name[2])

            print("---epoch :[%d/%d]--file :[%d/%d]" % (k + 1, args.epoch, j, 10))
            t1 = time()
            model.fit_once(Xi_train, Xv_train, y_train, k + 1, file_count,
                           Xi_valid, Xv_valid, y_valid)
            time_epoch += time() - t1
        print("--epoch:%d,time:%d" % (k + 1, time_epoch))

    # -----------------------------testing model-----------------------------#
    print('----------start testing!...')
    Xi_test = np.load(path_prefix + '/part1/' + file_name[0])
    Xv_test = np.load(path_prefix + '/part1/' + file_name[1])
    y_test = np.load(path_prefix + '/part1/' + file_name[2])

    model.restore()

    test_result, test_loss = model.evaluate(Xi_test, Xv_test, y_test)
    print("test-result = %.4lf, test-logloss = %.4lf" % (test_result, test_loss))
    return test_result, test_loss


if __name__ == "__main__":
    import os

    # os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    args = parse_args()
    print(args.__dict__)
    print('***************************')

    if args.data in ['Avazu']:
        # Avazu does not have numerical features so we didn't scale the data.
        file_name = ['train_i.npy', 'train_x.npy', 'train_y.npy']
    elif args.data in ['Criteo']:
        file_name = ['train_i.npy', 'train_x2.npy', 'train_y.npy']
    else:
        raise ValueError("args.data must in ['Avazu', 'Criteo']")

    test_auc, test_log = [], []

    print('run time : %d' % args.run_times)
    for run_cnt in range(1, args.run_times + 1):
        test_result, test_loss = __run__(args=args, file_name=file_name, run_cnt=run_cnt)
        test_auc.append(test_result)
        test_log.append(test_loss)

    print('test_auc', test_auc)
    print('test_log_loss', test_log)
    print('avg_auc', sum(test_auc) / len(test_auc))
    print('avg_log_loss', sum(test_log) / len(test_log))
