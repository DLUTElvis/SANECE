from config import *
import argparse
from utils import *
import networkx as nx
import tensorflow as tf
from model import *
from evaluate import *
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weighted', default=False)
    parser.add_argument('--directed', default=False)
    parser.add_argument('--dataset', default='cora')
    parser.add_argument('--neg_mean', default='aa')#'random' or 'aa'
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--area', default=3, type=int)#area for negative sample if 'aa'
    args = parser.parse_args()
    config = Config()

    if args.dataset == 'cora':
        config.n = 2708
        config.m = 1433
        config.layers[0] = config.m
        config.gcnlayers[0] = config.m
        input_edge_file = 'data/cora/cora.edgelist'
        input_label_file = 'data/cora/cora.label'
        output_embedding_file = 'data/cora/embed/cora.emb'

    G = nx.read_edgelist(input_edge_file, nodetype=int, create_using=nx.DiGraph())
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    if args.directed == False:
        print('undirected graph')
        G = G.to_undirected()
    Adj = nx.to_numpy_array(G)
    model = Model(config, Adj)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    X = np.loadtxt('data/'+str(args.dataset)+'/original_feature_mtx_tfidf')
    if args.neg_mean == 'R':
        print('generating negative sample by random......')
    elif args.neg_mean == 'S':
        print('generating negative sample by aa in '+str(args.area)+'-hop neighbors......' )
        aa_list = get_neg_list_aa(G, args.area)
        X_neg = get_neg_aa(X, aa_list, args.dataset)
    elif args.neg_mean == 'A':
        print('generating negative sample by attribute in ' + str(args.area) + '-hop neighbors......')
        attr_list = get_neg_list_attr(G, X, args.area)
        X_neg = get_neg_attr(X, attr_list)
    else:
        print('Error: no proper negagtive sampling strategy')
    all_pairs = getRandomWalkWholePairs(G, args.directed, config.p, config.q, config.num_walks, config.walk_length, config.window_size)
    for i in range(args.epoch):
        start = 0
        loss_all = 0
        if args.neg_mean == 'R':
            neg_list = get_neg_list_random(G,2)
            X_neg = get_neg_random(X, neg_list)
        while start < config.n:
            end = min(start + config.batch_size, config.n)
            batch_idx = np.array(range(start, end))
            batch_idx = np.random.permutation(batch_idx)
            batch_X = X[batch_idx]
            batch_X_neg = X_neg[batch_idx]
            feed_dict = {model.keep_prob: config.keep_prob, model.batch_idx: batch_idx, model.X: batch_X,
                         model.X_neg: batch_X_neg, model.batch_size: len(batch_idx)}
            _, loss_network = sess.run([model.train_net, model.loss_network], feed_dict=feed_dict)
            loss_all += loss_network


            start_idx = np.random.randint(0, len(all_pairs) - config.batch_size_sg)
            batch_idx = np.array(range(start_idx, start_idx + config.batch_size_sg))
            batch_idx = np.random.permutation(batch_idx)
            batch = np.zeros((config.batch_size_sg), dtype=np.int32)
            labels = np.zeros((config.batch_size_sg, 1), dtype=np.int32)
            batch[:] = all_pairs[batch_idx, 0]
            labels[:, 0] = all_pairs[batch_idx, 1]
            batch_X = X[batch]
            feed_dict = {model.keep_prob: config.keep_prob, model.batch_idx: batch, model.X: batch_X, model.batch_size: len(batch_idx),
                         model.labels: labels}
            _, loss_sg = sess.run([model.train_sg, model.loss_sg], feed_dict=feed_dict)
            loss_all += loss_sg
            start = end
        print('Epoch = {:d} Loss = {:.5f}'.format(i + 1, loss_all / config.n))
        feed_dict = {model.keep_prob: 1.0, model.batch_idx: np.array(range(config.n)), model.X: X, model.batch_size: config.n}
        embedding_result = sess.run(model.Y_final, feed_dict=feed_dict)
        y = read_label(input_label_file)
        test_macro_f1_neigh, test_micro_f1_neigh = multiclass_node_classification_eval(
            embedding_result, y, 0.3)
        print('test micro f1 = {:.3f} macro f1 = {:.3f} '.format(test_micro_f1_neigh, test_macro_f1_neigh))
    print('Learning finished.....')
    write_embedding(embedding_result, output_embedding_file)
if __name__ == '__main__':
    main()
