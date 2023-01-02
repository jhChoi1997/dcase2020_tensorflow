import os
import numpy as np
import tensorflow as tf

import utils
import dataset

from sklearn import metrics
from tqdm import tqdm
from tensorflow.keras.optimizers.schedules import CosineDecay


class WaveNetTrainer(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.machine_type = kwargs['machine_type']
        self.visualizer = kwargs['visualizer']
        self.model = kwargs['model']
        self.train_loss = kwargs['train_loss']
        self.csv_lines = []

    def loss_function(self, true, pred):
        receptive_field = true.shape[1] - pred.shape[1]
        mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        loss = tf.reduce_mean(mse(true[:, receptive_field:], pred), axis=1)
        return loss

    def train(self, train_dataset, val_dataset):
        os.makedirs(os.path.join(self.args.model_dir, self.args.version), exist_ok=True)
        print(f'Start Training - {self.machine_type}, {self.args.epochs} epochs')

        best_auc = 0
        a = 0
        p = 0
        e = 0
        no_better = 0
        final_mean, final_inv_cov = 0, 0
        epoch_loss = []

        for epoch in range(self.args.epochs):
            self.train_loss.reset_states()

            lr = CosineDecay(self.args.lr, decay_steps=len(train_dataset))
            optimizer = tf.keras.optimizers.Adam(lr)

            pbar = tqdm(train_dataset, total=len(train_dataset), ncols=200)

            for data, label in pbar:
                with tf.GradientTape() as tape:
                    output = self.model(data, training=True)
                    loss = self.loss_function(data, output)
                    training_loss = tf.reduce_mean(loss)
                    regularization_loss = tf.math.add_n(self.model.losses)
                    total_loss = training_loss + regularization_loss
                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, self.model.trainable_variables) if grad is not None)
                self.train_loss.update_state(loss)
                pbar.set_description(f'Epoch: {epoch + 1}\tLoss: {self.train_loss.result():.4f}')
            epoch_loss.append(self.train_loss.result())
            self.visualizer.add_train_loss1(self.train_loss.result())

            if epoch % 1 == 0:
                mean, inv_cov = self.eval_preprocess(val_dataset)
                auc, pauc = self.eval(mean, inv_cov)
                if auc + pauc > best_auc:
                    no_better = 0
                    best_auc = auc + pauc
                    a, p, e = auc, pauc, epoch
                    checkpoint_path = 'checkpoint_best_model'
                    utils.save_model(self.args, self.model, self.machine_type, checkpoint_path, self.visualizer)
                    print(f'Model saved! \t mean AUC: {a}, mean pAUC: {p}')
                    final_mean, final_inv_cov = mean, inv_cov
                else:
                    no_better += 1
                if no_better > self.args.early_stop:
                    break

        print(f'Training {self.machine_type} completed! \t Best Epoch: {e:4d}\tBest AUC: {a:3.3f}\tpAUC: {p:3.3f}')
        return final_mean, final_inv_cov

    def eval(self, mean, inv_cov):
        os.makedirs(os.path.join(self.args.result_dir, self.args.version), exist_ok=True)
        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))

        for id_str in machine_id_list:
            y_pred = []
            batch_data, y_true = dataset.get_wavenet_eval_test_dataset(self.args, self.machine_type, id_str, is_eval=True)
            for batch, name in batch_data:
                output = self.model(np.moveaxis(batch, 1, 2), training=False)
                error_vector = self.get_error_vector(np.moveaxis(batch, 1, 2), output)
                score = self.anomaly_score(error_vector, mean, inv_cov)
                y_pred.extend(score)

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        return mean_auc, mean_p_auc

    def test(self, mean, inv_cov):
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        os.makedirs(result_dir, exist_ok=True)

        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        eval_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))
        test_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.test_dir, self.machine_type, 'test'))
        machine_id_list = eval_machine_id_list + test_machine_id_list
        machine_id_list.sort()
        for id_str in machine_id_list:
            y_pred = []
            anomaly_score_csv = f'{self.args.result_dir}/{self.args.version}/anomaly_score_{self.machine_type}_{id_str}.csv'
            anomaly_score_list = []
            anomaly_score_list.append(['Name', 'Anomaly Score'])
            if id_str in eval_machine_id_list:
                batch_data, y_true = dataset.get_wavenet_eval_test_dataset(self.args, self.machine_type, id_str, is_eval=True)
            else:
                batch_data, y_true = dataset.get_wavenet_eval_test_dataset(self.args, self.machine_type, id_str, is_eval=False)
            for batch, name in batch_data:
                output = self.model(np.moveaxis(batch, 1, 2), training=False)
                error_vector = self.get_error_vector(np.moveaxis(batch, 1, 2), output)
                score = self.anomaly_score(error_vector, mean, inv_cov)
                y_pred.extend(score)
                anomaly_score_list.extend([[os.path.split(a.numpy().decode('utf-8'))[1], b] for a, b in zip(name, score)])

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
            id_AUC.append(auc)
            print(f'{id_str} AUC: {auc:3.3f} \t pAUC: {p_auc:3.3f}')
            utils.save_csv(anomaly_score_csv, anomaly_score_list)
            self.csv_lines.append([id_str, auc, p_auc])

        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        print(self.machine_type, 'AUC:', mean_auc, 'pAUC:', mean_p_auc)

        self.csv_lines.append(['Average'] + [mean_auc, mean_p_auc])
        self.csv_lines.append([])

        result_path = os.path.join(result_dir, f'{self.machine_type}_{self.args.result_file}')
        utils.save_csv(result_path, self.csv_lines)

    def eval_preprocess(self, val_dataset):
        loss_list = []
        for data, label in val_dataset:
            output = self.model(data, training=False)
            error_vector = self.get_error_vector(data, output)
            loss_list.extend(error_vector)

        loss_array = np.array(loss_list)
        mean = np.mean(loss_list, axis=0)
        cov = np.cov(np.transpose(loss_array))
        inv_cov = np.linalg.inv(cov)
        return mean, inv_cov

    def get_error_vector(self, data, output):
        receptive_field = data.shape[1] - output.shape[1]
        error = tf.cast(data[:, receptive_field:], tf.float32) - output
        return np.mean(error, axis=1)

    def anomaly_score(self, error_vector, mean, inv_cov):
        x = error_vector - mean
        tmp = np.matmul(x, inv_cov)
        tmp1 = np.matmul(tmp, np.transpose(x))
        dist = np.sqrt(np.diag(tmp1))
        return dist


class ResNetTrainer(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.machine_type = kwargs['machine_type']
        self.visualizer = kwargs['visualizer']
        self.model = kwargs['model']
        self.train_loss = kwargs['train_loss']
        self.n_class = kwargs['n_class']
        self.csv_lines = []

    def loss_function(self, true, pred):
        cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False,
                                                                reduction=tf.keras.losses.Reduction.NONE,
                                                                label_smoothing=0.1)
        loss = cross_entropy(true, tf.cast(pred, tf.float32))
        return loss

    def train(self, train_dataset, val_dataset):
        os.makedirs(os.path.join(self.args.model_dir, self.args.version), exist_ok=True)
        print(f'Start Training - {self.machine_type}, {self.args.epochs} epochs')

        best_auc = 0
        a = 0
        p = 0
        e = 0
        no_better = 0
        epoch_loss = []

        for epoch in range(self.args.epochs):
            self.train_loss.reset_states()

            lr = CosineDecay(self.args.lr, decay_steps=len(train_dataset))
            optimizer = tf.keras.optimizers.Adam(lr)

            pbar = tqdm(train_dataset, total=len(train_dataset), ncols=200)
            for data, label in pbar:
                with tf.GradientTape() as tape:
                    output = self.model(data, training=True)
                    loss = self.loss_function(label, output)
                    total_loss = tf.reduce_mean(loss)
                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                optimizer.apply_gradients(
                    (grad, var) for (grad, var) in zip(gradients, self.model.trainable_variables) if
                    grad is not None)
                self.train_loss.update_state(loss)
                pbar.set_description(f'Epoch: {epoch + 1}\tLoss: {self.train_loss.result():.4f}')
            epoch_loss.append(self.train_loss.result())
            self.visualizer.add_train_loss1(self.train_loss.result())

            if epoch % 1 == 0:
                auc, pauc = self.eval()
                if auc + pauc > best_auc:
                    no_better = 0
                    best_auc = auc + pauc
                    a, p, e = auc, pauc, epoch
                    checkpoint_path = 'checkpoint_best_model'
                    utils.save_model(self.args, self.model, self.machine_type, checkpoint_path, self.visualizer)
                    print(f'Model saved! \t mean AUC: {a}, mean pAUC: {p}')
                else:
                    no_better += 1
                if no_better > self.args.early_stop:
                    break
        print(f'Training {self.machine_type} completed! \t Best Epoch: {e:4d}\tBest AUC: {a:3.3f}\tpAUC: {p:3.3f}')

    def eval(self):
        os.makedirs(os.path.join(self.args.result_dir, self.args.version), exist_ok=True)
        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))

        for id_str in machine_id_list:
            y_pred = []
            batch_data, y_true = dataset.get_resnet_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=True)
            for batch, label, name in batch_data:
                output = self.model(batch, training=False)
                score = self.anomaly_score(label, output)
                y_pred.extend(score)

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        return mean_auc, mean_p_auc

    def test(self):
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        os.makedirs(result_dir, exist_ok=True)

        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        eval_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))
        test_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.test_dir, self.machine_type, 'test'))
        machine_id_list = eval_machine_id_list + test_machine_id_list
        machine_id_list.sort()
        for id_str in machine_id_list:
            y_pred = []
            anomaly_score_csv = f'{self.args.result_dir}/{self.args.version}/anomaly_score_{self.machine_type}_{id_str}.csv'
            anomaly_score_list = []
            anomaly_score_list.append(['Name', 'Anomaly Score'])
            if id_str in eval_machine_id_list:
                batch_data, y_true = dataset.get_resnet_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=True)
            else:
                batch_data, y_true = dataset.get_resnet_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=False)
            for batch, label, name in batch_data:
                output = self.model(batch, training=False)
                score = self.anomaly_score(label, output)
                y_pred.extend(score)
                anomaly_score_list.extend([[os.path.split(a.numpy().decode('utf-8'))[1], b.numpy()] for a, b in zip(name, score)])

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
            id_AUC.append(auc)
            print(f'{id_str} AUC: {auc:3.3f} \t pAUC: {p_auc:3.3f}')
            utils.save_csv(anomaly_score_csv, anomaly_score_list)
            self.csv_lines.append([id_str, auc, p_auc])

        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        print(self.machine_type, 'AUC:', mean_auc, 'pAUC:', mean_p_auc)

        self.csv_lines.append(['Average'] + [mean_auc, mean_p_auc])
        self.csv_lines.append([])

        result_path = os.path.join(result_dir, f'{self.machine_type}_{self.args.result_file}')
        utils.save_csv(result_path, self.csv_lines)

    def anomaly_score(self, label, output):
        # loss = tf.keras.metrics.categorical_crossentropy(label, output)
        loss = tf.reduce_sum(tf.cast(label, tf.float32) * output, axis=1)
        return loss


class MTLClassTrainer(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.machine_type = kwargs['machine_type']
        self.visualizer = kwargs['visualizer']
        self.model = kwargs['model']
        self.train_loss = kwargs['train_loss']
        self.train_loss1 = kwargs['train_loss1']
        self.train_loss2 = kwargs['train_loss2']
        self.n_class = kwargs['n_class']
        self.csv_lines = []
        self.loss_value = []
        self.weight_value = []

    def loss_function(self, true_mel, output_mel, true_label, pred_label, weight):
        receptive_field = true_mel.shape[1] - output_mel.shape[1]

        mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        loss1 = tf.reduce_mean(mse(true_mel[:, receptive_field:], output_mel), axis=1)

        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False,
                                                      reduction=tf.keras.losses.Reduction.NONE,
                                                      label_smoothing=0.1)
        loss2 = cce(true_label, tf.cast(pred_label, tf.float32))

        w1, w2 = weight
        return w1 * loss1 + w2 * loss2, loss1, loss2

    def train(self, train_dataset, val_dataset):
        os.makedirs(os.path.join(self.args.model_dir, self.args.version), exist_ok=True)
        print(f'Start Training - {self.machine_type}, {self.args.epochs} epochs')

        best_auc = 0
        a = 0
        p = 0
        e = 0
        no_better = 0
        final_mean, final_inv_cov, final_score_mean, final_score_inv_cov = 0, 0, 0, 0
        epoch_loss = []

        for epoch in range(self.args.epochs):
            self.train_loss.reset_states()
            self.train_loss1.reset_states()
            self.train_loss2.reset_states()

            lr = CosineDecay(self.args.lr, decay_steps=len(train_dataset))
            optimizer = tf.keras.optimizers.Adam(lr)
            pbar = tqdm(train_dataset, total=len(train_dataset), ncols=200)

            self.weight_value = self.loss_weighting(self.loss_value, self.weight_value)

            for data, label in pbar:
                with tf.GradientTape() as tape:
                    output1, output2 = self.model([data, label], training=True)
                    loss, loss1, loss2 = self.loss_function(data, output1, label, output2, self.weight_value[-1])
                    total_loss = tf.reduce_mean(loss)
                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, self.model.trainable_variables) if grad is not None)
                self.train_loss.update_state(loss)
                self.train_loss1.update_state(loss1)
                self.train_loss2.update_state(loss2)
                pbar.set_description(f'Epoch: {epoch + 1}\tLoss: {self.train_loss.result():.4f}\tLoss1: {self.train_loss1.result():.4f}\tLoss2: {self.train_loss2.result():.4f}')
            epoch_loss.append(self.train_loss.result())
            self.visualizer.add_train_loss1(self.train_loss1.result())
            self.visualizer.add_train_loss2(self.train_loss2.result())
            self.loss_value.append([self.train_loss1.result(), self.train_loss2.result()])

            if epoch % 1 == 0:
                mean, inv_cov, score_mean, score_inv_cov = self.eval_preprocess(val_dataset)
                auc, pauc = self.eval(mean, inv_cov, score_mean, score_inv_cov)
                if auc + pauc > best_auc:
                    no_better = 0
                    best_auc = auc + pauc
                    a, p, e = auc, pauc, epoch
                    checkpoint_path = 'checkpoint_best_model'
                    utils.save_model(self.args, self.model, self.machine_type, checkpoint_path, self.visualizer)
                    print(f'Model saved! \t mean AUC: {a}, mean pAUC: {p}')
                    final_mean, final_inv_cov, final_score_mean, final_score_inv_cov = mean, inv_cov, score_mean, score_inv_cov
                else:
                    no_better += 1
                if no_better > self.args.early_stop:
                    break

        print(f'Training {self.machine_type} completed! \t Best Epoch: {e:4d}\tBest AUC: {a:3.3f}\tpAUC: {p:3.3f}')
        return final_mean, final_inv_cov, final_score_mean, final_score_inv_cov

    def eval(self, mean, inv_cov, score_mean, score_inv_cov):
        os.makedirs(os.path.join(self.args.result_dir, self.args.version), exist_ok=True)
        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))

        for id_str in machine_id_list:
            y_pred = []
            batch_data, y_true = dataset.get_mtl_class_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=True)
            for batch, label, name in batch_data:
                output1, output2 = self.model([np.moveaxis(batch, 1, 2), label], training=False)
                error_vector = self.get_error_vector(np.moveaxis(batch, 1, 2), output1)
                score1 = self.anomaly_score1(error_vector, mean, inv_cov)
                score2 = self.anomaly_score2(label, output2)
                score = self.anomaly_score(score1, score2, score_mean, score_inv_cov)
                y_pred.extend(score)

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        return mean_auc, mean_p_auc

    def test(self, mean, inv_cov, score_mean, score_inv_cov):
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        os.makedirs(result_dir, exist_ok=True)

        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        eval_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))
        test_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.test_dir, self.machine_type, 'test'))
        machine_id_list = eval_machine_id_list + test_machine_id_list
        machine_id_list.sort()
        for id_str in machine_id_list:
            y_pred = []
            anomaly_score_csv = f'{self.args.result_dir}/{self.args.version}/anomaly_score_{self.machine_type}_{id_str}.csv'
            anomaly_score_list = []
            anomaly_score_list.append(['Name', 'Anomaly Score', 'Score 1', 'Score 2'])
            if id_str in eval_machine_id_list:
                batch_data, y_true = dataset.get_mtl_class_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=True)
            else:
                batch_data, y_true = dataset.get_mtl_class_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=False)
            for batch, label, name in batch_data:
                output1, output2 = self.model([np.moveaxis(batch, 1, 2), label], training=False)
                error_vector = self.get_error_vector(np.moveaxis(batch, 1, 2), output1)
                score1 = self.anomaly_score1(error_vector, mean, inv_cov)
                score2 = self.anomaly_score2(label, output2)
                score = self.anomaly_score(score1, score2, score_mean, score_inv_cov)
                y_pred.extend(score)
                anomaly_score_list.extend([[os.path.split(a.numpy().decode('utf-8'))[1], b, c, d.numpy()] for a, b, c, d in zip(name, score, score1, score2)])

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
            id_AUC.append(auc)
            print(f'{id_str} AUC: {auc:3.3f} \t pAUC: {p_auc:3.3f}')
            utils.save_csv(anomaly_score_csv, anomaly_score_list)
            self.csv_lines.append([id_str, auc, p_auc])

        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        print(self.machine_type, 'AUC:', mean_auc, 'pAUC:', mean_p_auc)

        self.csv_lines.append(['Average'] + [mean_auc, mean_p_auc])
        self.csv_lines.append([])

        result_path = os.path.join(result_dir, f'{self.machine_type}_{self.args.result_file}')
        utils.save_csv(result_path, self.csv_lines)

    def loss_weighting(self, loss_list, weight_list):
        if len(weight_list) < 2:
            weight_list.append([1, 1])
        else:
            T = 2
            r1 = loss_list[-1][0] / loss_list[-2][0]
            r2 = loss_list[-1][1] / loss_list[-2][1]
            w1 = 2 * np.exp(r1 / T) / (np.exp(r1 / T) + np.exp(r2 / T))
            w2 = 2 * np.exp(r2 / T) / (np.exp(r1 / T) + np.exp(r2 / T))
            weight_list.append([w1, w2])
        return weight_list

    def eval_preprocess(self, val_dataset):
        error_vector_list = []
        score_list1 = []
        score_list2 = []
        for data, label in val_dataset:
            output1, output2 = self.model(data, training=False)
            error_vector = self.get_error_vector(data, output1)
            score2 = self.anomaly_score2(label, output2)
            error_vector_list.extend(error_vector)
            score_list2.extend(score2)

        error_vector_array = np.array(error_vector_list)
        mean = np.mean(error_vector_array, axis=0)
        cov = np.cov(np.transpose(error_vector_array))
        inv_cov = np.linalg.inv(cov)

        # score_list1.extend([self.anomaly_score1(v[np.newaxis], mean, inv_cov)[0] for v in error_vector_list])
        score_list1.extend(self.anomaly_score1(error_vector_array, mean, inv_cov))

        score_array = np.array([(u, v) for u, v in zip(score_list1, score_list2)])
        score_mean = np.mean(score_array, axis=0)
        score_cov = np.cov(np.transpose(score_array))
        score_inv_cov = np.linalg.inv(score_cov)

        return mean, inv_cov, score_mean, score_inv_cov

    def get_error_vector(self, data, output):
        receptive_field = data.shape[1] - output.shape[1]
        error = tf.cast(data[:, receptive_field:], tf.float32) - output
        return np.mean(error, axis=1)

    def anomaly_score1(self, error_vector, mean, inv_cov):
        x = error_vector - mean
        tmp = np.matmul(x, inv_cov)
        tmp1 = np.matmul(tmp, np.transpose(x))
        dist = np.sqrt(np.diag(tmp1))
        return dist

    def anomaly_score2(self, true, pred):
        score = tf.reduce_sum(tf.cast(true, tf.float32) * pred, axis=1)
        return score

    def anomaly_score(self, score1, score2, score_mean, score_inv_cov):
        score = np.array([(u, v) for u, v in zip(score1, score2)])
        x = score - score_mean
        tmp = np.matmul(x, score_inv_cov)
        tmp1 = np.matmul(tmp, np.transpose(x))
        dist = np.sqrt(np.diag(tmp1))
        return dist


class MultiResolutionWaveNetTrainer(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.machine_type = kwargs['machine_type']
        self.visualizer = kwargs['visualizer']
        self.model = kwargs['model']
        self.train_loss = kwargs['train_loss']
        self.csv_lines = []

    def loss_function(self, true_mel, output_mel):
        n_blocks = output_mel.shape[3]
        receptive_field = true_mel.shape[1] - output_mel.shape[1]
        true_mel_blocks = tf.stack([true_mel for _ in range(n_blocks)], axis=-1)

        mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        loss = tf.reduce_mean(mse(true_mel_blocks[:, receptive_field:], output_mel), axis=(1, 2))

        return loss

    def train(self, train_dataset, val_dataset):
        os.makedirs(os.path.join(self.args.model_dir, self.args.version), exist_ok=True)
        print(f'Start Training - {self.machine_type}, {self.args.epochs} epochs')

        best_auc = 0
        a = 0
        p = 0
        e = 0
        no_better = 0
        final_mean, final_inv_cov, final_block_mean, final_block_inv_cov = 0, 0, 0, 0
        epoch_loss = []

        for epoch in range(self.args.epochs):
            self.train_loss.reset_states()

            lr = CosineDecay(self.args.lr, decay_steps=len(train_dataset))
            optimizer = tf.keras.optimizers.Adam(lr)
            pbar = tqdm(train_dataset, total=len(train_dataset), ncols=200)

            for data, label in pbar:
                with tf.GradientTape() as tape:
                    output = self.model(data, training=True)
                    loss = self.loss_function(data, output)
                    total_loss = tf.reduce_mean(loss)
                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, self.model.trainable_variables) if grad is not None)
                self.train_loss.update_state(loss)
                pbar.set_description(f'Epoch: {epoch + 1}\tLoss: {self.train_loss.result():.4f}')
            epoch_loss.append(self.train_loss.result())
            self.visualizer.add_train_loss1(self.train_loss.result())

            if epoch % 1 == 0:
                mean, inv_cov, block_mean, block_inv_cov = self.eval_preprocess(val_dataset)
                auc, pauc = self.eval(mean, inv_cov, block_mean, block_inv_cov)
                if auc + pauc > best_auc:
                    no_better = 0
                    best_auc = auc + pauc
                    a, p, e = auc, pauc, epoch
                    checkpoint_path = 'checkpoint_best_model'
                    utils.save_model(self.args, self.model, self.machine_type, checkpoint_path, self.visualizer)
                    print(f'Model saved! \t mean AUC: {a}, mean pAUC: {p}')
                    final_mean, final_inv_cov, final_block_mean, final_block_inv_cov = mean, inv_cov, block_mean, block_inv_cov
                else:
                    no_better += 1
                if no_better > self.args.early_stop:
                    break

        print(f'Training {self.machine_type} completed! \t Best Epoch: {e:4d}\tBest AUC: {a:3.3f}\tpAUC: {p:3.3f}')
        return final_mean, final_inv_cov, final_block_mean, final_block_inv_cov

    def eval(self, mean, inv_cov, block_mean, block_inv_cov):
        os.makedirs(os.path.join(self.args.result_dir, self.args.version), exist_ok=True)
        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))

        for id_str in machine_id_list:
            y_pred = []
            batch_data, y_true = dataset.get_wavenet_eval_test_dataset(self.args, self.machine_type, id_str, is_eval=True)
            for batch, name in batch_data:
                output = self.model(np.moveaxis(batch, 1, 2), training=False)
                error_vector = self.get_error_vectors(np.moveaxis(batch, 1, 2), output)
                n_block = error_vector.shape[2]

                score_vector = np.zeros((error_vector.shape[0], n_block))
                for i in range(n_block):
                    score_vector[:, i] = self.block_score(error_vector[..., i], mean[i], inv_cov[i])
                score = self.anomaly_score(score_vector, block_mean, block_inv_cov)
                y_pred.extend(score)

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        return mean_auc, mean_p_auc

    def test(self, mean, inv_cov, block_mean, block_inv_cov):
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        os.makedirs(result_dir, exist_ok=True)

        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        eval_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))
        test_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.test_dir, self.machine_type, 'test'))
        machine_id_list = eval_machine_id_list + test_machine_id_list
        machine_id_list.sort()
        for id_str in machine_id_list:
            y_pred = []
            anomaly_score_csv = f'{self.args.result_dir}/{self.args.version}/anomaly_score_{self.machine_type}_{id_str}.csv'
            anomaly_score_list = []
            anomaly_score_list.append(['Name', 'Anomaly Score', 'Score'])
            if id_str in eval_machine_id_list:
                batch_data, y_true = dataset.get_wavenet_eval_test_dataset(self.args, self.machine_type, id_str, is_eval=True)
            else:
                batch_data, y_true = dataset.get_wavenet_eval_test_dataset(self.args, self.machine_type, id_str, is_eval=False)
            for batch, name in batch_data:
                output = self.model(np.moveaxis(batch, 1, 2), training=False)
                error_vector = self.get_error_vectors(np.moveaxis(batch, 1, 2), output)
                n_block = error_vector.shape[2]

                score_vector = np.zeros((error_vector.shape[0], n_block))
                for i in range(n_block):
                    score_vector[:, i] = self.block_score(error_vector[..., i], mean[i], inv_cov[i])
                score = self.anomaly_score(score_vector, block_mean, block_inv_cov)
                y_pred.extend(score)
                anomaly_score_list.extend([[os.path.split(a.numpy().decode('utf-8'))[1], b] for a, b in zip(name, score)])

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
            id_AUC.append(auc)
            print(f'{id_str} AUC: {auc:3.3f} \t pAUC: {p_auc:3.3f}')
            utils.save_csv(anomaly_score_csv, anomaly_score_list)
            self.csv_lines.append([id_str, auc, p_auc])

        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        print(self.machine_type, 'AUC:', mean_auc, 'pAUC:', mean_p_auc)

        self.csv_lines.append(['Average'] + [mean_auc, mean_p_auc])
        self.csv_lines.append([])

        result_path = os.path.join(result_dir, f'{self.machine_type}_{self.args.result_file}')
        utils.save_csv(result_path, self.csv_lines)

    def eval_preprocess(self, val_dataset):
        error_vector_list = []

        for data, label in val_dataset:
            output = self.model(data, training=False)
            error_vectors = self.get_error_vectors(data, output)
            error_vector_list.extend(error_vectors)

        error_vector_array = np.array(error_vector_list)
        n_blocks = error_vector_array.shape[2]
        mean_list, inv_cov_list = [], []
        score_list = []

        for block_ids in range(n_blocks):
            mean = np.mean(error_vector_array[:, :, block_ids], axis=0)
            cov = np.cov(np.transpose(error_vector_array[:, :, block_ids]))
            inv_cov = np.linalg.inv(cov)
            mean_list.append(mean)
            inv_cov_list.append(inv_cov)

            score = self.anomaly_score(error_vector_array[..., block_ids], mean, inv_cov)
            score_list.append(score)

        score_array = np.array(score_list)
        score_mean = np.mean(score_array, axis=1)
        score_cov = np.cov(score_array)
        score_inv_cov = np.linalg.inv(score_cov)

        return np.array(mean_list), np.array(inv_cov_list), score_mean, score_inv_cov

    def get_error_vectors(self, data, output):
        n_blocks = output.shape[3]
        receptive_field = data.shape[1] - output.shape[1]
        data_copy = tf.stack([data for _ in range(n_blocks)], axis=-1)
        spec_diff = tf.cast(data_copy[:, receptive_field:], tf.float32) - output
        freq_mean = tf.reduce_mean(spec_diff, axis=1)
        return freq_mean.numpy()

    def block_score(self, spec, mean, inv_cov):
        x = spec - mean
        tmp = np.matmul(x, inv_cov)
        tmp1 = np.matmul(tmp, np.transpose(x))
        dist = np.sqrt(np.diag(tmp1))
        return dist

    def anomaly_score(self, vector, mean, inv_cov):
        x = vector - mean
        tmp = np.matmul(x, inv_cov)
        tmp1 = np.matmul(tmp, np.transpose(x))
        dist = np.sqrt(np.diag(tmp1))
        return dist


class MTLSegmentationTrainer(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.machine_type = kwargs['machine_type']
        self.visualizer = kwargs['visualizer']
        self.model = kwargs['model']
        self.train_loss = kwargs['train_loss']
        self.train_loss1 = kwargs['train_loss1']
        self.train_loss2 = kwargs['train_loss2']
        self.n_class = kwargs['n_class']
        self.csv_lines = []
        self.loss_value = []
        self.weight_value = []

    def loss_function(self, true_mel, output_mel, true_label, pred_label, weight):
        receptive_field = true_mel.shape[1] - output_mel.shape[1]

        mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        loss1 = tf.reduce_mean(mse(true_mel[:, receptive_field:], output_mel), axis=1)

        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False,
                                                      reduction=tf.keras.losses.Reduction.NONE,
                                                      label_smoothing=0.1)
        loss2 = tf.reduce_mean(cce(true_label[:, receptive_field:], tf.cast(pred_label, tf.float32)), axis=1)

        w1, w2 = weight
        return w1 * loss1 + w2 * loss2, loss1, loss2

    def train(self, train_dataset, val_dataset):
        os.makedirs(os.path.join(self.args.model_dir, self.args.version), exist_ok=True)
        print(f'Start Training - {self.machine_type}, {self.args.epochs} epochs')

        best_auc = 0
        a = 0
        p = 0
        e = 0
        no_better = 0
        final_mean, final_inv_cov, final_score_mean, final_score_inv_cov = 0, 0, 0, 0
        epoch_loss = []

        for epoch in range(self.args.epochs):
            self.train_loss.reset_states()
            self.train_loss1.reset_states()
            self.train_loss2.reset_states()

            lr = CosineDecay(self.args.lr, decay_steps=len(train_dataset))
            optimizer = tf.keras.optimizers.Adam(lr)
            pbar = tqdm(train_dataset, total=len(train_dataset), ncols=200)

            self.weight_value = self.loss_weighting(self.loss_value, self.weight_value)

            for data, label in pbar:
                with tf.GradientTape() as tape:
                    output1, output2 = self.model(data, training=True)
                    loss, loss1, loss2 = self.loss_function(data, output1, label, output2, self.weight_value[-1])
                    total_loss = tf.reduce_mean(loss)
                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, self.model.trainable_variables) if grad is not None)
                self.train_loss.update_state(loss)
                self.train_loss1.update_state(loss1)
                self.train_loss2.update_state(loss2)
                pbar.set_description(f'Epoch: {epoch + 1}\tLoss: {self.train_loss.result():.4f}\tLoss1: {self.train_loss1.result():.4f}\tLoss2: {self.train_loss2.result():.4f}')
            epoch_loss.append(self.train_loss.result())
            self.visualizer.add_train_loss1(self.train_loss1.result())
            self.visualizer.add_train_loss2(self.train_loss2.result())
            self.loss_value.append([self.train_loss1.result(), self.train_loss2.result()])

            if epoch % 1 == 0:
                mean, inv_cov, score_mean, score_inv_cov = self.eval_preprocess(val_dataset)
                auc, pauc = self.eval(mean, inv_cov, score_mean, score_inv_cov)
                if auc + pauc > best_auc:
                    no_better = 0
                    best_auc = auc + pauc
                    a, p, e = auc, pauc, epoch
                    checkpoint_path = 'checkpoint_best_model'
                    utils.save_model(self.args, self.model, self.machine_type, checkpoint_path, self.visualizer)
                    print(f'Model saved! \t mean AUC: {a}, mean pAUC: {p}')
                    final_mean, final_inv_cov, final_score_mean, final_score_inv_cov = mean, inv_cov, score_mean, score_inv_cov
                else:
                    no_better += 1
                if no_better > self.args.early_stop:
                    break

        print(f'Training {self.machine_type} completed! \t Best Epoch: {e:4d}\tBest AUC: {a:3.3f}\tpAUC: {p:3.3f}')
        return final_mean, final_inv_cov, final_score_mean, final_score_inv_cov

    def eval(self, mean, inv_cov, score_mean, score_inv_cov):
        os.makedirs(os.path.join(self.args.result_dir, self.args.version), exist_ok=True)
        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))

        for id_str in machine_id_list:
            y_pred = []
            batch_data, y_true = dataset.get_mtl_class_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=True, is_seg=True)
            for batch, label, name in batch_data:
                output1, output2 = self.model(np.moveaxis(batch, 1, 2), training=False)
                error_vector = self.get_error_vector(np.moveaxis(batch, 1, 2), output1)
                score1 = self.anomaly_score1(error_vector, mean, inv_cov)
                score2 = self.anomaly_score2(label, output2)
                score = self.anomaly_score(score1, score2, score_mean, score_inv_cov)
                y_pred.extend(score)

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        return mean_auc, mean_p_auc

    def test(self, mean, inv_cov, score_mean, score_inv_cov):
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        os.makedirs(result_dir, exist_ok=True)

        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        eval_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))
        test_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.test_dir, self.machine_type, 'test'))
        machine_id_list = eval_machine_id_list + test_machine_id_list
        machine_id_list.sort()
        for id_str in machine_id_list:
            y_pred = []
            anomaly_score_csv = f'{self.args.result_dir}/{self.args.version}/anomaly_score_{self.machine_type}_{id_str}.csv'
            anomaly_score_list = []
            anomaly_score_list.append(['Name', 'Anomaly Score', 'Score 1', 'Score 2'])
            if id_str in eval_machine_id_list:
                batch_data, y_true = dataset.get_mtl_class_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=True, is_seg=True)
            else:
                batch_data, y_true = dataset.get_mtl_class_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=False, is_seg=True)
            for batch, label, name in batch_data:
                output1, output2 = self.model(np.moveaxis(batch, 1, 2), training=False)
                error_vector = self.get_error_vector(np.moveaxis(batch, 1, 2), output1)
                score1 = self.anomaly_score1(error_vector, mean, inv_cov)
                score2 = self.anomaly_score2(label, output2)
                score = self.anomaly_score(score1, score2, score_mean, score_inv_cov)
                y_pred.extend(score)
                anomaly_score_list.extend([[os.path.split(a.numpy().decode('utf-8'))[1], b, c, d] for a, b, c, d in zip(name, score, score1, score2)])

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
            id_AUC.append(auc)
            print(f'{id_str} AUC: {auc:3.3f} \t pAUC: {p_auc:3.3f}')
            utils.save_csv(anomaly_score_csv, anomaly_score_list)
            self.csv_lines.append([id_str, auc, p_auc])

        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        print(self.machine_type, 'AUC:', mean_auc, 'pAUC:', mean_p_auc)

        self.csv_lines.append(['Average'] + [mean_auc, mean_p_auc])
        self.csv_lines.append([])

        result_path = os.path.join(result_dir, f'{self.machine_type}_{self.args.result_file}')
        utils.save_csv(result_path, self.csv_lines)

    def loss_weighting(self, loss_list, weight_list):
        if len(weight_list) < 2:
            weight_list.append([1, 1])
        else:
            T = 2
            r1 = loss_list[-1][0] / loss_list[-2][0]
            r2 = loss_list[-1][1] / loss_list[-2][1]
            w1 = 2 * np.exp(r1 / T) / (np.exp(r1 / T) + np.exp(r2 / T))
            w2 = 2 * np.exp(r2 / T) / (np.exp(r1 / T) + np.exp(r2 / T))
            weight_list.append([w1, w2])
        return weight_list

    def eval_preprocess(self, val_dataset):
        error_vector_list = []
        score_list1 = []
        score_list2 = []
        for data, label in val_dataset:
            output1, output2 = self.model(data, training=False)
            error_vector = self.get_error_vector(data, output1)
            score2 = self.anomaly_score2(label, output2)
            error_vector_list.extend(error_vector)
            score_list2.extend(score2)

        error_vector_array = np.array(error_vector_list)
        mean = np.mean(error_vector_array, axis=0)
        cov = np.cov(np.transpose(error_vector_array))
        inv_cov = np.linalg.inv(cov)

        # score_list1.extend([self.anomaly_score1(v[np.newaxis], mean, inv_cov)[0] for v in error_vector_list])
        score_list1.extend(self.anomaly_score1(error_vector_array, mean, inv_cov))

        score_array = np.array([(u, v) for u, v in zip(score_list1, score_list2)])
        score_mean = np.mean(score_array, axis=0)
        score_cov = np.cov(np.transpose(score_array))
        score_inv_cov = np.linalg.inv(score_cov)

        return mean, inv_cov, score_mean, score_inv_cov

    def get_error_vector(self, data, output):
        receptive_field = data.shape[1] - output.shape[1]
        error = tf.cast(data[:, receptive_field:], tf.float32) - output
        return np.mean(error, axis=1)

    def anomaly_score1(self, error_vector, mean, inv_cov):
        x = error_vector - mean
        tmp = np.matmul(x, inv_cov)
        tmp1 = np.matmul(tmp, np.transpose(x))
        dist = np.sqrt(np.diag(tmp1))
        return dist

    def anomaly_score2(self, true, pred):
        receptive_field = true.shape[1] - pred.shape[1]
        prob = tf.cast(true[:, receptive_field:], tf.float32) * pred
        class_prob = np.sum(prob, axis=-1)
        prob_mean = np.mean(class_prob, axis=1)
        return prob_mean

    def anomaly_score(self, score1, score2, score_mean, score_inv_cov):
        score = np.array([(u, v) for u, v in zip(score1, score2)])
        x = score - score_mean
        tmp = np.matmul(x, score_inv_cov)
        tmp1 = np.matmul(tmp, np.transpose(x))
        dist = np.sqrt(np.diag(tmp1))
        return dist


class MTLClassSegTrainer(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.machine_type = kwargs['machine_type']
        self.visualizer = kwargs['visualizer']
        self.model = kwargs['model']
        self.train_loss = kwargs['train_loss']
        self.train_loss1 = kwargs['train_loss1']
        self.train_loss2 = kwargs['train_loss2']
        self.train_loss3 = kwargs['train_loss3']
        self.n_class = kwargs['n_class']
        self.csv_lines = []
        self.loss_value = []
        self.weight_value = []

    def loss_function(self, true_mel, output_mel, true_label1, pred_label1, true_label2, pred_label2, weight):
        receptive_field = true_mel.shape[1] - output_mel.shape[1]

        mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False,
                                                      reduction=tf.keras.losses.Reduction.NONE,
                                                      label_smoothing=0.1)

        loss1 = tf.reduce_mean(mse(true_mel[:, receptive_field:], output_mel), axis=1)
        loss2 = cce(true_label1, tf.cast(pred_label1, tf.float32))
        loss3 = tf.reduce_mean(cce(true_label2[:, receptive_field:], tf.cast(pred_label2, tf.float32)), axis=1)

        w1, w2, w3 = weight
        return w1 * loss1 + w2 * loss2 + w3 * loss3, loss1, loss2, loss3

    def train(self, train_dataset, val_dataset):
        os.makedirs(os.path.join(self.args.model_dir, self.args.version), exist_ok=True)
        print(f'Start Training - {self.machine_type}, {self.args.epochs} epochs')

        best_auc = 0
        a = 0
        p = 0
        e = 0
        no_better = 0
        final_mean, final_inv_cov, final_score_mean, final_score_inv_cov = 0, 0, 0, 0
        epoch_loss = []

        for epoch in range(self.args.epochs):
            self.train_loss.reset_states()
            self.train_loss1.reset_states()
            self.train_loss2.reset_states()
            self.train_loss3.reset_states()

            lr = CosineDecay(self.args.lr, decay_steps=len(train_dataset))
            optimizer = tf.keras.optimizers.Adam(lr)
            pbar = tqdm(train_dataset, total=len(train_dataset), ncols=200)

            self.weight_value = self.loss_weighting(self.loss_value, self.weight_value)

            for data, label1, label2 in pbar:
                with tf.GradientTape() as tape:
                    output1, output2, output3 = self.model([data, label1], training=True)
                    loss, loss1, loss2, loss3 = self.loss_function(data, output1, label1, output2, label2, output3, self.weight_value[-1])
                    total_loss = tf.reduce_mean(loss)
                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, self.model.trainable_variables) if grad is not None)
                self.train_loss.update_state(loss)
                self.train_loss1.update_state(loss1)
                self.train_loss2.update_state(loss2)
                self.train_loss3.update_state(loss3)
                pbar.set_description(f'Epoch: {epoch + 1}\tLoss: {self.train_loss.result():.4f}\tLoss1: {self.train_loss1.result():.4f}\tLoss2: {self.train_loss2.result():.4f}\tLoss3: {self.train_loss3.result():.4f}')
            epoch_loss.append(self.train_loss.result())
            self.visualizer.add_train_loss1(self.train_loss1.result())
            self.visualizer.add_train_loss2(self.train_loss2.result())
            self.visualizer.add_train_loss3(self.train_loss3.result())
            self.loss_value.append([self.train_loss1.result(), self.train_loss2.result(), self.train_loss3.result()])

            if epoch % 1 == 0:
                mean, inv_cov, score_mean, score_inv_cov = self.eval_preprocess(val_dataset)
                auc, pauc = self.eval(mean, inv_cov, score_mean, score_inv_cov)
                if auc + pauc > best_auc:
                    no_better = 0
                    best_auc = auc + pauc
                    a, p, e = auc, pauc, epoch
                    checkpoint_path = 'checkpoint_best_model'
                    utils.save_model(self.args, self.model, self.machine_type, checkpoint_path, self.visualizer)
                    print(f'Model saved! \t mean AUC: {a}, mean pAUC: {p}')
                    final_mean, final_inv_cov, final_score_mean, final_score_inv_cov = mean, inv_cov, score_mean, score_inv_cov
                else:
                    no_better += 1
                if no_better > self.args.early_stop:
                    break

        print(f'Training {self.machine_type} completed! \t Best Epoch: {e:4d}\tBest AUC: {a:3.3f}\tpAUC: {p:3.3f}')
        return final_mean, final_inv_cov, final_score_mean, final_score_inv_cov

    def eval(self, mean, inv_cov, score_mean, score_inv_cov):
        os.makedirs(os.path.join(self.args.result_dir, self.args.version), exist_ok=True)
        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))

        for id_str in machine_id_list:
            y_pred = []
            batch_data, y_true = dataset.get_mtl_class_seg_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=True)
            for batch, label1, label2, name in batch_data:
                output1, output2, output3 = self.model([np.moveaxis(batch, 1, 2), label1], training=False)
                error_vector = self.get_error_vector(np.moveaxis(batch, 1, 2), output1)
                score1 = self.anomaly_score1(error_vector, mean, inv_cov)
                score2 = self.anomaly_score2(label1, output2)
                score3 = self.anomaly_score3(label2, output3)
                score = self.anomaly_score(score1, score2, score3, score_mean, score_inv_cov)
                y_pred.extend(score)

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        return mean_auc, mean_p_auc

    def test(self, mean, inv_cov, score_mean, score_inv_cov):
        result_dir = os.path.join(self.args.result_dir, self.args.version)
        os.makedirs(result_dir, exist_ok=True)

        id_AUC, performance = [], []
        self.csv_lines.append([self.machine_type])
        self.csv_lines.append(['id', 'AUC', 'pAUC'])

        eval_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.dataset_dir, self.machine_type, 'test'))
        test_machine_id_list = utils.get_machine_id_list(os.path.join(self.args.test_dir, self.machine_type, 'test'))
        machine_id_list = eval_machine_id_list + test_machine_id_list
        machine_id_list.sort()
        for id_str in machine_id_list:
            y_pred = []
            anomaly_score_csv = f'{self.args.result_dir}/{self.args.version}/anomaly_score_{self.machine_type}_{id_str}.csv'
            anomaly_score_list = []
            anomaly_score_list.append(['Name', 'Anomaly Score', 'Score 1', 'Score 2', 'Score 3'])
            if id_str in eval_machine_id_list:
                batch_data, y_true = dataset.get_mtl_class_seg_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=True)
            else:
                batch_data, y_true = dataset.get_mtl_class_seg_eval_test_dataset(self.args, self.machine_type, id_str, self.n_class, is_eval=False)
            for batch, label1, label2, name in batch_data:
                output1, output2, output3 = self.model([np.moveaxis(batch, 1, 2), label1], training=False)
                error_vector = self.get_error_vector(np.moveaxis(batch, 1, 2), output1)
                score1 = self.anomaly_score1(error_vector, mean, inv_cov)
                score2 = self.anomaly_score2(label1, output2)
                score3 = self.anomaly_score3(label2, output3)
                score = self.anomaly_score(score1, score2, score3, score_mean, score_inv_cov)
                y_pred.extend(score)
                anomaly_score_list.extend([[os.path.split(a.numpy().decode('utf-8'))[1], b, c, d, e.numpy()] for a, b, c, d, e in zip(name, score, score1, score2, score3)])

            max_fpr = 0.1
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
            performance.append([auc, p_auc])
            id_AUC.append(auc)
            print(f'{id_str} AUC: {auc:3.3f} \t pAUC: {p_auc:3.3f}')
            utils.save_csv(anomaly_score_csv, anomaly_score_list)
            self.csv_lines.append([id_str, auc, p_auc])

        mean_auc, mean_p_auc = np.mean(np.array(performance, dtype=float), axis=0)
        print(self.machine_type, 'AUC:', mean_auc, 'pAUC:', mean_p_auc)

        self.csv_lines.append(['Average'] + [mean_auc, mean_p_auc])
        self.csv_lines.append([])

        result_path = os.path.join(result_dir, f'{self.machine_type}_{self.args.result_file}')
        utils.save_csv(result_path, self.csv_lines)

    def loss_weighting(self, loss_list, weight_list):
        if len(weight_list) < 2:
            weight_list.append([1, 1, 1])
        else:
            T = 2
            r1 = loss_list[-1][0] / loss_list[-2][0]
            r2 = loss_list[-1][1] / loss_list[-2][1]
            r3 = loss_list[-1][2] / loss_list[-2][2]
            w1 = 2 * np.exp(r1 / T) / (np.exp(r1 / T) + np.exp(r2 / T) + np.exp(r3 / T))
            w2 = 2 * np.exp(r2 / T) / (np.exp(r1 / T) + np.exp(r2 / T) + np.exp(r3 / T))
            w3 = 2 * np.exp(r3 / T) / (np.exp(r1 / T) + np.exp(r2 / T) + np.exp(r3 / T))
            weight_list.append([w1, w2, w3])
        return weight_list

    def eval_preprocess(self, val_dataset):
        error_vector_list = []
        score_list1 = []
        score_list2 = []
        score_list3 = []
        for data, label1, label2 in val_dataset:
            output1, output2, output3 = self.model([data, label1], training=False)
            error_vector = self.get_error_vector(data, output1)
            score2 = self.anomaly_score2(label1, output2)
            score3 = self.anomaly_score3(label2, output3)
            error_vector_list.extend(error_vector)
            score_list2.extend(score2)
            score_list3.extend(score3)

        error_vector_array = np.array(error_vector_list)
        mean = np.mean(error_vector_array, axis=0)
        cov = np.cov(np.transpose(error_vector_array))
        inv_cov = np.linalg.inv(cov)

        score_list1.extend(self.anomaly_score1(error_vector_array, mean, inv_cov))

        score_array = np.array([(u, v, w) for u, v, w in zip(score_list1, score_list2, score_list3)])
        score_mean = np.mean(score_array, axis=0)
        score_cov = np.cov(np.transpose(score_array))
        score_inv_cov = np.linalg.inv(score_cov)

        return mean, inv_cov, score_mean, score_inv_cov

    def get_error_vector(self, data, output):
        receptive_field = data.shape[1] - output.shape[1]
        error = tf.cast(data[:, receptive_field:], tf.float32) - output
        return np.mean(error, axis=1)

    def anomaly_score1(self, error_vector, mean, inv_cov):
        x = error_vector - mean
        tmp = np.matmul(x, inv_cov)
        tmp1 = np.matmul(tmp, np.transpose(x))
        dist = np.sqrt(np.diag(tmp1))
        return dist

    def anomaly_score2(self, true, pred):
        score = tf.reduce_sum(tf.cast(true, tf.float32) * pred, axis=1)
        return score

    def anomaly_score3(self, true, pred):
        receptive_field = true.shape[1] - pred.shape[1]
        prob = tf.cast(true[:, receptive_field:], tf.float32) * pred
        class_prob = np.sum(prob, axis=-1)
        prob_mean = np.mean(class_prob, axis=1)
        return prob_mean

    def anomaly_score(self, score1, score2, score3, score_mean, score_inv_cov):
        score = np.array([(u, v, w) for u, v, w in zip(score1, score2, score3)])
        x = score - score_mean
        tmp = np.matmul(x, score_inv_cov)
        tmp1 = np.matmul(tmp, np.transpose(x))
        dist = np.sqrt(np.diag(tmp1))
        return dist
