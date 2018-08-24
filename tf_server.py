import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def server_start(tf_num):
    worker_hosts = []
    port = 10935
    for i in range(tf_num):
        worker = "localhost:%d" % (port + i)
        worker_hosts.append(worker)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    # config.log_device_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    config.gpu_options.allow_growth = True

    cluster_spec = tf.train.ClusterSpec({"worker": worker_hosts})

    servers = []
    for i in range(len(worker_hosts)):
        servers.append(tf.train.Server(cluster_spec, job_name="worker", task_index=i, config=config))
    for i in servers:
        i.join()


if __name__ == '__main__':
    server_start(1)
