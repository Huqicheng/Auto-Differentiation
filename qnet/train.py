from qnet.autodiff import gradients, Executor
from qnet.data import BatchIterator


def compile(net, inputs, targets, loss):
    # computation graph
    y = net.forward(inputs, training=True)

    # get loss
    loss_node = loss.loss(y, targets)

    # get gradient nodes
    print(net.get_all_placeholders())
    grad_node_list = gradients(loss_node, net.get_all_placeholders())

    return inputs, targets, loss_node, grad_node_list


def train(net, inputs, targets, loss, grad_node_list,
          data_x,
          data_y,
          num_epochs=1,
          iterator=BatchIterator()):
    node_list = grad_node_list.copy()
    node_list.insert(0,loss)
    print(node_list)
    executor = Executor(node_list)
    feed_dict = net.get_params()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(data_x, data_y):
            feed_dict[inputs] = batch["inputs"]
            feed_dict[targets] = batch["targets"]
            grads = executor.run(feed_dict=feed_dict)
            params = net.get_params_as_list()
            epoch_loss += grads[0][0]
            # print('\t epoch_loss:', grads[0][0])
            grads = grads[1:]
            for i, grad in enumerate(grads):
                params[i] -= 0.2 * grad
        if epoch % 100 == 0:
            print("epoch:", epoch, ", loss: ", epoch_loss)
            


