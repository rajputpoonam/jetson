python freeze_graph.py --input_graph=mnist_net.pb --input_checkpoint=./training_1/cp.ckpt --output_graph=./frozen_graph.pb --output_node_names=softmax
