import keras
from model_cls import PointNet
from data_loader import DataGenerator

nb_classes = 40
test_file = './ModelNet40/ply_data_test.h5'
epochs = 100
batch_size = 32
model = PointNet(nb_classes)
val = DataGenerator(test_file, batch_size, nb_classes, train=False)
model.load_weights("./results/pointnet.h5")
loss, acc = model.evaluate_generator(val.generator(), verbose=1)
print("loss is {}, and acc is {}".format(loss, acc))

