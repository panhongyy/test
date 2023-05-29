from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import mymodel
import tensorflow as tf
import json
import os

def main():
    data_root = 'D:/'
    image_path = os.path.join(data_root, "oct dataset", "test OCT2017")  # flower data set path
    train_dir = os.path.join(image_path, "train")
    validation_dir = os.path.join(image_path, "val")
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    if not os.path.exists("save_weights"):   # 创建文件夹保存权重数据,判断当前路径下是否有save_weights
        os.makedirs("save_weights")          # 判断是否有此文件夹，没有就创建

    im_height = 300
    im_width = 500
    # batch_size = 60
    batch_size = 30
    epochs = 10

    train_image_generator = ImageDataGenerator(rescale=1. / 255,  # 对训练集图片进行缩放，从0-255缩小为0-1
                                               horizontal_flip=False)
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)

    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                               # 指向训练集目录，通过train_image_generator.flow_from_directory函数读取文件
                                                               batch_size=batch_size,  # 定义batch_size
                                                               shuffle=True,  # 随机打乱设置为是
                                                               target_size=(im_height, im_width),  # 定义目标尺寸，即输入网络的图像大小
                                                               class_mode='categorical')  # 模型设置为分类的方式
    total_train = train_data_gen.n

    class_indices = train_data_gen.class_indices

    inverse_dict = dict((val, key) for key, val in class_indices.items())
    json_str = json.dumps(inverse_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,  # 设置验证集生成器
                                                                  batch_size=batch_size,
                                                                  shuffle=False,
                                                                  target_size=(im_height, im_width),
                                                                  class_mode='categorical')
    total_val = val_data_gen.n
    print("using {} images for training, {} images for validation.".format(total_train, total_val))


    model = mymodel(im_height=im_height, im_width=im_width, num_class=4)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  # 因为已经在模型里使用了softmax函数，所以from_logits=False
                  metrics=["accuracy"])  # metrics代表需要监控的指标，这里监控准确率

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/mymodel.h5',   # 定义了一个回调函数的列表，tf.keras.callbacks.ModelCheckpoint用来控制模型运行过程中保存模型的参数
                                                    save_best_only=True,                   # 保存格式.keras里的为.h5, tensorflow中的为.ckpt格式
                                                    save_weights_only=True,                # 只保存效果最好的权重
                                                    monitor='val_loss')]                   # 监控验证集的损失
    history = model.fit(x=train_data_gen,  # x=训练集的生成器
                        steps_per_epoch=total_train // batch_size,  # 每一轮要迭代多少次
                        epochs=epochs,
                        validation_data=val_data_gen,
                        validation_steps=total_val // batch_size,
                        callbacks=callbacks)

    history_dict = history.history  # 通过history.history，可以获取history_dict训练字典
    train_loss = history_dict["loss"]  # 训练字典中有训练集的损失等参数
    train_accuracy = history_dict["accuracy"]
    val_loss = history_dict["val_loss"]
    val_accuracy = history_dict["val_accuracy"]

    # figure 1
    plt.figure()
    plt.plot(range(epochs), train_loss, label='train_loss')
    plt.plot(range(epochs), val_loss, label='val_loss')  # 将训练损失和验证损失绘制在一张图片上
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')

    # figure 2
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='train_accuracy')  # 将训练准确率和验证准确率绘制在一张图片上
    plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()



if __name__ == '__main__':
    main()
