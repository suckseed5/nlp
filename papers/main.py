import copy, numpy as np

np.random.seed(0)


# 定义一个非线性函数以及这个非线性函数的导数

def sigmoid(x):
    output = 1 / (1 + np.exp(-x))

    return output


# convert output of sigmoid function to its derivative

def sigmoid_output_to_derivative(output):
    return output * (1 - output)


# training dataset generation
#定义一个查找表，是实数与对应二进制表示的映射
int2binary = {}
#设置了二进制书的最大长度8
binary_dim = 8
#可以表示的最大的十进制数
largest_number = pow(2, binary_dim)
#生成十进制转化为二进制的查找表
binary = np.unpackbits(

    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)

for i in range(largest_number):
    int2binary[i] = binary[i]

# input variables
#设置学习速率
alpha = 0.1
#因为网络需要两个输入 也就是两个字符
input_dim = 2
#设定隐藏层的大小
hidden_dim = 16
#由于在整个过程中只需要一个输出，所以设定其数据尺寸为1
output_dim = 1

# initialize neural network weights
#这个权值矩阵链接了输入层与隐藏层
synapse_0 = 2 * np.random.random((input_dim, hidden_dim)) - 1
#这个权值矩阵链接了输出层与隐藏层
synapse_1 = 2 * np.random.random((hidden_dim, output_dim)) - 1
#这个权值矩阵链接了不同时刻的两个隐藏层
synapse_h = 2 * np.random.random((hidden_dim, hidden_dim)) - 1

#下面三行代码储存更新的权值
synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# 训练逻辑  迭代训练10000次

for j in range(10000):

    # generate a simple addition problem (a + b = c)
#控制数值范围 参与加法的两个数据不能超过最大值的一半，主要目的是为了防止数据溢出
    a_int = np.random.randint(largest_number / 2)  # int version
#查找获取输入的二进制表示，并将其存入到变量a中
    a = int2binary[a_int]  # binary encoding
#原理同64行代码
    b_int = np.random.randint(largest_number / 2)  # int version
#原理同66行代码
    b = int2binary[b_int]  # binary encoding

    # true answer
#计算加法的正确结果
    c_int = a_int + b_int
    #将正确的结果转化为二进制
    c = int2binary[c_int]

    # where we'll store our best guess (binary encoded)
#初始化一个空的二维数组，用于存储神经网络的预测值，也就是输出值
    d = np.zeros_like(c)
#将误差值设定为0（重置误差值）
    overallError = 0
#下面的两个列表将不停计算layer2的导数值与layer1的值
    layer_2_deltas = list()
    layer_1_values = list()
#在0时刻并不存在隐藏层，所以初始化一个空的隐藏层
    layer_1_values.append(np.zeros(hidden_dim))

    # moving along the positions in the binary encoding
#设定一个循环，循环的次数就是二进制数据的长度，对二进制数进行遍历
    for position in range(binary_dim):
        # generate input and output
#x数组中的每个元素包含两个二进制数，是变量ab从右向左进行检索的过程。
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])

        y = np.array([[c[binary_dim - position - 1]]]).T

        # hidden layer (input ~+ prev_hidden)

        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))

        # output layer (new binary representation)

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # did we miss?... if so by how much?

        layer_2_error = y - layer_2

        layer_2_deltas.append((layer_2_error) * sigmoid_output_to_derivative(layer_2))

        overallError += np.abs(layer_2_error[0])

        # decode estimate so we can print it out

        d[binary_dim - position - 1] = np.round(layer_2[0][0])

        # store hidden layer so we can use it in the next timestep

        layer_1_values.append(copy.deepcopy(layer_1))

    future_layer_1_delta = np.zeros(hidden_dim)

    for position in range(binary_dim):
        X = np.array([[a[position], b[position]]])

        layer_1 = layer_1_values[-position - 1]

        prev_layer_1 = layer_1_values[-position - 2]

        # error at output layer

        layer_2_delta = layer_2_deltas[-position - 1]

        # error at hidden layer

        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + \
 \
                         layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # let's update all our weights so we can try again

        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)

        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)

        synapse_0_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    synapse_0 += synapse_0_update * alpha

    synapse_1 += synapse_1_update * alpha

    synapse_h += synapse_h_update * alpha

    synapse_0_update *= 0

    synapse_1_update *= 0

    synapse_h_update *= 0

    # print out progress

    if (j % 1000 == 0):

        print ("error"+str(overallError))
        print ("pred"+str(d))
        print ("true"+str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print (str(a_int) + " + " + str(b_int) + " = " + str(out))
        print ("------------")