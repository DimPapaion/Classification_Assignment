import numpy as np 
import torch.nn as nn


def pad_image(A, size):
    # Apply equal padding to all sides
    A_pad = np.zeros((A.shape[0] + 2 * size, A.shape[1] + 2 * size), dtype=np.float32)
    A_pad[1 * size:-1 * size,1 * size:-1 * size] = A
    return A_pad




class MyConv2D(object):
    def __init__(self, in_channels, out_channels, kernel, stride, padding=0):
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.pad = padding
        self.weights = np.random.randn(out_channels, in_channels, kernel[0], kernel[1]) * 0.01
        self.bias = np.zeros(out_channels)


    def myConv2D(self, x):
    

        batch_size, channels, x_height, x_width = x.shape
      
        xKernShape = self.kernel[0]
        yKernShape = self.kernel[1]

     
        
        
        pad = self.pad
        strides = self.stride
       

        xOutput = int(((x_height - xKernShape + 2 * pad) / strides) + 1)
        yOutput = int(((x_width - yKernShape + 2 * pad) / strides) + 1)

        output = np.zeros((batch_size, self.out_channels, xOutput, yOutput),dtype=np.float32)  # Filling with zeros our new output matrix

        if self.pad > 0 :
            x = pad_image(x, self.pad)
       

        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for x_h in range(xOutput):
                    for x_w in range(yOutput):
                        x_h_ = x_h * strides
                        x_w_ = x_w * strides
                        oper = x[b, :, x_h_: x_h_ + xKernShape, x_w_: x_w_ + yKernShape]
                        output[b,c_out, x_h,  x_w] = (oper * self.weights[c_out]).sum() + self.bias[c_out] 

        return output



class MyLinear(object):
    def __init__(self, in_feats, out_feats):
        self.weights = np.random.randn(in_feats, out_feats) * 0.01
        self.bias = np.zeros(out_feats)
    
    def forward(self, x):
        return np.dot(x, self.weights) + self.bias



class MyModel(object):
    def __init__(self, conv, fc):

        self.conv_1 = conv(3, 8, (3,3), 1)
        self.conv_2 = conv(8, 16, (3,3), 1)

        self.fc = fc(12544, 1024)
        self.fc_2 = fc(1024, 256)
        self.fc_3 = fc(256, 5)

    def forward(self, x):
        x = self.conv_1.myConv2D(x)
        x = self.conv_2.myConv2D(x)
        # print(x.shape)
        x = x.reshape(x.shape[0], -1)
        # print(x.shape)
        x = self.fc.forward(x)
        x = self.fc_2.forward(x)
        x = self.fc_3.forward(x)
        return x
        
class BCELoss(object):
  def __init__(self, num_classes=5):
    super(BCELoss, self).__init__()
    # self.bceloss = nn.BCELoss()
    self.num_classes = num_classes

  def forward(self, x, y, avg=True):
    x = np.clip(x, a_min=0.00001, a_max=0.99999)
    loss = y * np.log(x) + (1 - y) * np.log(1 - x)

    return -loss.mean() if avg else -loss.sum()
        

x = np.random.randn(4, 3, 32, 32)
y_true = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0]
])
print(y_true.shape)
ce_loss = BCELoss(num_classes=5)
model = MyModel(MyConv2D, MyLinear)


probs = model.forward(x)

loss = ce_loss.forward(probs, y_true)
print("Loss:", loss)
print("Probs:", probs)

